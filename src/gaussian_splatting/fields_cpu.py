"""
fields_cpu.py - the five fields in closed form, on the CPU (reference)
======================================================================
The run-averaged fields of the paper at any set of points (the seeds of a run, or a display grid),
from the positions and velocities of N agents over T instants:

    rho   density of the Gaussian footprints (agents within the window of kappa sigma)
    K     kinetic intensity: the footprints weighted by |v|^2 / <v^2>
    phi   divergence potential and psi vorticity potential of the closed-form Helmholtz-Hodge
          decomposition, through the radial kernel
          kappa_sigma(d) = sigma^2 (1 - exp(-|d|^2 / (2 sigma^2))) / |d|^2, kappa_sigma(0) = 1/2,
          summed over all the agents
    J     the Jacobi coupling sigma^2 det(grad phi, grad psi), with the gradients in closed form

fields_gpu.py computes the same on the GPU; analysis/verify_code.py checks that they agree.

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations
from typing import Iterable
import numpy as np

# ── Field registries ──────────────────────────────────────────────
AVAILABLE_FIELDS: tuple = ("rho", "K", "phi", "psi", "J")
POTENTIAL_FIELDS: tuple = ("phi", "psi")

# ==================================================================
# 1. Internal: prepare shared quantities
# ==================================================================

def _prepare_batch(
    pos: np.ndarray,
    vel: np.ndarray,
    grid,
    kappa: float,
    global_N: int,
    v2_mean: float,
) -> dict:
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    T, N, _ = pos.shape

    sigma  = grid.sigma
    sigma2 = float(sigma) ** 2
    Ny, Nx = grid.shape
    gx = grid.X.ravel().astype(np.float32)
    gy = grid.Y.ravel().astype(np.float32)

    dx = gx[None, :, None] - pos[:, None, :, 0]
    dy = gy[None, :, None] - pos[:, None, :, 1]
    d2 = dx * dx + dy * dy

    Vi    = np.exp(-d2 / (2.0 * sigma2))
    valid = d2 < (kappa * sigma) ** 2
    Vi_m  = np.where(valid, Vi, 0.0)

    ever_in_domain = np.any(np.any(valid, axis=2), axis=0)

    vx = vel[:, None, :, 0]
    vy = vel[:, None, :, 1]
    v2 = vx * vx + vy * vy

    rho_t = Vi_m.sum(axis=2) / global_N
    K_t   = (Vi_m * v2).sum(axis=2) / (global_N * v2_mean)

    return dict(
        T=T, N=N, Ny=Ny, Nx=Nx, sigma=sigma, sigma2=sigma2,
        dx=dx, dy=dy, d2=d2, Vi=Vi, valid=valid, Vi_m=Vi_m,
        ever_in_domain=ever_in_domain,
        vx=vx, vy=vy, v2=v2,
        rho_t=rho_t, K_t=K_t,
        global_N=global_N,
    )


def _field_rho_sum(b: dict) -> np.ndarray:
    return b["rho_t"].sum(axis=0)


def _field_K_sum(b: dict) -> np.ndarray:
    return b["K_t"].sum(axis=0)


def _fields_HHD_with_jacobi_sums(b: dict, vref: float) -> dict:
    d2_safe = np.where(b["d2"] < 1e-12, 1e-12, b["d2"])
    g = b["sigma2"] * (-np.expm1(-d2_safe / (2.0 * b["sigma2"])) / d2_safe)   # kappa(0) = 1/2

    sigma_val = np.sqrt(b["sigma2"])
    norm = 1.0 / (b["global_N"] * sigma_val * vref)

    vdot_d = b["vx"] * b["dx"] + b["vy"] * b["dy"]
    det_vd = b["vx"] * b["dy"] - b["vy"] * b["dx"]

    phi_raw = (vdot_d * g).sum(axis=2)
    psi_raw = -(det_vd * g).sum(axis=2)

    phi_t = norm * phi_raw
    psi_t = norm * psi_raw

    term2 = (b["Vi"] - 2.0 * g) / d2_safe

    df_dx = b["vx"] * g + vdot_d * b["dx"] * term2
    df_dy = b["vy"] * g + vdot_d * b["dy"] * term2

    dh_dx = -b["vy"] * g + det_vd * b["dx"] * term2
    dh_dy =  b["vx"] * g + det_vd * b["dy"] * term2

    grad_phi_x = norm * df_dx.sum(axis=2)
    grad_phi_y = norm * df_dy.sum(axis=2)
    
    grad_psi_x = -norm * dh_dx.sum(axis=2)
    grad_psi_y = -norm * dh_dy.sum(axis=2)

    return {
        "phi": phi_t.sum(axis=0),
        "psi": psi_t.sum(axis=0),
        "grad_phi_x": grad_phi_x.sum(axis=0),
        "grad_phi_y": grad_phi_y.sum(axis=0),
        "grad_psi_x": grad_psi_x.sum(axis=0),
        "grad_psi_y": grad_psi_y.sum(axis=0),
    }


class FieldComputer:
    def __init__(self, fields: Iterable[str], kappa: float = 5.0, chunk_size: int = 100):
        self.fields: tuple = tuple(fields)
        self.kappa = float(kappa)
        self.chunk_size = int(chunk_size)
        self._needs_potentials = any(k in self.fields for k in ("phi", "psi", "J"))

    def compute(self, pos: np.ndarray, vel: np.ndarray, grid) -> dict:
        T = pos.shape[0]
        N = pos.shape[1]
        G = grid.n_vertices

        v2_all = (np.asarray(vel, dtype=np.float64) ** 2).sum(-1)     # norms in float64, as on the GPU
        vref = float(np.sqrt(v2_all).mean())
        v2_mean = float(v2_all.mean())
        if vref < 1e-12: vref = 1e-12
        if v2_mean < 1e-12: v2_mean = 1e-12

        acc = {k: np.zeros(G, dtype=np.float64) for k in self.fields if k != "J"}
        if "J" in self.fields:
            acc["grad_phi_x"] = np.zeros(G, dtype=np.float64)
            acc["grad_phi_y"] = np.zeros(G, dtype=np.float64)
            acc["grad_psi_x"] = np.zeros(G, dtype=np.float64)
            acc["grad_psi_y"] = np.zeros(G, dtype=np.float64)

        ever_in = np.zeros(G, dtype=bool)

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            
            b = _prepare_batch(pos[start:end], vel[start:end], grid, self.kappa, N, v2_mean)
            ever_in |= b["ever_in_domain"]

            if "rho" in self.fields:
                acc["rho"] += _field_rho_sum(b).astype(np.float64).ravel()
            if "K" in self.fields:
                acc["K"] += _field_K_sum(b).astype(np.float64).ravel()
            
            if self._needs_potentials:
                pot_sums = _fields_HHD_with_jacobi_sums(b, vref)
                if "phi" in self.fields:
                    acc["phi"] += pot_sums["phi"].astype(np.float64).ravel()
                if "psi" in self.fields:
                    acc["psi"] += pot_sums["psi"].astype(np.float64).ravel()
                if "J" in self.fields:
                    acc["grad_phi_x"] += pot_sums["grad_phi_x"].astype(np.float64).ravel()
                    acc["grad_phi_y"] += pot_sums["grad_phi_y"].astype(np.float64).ravel()
                    acc["grad_psi_x"] += pot_sums["grad_psi_x"].astype(np.float64).ravel()
                    acc["grad_psi_y"] += pot_sums["grad_psi_y"].astype(np.float64).ravel()

        out = {}
        for k in self.fields:
            if k == "J":
                gpx = acc["grad_phi_x"] / T
                gpy = acc["grad_phi_y"] / T
                gsx = acc["grad_psi_x"] / T
                gsy = acc["grad_psi_y"] / T
                val = float(grid.sigma)**2 * (gpx * gsy - gpy * gsx)
                val = np.where(ever_in, val, np.nan)
                out[k] = val.reshape(grid.shape).astype(np.float32)
            else:
                mean = np.where(ever_in, acc[k] / T, np.nan)
                out[k] = mean.reshape(grid.shape).astype(np.float32)

        return out
