"""
fields_gpu.py - the five fields on the GPU, one fused kernel
============================================================
The same closed forms as fields_cpu.py, evaluated at any set of points (the seeds of a run, or the
vertices of a display grid) for a batch of runs at once. One Triton kernel holds, for a block of
points, the eight sums of the fields in registers and runs over the instants and the agents; nothing
of size (instants x points x agents) is ever stored, so the cost is the arithmetic alone.

Precision: each term in float32, the sum over the agents of one instant in float32, the sum over the
instants in float64 (the agreement with the CPU code, to float32 rounding, is checked by
analysis/verify_code.py; analysis/precision_check.py measures that float32 terms are enough).

    rho   = (1/(N T)) sum_t sum_i exp(-|r - x_i|^2 / (2 sigma^2))             (agents within the window)
    K     = (1/(N T <v^2>)) sum_t sum_i |v_i|^2 exp(...)                       (agents within the window)
    phi   = (1/(N T sigma vbar)) sum_t sum_i  (v_i . d) kappa_sigma(d),        d = r - x_i
    psi   = (1/(N T sigma vbar)) sum_t sum_i -(v_i x d) kappa_sigma(d)
    J     = sigma^2 det(grad phi, grad psi), with the gradients in closed form
    kappa_sigma(d) = sigma^2 (1 - exp(-|d|^2 / (2 sigma^2))) / |d|^2, kappa_sigma(0) = 1/2

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    torch = None

from .fields_cpu import AVAILABLE_FIELDS

BLOCK = 32             # points per program

if torch is not None:
    @triton.jit
    def _fields_kernel(P, V, GX, GY, OUT, EVER, T, N, G, inv2s2, sigma2, win2,
                       NP: tl.constexpr, BLOCK: tl.constexpr, DENS: tl.constexpr, POT: tl.constexpr,
                       JAC: tl.constexpr):
        """The eight sums of the fields at a block of points of run b; OUT (B, 8, G) float64."""
        b = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        gm = offs < G
        gx = tl.load(GX + b * G + offs, mask=gm, other=0.0)
        gy = tl.load(GY + b * G + offs, mask=gm, other=0.0)
        n = tl.arange(0, NP)
        nm = n < N
        live = nm[None, :]
        a_rho = tl.zeros([BLOCK], dtype=tl.float64)
        a_k = tl.zeros([BLOCK], dtype=tl.float64)
        a_phi = tl.zeros([BLOCK], dtype=tl.float64)
        a_psi = tl.zeros([BLOCK], dtype=tl.float64)
        a_fx = tl.zeros([BLOCK], dtype=tl.float64)
        a_fy = tl.zeros([BLOCK], dtype=tl.float64)
        a_hx = tl.zeros([BLOCK], dtype=tl.float64)
        a_hy = tl.zeros([BLOCK], dtype=tl.float64)
        ever = tl.zeros([BLOCK], dtype=tl.int32)
        base = b * T * N * 2
        for t in range(T):
            idx = base + (t * N + n) * 2
            px = tl.load(P + idx, mask=nm, other=0.0)
            py = tl.load(P + idx + 1, mask=nm, other=0.0)
            vx = tl.load(V + idx, mask=nm, other=0.0)
            vy = tl.load(V + idx + 1, mask=nm, other=0.0)
            dx = gx[:, None] - px[None, :]
            dy = gy[:, None] - py[None, :]
            d2 = dx * dx + dy * dy
            valid = (d2 < win2) & live
            ever = tl.maximum(ever, tl.max(valid.to(tl.int32), axis=1))
            e = libdevice.exp(-d2 * inv2s2)
            if DENS:
                em = tl.where(valid, e, 0.0)
                a_rho += tl.sum(em, axis=1).to(tl.float64)
                a_k += tl.sum(em * (vx * vx + vy * vy)[None, :], axis=1).to(tl.float64)
            if POT:
                d2s = tl.maximum(d2, 1e-12)
                kap = tl.where(live, sigma2 * (-libdevice.expm1(-d2s * inv2s2)) / d2s, 0.0)
                u = vx[None, :] * dx + vy[None, :] * dy
                w = vx[None, :] * dy - vy[None, :] * dx
                a_phi += tl.sum(u * kap, axis=1).to(tl.float64)
                a_psi -= tl.sum(w * kap, axis=1).to(tl.float64)
                if JAC:
                    t2 = tl.where(live, (e - 2.0 * kap) / d2s, 0.0)
                    a_fx += tl.sum(vx[None, :] * kap + u * dx * t2, axis=1).to(tl.float64)
                    a_fy += tl.sum(vy[None, :] * kap + u * dy * t2, axis=1).to(tl.float64)
                    a_hx -= tl.sum(-vy[None, :] * kap + w * dx * t2, axis=1).to(tl.float64)
                    a_hy -= tl.sum(vx[None, :] * kap + w * dy * t2, axis=1).to(tl.float64)
        o = OUT + b * 8 * G + offs
        tl.store(o, a_rho, mask=gm)
        tl.store(o + G, a_k, mask=gm)
        tl.store(o + 2 * G, a_phi, mask=gm)
        tl.store(o + 3 * G, a_psi, mask=gm)
        tl.store(o + 4 * G, a_fx, mask=gm)
        tl.store(o + 5 * G, a_fy, mask=gm)
        tl.store(o + 6 * G, a_hx, mask=gm)
        tl.store(o + 7 * G, a_hy, mask=gm)
        tl.store(EVER + b * G + offs, ever, mask=gm)


class TorchFieldComputer:
    """The fields of `fields` at points, on the GPU. `kappa` is the window of rho and K, in sigma
    (points farther than kappa * sigma from every agent at every instant get NaN)."""

    def __init__(self, fields: Iterable[str], kappa: float = 5.0, device: str | None = None):
        if torch is None:
            raise ImportError("torch and triton are needed for TorchFieldComputer")
        self.fields = tuple(fields)
        unknown = set(self.fields) - set(AVAILABLE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown fields: {sorted(unknown)}. Available: {AVAILABLE_FIELDS}")
        self.kappa = float(kappa)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _evaluate(self, pos, vel, gx, gy, sigma):
        """pos, vel (B, T, N, 2) float32 and points gx, gy (B, G) float32, on the device."""
        B, T, N, _ = pos.shape
        G = gx.shape[1]
        out = torch.empty((B, 8, G), dtype=torch.float64, device=self.device)
        ever = torch.empty((B, G), dtype=torch.int32, device=self.device)
        s = float(sigma)
        pot = any(k in self.fields for k in ("phi", "psi", "J"))
        _fields_kernel[(B, triton.cdiv(G, BLOCK))](
            pos, vel, gx, gy, out, ever, T, N, G, 1.0 / (2.0 * s * s), s * s, (self.kappa * s) ** 2,
            NP=triton.next_power_of_2(N), BLOCK=BLOCK, DENS=any(k in self.fields for k in ("rho", "K")),
            POT=pot, JAC="J" in self.fields, num_warps=4)
        v2 = (vel.double() ** 2).sum(-1)
        vbar = torch.sqrt(v2).mean(dim=(1, 2)).clamp(min=1e-12)[:, None]
        v2m = v2.mean(dim=(1, 2)).clamp(min=1e-12)[:, None]
        norm = 1.0 / (N * s * vbar * T)
        res = {}
        if "rho" in self.fields:
            res["rho"] = out[:, 0] / (N * T)
        if "K" in self.fields:
            res["K"] = out[:, 1] / (N * v2m * T)
        if "phi" in self.fields:
            res["phi"] = out[:, 2] * norm
        if "psi" in self.fields:
            res["psi"] = out[:, 3] * norm
        if "J" in self.fields:
            res["J"] = s * s * ((out[:, 4] * norm) * (out[:, 7] * norm) - (out[:, 5] * norm) * (out[:, 6] * norm))
        nan = torch.tensor(float("nan"), dtype=torch.float64, device=self.device)
        return {k: torch.where(ever > 0, v, nan).cpu().numpy().astype(np.float32) for k, v in res.items()}

    def compute(self, pos: np.ndarray, vel: np.ndarray, grid) -> dict:
        """The fields of one run at the vertices of `grid` (attributes X, Y, shape, sigma)."""
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=self.device)
        res = self._evaluate(t(pos)[None], t(vel)[None], t(np.ravel(grid.X))[None], t(np.ravel(grid.Y))[None], grid.sigma)
        return {k: v[0].reshape(grid.shape) for k, v in res.items()}

    def compute_batch_points(self, pos_list, vel_list, pe_list) -> list[dict]:
        """The fields of B runs, each at its own points (pe.X, pe.Y; one sigma). Runs of different
        lengths are evaluated by groups of equal length."""
        B = len(pos_list)
        lengths = {np.shape(p)[:2] for p in pos_list}
        if len(lengths) > 1:
            out = [None] * B
            for shape in lengths:
                ids = [b for b in range(B) if np.shape(pos_list[b])[:2] == shape]
                for b, r in zip(ids, self.compute_batch_points([pos_list[b] for b in ids], [vel_list[b] for b in ids],
                                                               [pe_list[b] for b in ids])):
                    out[b] = r
            return out
        G = max(1, max(pe.n_vertices for pe in pe_list))
        gx = np.zeros((B, G), dtype=np.float32)
        gy = np.zeros((B, G), dtype=np.float32)
        for b, pe in enumerate(pe_list):
            gx[b, :pe.n_vertices] = np.ravel(pe.X)
            gy[b, :pe.n_vertices] = np.ravel(pe.Y)
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=self.device)
        res = self._evaluate(t(np.stack(pos_list)), t(np.stack(vel_list)), t(gx), t(gy), pe_list[0].sigma)
        return [{k: v[b, :pe.n_vertices].reshape(1, pe.n_vertices) for k, v in res.items()}
                for b, pe in enumerate(pe_list)]
