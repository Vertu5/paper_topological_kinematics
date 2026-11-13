#!/usr/bin/env python3
"""
precision_check.py - where is float64 needed?
=============================================
The heavy parts of stage 1 compute in float32. This script measures, on real runs, what float32 changes
against a float64 reference:
  1. the five fields at the seeds (relative difference);
  2. downstream, the sliced-Wasserstein distance between the diagrams computed from float32 fields and
     from float64 fields, relative to the median distance between two runs of the same mission;
  3. the zeros of J: the signs of J at the coreset vertices (which decide the edges to bisect) and the
     positions of the zeros after 12 bisection steps, in units of sigma.

Author: Olivier Vertu Ndinga Oba
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from gaussian_splatting.alpha import AlphaSimplexTree  # noqa: E402
from gaussian_splatting.persistence import PersistenceComputer  # noqa: E402
from gaussian_splatting.distance import sample_directions, swd  # noqa: E402
from gaussian_splatting.seeds_gridfree import _eval_J_local, delaunay_structure  # noqa: E402

DIRS = sample_directions(25)
FIELDS = ("rho", "K", "phi", "psi", "J")


def fields_at(S, pos, vel, sigma, dtype, chunk=40, window=5.0):
    """The five fields at points S, every operation in `dtype`, sums in float64."""
    S = np.asarray(S, dtype); pos = np.asarray(pos, dtype); vel = np.asarray(vel, dtype)
    T, N, _ = pos.shape
    s2 = dtype(sigma) ** 2
    sp2 = (vel ** 2).sum(-1)
    vbar = float(np.sqrt(sp2.astype(np.float64)).mean()); v2bar = float(sp2.astype(np.float64).mean())
    acc = np.zeros((len(S), 8)); ever = np.zeros(len(S), bool)
    for t0 in range(0, T, chunk):
        p, v = pos[t0:t0 + chunk], vel[t0:t0 + chunk]
        dx = S[None, :, None, 0] - p[:, None, :, 0]; dy = S[None, :, None, 1] - p[:, None, :, 1]
        d2 = dx * dx + dy * dy
        d2s = np.maximum(d2, dtype(1e-12))
        Vi = np.exp(-d2 / (2 * s2)); valid = d2 < dtype(window * window) * s2
        Vm = np.where(valid, Vi, dtype(0)); ever |= valid.any(axis=(0, 2))
        g = s2 * (-np.expm1(-d2s / (2 * s2))) / d2s
        t2 = (Vi - 2 * g) / d2s
        vx, vy = v[:, None, :, 0], v[:, None, :, 1]
        u = vx * dx + vy * dy; w = vx * dy - vy * dx
        sq = vx * vx + vy * vy
        terms = (Vm, Vm * sq, u * g, -w * g, vx * g + u * dx * t2, vy * g + u * dy * t2,
                 -(-vy * g + w * dx * t2), -(vx * g + w * dy * t2))
        for k, q in enumerate(terms):
            acc[:, k] += q.sum(axis=(0, 2), dtype=np.float64)
    acc /= T
    out = {"rho": acc[:, 0] / N, "K": acc[:, 1] / (N * v2bar)}
    nrm = 1.0 / (N * sigma * vbar)
    out["phi"], out["psi"] = acc[:, 2] * nrm, acc[:, 3] * nrm
    out["J"] = sigma ** 2 * (acc[:, 4] * acc[:, 7] - acc[:, 5] * acc[:, 6]) * nrm ** 2
    for k in out:
        out[k] = np.where(ever, out[k], np.nan)
    return out


def J64(points, pos, vel, sigma, chunk=32):
    P = np.asarray(points, np.float64); pos = np.asarray(pos, np.float64); vel = np.asarray(vel, np.float64)
    s2 = sigma ** 2; vx, vy = vel[None, ..., 0], vel[None, ..., 1]; out = np.empty(len(P))
    for k0 in range(0, len(P), chunk):
        Q = P[k0:k0 + chunk]
        dx = Q[:, 0, None, None] - pos[None, ..., 0]; dy = Q[:, 1, None, None] - pos[None, ..., 1]
        r2 = dx * dx + dy * dy; r2s = np.maximum(r2, 1e-12)
        ka = s2 * (-np.expm1(-r2s / (2 * s2))) / r2s; gk = (np.exp(-r2 / (2 * s2)) - 2 * ka) / r2s
        u = vx * dx + vy * dy; w = vx * dy - vy * dx
        a = (vx * ka + u * dx * gk).sum(axis=(1, 2)); b = (vy * ka + u * dy * gk).sum(axis=(1, 2))
        c = -(-vy * ka + w * dx * gk).sum(axis=(1, 2)); d = -(vx * ka + w * dy * gk).sum(axis=(1, 2))
        out[k0:k0 + chunk] = s2 * (a * d - b * c)
    return out


def bisect(fun, pa, pb, Ja, n=12):
    lo = np.zeros(len(pa)); hi = np.ones(len(pa))
    for _ in range(n):
        mid = 0.5 * (lo + hi); Jm = fun(pa + mid[:, None] * (pb - pa))
        same = Ja * Jm > 0; lo = np.where(same, mid, lo); hi = np.where(~same, mid, hi)
    return 0.5 * (lo + hi)


def main():
    lab, rk, _ = RS.load_labels()
    D, ladder = RS.load_distances()
    idx = RS.subset(1, rk); y = lab[idx]
    meta = pickle.load(open(RS.RES / "meta.pkl", "rb"))["trajectories"]
    picks = [int(idx[np.where(y == m)[0][0]]) for m in range(5)]
    fld, down, zeros = [], [], []
    for g in picks:
        m = int(lab[g])
        for si in (0, 3):
            st = pickle.load(open(RS.RES / "stage1_alpha_fields" / f"traj{g:04d}_s{si}.pkl", "rb"))
            sig = float(st["sigma"]); S = np.asarray(st["seeds"], np.float64)
            pos, vel = meta[g]["pos"], meta[g]["vel"]
            f32 = fields_at(S, pos, vel, sig, np.float32); f64 = fields_at(S, pos, vel, sig, np.float64)
            pc = PersistenceComputer(AlphaSimplexTree(S, alpha_sq_max=(3 * sig) ** 2))
            same = [int(i) for i in idx[(y == m)] if i != g][:10]
            for f in FIELDS:
                a, b = f32[f], f64[f]
                fld.append((f, si, np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(b))))
                A, B = pc.compute_field(np.nan_to_num(a)), pc.compute_field(np.nan_to_num(b))
                d = sum(swd(A[s], [B[s]], DIRS)[0] for s in range(6))
                within = np.median(np.asarray(D[f][si])[g, same])
                down.append((f, si, d / within))
            # zeros of J: signs at the coreset vertices and positions after bisection
            core = S[:int(st.get("n_core", 300))]
            struct = delaunay_structure(core)
            if struct is None:
                continue
            _, edges = struct
            j32, j64 = _eval_J_local(core, pos, vel, sig), J64(core, pos, vel, sig)
            flips = int(np.sum(np.sign(j32) != np.sign(j64)))
            a_, b_ = edges[:, 0], edges[:, 1]
            m64 = j64[a_] * j64[b_] < 0; m32 = j32[a_] * j32[b_] < 0
            both = m64 & m32
            pa, pb = core[a_[both]], core[b_[both]]
            t32 = bisect(lambda q: _eval_J_local(q, pos, vel, sig), pa, pb, j32[a_[both]])
            t64 = bisect(lambda q: J64(q, pos, vel, sig), pa, pb, j64[a_[both]])
            shift = np.abs(t32 - t64) * np.linalg.norm(pb - pa, axis=1) / sig
            zeros.append((si, flips, len(core), int(m64.sum()), int(m32.sum()), int((m64 != m32).sum()),
                          float(np.max(shift)) if len(shift) else 0.0))
    print("1. fields at the seeds, float32 against float64 (max relative difference, worst over runs):")
    for f in FIELDS:
        for si in (0, 3):
            v = max(r for ff, s, r in fld if ff == f and s == si)
            print(f"   {f:4s} sigma_{si + 1}: {v:.1e}")
    print("2. downstream: distance between the float32 and float64 diagrams / median within-mission distance:")
    for f in FIELDS:
        v = max(r for ff, s, r in down if ff == f)
        print(f"   {f:4s}: max {v:.1e}")
    print("3. zeros of J (per run and scale): sign flips at the coreset vertices, edges with a sign change "
          "(float64 / float32 / differing), largest shift of a zero in units of sigma")
    for si, flips, n, e64, e32, diff, sh in zeros:
        print(f"   sigma_{si + 1}: {flips}/{n} vertices change sign; edges {e64}/{e32}/{diff}; max shift {sh:.1e} sigma")


if __name__ == "__main__":
    main()
