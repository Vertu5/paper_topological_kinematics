"""
seeds_gridfree.py - the seeds of a run at a scale, without a grid
==================================================================
Per run and scale sigma (batched over runs of equal length):
  1. Coreset  : farthest-point sampling among all the agent positions of the recording, until no
                position lies farther than cover * sigma from the coreset (the method: cover = 1/2, at
                most n_max points; coreset_orders_gpu and prefix_for_radius). With cover None, a fixed
                number n_core of points instead, which does not depend on sigma (coreset, coresets_gpu),
                as in the parameter study.
  2. Delaunay : triangulation of the coreset (scipy).
  3. Fields   : the five fields at the coreset vertices, in closed form.
  4. Extras   : (a) barycentres of the frac_low_grad flattest triangles of each field (the piecewise
                    linear gradient is constant on a triangle);
                (b) zeros of J on the Delaunay edges where J changes sign, by n_bis bisection steps on
                    the closed form of J (only its sign is used), at most n_jacobi of them kept by
                    farthest-point sampling.
  5. The seeds are assembled, duplicates removed (one seed per cell of a grid of side dedup_tol sigma),
     the five fields evaluated at the final seeds, and seeds whose fields are not finite (farther than
     the window from every agent at every instant) dropped.

`computer` is any object with an attribute `fields` and a method
    compute_batch_points(pos_list, vel_list, pe_list) -> list of dict, dict[field] of shape (1, G)
(fields_gpu.TorchFieldComputer, or CPUPointComputer below).

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations
from types import SimpleNamespace
import numpy as np
from scipy.spatial import Delaunay
from joblib import Parallel, delayed

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_pe(points: np.ndarray, sigma: float) -> SimpleNamespace:
    """Point evaluator understood by compute_batch_points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return SimpleNamespace(
        X=pts[:, 0].reshape(1, -1).astype(np.float32),
        Y=pts[:, 1].reshape(1, -1).astype(np.float32),
        n_vertices=len(pts),
        sigma=float(sigma),
        shape=(1, len(pts)),
    )


def fps(points: np.ndarray, n_max: int, start: int = 0) -> np.ndarray:
    """Deterministic farthest-point sampling, starting from points[start]."""
    M = len(points)
    if M <= n_max:
        return points.copy()
    idx = np.empty(n_max, dtype=np.int64)
    idx[0] = start
    d2 = ((points - points[start]) ** 2).sum(1)
    k = 1
    while k < n_max:
        i = int(np.argmax(d2))
        if d2[i] <= 0.0:
            break
        idx[k] = i
        k += 1
        np.minimum(d2, ((points - points[i]) ** 2).sum(1), out=d2)
    return points[idx[:k]]


def delaunay_structure(points: np.ndarray):
    """(simplices, unique edges) of the Delaunay triangulation, or None."""
    if len(points) < 4:
        return None
    try:
        tri = Delaunay(points)
    except (QhullError, ValueError):
        return None
    s = tri.simplices
    e = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    return s, np.unique(e, axis=0)


def low_gradient_barycenters(simplices: np.ndarray, pts: np.ndarray,
                             values: np.ndarray, frac: float) -> np.ndarray:
    """Barycentres of the frac fraction of triangles with smallest PL gradient."""
    p0, p1, p2 = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    dv1 = values[simplices[:, 1]] - values[simplices[:, 0]]
    dv2 = values[simplices[:, 2]] - values[simplices[:, 0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        gx = (e2[:, 1] * dv1 - e1[:, 1] * dv2) / det
        gy = (-e2[:, 0] * dv1 + e1[:, 0] * dv2) / det
    gn = np.hypot(gx, gy)
    gn[~np.isfinite(gn)] = np.inf
    n_keep = max(1, int(frac * len(gn)))
    order = np.argsort(gn, kind="stable")[:n_keep]
    return (p0[order] + p1[order] + p2[order]) / 3.0


def _dedup(seeds: np.ndarray, sigma: float, tol: float = 1e-3) -> np.ndarray:
    key = np.round(seeds / max(tol * sigma, 1e-12)).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return seeds[np.sort(idx)]


def coresets_gpu(pos_list, n_core, start=0, device="cuda"):
    """coreset() of several recordings at once on the GPU, with the same arithmetic (float64 squared
    distances, first index of the maximum), hence the same points bit for bit."""
    import torch
    pts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in pos_list]
    pts = [q[np.isfinite(q).all(1)] for q in pts]
    out = [None] * len(pts)
    todo = []
    for r, q in enumerate(pts):
        if len(q) <= int(n_core):
            out[r] = q.copy()
        else:
            todo.append(r)
    if not todo:
        return out
    M = max(len(pts[r]) for r in todo)
    P = np.zeros((len(todo), M, 2))
    for k, r in enumerate(todo):
        P[k, :len(pts[r])] = pts[r]
    P = torch.as_tensor(P, device=device)
    valid = torch.zeros((len(todo), M), dtype=torch.bool, device=device)
    for k, r in enumerate(todo):
        valid[k, :len(pts[r])] = True
    ar = torch.arange(len(todo), device=device)
    first = torch.as_tensor([int(start) % len(pts[r]) for r in todo], device=device)
    idx = torch.empty((len(todo), int(n_core)), dtype=torch.long, device=device)
    idx[:, 0] = first
    ninf = torch.tensor(float("-inf"), dtype=torch.float64, device=device)
    d2 = torch.where(valid, ((P - P[ar, first][:, None]) ** 2).sum(-1), ninf)
    count = torch.ones(len(todo), dtype=torch.long, device=device)
    alive = torch.ones(len(todo), dtype=torch.bool, device=device)
    for k in range(1, int(n_core)):
        i = torch.argmax(d2, dim=1)
        alive &= d2[ar, i] > 0.0                     # every remaining point already selected: stop
        idx[:, k] = i
        count += alive.long()
        d2 = torch.minimum(d2, torch.where(valid, ((P - P[ar, i][:, None]) ** 2).sum(-1), ninf))
    idx, count = idx.cpu().numpy(), count.cpu().numpy()
    for k, r in enumerate(todo):
        out[r] = pts[r][idx[k, :count[k]]]
    return out


def coreset_orders_gpu(pos_list, n_max, start=0, device="cuda", stop_radius=None):
    """Farthest-point order of the agent positions of each recording, up to n_max points, with the
    covering radius after each point: R[k-1] is the largest distance from a position to the first k
    points. The coreset that leaves no position farther than r is the shortest prefix with R <= r;
    its first n points are coresets_gpu(pos, n), bit for bit. With stop_radius, the order stops once
    every recording is covered within that radius (the prefixes needed are then all available).
    Returns a list of (points, R)."""
    import torch
    pts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in pos_list]
    pts = [q[np.isfinite(q).all(1)] for q in pts]
    M = max(len(q) for q in pts)
    n = int(min(n_max, M))
    P = np.zeros((len(pts), M, 2))
    for k, q in enumerate(pts):
        P[k, :len(q)] = q
    P = torch.as_tensor(P, device=device)
    valid = torch.zeros((len(pts), M), dtype=torch.bool, device=device)
    for k, q in enumerate(pts):
        valid[k, :len(q)] = True
    ar = torch.arange(len(pts), device=device)
    first = torch.as_tensor([int(start) % len(q) for q in pts], device=device)
    ninf = torch.tensor(float("-inf"), dtype=torch.float64, device=device)
    idx = torch.empty((len(pts), n), dtype=torch.long, device=device)
    rad = torch.empty((len(pts), n), dtype=torch.float64, device=device)
    idx[:, 0] = first
    d2 = torch.where(valid, ((P - P[ar, first][:, None]) ** 2).sum(-1), ninf)
    stop2 = None if stop_radius is None else float(stop_radius) ** 2
    for k in range(1, n):
        i = torch.argmax(d2, dim=1)
        rad[:, k - 1] = d2[ar, i]
        idx[:, k] = i
        d2 = torch.minimum(d2, torch.where(valid, ((P - P[ar, i][:, None]) ** 2).sum(-1), ninf))
        if stop2 is not None and k % 256 == 0 and bool((rad[:, k - 1] <= stop2).all()):
            n = k + 1
            idx, rad = idx[:, :n], rad[:, :n]
            break
    rad[:, n - 1] = d2.max(dim=1).values
    idx, rad = idx.cpu().numpy(), np.sqrt(np.maximum(rad.cpu().numpy(), 0.0))
    out = []
    for k, q in enumerate(pts):
        m = min(n, len(q))
        out.append((q[idx[k, :m]], rad[k, :m]))
    return out


def prefix_for_radius(R, r):
    """Length of the shortest farthest-point prefix whose covering radius is at most r (all if none)."""
    ok = np.flatnonzero(R <= r)
    return int(ok[0] + 1) if len(ok) else len(R)


def coreset(pos, n_core, start=0) -> np.ndarray:
    """Farthest-point sampling of n_core points among all the agent positions of a recording. It does
    not depend on sigma, so one coreset serves every scale."""
    pts = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) == 0:
        return pts
    return fps(pts, min(len(pts), int(n_core)), start=int(start) % len(pts))


# ---------------------------------------------------------------------------
# Local float64 evaluation of J and true bisection
# ---------------------------------------------------------------------------
def _eval_J_local(points, pos, vel, sigma, chunk=64):
    """CPU reference used by the checks (analysis/precision_check.py). J at the given points, up to a positive factor (only its sign is used). Closed form computed in
    float32 with sums in float64; an agent exactly at a point contributes with the limit
    kappa(0) = 1/2. Points are processed in blocks to bound the memory."""
    s2 = np.float32(sigma) ** 2
    P_all = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    vx, vy = vel[None, :, :, 0], vel[None, :, :, 1]
    out = np.empty(len(P_all), dtype=np.float64)
    for k0 in range(0, len(P_all), chunk):
        P = P_all[k0:k0 + chunk]
        dx = P[:, 0, None, None] - pos[None, :, :, 0]
        dy = P[:, 1, None, None] - pos[None, :, :, 1]
        r2 = dx * dx + dy * dy
        r2s = np.maximum(r2, np.float32(1e-12))
        kappa = s2 * (-np.expm1(-r2s / (2 * s2))) / r2s
        gk = (np.exp(-r2 / (2 * s2)) - 2 * kappa) / r2s
        u = vx * dx + vy * dy
        w = vx * dy - vy * dx
        gpx = (vx * kappa + u * dx * gk).sum(axis=(1, 2), dtype=np.float64)
        gpy = (vy * kappa + u * dy * gk).sum(axis=(1, 2), dtype=np.float64)
        gsx = -(-vy * kappa + w * dx * gk).sum(axis=(1, 2), dtype=np.float64)
        gsy = -(vx * kappa + w * dy * gk).sum(axis=(1, 2), dtype=np.float64)
        out[k0:k0 + chunk] = float(s2) * (gpx * gsy - gpy * gsx)
    return out


# ---------------------------------------------------------------------------
# Parallel worker: barycentres + local J + bisection for one run
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# main entry points
# ---------------------------------------------------------------------------
def _bisect_batch(computer_J, pos_list, vel_list, pa_l, pb_l, Ja_l, sigma, n_iter):
    """Bisection of J = 0 on the edges of every run at once: one evaluation of J per step for the whole
    batch (on the GPU with the torch computer). computer_J has no evaluation window, so that J is
    defined at every midpoint; only its sign is used."""
    B = len(pa_l)
    lo = [np.zeros(len(p)) for p in pa_l]
    hi = [np.ones(len(p)) for p in pa_l]
    active = [r for r in range(B) if len(pa_l[r])]
    for _ in range(n_iter if active else 0):
        mids = [0.5 * (lo[r] + hi[r]) for r in active]
        pts = [pa_l[r] + m[:, None] * (pb_l[r] - pa_l[r]) for m, r in zip(mids, active)]
        Jm = computer_J.compute_batch_points([pos_list[r] for r in active], [vel_list[r] for r in active],
                                             [make_pe(p, sigma) for p in pts])
        for k, r in enumerate(active):
            same = (Ja_l[r] * np.asarray(Jm[k]["J"], dtype=np.float64).ravel()) > 0.0
            lo[r] = np.where(same, mids[k], lo[r])
            hi[r] = np.where(~same, mids[k], hi[r])
    return [pa_l[r] + (0.5 * (lo[r] + hi[r]))[:, None] * (pb_l[r] - pa_l[r]) if len(pa_l[r]) else None
            for r in range(B)]


def seeds_from_coresets(pos_list, vel_list, coresets, computer, sigma, *,
                        computer_J=None, frac_low_grad=0.05,
                        n_jacobi=150, n_bis=12, dedup_tol=1e-3, structures=None, **_unused):
    """
    Seeds and fields of B runs at one sigma, from their coresets.

    1. the five fields at the coreset vertices (one batched evaluation);
    2. barycentres of the frac_low_grad flattest Delaunay triangles of each field, and the edges where
       J (from step 1) changes sign;
    3. bisection of J = 0 on those edges, n_bis steps, batched over the runs;
    4. at most n_jacobi zeros kept by farthest-point sampling, duplicates removed (dedup_tol * sigma),
       the five fields evaluated at the final seeds, seeds with non-finite fields dropped.
    `structures` may give the Delaunay triangulation of each coreset (it does not depend on sigma).
    """
    B = len(pos_list)
    names = tuple(getattr(computer, "fields", ("rho", "K", "phi", "psi", "J")))
    if computer_J is None:
        kw = dict(fields=("J",), kappa=1e6)
        if hasattr(computer, "device"):
            kw["device"] = computer.device
        computer_J = type(computer)(**kw)

    vals0 = computer.compute_batch_points(pos_list, vel_list, [make_pe(c, sigma) for c in coresets])

    bary, pa_l, pb_l, Ja_l = [], [], [], []
    for r in range(B):
        struct = structures[r] if structures is not None else delaunay_structure(coresets[r])
        if struct is None:
            bary.append([]); pa_l.append(np.zeros((0, 2))); pb_l.append(np.zeros((0, 2))); Ja_l.append(np.zeros(0))
            continue
        simplices, edges = struct
        bary.append([low_gradient_barycenters(simplices, coresets[r], np.asarray(vals0[r][f], dtype=np.float64).ravel(),
                                              frac_low_grad) for f in names])
        J0 = np.asarray(vals0[r]["J"], dtype=np.float64).ravel()
        a, b = edges[:, 0], edges[:, 1]
        m = np.isfinite(J0[a]) & np.isfinite(J0[b]) & ((J0[a] * J0[b]) < 0.0)
        pa_l.append(coresets[r][a[m]]); pb_l.append(coresets[r][b[m]]); Ja_l.append(J0[a[m]])

    zeros_list = _bisect_batch(computer_J, pos_list, vel_list, pa_l, pb_l, Ja_l, sigma, n_bis)

    seeds_list = []
    for r in range(B):
        parts = [coresets[r]] + bary[r]
        z = zeros_list[r]
        if z is not None and len(z) > 0 and n_jacobi > 0:
            parts.append(fps(z, n_jacobi) if len(z) > n_jacobi else z)
        seeds_list.append(_dedup(np.vstack(parts), sigma, dedup_tol).astype(np.float32))

    fin = computer.compute_batch_points(pos_list, vel_list, [make_pe(s_, sigma) for s_ in seeds_list])
    result = []
    for r in range(B):
        fields = {k: fin[r][k].ravel() for k in names}
        valid = np.all([np.isfinite(v) for v in fields.values()], axis=0)
        result.append((seeds_list[r][valid], {k: v[valid] for k, v in fields.items()}))
    return result


def gridfree_seeds_batch(pos_list, vel_list, computer, sigma, *, cover=None, n_min=300, frac_low_grad=0.05,
                         n_jacobi=150, n_bis=12, dedup_tol=1e-3, coresets=None, n_threads=8, n_max=20000):
    """Grid-free seeds and fields for B runs at one sigma. The coreset leaves no agent position farther
    than cover * sigma (or, with cover None, holds n_min positions); coresets may be passed instead."""
    if coresets is None and cover is not None:
        device = "cpu" if isinstance(computer, CPUPointComputer) else "cuda"
        coresets = [o[0][:prefix_for_radius(o[1], cover * float(sigma))]
                    for o in coreset_orders_gpu(pos_list, n_max, stop_radius=cover * float(sigma), device=device)]
    if coresets is None:
        coresets = Parallel(n_jobs=n_threads, backend='loky')(
            delayed(coreset)(pos_list[i], n_min) for i in range(len(pos_list)))
    return seeds_from_coresets(pos_list, vel_list, coresets, computer, sigma, frac_low_grad=frac_low_grad,
                               n_jacobi=n_jacobi, n_bis=n_bis, dedup_tol=dedup_tol)


class CPUPointComputer:
    """CPU twin of TorchFieldComputer.compute_batch_points (same interface)."""

    def __init__(self, fields, kappa: float = 5.0, chunk_size: int = 25):
        from .fields_cpu import FieldComputer
        self.fields = tuple(fields)
        self._fc = FieldComputer(fields=self.fields, kappa=kappa,
                                 chunk_size=chunk_size)

    def compute_batch_points(self, pos_list, vel_list, pe_list):
        out = []
        for pos, vel, pe in zip(pos_list, vel_list, pe_list):
            if pe.n_vertices == 0:
                out.append({k: np.zeros((1, 0), dtype=np.float32)
                            for k in self.fields})
                continue
            res = self._fc.compute(pos, vel, pe)
            out.append({k: np.asarray(v).reshape(1, -1)
                        for k, v in res.items()})
        return out
