#!/usr/bin/env python3
"""
verify_code.py - checks that the code computes the formulas of the paper
========================================================================
1. Footprint kernel: kappa_sigma(0) = 1/2 (limit), and the CPU and GPU field codes give the same
   five fields at the seeds of a real run, including seeds that coincide with an agent.
2. Proposition 1 on a grid: grad(phi) + perp-grad(psi) = (1 / (N T sigma vbar)) sum_t V_sigma(r, t).
3. J against an independent direct evaluation (float64, explicit limit at d = 0).
4. Extended persistence: the filtration is lower-star (every simplex enters at the largest value of its
   vertices) and every pair returned by GUDHI lands in one of the six diagrams.
5. Sliced Wasserstein (order one): GPU (all pairs, vectorised) against the CPU routine, empty diagrams
   included, bit for bit.
6. Similarity network fusion: the sparse diffusion against the dense formula, and GPU against CPU bit for
   bit, at every t of the sweep, on the scale views of Top-1.
7. Affinity propagation: the GPU version against scikit-learn (labels, exemplars, iterations), bit for bit.
8. Each formula of the paper against its definition, written out again here: the fields (1)-(6) by
   explicit sums over agents and instants (J from finite differences of those potentials), the sliced
   distance (8) and one entry of a stored distance matrix, the view W_m (9), the ladder (7) from d_min and
   D_s, and the rules of the domain (covering radius, cells of the duplicates, window of the seeds).
Exit status 1 if a check fails.

Author: Olivier Vertu Ndinga Oba
"""
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from gaussian_splatting.fields_cpu import FieldComputer  # noqa: E402
from gaussian_splatting.alpha import AlphaSimplexTree  # noqa: E402
from gaussian_splatting.persistence import _extract_sub_diagrams  # noqa: E402
from gaussian_splatting.distance import sample_directions, swd  # noqa: E402

FAIL = []


def check(name, ok, detail):
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAIL.append(name)


meta = pickle.load(open(RS.RES / "meta.pkl", "rb"))["trajectories"]
g, si = 1525, 0                                         # a Shelter run (robots resting in the shelter)
st1 = pickle.load(open(RS.RES / "stage1_alpha_fields" / f"traj{g:04d}_s{si}.pkl", "rb"))
sigma = float(st1["sigma"])
pos = np.asarray(meta[g]["pos"], np.float32); vel = np.asarray(meta[g]["vel"], np.float32)
S = np.asarray(st1["seeds"], np.float32)
pe = SimpleNamespace(X=S[None, :, 0], Y=S[None, :, 1], shape=(1, len(S)), sigma=sigma, n_vertices=len(S))

# 1. kernel limit and CPU against GPU
s2 = sigma ** 2
d2 = np.array([0.0, 1e-14, 1e-10, 1e-6])
d2s = np.maximum(d2, 1e-12)
kap = s2 * (-np.expm1(-d2s / (2 * s2))) / d2s
check("kappa(0) = 1/2", np.allclose(kap[:3], 0.5, atol=1e-6), f"kappa at d^2 = {d2.tolist()}: {np.round(kap, 6).tolist()}")
cpu = FieldComputer(fields=("rho", "K", "phi", "psi", "J"), kappa=5.0, chunk_size=25).compute(pos, vel, pe)
try:
    import torch
    from gaussian_splatting.fields_gpu import TorchFieldComputer
    if torch.cuda.is_available():
        gpu = TorchFieldComputer(fields=("rho", "K", "phi", "psi", "J")).compute_batch_points([pos], [vel], [pe])[0]
        for f in ("rho", "K", "phi", "psi", "J"):
            a, b = np.asarray(cpu[f], float).ravel(), np.asarray(gpu[f], float).ravel()
            rel = np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(a))
            check(f"CPU = GPU for {f}", rel < 1e-4, f"max relative difference {rel:.1e}")
except ImportError:
    print("(no torch: GPU check skipped)")
coinc = int((np.min(((S[:, None, None, :] - pos[None]) ** 2).sum(-1).reshape(len(S), -1), axis=1) == 0).sum())
print(f"    seeds of this run that coincide with an agent at some instant: {coinc} of {len(S)}")

# 3. J against an independent direct evaluation (float64, explicit limit)
def J_direct(P, pos, vel, sigma):
    P = P.astype(np.float64); pos = pos.astype(np.float64); vel = vel.astype(np.float64)
    T, N, _ = pos.shape; s2 = sigma ** 2
    vbar = np.sqrt((vel ** 2).sum(-1)).mean(); norm = 1.0 / (N * T * sigma * vbar)
    G = np.zeros((len(P), 4))
    for t in range(T):
        d = P[:, None, :] - pos[t][None]                       # (P, N, 2)
        r2 = (d ** 2).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            kap = np.where(r2 > 0, s2 * (1 - np.exp(-r2 / (2 * s2))) / r2, 0.5)
            dk = np.where(r2 > 0, (np.exp(-r2 / (2 * s2)) - 2 * kap) / r2, -1.0 / (4 * s2))
        v = vel[t][None]
        u = (v * d).sum(-1); w = v[..., 0] * d[..., 1] - v[..., 1] * d[..., 0]
        G[:, 0] += (v[..., 0] * kap + u * d[..., 0] * dk).sum(1); G[:, 1] += (v[..., 1] * kap + u * d[..., 1] * dk).sum(1)
        G[:, 2] -= (-v[..., 1] * kap + w * d[..., 0] * dk).sum(1); G[:, 3] -= (v[..., 0] * kap + w * d[..., 1] * dk).sum(1)
    G *= norm
    return s2 * (G[:, 0] * G[:, 3] - G[:, 1] * G[:, 2])


sel = np.arange(0, len(S), max(1, len(S) // 60))
Jd = J_direct(S[sel], pos, vel, sigma)
Jc = np.asarray(cpu["J"], float).ravel()[sel]
rel = np.max(np.abs(Jd - Jc)) / np.max(np.abs(Jd))
check("J = direct evaluation", rel < 1e-4, f"max relative difference {rel:.1e} on {len(sel)} seeds")

# 2. Proposition 1 on a local fine grid (a numerical derivative needs a spacing much smaller than sigma)
c0 = np.median(pos.reshape(-1, 2), axis=0)
xs = np.linspace(c0[0] - 3 * sigma, c0[0] + 3 * sigma, 121); ys = np.linspace(c0[1] - 3 * sigma, c0[1] + 3 * sigma, 121)
X, Y = np.meshgrid(xs, ys); h = (xs[1] - xs[0], ys[1] - ys[0])
grid = SimpleNamespace(X=X, Y=Y, shape=X.shape, sigma=sigma, n_vertices=X.size)
F = FieldComputer(fields=("phi", "psi"), kappa=1e6, chunk_size=10).compute(pos[::10], vel[::10], grid)
p_, s_ = F["phi"].astype(float), F["psi"].astype(float)
lx = np.gradient(p_, h[0], axis=1) - np.gradient(s_, h[1], axis=0)
ly = np.gradient(p_, h[1], axis=0) + np.gradient(s_, h[0], axis=1)
P_, V_ = pos[::10].astype(float), vel[::10].astype(float)
O = np.exp(-((X[None, ..., None] - P_[:, None, None, :, 0]) ** 2 + (Y[None, ..., None] - P_[:, None, None, :, 1]) ** 2) / (2 * s2))
Vx = (O * V_[:, None, None, :, 0]).sum((0, 3)); Vy = (O * V_[:, None, None, :, 1]).sum((0, 3))
T_, N_ = P_.shape[:2]; vbar = np.sqrt((V_ ** 2).sum(-1)).mean()
rx, ry = Vx / (N_ * T_ * sigma * vbar), Vy / (N_ * T_ * sigma * vbar)
sl = (slice(3, -3), slice(3, -3))
rel = np.max(np.hypot(lx - rx, ly - ry)[sl]) / np.max(np.hypot(rx, ry)[sl])
check("Proposition 1 on a fine grid", rel < 1e-2, f"max relative error {rel:.1e} (spacing sigma/20, finite differences)")

# 4. every persistence pair lands in a diagram
n_raw = n_kept = 0
ast = AlphaSimplexTree(np.asarray(st1["seeds"], float), alpha_sq_max=(3 * sigma) ** 2)
for f in ("rho", "K", "phi", "psi", "J"):
    raw = ast.compute_raw_persistence(np.asarray(st1["fields"][f], float).ravel())
    n_raw += sum(len(gp) for gp in raw)
    n_kept += sum(len(d) for d in _extract_sub_diagrams(raw))
check("no persistence pair dropped", n_raw == n_kept, f"{n_kept} of {n_raw} pairs in the six diagrams")
import gudhi  # noqa: E402
bad = total = 0
for fname in ("psi", "J"):
    Fv = np.asarray(st1["fields"][fname], float).ravel()
    st = gudhi.SimplexTree(ast._template)
    for v in ast.active_verts:
        st.assign_filtration([v], float(Fv[v]))
    st.make_filtration_non_decreasing()
    for simplex, fval in st.get_simplices():
        total += 1
        bad += fval != max(Fv[v] for v in simplex)
check("lower-star filtration", bad == 0, f"{total - bad} of {total} simplices enter at the largest value of their vertices")

# 5. sliced Wasserstein, GPU against CPU
try:
    import torch
    if torch.cuda.is_available():
        from gaussian_splatting.swd_gpu import swd_condensed_gpu
        dirs = sample_directions(25)
        rng = np.random.default_rng(0)
        D = [np.sort(rng.normal(size=(rng.integers(0, 8), 2)), axis=1) for _ in range(40)]
        gq = swd_condensed_gpu(D)
        iu, ju = np.triu_indices(len(D), 1)
        cq = np.array([swd(D[i], [D[j]], dirs)[0] for i, j in zip(iu, ju)])
        check("sliced Wasserstein GPU = CPU", np.array_equal(gq, cq), f"{len(iu)} pairs identical bit for bit")
        from gaussian_splatting.swd_gpu import swd_field_condensed_gpu
        sets = [[np.sort(rng.normal(size=(rng.integers(0, 6) * (s_ != 3), 2)), axis=1) for s_ in range(6)] for _ in range(30)]
        g6 = swd_field_condensed_gpu(sets)
        i6, j6 = np.triu_indices(len(sets), 1)
        c6 = np.array([sum(swd(sets[i][s_], [sets[j][s_]], dirs)[0] for s_ in range(6)) for i, j in zip(i6, j6)])
        check("stage-3 distance GPU = CPU", np.array_equal(g6, c6),
              f"six sub-diagrams summed, one empty in every run, {len(i6)} pairs identical bit for bit")
        # runs of different lengths in one call: evaluated by groups of equal length, as when separate
        meta_short = [g_ for g_ in range(0, 2500, 97) if meta[g_]["pos"].shape[0] != meta[g]["pos"].shape[0]][:1]
        if meta_short:
            h = meta_short[0]
            ph, vh = np.asarray(meta[h]["pos"], np.float32), np.asarray(meta[h]["vel"], np.float32)
            comp = TorchFieldComputer(fields=("rho", "K", "phi", "psi", "J"))
            mixed = comp.compute_batch_points([pos, ph], [vel, vh], [pe, pe])
            alone = [comp.compute_batch_points([pos], [vel], [pe])[0], comp.compute_batch_points([ph], [vh], [pe])[0]]
            same = all(np.array_equal(mixed[k][f], alone[k][f], equal_nan=True) for k in range(2) for f in mixed[0])
            check("fields of runs of different lengths", same, "one call on mixed lengths = separate calls")
except ImportError:
    pass

# 5b. the mutual rank of the rank normalisation (Obayashi and Kinoshita 2009), against its definition
from classification.affinity import mutual_rank  # noqa: E402
rng = np.random.default_rng(3)
X = rng.random((40, 2))
Dm = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
brute = np.zeros_like(Dm)
for i in range(len(Dm)):
    for j in range(len(Dm)):
        if i != j:                                 # rank 1 for the nearest neighbour, the recording itself left out
            brute[i, j] = np.sqrt((1 + np.sum(np.delete(Dm[i], [i, j]) < Dm[i, j])) * (1 + np.sum(np.delete(Dm[j], [i, j]) < Dm[j, i])))
MR = mutual_rank(Dm)
check("mutual rank = definition", np.allclose(MR, brute) and np.allclose(MR, MR.T) and np.allclose(mutual_rank(Dm ** 3 + Dm), MR),
      "sqrt(rank of j among the neighbours of i x rank of i among those of j), symmetric, unchanged by an increasing map of the distances")

# 6. similarity network fusion on the scale views of Top-1
from classification import fusion as Fu  # noqa: E402
from classification.snf import snf_dense, snf_path  # noqa: E402
lab_, rk_, _ = RS.load_labels()
Dist, _ = RS.load_distances()
idx1 = RS.subset(1, rk_)
views = Fu.scale_views(Fu.rank_normalise(Dist, Fu.ALL_FIELDS, idx1))
cpu_path = snf_path(views, 7, RS.TS, "cpu")
dense = snf_dense(views, 7, 10)
rel = np.abs(dense - cpu_path[10]).max() / np.abs(dense).max()
check("SNF sparse = dense formula", rel < 1e-12, f"max relative difference {rel:.1e} (knn 7, t 10)")
try:
    import torch
    if torch.cuda.is_available():
        gpu_path = snf_path(views, 7, RS.TS, "cuda")
        check("SNF GPU = CPU", all(np.array_equal(cpu_path[t], gpu_path[t]) for t in RS.TS),
              f"all {len(RS.TS)} values of t identical bit for bit")

        # 7. affinity propagation, GPU against scikit-learn
        from classification.ap import affinity_propagation_cpu, affinity_propagation_gpu
        mats = [cpu_path[t] for t in (1, 5, 10, 25)]
        g = affinity_propagation_gpu(mats)
        c = [affinity_propagation_cpu(S) for S in mats]
        same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2:] == b[2:] for a, b in zip(c, g))
        check("affinity propagation GPU = scikit-learn", same,
              f"labels, exemplars, convergence and iterations identical on {len(mats)} consensus matrices")
except ImportError:
    pass

# 8. every formula of the paper against its definition, written out again here
# 8a. the fields (1)-(6), by explicit sums over agents and instants (float64), at 40 seeds of the run
rs = np.random.default_rng(5).choice(len(S), 40, replace=False)
P64, X64, V64 = S[rs].astype(np.float64), pos.astype(np.float64), vel.astype(np.float64)
T_, N_ = X64.shape[:2]
sp2 = (V64 ** 2).sum(-1)
vbar, v2bar = np.sqrt(sp2).mean(), sp2.mean()


def direct_fields(r):
    """rho, K, phi, psi at the points r from formulas (1), (2), (4) and (5) of the paper."""
    d = r[:, None, None, :] - X64[None]                         # d_i(t) = r - x_i(t)
    dd = (d ** 2).sum(-1)
    O = np.exp(-dd / (2 * sigma ** 2))                          # footprint (1)
    win = dd < (5 * sigma) ** 2                                 # rho and K add the agents within 5 sigma
    kap = np.where(dd > 0, sigma ** 2 * (1 - np.exp(-dd / (2 * sigma ** 2))) / np.where(dd > 0, dd, 1), 0.5)   # (4)
    u = (V64[None] * d).sum(-1)
    w = V64[None, ..., 0] * d[..., 1] - V64[None, ..., 1] * d[..., 0]
    return dict(rho=(O * win).sum((1, 2)) / (N_ * T_), K=(O * win * sp2[None]).sum((1, 2)) / (N_ * T_ * v2bar),
                phi=(u * kap).sum((1, 2)) / (N_ * T_ * sigma * vbar), psi=-(w * kap).sum((1, 2)) / (N_ * T_ * sigma * vbar))


ref = direct_fields(P64)
h = 1e-4 * sigma                                                # J (6) from central differences of phi and psi
grad = {k: [(direct_fields(P64 + e)[k] - direct_fields(P64 - e)[k]) / (2 * h) for e in (np.array([h, 0]), np.array([0, h]))]
        for k in ("phi", "psi")}
ref["J"] = sigma ** 2 * (grad["phi"][0] * grad["psi"][1] - grad["phi"][1] * grad["psi"][0])
errs = {f: np.abs(np.asarray(cpu[f]).ravel()[rs] - ref[f]).max() / np.abs(ref[f]).max() for f in ref}
check("fields = formulas (1)-(6) written out", all(e < (1e-3 if f == "J" else 1e-5) for f, e in errs.items()),
      "max relative difference " + ", ".join(f"{f} {e:.1e}" for f, e in errs.items()) + " (J by finite differences)")

# 8b. the sliced Wasserstein distance (8) from its definition, then one stored distance from the six diagrams
def sw1_definition(A, B, n_dirs=25):
    th = -np.pi / 2 + np.pi * np.arange(n_dirs) / n_dirs            # evenly spaced in [-pi/2, pi/2)
    A, B = np.asarray(A, float).reshape(-1, 2), np.asarray(B, float).reshape(-1, 2)
    diag = lambda Z: np.repeat(Z.mean(1, keepdims=True), 2, axis=1)   # projection onto the diagonal
    Aa, Ba = np.vstack([A, diag(B)]), np.vstack([B, diag(A)])        # each augmented with the other's projections
    tot = 0.0
    for t in th:
        u = np.array([np.cos(t), np.sin(t)])
        tot += np.abs(np.sort(Aa @ u) - np.sort(Ba @ u)).sum()
    return tot / n_dirs


rng = np.random.default_rng(11)
pairs_ = [(rng.random((rng.integers(0, 9), 2)), rng.random((rng.integers(0, 9), 2))) for _ in range(30)]
pairs_ = [(np.sort(a, 1), np.sort(b, 1)) for a, b in pairs_]
dirs = sample_directions(25)
e8 = max(abs(swd(a, [b], dirs)[0] - sw1_definition(a, b)) for a, b in pairs_)
st2a = pickle.load(open(RS.RES / "stage2_alpha_dgms" / "traj0000_s0.pkl", "rb"))["dgms"]["rho"]
st2b = pickle.load(open(RS.RES / "stage2_alpha_dgms" / "traj0001_s0.pkl", "rb"))["dgms"]["rho"]
d01 = sum(sw1_definition(st2a[k], st2b[k]) for k in range(6))
stored = float(np.asarray(Dist["rho"][0])[0, 1])
check("sliced distance = formula (8)", e8 < 1e-12 and abs(d01 - stored) <= 1e-9 * max(1.0, stored),
      f"30 random pairs (empty diagrams included) within {e8:.1e}; runs 0 and 1 on rho at sigma_1: "
      f"{d01:.10f} from the six diagrams, {stored:.10f} stored")

# 8c. the view W_m (9) from the ranks: 1 - (1/3 rho + 1/3 K + 1/9 (phi + psi + J)), unit diagonal
from classification.affinity import quantile_norm  # noqa: E402
Qr = Fu.rank_normalise(Dist, Fu.ALL_FIELDS, idx1)
pct = quantile_norm(mutual_rank(np.asarray(Dist["psi"][2], float)[np.ix_(idx1, idx1)]))
W9 = [1.0 - (Qr["rho"][s_] / 3 + Qr["K"][s_] / 3 + (Qr["phi"][s_] + Qr["psi"][s_] + Qr["J"][s_]) / 9) for s_ in Qr["rho"]]
e9 = max(np.abs(a - b).max() for a, b in zip(W9, views))
check("view W_m = formula (9)", e9 < 1e-6 and np.allclose(np.asarray(Qr["psi"][2], float), pct, atol=1e-6)
      and all(np.allclose(np.diag(v), 1.0) for v in views),
      f"max difference {e9:.1e} over the four scales of Top-1; ranks = percentile ranks of the mutual ranks; unit diagonal")

# 8d. the ladder (7) from d_min and D_s over every instant of every run
from pipeline.config import EPS_SIGMA, M_SCALES, SIGMA_MAX_FACTOR  # noqa: E402
dmin, Ds = np.inf, 0.0
for tr in meta:
    X_ = np.asarray(tr["pos"], np.float64)
    iu_ = np.triu_indices(X_.shape[1], 1)
    dd_ = np.sqrt(((X_[:, iu_[0]] - X_[:, iu_[1]]) ** 2).sum(-1))
    dmin, Ds = min(dmin, dd_.min()), max(Ds, dd_.max())
lad = np.asarray(pickle.load(open(RS.RES / "scale_ladder.pkl", "rb"))["ladder"], float)
smin, smax = max(dmin / 2, EPS_SIGMA), Ds / (2.35 * SIGMA_MAX_FACTOR)
expected = smin * (smax / smin) ** (np.arange(M_SCALES) / (M_SCALES - 1))
check("ladder = formula (7)", np.allclose(lad, expected, rtol=1e-12),
      f"d_min {100 * dmin:.2f} cm, D_s {100 * Ds:.1f} cm, k {SIGMA_MAX_FACTOR:g}: ladder {np.round(100 * lad, 3).tolist()} cm")

# 8e. the rules of the domain, on the seeds of the run at sigma_1
from scipy.spatial import cKDTree  # noqa: E402
cover = cKDTree(S.astype(np.float64)).query(pos.reshape(-1, 2).astype(np.float64))[0].max()
cells = np.round(S.astype(np.float64) / (1e-3 * sigma)).astype(np.int64)
near = cKDTree(pos.reshape(-1, 2).astype(np.float64)).query(S.astype(np.float64))[0].max()
check("domain = its description", cover <= sigma / 2 + 1e-6 and len(np.unique(cells, axis=0)) == len(S) and near < 5 * sigma,
      f"every robot position within {cover / sigma:.3f} sigma of a seed (at most 1/2), {len(S)} seeds in as many cells of "
      f"side 1e-3 sigma, every seed within {near / sigma:.2f} sigma of a robot position (less than 5)")

print("ALL CHECKS PASSED" if not FAIL else f"FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
