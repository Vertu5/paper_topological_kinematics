#!/usr/bin/env python3
"""
attribution.py - back-propagating the distance between two recordings to the robots
====================================================================================
Once the seeds are fixed, the chain from the robot positions to the sliced-Wasserstein distance is
differentiable almost everywhere (Section 4). This script makes it explicit on the Top-1 exemplars of
two missions and one field at one scale (any of the five; by default the vorticity potential psi at sigma_3):

  positions -> velocities (forward differences) -> field at the seeds (closed form, float64)
  -> extended persistence (pairing by GUDHI; every coordinate of a diagram point is the field value
     at one seed, gathered from the field tensor) -> sliced-Wasserstein distance of stage 3

Every step is first checked against the stored outputs of the pipeline (fields at the seeds, diagrams,
distance). The distance is a sum of terms, one per matched pair and direction, so it splits exactly over the
points of the diagrams, and through them over the seeds that realize their critical values (sw1_split). Each
seed's share is then carried back to the robots by the derivative of the field value at that seed with respect
to every robot position (robot_flow). The gradient of the distance itself is also recorded. The value reader
(32 quantiles of the field values, 1-D Wasserstein distance) is split and carried back in the same way, for
comparison.

Usage: python analysis/attribution.py            (ATTR_FIELD=psi ATTR_SCALE=2 ATTR_PAIR=0,1 by default)
Output: results/epuck/paper/attribution.pkl (ATTR_OUT sets another path, as analysis/attribution_pairs.py does)

Author: Olivier Vertu Ndinga Oba
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from pipeline.checkpoint import save_pickle  # noqa: E402
from gaussian_splatting.alpha import AlphaSimplexTree  # noqa: E402
from gaussian_splatting.distance import sample_directions  # noqa: E402

PAPER = RS.OUT
DT = 0.1                      # ARGoS time step (s)
C_ALPHA = 3.0                 # alpha complex cut (3 sigma)^2
N_QUANT = 32                  # quantiles of the value reader
FIELD = os.environ.get("ATTR_FIELD", "psi")
SI = int(os.environ.get("ATTR_SCALE", "2"))
PAIR = tuple(int(x) for x in os.environ.get("ATTR_PAIR", "0,1").split(","))
DEV = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("ATTR_CPU") != "1" else "cpu")
TD = torch.float64
CHUNK = 100
MISSIONS = ["D-Gate", "Foraging", "Homing", "Shelter", "XOR"]
# (GUDHI group, dimension, swap) of the pairs gathered in each of the six diagrams, as in persistence.py
GROUPS = [[(0, 0, False)], [(0, 1, False)], [(1, 1, True)], [(1, 2, True)],
          [(2, 0, False), (3, 0, True)], [(3, 1, True), (2, 1, False)]]
DIRS = torch.tensor(sample_directions(25), dtype=TD)


# ----------------------------------------------------------------------------- inputs
def exemplars(pair):
    """(position in Top-1, global run index) of the Top-1 exemplar of each mission in `pair`."""
    tax = pickle.load(open(os.environ.get("ATTR_TAXONOMY", PAPER / "taxonomy.pkl"), "rb"))[1]
    lab = np.load(RS.RES / "labels.npz")["labels"]
    by_mission = {int(lab[tax["idx"][e]]): (int(e), int(tax["idx"][e])) for e in tax["exemplars"]}
    return [by_mission[m] for m in pair]


def load_run(e, g, tracks):
    assert int(tracks["idx"][e]) == g, "Top-1 track order differs from the taxonomy index"
    tr = tracks["tracks"][e]
    with open(RS.RES / "stage1_alpha_fields" / f"traj{g:04d}_s{SI}.pkl", "rb") as fh:
        st1 = pickle.load(fh)
    with open(RS.RES / "stage2_alpha_dgms" / f"traj{g:04d}_s{SI}.pkl", "rb") as fh:
        st2 = pickle.load(fh)
    pos = np.asarray(tr["pos"], np.float64)
    vel = np.asarray(tr["vel"], np.float64)
    full = np.concatenate([pos, pos[-1:] + DT * vel[-1:]], axis=0)       # positions at T+1 instants
    return dict(pos=pos, vel=vel, P=full, seeds=np.asarray(st1["seeds"], np.float64),
                stored=np.asarray(st1["fields"][FIELD], np.float64).ravel(), sigma=float(st1["sigma"]),
                dgms=[np.asarray(d, np.float64).reshape(-1, 2) for d in st2["dgms"][FIELD]])


# ----------------------------------------------------------------------------- fields
def _norm(vel, sigma):
    """1 / (N sigma vbar): the normalisation of the potentials, through the mean speed of the run."""
    speed = torch.sqrt(torch.clamp((vel ** 2).sum(-1), min=1e-30))       # no NaN gradient for a robot at rest
    return 1.0 / (vel.shape[1] * sigma * torch.clamp(speed.mean(), min=1e-12))


def _scale(vel, sigma, key):
    """The factor of the sums of `key`, the division by the number of instants aside (fields_gpu.py): 1/N for rho,
    1/(N <v^2>) for K, 1/(N sigma vbar) for the potentials and for the gradients of which J is made."""
    if key == "rho":
        return torch.tensor(1.0 / vel.shape[1], dtype=TD, device=vel.device)
    if key == "K":
        return 1.0 / (vel.shape[1] * torch.clamp((vel ** 2).sum(-1).mean(), min=1e-12))
    return _norm(vel, sigma)


def _sums(pos, vel, seeds, sigma, key):
    """The sums over the given instants and robots at the seeds, as in the kernel of fields_gpu.py, shape (k, G):
    one sum for rho, K, phi and psi (rho and K within 5 sigma of the robot), four for J (the gradients of phi and
    psi, from which J = sigma^2 det(grad phi, grad psi))."""
    sx, sy = seeds[:, 0], seeds[:, 1]
    acc = [torch.zeros(len(seeds), dtype=TD, device=pos.device) for _ in range(4 if key == "J" else 1)]
    for t0 in range(0, pos.shape[0], CHUNK):
        p, v = pos[t0:t0 + CHUNK], vel[t0:t0 + CHUNK]
        dx = sx[None, :, None] - p[:, None, :, 0]
        dy = sy[None, :, None] - p[:, None, :, 1]
        d2 = dx * dx + dy * dy
        vx, vy = v[:, None, :, 0], v[:, None, :, 1]
        e = torch.exp(-d2 / (2.0 * sigma ** 2))
        if key in ("rho", "K"):
            em = e * (d2 < (5.0 * sigma) ** 2)
            acc[0] = acc[0] + (em if key == "rho" else em * (vx * vx + vy * vy)).sum(dim=(0, 2))
            continue
        d2s = torch.clamp(d2, min=1e-12)                                     # kappa(0) = 1/2
        g = sigma ** 2 * (-torch.expm1(-d2s / (2.0 * sigma ** 2))) / d2s
        u, w = vx * dx + vy * dy, vx * dy - vy * dx
        if key == "phi":
            acc[0] = acc[0] + (u * g).sum(dim=(0, 2))
        elif key == "psi":
            acc[0] = acc[0] - (w * g).sum(dim=(0, 2))
        else:
            t2 = (e - 2.0 * g) / d2s
            acc[0] = acc[0] + (vx * g + u * dx * t2).sum(dim=(0, 2))
            acc[1] = acc[1] + (vy * g + u * dy * t2).sum(dim=(0, 2))
            acc[2] = acc[2] - (-vy * g + w * dx * t2).sum(dim=(0, 2))
            acc[3] = acc[3] - (vx * g + w * dy * t2).sum(dim=(0, 2))
    return torch.stack(acc)


def _combine(S, c, T, sigma, key):
    """The field at the seeds from its sums S (k, G), their factor c and the number of instants T."""
    if key == "J":
        a = S * (c / T)
        return sigma ** 2 * (a[0] * a[3] - a[1] * a[2])
    return S[0] * (c / T)


def potentials(pos, vel, seeds, sigma, key):
    """The field `key` (any of the five) at the seeds, same closed form as fields_gpu.TorchFieldComputer."""
    return _combine(_sums(pos, vel, seeds, sigma, key), _scale(vel, sigma, key), pos.shape[0], sigma, key)


def field_from_positions(P, seeds, sigma):
    return potentials(P[:-1], (P[1:] - P[:-1]) / DT, seeds, sigma, FIELD)


def field_vjp(P, seeds, sigma, g):
    """d<g, F>/dP for F = field_from_positions(P). The field is a function of a few sums over the instants and of
    one factor that depends on the speeds, so the gradient is that of the factor, then that of the sums chunk by
    chunk of instants: only the graph of one chunk is held in memory, and the value is that of autograd on the
    whole run (up to the order of the sums)."""
    P = P.detach()
    T = P.shape[0] - 1
    with torch.no_grad():
        S = _sums(P[:-1], (P[1:] - P[:-1]) / DT, seeds, sigma, FIELD)
    Pn = P.clone().requires_grad_(True)
    Sl = S.clone().requires_grad_(True)
    F = _combine(Sl, _scale((Pn[1:] - Pn[:-1]) / DT, sigma, FIELD), T, sigma, FIELD)
    gS, gP = torch.autograd.grad((g * F).sum(), (Sl, Pn), allow_unused=True)
    gP = torch.zeros_like(P) if gP is None else gP
    for t0 in range(0, T, CHUNK):
        t1 = min(T, t0 + CHUNK)
        Pc = P[t0:t1 + 1].clone().requires_grad_(True)               # velocities need the next position too
        sc = _sums(Pc[:-1], (Pc[1:] - Pc[:-1]) / DT, seeds, sigma, FIELD)
        gc, = torch.autograd.grad(sc, Pc, grad_outputs=gS)
        gP[t0:t1 + 1] += gc
    return gP


# ----------------------------------------------------------------------------- persistence
def pairing(seeds, values, sigma):
    """Vertex indices (birth seed, death seed) of every point of the six diagrams."""
    ast = AlphaSimplexTree(seeds, alpha_sq_max=(C_ALPHA * sigma) ** 2)
    raw = ast.compute_raw_persistence(values)
    order = np.argsort(values, kind="stable")
    sv = values[order]
    worst = 0.0

    def vid(c):
        nonlocal worst
        k = int(np.searchsorted(sv, c))
        j = min((j for j in (k - 1, k, k + 1) if 0 <= j < len(sv)), key=lambda j: abs(sv[j] - c))
        worst = max(worst, abs(sv[j] - c))
        return int(order[j])

    idx = []
    for parts in GROUPS:
        pts = [p[1][::-1] if swap else p[1] for g, dim, swap in parts
               for p in (raw[g] if g < len(raw) else []) if p[0] == dim]
        idx.append(np.array([(vid(b), vid(d)) for b, d in pts], dtype=np.int64).reshape(-1, 2))
    simp = [s for s, _ in ast._template.get_skeleton(2)]
    cx = dict(edges=np.array([s for s in simp if len(s) == 2], dtype=np.int64).reshape(-1, 2),
              tris=np.array([s for s in simp if len(s) == 3], dtype=np.int64).reshape(-1, 3))
    return idx, worst, cx


def diagrams(F, idx):
    return [torch.stack([F[i[:, 0]], F[i[:, 1]]], dim=1) if len(i) else F.new_zeros((0, 2)) for i in idx]


def sw1(A, B):
    c, s = DIRS[:, 0].to(A.device), DIRS[:, 1].to(A.device)
    pa, pb = A[:, :1] * c + A[:, 1:] * s, B[:, :1] * c + B[:, 1:] * s
    da = (0.5 * (A[:, 0] + A[:, 1]))[:, None] * (c + s)
    db = (0.5 * (B[:, 0] + B[:, 1]))[:, None] * (c + s)
    X = torch.sort(torch.cat([pa, db], 0), dim=0).values
    Y = torch.sort(torch.cat([pb, da], 0), dim=0).values
    return (X - Y).abs().sum(0).mean()


def persistence_distance(FA, FB, idxA, idxB):
    return sum(sw1(a, b) for a, b in zip(diagrams(FA, idxA), diagrams(FB, idxB)))


def sw1_split(A, B):
    """Exact split of sw1(A, B) over the points of A and of B. In each direction, the term |x_(k) - y_(k)| of
    every matched pair is shared equally between its two elements, a projection onto the diagonal standing for
    the point it projects; the shares of the points of A and of B sum to the distance."""
    c, s = DIRS[:, 0].to(A.device), DIRS[:, 1].to(A.device)
    nA, nB = len(A), len(B)
    pa, pb = A[:, :1] * c + A[:, 1:] * s, B[:, :1] * c + B[:, 1:] * s
    da = (0.5 * (A[:, 0] + A[:, 1]))[:, None] * (c + s)
    db = (0.5 * (B[:, 0] + B[:, 1]))[:, None] * (c + s)
    X, Y = torch.cat([pa, db], 0), torch.cat([pb, da], 0)
    own_x = torch.cat([torch.arange(nA), nA + torch.arange(nB)]).to(A.device)   # X: points of A, projections of B
    own_y = torch.cat([nA + torch.arange(nB), torch.arange(nA)]).to(A.device)   # Y: points of B, projections of A
    ix, iy = torch.argsort(X, 0), torch.argsort(Y, 0)
    term = 0.5 * (torch.gather(X, 0, ix) - torch.gather(Y, 0, iy)).abs() / X.shape[1]
    out = torch.zeros(nA + nB, dtype=A.dtype, device=A.device)
    out.index_add_(0, own_x[ix].ravel(), term.ravel())
    out.index_add_(0, own_y[iy].ravel(), term.ravel())
    return out[:nA], out[nA:]


def persistence_split(F_A, F_B, idxA, idxB):
    """Share of the distance carried by each seed of A: the share of every point of its six diagrams, half to
    its birth seed and half to its death seed. With the shares of B, they sum to the distance."""
    FA, FB = torch.as_tensor(F_A, dtype=TD), torch.as_tensor(F_B, dtype=TD)
    out = np.zeros(len(F_A))
    for dA, dB, iA in zip(diagrams(FA, idxA), diagrams(FB, idxB), idxA):
        if len(dA) + len(dB):
            cA = 0.5 * sw1_split(dA, dB)[0].cpu().numpy()
            np.add.at(out, iA[:, 0], cA)
            np.add.at(out, iA[:, 1], cA)
    return out


def value_split(F_A, F_B):
    """Share of the value distance carried by each seed of A: the term of every quantile is shared equally
    between the two runs, and the half of A between the two order statistics that the quantile interpolates."""
    FA, FB = np.asarray(F_A, np.float64), np.asarray(F_B, np.float64)
    q = np.linspace(0.0, 1.0, N_QUANT)
    term = 0.5 * np.abs(np.quantile(FA, q) - np.quantile(FB, q)) / N_QUANT
    order, h = np.argsort(FA, kind="stable"), (len(FA) - 1) * q
    lo = np.floor(h).astype(int)
    hi, frac = np.minimum(lo + 1, len(FA) - 1), h - lo
    out = np.zeros(len(FA))
    np.add.at(out, order[lo], term * (1 - frac))
    np.add.at(out, order[hi], term * frac)
    return out


def value_distance(FA, FB):
    q = torch.linspace(0.0, 1.0, N_QUANT, dtype=TD, device=FA.device)
    return (torch.quantile(FA, q) - torch.quantile(FB, q)).abs().mean()


def vertex_types(values, cx):
    """Piecewise-linear type of every seed from its link in the complex: min (empty lower link),
    max (empty upper link), saddle (lower or upper link in two pieces or more), otherwise regular."""
    n = len(values)
    nbrs = [[] for _ in range(n)]
    for a, b in cx["edges"]:
        nbrs[a].append(b); nbrs[b].append(a)
    link = [[] for _ in range(n)]
    for a, b, c in cx["tris"]:
        link[a].append((b, c)); link[b].append((a, c)); link[c].append((a, b))

    def pieces(v, part):
        parent = {u: u for u in part}

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        for a, b in link[v]:
            if a in parent and b in parent:
                parent[find(a)] = find(b)
        return len({find(u) for u in part})

    out = []
    for v in range(n):
        if not nbrs[v]:
            out.append("isolated")
            continue
        low = {u for u in nbrs[v] if values[u] < values[v]}
        up = set(nbrs[v]) - low
        cl, cu = pieces(v, low), pieces(v, up)
        out.append("min" if cl == 0 else "max" if cu == 0 else "saddle" if max(cl, cu) >= 2 else "regular")
    return np.array(out)


def shares(g, types):
    g = np.abs(g)
    return dict(n=int((g > 0).sum()), extrema=float(g[np.isin(types, ("min", "max"))].sum() / g.sum()),
                critical=float(g[np.isin(types, ("min", "max", "saddle"))].sum() / g.sum()))


def seed_shares(FA_np, FB_np, idxA, idxB):
    """Share of the distance carried by each seed of A, normalised to sum one over A, for both readers."""
    out = dict(persistence=persistence_split(FA_np, FB_np, idxA, idxB), values=value_split(FA_np, FB_np))
    return {k: v / v.sum() for k, v in out.items()}


def blurred(a, seeds, width):
    K = np.exp(-((seeds[:, None, :] - seeds[None, :, :]) ** 2).sum(-1) / (2 * width ** 2))
    b = K @ a
    return b / b.sum()


def kde_map(a, seeds, width, n=120):
    X, Y = grid_centres(n)
    G = np.exp(-((X.ravel()[:, None] - seeds[None, :, 0]) ** 2 + (Y.ravel()[:, None] - seeds[None, :, 1]) ** 2)
               / (2 * width ** 2))
    m = (G @ a).reshape(n, n) * in_arena(n)
    return m / m.sum()


ARENA = 2.462
DISC = (1.231, 1.231, 1.236)          # centre and radius of the arena (m), for area fractions


def grid_centres(n):
    c = (np.arange(n) + 0.5) * ARENA / n
    return np.meshgrid(c, c)


def in_arena(n):
    X, Y = grid_centres(n)
    return (X - DISC[0]) ** 2 + (Y - DISC[1]) ** 2 <= DISC[2] ** 2


def robot_flow(P, seeds, sigma, w):
    """Each seed's share w of the distance divided among the robot positions (instant by instant), in proportion
    to the size of the derivative of the field value at that seed with respect to each of them."""
    acc = torch.zeros(P.shape[:2], dtype=TD, device=P.device)
    one = torch.ones(1, dtype=TD, device=P.device)
    for v in np.flatnonzero(w > 0):
        m = torch.linalg.norm(field_vjp(P, seeds[v:v + 1], sigma, one), dim=-1)
        acc += float(w[v]) * m / m.sum()
    return acc.cpu().numpy()


def robot_map(P, a, n=80, smooth=1.5):
    """An attribution to the robot positions (instants x robots) accumulated where the robots were (end instants
    dropped: with forward differences the first and last positions are not compensated by a neighbour)."""
    from scipy.ndimage import gaussian_filter
    x, w = P[1:-1].reshape(-1, 2), a[1:-1].ravel()
    H, _, _ = np.histogram2d(x[:, 1], x[:, 0], bins=n, range=[[0, ARENA], [0, ARENA]], weights=w)
    H = gaussian_filter(H, smooth) * in_arena(n)
    return H / H.sum()


def area_half(M):
    """Fraction of the arena that holds half of the mass of a map."""
    inside = in_arena(M.shape[0])
    v = np.sort(M[inside])[::-1]
    return float((np.searchsorted(np.cumsum(v) / v.sum(), 0.5) + 1) / inside.sum())


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    tracks = pickle.load(open(PAPER / "top1_tracks.pkl", "rb"))
    D_swd, _ = RS.load_distances()
    base = pickle.load(open(PAPER / "baselines.pkl", "rb"))
    runs = [load_run(e, g, tracks) for e, g in exemplars(PAIR)]
    A, B = runs
    check = {}

    # 1. fields at the seeds, from the stored positions and velocities
    for name, r in zip(("A", "B"), runs):
        pos, vel, S = (torch.tensor(r[k], dtype=TD, device=DEV) for k in ("pos", "vel", "seeds"))
        with torch.no_grad():
            F_st = potentials(pos, vel, S, r["sigma"], FIELD).cpu().numpy()
            F_P = field_from_positions(torch.tensor(r["P"], dtype=TD, device=DEV), S, r["sigma"]).cpu().numpy()
        scale = np.abs(r["stored"]).max()
        check[f"field_{name}_vs_stored"] = float(np.abs(F_st - r["stored"]).max() / scale)
        check[f"field_{name}_positions_vs_stored"] = float(np.abs(F_P - r["stored"]).max() / scale)
        r["F"] = F_P

    # 2. diagrams: pairing on the stored values reproduces the stored diagrams; pairing on the recomputed values
    for name, r in zip(("A", "B"), runs):
        idx_st, err_st, _ = pairing(r["seeds"], r["stored"], r["sigma"])
        got = [np.sort(np.c_[r["stored"][i[:, 0]], r["stored"][i[:, 1]]], axis=0) if len(i) else np.zeros((0, 2))
               for i in idx_st]
        want = [np.sort(d, axis=0) if len(d) else np.zeros((0, 2)) for d in r["dgms"]]
        check[f"diagrams_{name}_vs_stored"] = float(max(np.abs(a - b).max() if len(a) else 0.0 for a, b in zip(got, want)))
        check[f"diagrams_{name}_sizes_equal"] = all(len(a) == len(b) for a, b in zip(got, want))
        r["idx"], err, r["cx"] = pairing(r["seeds"], r["F"], r["sigma"])
        check[f"value_match_{name}"] = float(max(err, err_st))
        check[f"pairing_{name}_unchanged"] = all(np.array_equal(np.sort(a, axis=0), np.sort(b, axis=0))
                                                for a, b in zip(idx_st, r["idx"]))

    # 3. distances against the stored matrices
    gA, gB = (g for _, g in exemplars(PAIR))
    with torch.no_grad():
        st = [torch.tensor(r["stored"], dtype=TD) for r in runs]
        idx_st = [pairing(r["seeds"], r["stored"], r["sigma"])[0] for r in runs]
        d_st = float(persistence_distance(st[0], st[1], idx_st[0], idx_st[1]))
        v_st = float(value_distance(st[0], st[1]))
    check["persistence_distance_vs_stored"] = abs(d_st - D_swd[FIELD][SI][gA, gB]) / D_swd[FIELD][SI][gA, gB]
    check["value_distance_vs_stored"] = abs(v_st - base["fieldstat_D"][FIELD][SI][gA, gB]) / base["fieldstat_D"][FIELD][SI][gA, gB]

    # 4. gradients: first with respect to the field values at the seeds, then pushed back to the positions
    FA = torch.tensor(A["F"], dtype=TD, device=DEV, requires_grad=True)
    FB = torch.tensor(B["F"], dtype=TD, device=DEV, requires_grad=True)
    out = dict(field=FIELD, scale=SI, pair=PAIR, missions=[MISSIONS[m] for m in PAIR], check=check)
    for reader, dist in (("persistence", lambda a, b: persistence_distance(a, b, A["idx"], B["idx"])),
                         ("values", value_distance)):
        d = dist(FA, FB)
        gFA, gFB = torch.autograd.grad(d, [FA, FB])
        out[f"{reader}_distance"] = float(d.detach())
        for name, r, gF in (("A", A, gFA), ("B", B, gFB)):
            P = torch.tensor(r["P"], dtype=TD, device=DEV)
            gP = field_vjp(P, torch.tensor(r["seeds"], dtype=TD, device=DEV), r["sigma"], gF)
            out[f"{reader}_gF_{name}"] = gF.cpu().numpy()
            out[f"{reader}_gP_{name}"] = gP.cpu().numpy()
            del P, gP
            if DEV.type == "cuda":
                torch.cuda.empty_cache()

    # 5. the split of each distance over the seeds, and what carries it: the diagram points and the
    #    piecewise-linear type of their seeds
    FA_, FB_ = (torch.tensor(r["F"], dtype=TD) for r in runs)
    split = [sw1_split(a, b) for a, b in zip(diagrams(FA_, A["idx"]), diagrams(FB_, B["idx"]))]
    check["split_sums_to_distance"] = abs(float(sum(a.sum() + b.sum() for a, b in split)) - out["persistence_distance"])
    out["split_share_A"] = float(sum(a.sum() for a, _ in split) / out["persistence_distance"])
    for k, (name, r) in enumerate(zip(("A", "B"), runs)):
        types = vertex_types(r["F"], r["cx"])
        gF = out[f"persistence_gF_{name}"]
        other = runs[1 - k]
        sh = seed_shares(r["F"], other["F"], r["idx"], other["idx"])
        pts = []
        for s_, i in enumerate(r["idx"]):
            c_ = split[s_][k].cpu().numpy()
            for (vb, vd), cv in zip(i, c_):
                pts.append(dict(diagram=s_, birth_seed=int(vb), death_seed=int(vd), birth=float(r["F"][vb]),
                                death=float(r["F"][vd]), share=float(cv), grad=float(abs(gF[vb]) + abs(gF[vd])),
                                birth_type=str(types[vb]), death_type=str(types[vd])))
        out[f"points_{name}"] = sorted(pts, key=lambda p: -p["share"])
        out.update({f"seeds_{name}": r["seeds"], f"F_{name}": r["F"], f"types_{name}": types,
                    f"run_{name}": exemplars(PAIR)[0 if name == "A" else 1], "sigma": r["sigma"]})
        for reader in ("persistence", "values"):
            out[f"{reader}_share_{name}"] = sh[reader]
            out[f"{reader}_shares_{name}"] = shares(sh[reader], types)
            out[f"{reader}_grad_shares_{name}"] = shares(out[f"{reader}_gF_{name}"], types)

    # 6. stability: small noise on the positions of A, seeds fixed; does the attribution stay on the same seeds?
    rng = np.random.default_rng(0)
    base_maps = seed_shares(A["F"], B["F"], A["idx"], B["idx"])
    SA = torch.tensor(A["seeds"], dtype=TD, device=DEV)
    stab = {}
    for eps in (0.0005, 0.001, 0.002, 0.005):
        rows = []
        for k in range(10):
            P = torch.tensor(A["P"] + eps * rng.standard_normal(A["P"].shape), dtype=TD, device=DEV)
            with torch.no_grad():
                Fp = field_from_positions(P, SA, A["sigma"]).cpu().numpy()
            maps = seed_shares(Fp, B["F"], pairing(A["seeds"], Fp, A["sigma"])[0], B["idx"])
            row = dict(field_change=float(np.abs(Fp - A["F"]).max() / np.abs(A["F"]).max()))
            for reader in ("persistence", "values"):
                row[f"{reader}_same"] = float(np.minimum(base_maps[reader], maps[reader]).sum())
                row[f"{reader}_blur"] = float(np.minimum(blurred(base_maps[reader], A["seeds"], A["sigma"]),
                                                         blurred(maps[reader], A["seeds"], A["sigma"])).sum())
            rows.append(row)
        stab[eps] = {k: float(np.median([r_[k] for r_ in rows])) for k in rows[0]}
    out["stability"] = stab

    # 7. reproducibility: every run of the controller of A against the exemplar of B, attribution mapped on the arena
    tax = pickle.load(open(os.environ.get("ATTR_TAXONOMY", PAPER / "taxonomy.pkl"), "rb"))[1]
    lab = np.load(RS.RES / "labels.npz")["labels"]
    same = [int(g) for g in tax["idx"] if lab[g] == PAIR[0]]
    width = 2 * A["sigma"]
    kmaps = {"persistence": [], "values": []}
    per_run = []
    for g in same:
        with open(RS.RES / "stage1_alpha_fields" / f"traj{g:04d}_s{SI}.pkl", "rb") as fh:
            st1 = pickle.load(fh)
        seeds = np.asarray(st1["seeds"], np.float64)
        F = np.asarray(st1["fields"][FIELD], np.float64).ravel()
        idx_r, _, cx_r = pairing(seeds, F, A["sigma"])
        maps = seed_shares(F, B["F"], idx_r, B["idx"])
        for reader in kmaps:
            kmaps[reader].append(kde_map(maps[reader], seeds, width))
        types = vertex_types(F, cx_r)
        ext = np.where(np.isin(types, ("min", "max")))[0]
        top2 = ext[np.argsort(maps["persistence"][ext])[-2:]]
        per_run.append(dict(run=g, share=float(maps["persistence"][top2].sum()),
                            types=tuple(sorted(types[top2])), xy={str(types[v]): seeds[v] for v in top2}))
    rep = {}
    for reader, ms in kmaps.items():
        M = np.array(ms)
        ov = [np.minimum(M[i], M[j]).sum() for i in range(len(M)) for j in range(i + 1, len(M))]
        rep[reader] = dict(mean_overlap=float(np.mean(ov)), median_overlap=float(np.median(ov)),
                           mean_map=M.mean(0), area_half=area_half(M.mean(0)),
                           area_half_single=float(np.median([area_half(m) for m in M])))
    pair_ok = [r_ for r_ in per_run if r_["types"] == ("max", "min")]
    spread = {t: float(np.sqrt(np.mean(np.sum((np.array([r_["xy"][t] for r_ in pair_ok])
                                               - np.median([r_["xy"][t] for r_ in pair_ok], axis=0)) ** 2, axis=1))))
              for t in ("max", "min")} if pair_ok else {}
    rep["centres"] = dict(n_max_min=len(pair_ok), share_median=float(np.median([r_["share"] for r_ in per_run])),
                          spread_m=spread, per_run=per_run)
    out["reproducibility"] = dict(runs=same, width=width, **rep)

    # 8. back to the robots: each seed's share of the distance carried to the positions that shaped its value
    PA, SA = torch.tensor(A["P"], dtype=TD, device=DEV), torch.tensor(A["seeds"], dtype=TD, device=DEV)
    for reader in ("persistence", "values"):
        flow = robot_flow(PA, SA, A["sigma"], out[f"{reader}_share_A"])
        M = robot_map(A["P"], flow)
        prof = flow[1:-1].sum(1)
        out[f"{reader}_robot_map_A"] = M
        out[f"{reader}_robot_area_half_A"] = area_half(M)
        out[f"{reader}_top_decile_time_A"] = float(np.sort(prof)[-len(prof) // 10:].sum() / prof.sum())
    save_pickle(out, Path(os.environ.get("ATTR_OUT", PAPER / "attribution.pkl")))

    print(f"device {DEV}, {time.time() - t0:.0f}s; field {FIELD} at sigma_{SI + 1}; "
          f"{out['missions'][0]} run {gA} vs {out['missions'][1]} run {gB}")
    for k, v in check.items():
        print(f"  check {k:38s} {v}")
    for reader in ("persistence", "values"):
        sh = out[f"{reader}_shares_A"]
        print(f"  {reader:11s} distance {out[f'{reader}_distance']:.6f}; on A: {sh['n']} seeds carry its share, "
              f"{sh['extrema']:.2f} of it on extrema, {sh['critical']:.2f} on critical seeds")
    for p in out["points_A"][:5]:
        print("   ", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()})
    for eps, row in out["stability"].items():
        print(f"  noise {eps * 1000:.1f} mm (field change {row['field_change']:.4f}): same seeds pers {row['persistence_same']:.2f} "
              f"values {row['values_same']:.2f}; within sigma pers {row['persistence_blur']:.2f} values {row['values_blur']:.2f}")
    r_ = out["reproducibility"]
    print(f"  reproducibility over {len(r_['runs'])} runs: mean overlap pers {r_['persistence']['mean_overlap']:.2f}, "
          f"values {r_['values']['mean_overlap']:.2f}; half of the mean map in {r_['persistence']['area_half']:.3f} "
          f"vs {r_['values']['area_half']:.3f} of the arena (single runs {r_['persistence']['area_half_single']:.3f} "
          f"vs {r_['values']['area_half_single']:.3f})")
    c_ = r_["centres"]
    print(f"  centres: {c_['n_max_min']}/{len(r_['runs'])} runs have one max and one min as the two largest extrema; "
          f"median share {c_['share_median']:.2f}; rms spread {c_['spread_m']}")
    for reader in ("persistence", "values"):
        print(f"  {reader}: half of the robot-level attribution in {out[f'{reader}_robot_area_half_A']:.3f} of the arena; "
              f"top 10% of instants carry {out[f'{reader}_top_decile_time_A']:.3f}")


if __name__ == "__main__":
    main()
