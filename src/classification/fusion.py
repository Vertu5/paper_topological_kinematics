"""
fusion.py - the method of the paper, from the distances to the taxonomy
========================================================================
From the sliced-Wasserstein distance matrices d_(f,m) (field f, scale sigma_m) to the consensus network
S* and the affinity propagation taxonomy:

  1. Rank normalisation: in every matrix d_(f,m) each pair gets its mutual rank, the geometric mean of
     the rank of each recording among the neighbours of the other, and the mutual ranks are replaced
     by their percentile ranks among all pairs (values in (0, 1]; affinity.rank_distance).
  2. Physical sources: the fields are grouped by the moment they come from,
         density  rho            (zeroth moment)
         kinetic  K              (trace of the second moment)
         flux     phi, psi, J    (first moment: Helmholtz-Hodge potentials and their coupling)
     and at each scale the view distance is the mean of the ranks with the same weight per source
     present, shared equally by its fields (all five: 1/3 rho, 1/3 K, 1/9 phi, 1/9 psi, 1/9 J).
  3. One view per scale: W_m = 1 - dbar_m (unit diagonal).
  4. Similarity network fusion across the M scale views (knn, t) -> S*.
  5. Affinity propagation on S* (median preference; the settings of Frey and Dueck: damping 0.9,
     convergence after 100 unchanged iterations, at most 1000): the number of behaviours C, the
     partition and one exemplar per cluster.
  6. sweep runs steps 4-5 at every (knn, t) of the resolution map, on the GPU when available (same
     numbers as on the CPU), and plateau_consensus reads the taxonomy from the plateau of that map.

hierarchical_snf (three stacked SNF stages) is kept only for the comparison of fusion designs: stacking
compounds the diffusion and, on the paper's data, collapses the partition at moderate (knn, t).

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import rankdata

from .affinity import rank_affinity, rank_distance
from .ap import affinity_propagation_cpu, affinity_propagation_gpu, gpu_available, gpu_batch_size
from .snf import snf, snf_path

SOURCES: Dict[str, tuple] = {
    "density": ("rho",),
    "kinetic": ("K",),
    "flux": ("phi", "psi", "J"),
}
ALL_FIELDS = ("rho", "K", "phi", "psi", "J")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def rank_normalise(dist: Dict[str, Dict[int, np.ndarray]], fields=ALL_FIELDS,
                   idx: np.ndarray | None = None) -> Dict[str, Dict[int, np.ndarray]]:
    """Rank-normalise every (field, scale) distance matrix (mutual, then percentile ranks), optionally
    on a subset ``idx``."""
    out = {}
    for f in fields:
        out[f] = {}
        for s, d in dist[f].items():
            d = np.asarray(d, dtype=np.float64)
            if idx is not None:
                d = d[np.ix_(idx, idx)]
            out[f][s] = rank_distance(d).astype(np.float32)
    return out


def source_weights(fields: Sequence[str]) -> Dict[str, float]:
    """Equal weight per physical source present, shared equally by its fields."""
    fields = list(fields)
    present = {src: [f for f in fl if f in fields] for src, fl in SOURCES.items()}
    present = {src: fl for src, fl in present.items() if fl}
    w = {}
    for src, fl in present.items():
        for f in fl:
            w[f] = 1.0 / (len(present) * len(fl))
    return w


def scale_views(Q: Dict[str, Dict[int, np.ndarray]], fields: Sequence[str] = ALL_FIELDS):
    """One affinity view per scale: W_m = 1 - sum_f w_f * rank(d_{f,m})."""
    w = source_weights(fields)
    scales = list(Q[list(w)[0]].keys())
    views = []
    for s in scales:
        d = sum(w[f] * Q[f][s] for f in w)
        views.append(rank_affinity(np.asarray(d, dtype=np.float32)))
    return views


# --------------------------------------------------------------------------- #
# Consensus and clustering
# --------------------------------------------------------------------------- #
def consensus(views: Sequence[np.ndarray], knn: int, t: int, device: str | None = None) -> np.ndarray:
    """SNF of the scale views (a single view is returned as it is)."""
    return snf(list(views), knn, t, device)


def affinity_propagation(S: np.ndarray, damping: float = 0.9, max_iter: int = 1000,
                         convergence_iter: int = 100, random_state: int = 0):
    """AP on the consensus (median preference). Returns (labels, exemplar indices, converged).

    Defaults follow the original implementation of Frey & Dueck (2007): damping 0.9,
    convergence declared after 100 iterations without change, at most 1000 iterations.
    A shorter window (e.g. 20) can stop on a transient exemplar set at some settings.
    """
    labels, ex, converged, _ = affinity_propagation_cpu(S, damping, max_iter, convergence_iter, random_state)
    return labels, ex, converged


def ap_many(S_list, device: str = "auto", **ap):
    """AP on several similarity matrices of the same size, on the GPU when available (same result as
    on the CPU). Returns (labels, exemplar indices, converged) for each matrix."""
    if device == "auto":
        device = "cuda" if gpu_available() else "cpu"
    if str(device).startswith("cuda"):
        res = affinity_propagation_gpu(list(S_list), device=device, **ap)
    else:
        res = [affinity_propagation_cpu(S, **ap) for S in S_list]
    return [r[:3] for r in res]


def _sweep_knn(views, knn, ts, ap, device):
    """One diffusion for every t of this knn, then AP at each t."""
    path = snf_path(views, knn, ts, device)
    if device is not None and str(device).startswith("cuda"):
        res = affinity_propagation_gpu([path[t] for t in ts], device=device, **ap)
    else:
        res = [affinity_propagation_cpu(path[t], **ap) for t in ts]
    return {t: r for t, r in zip(ts, res)}


def sweep(views: Sequence[np.ndarray], grid, damping: float = 0.9, convergence_iter: int = 100,
          max_iter: int = 1000, random_state: int = 0, device: str = "auto", n_jobs: int = 9):
    """Affinity propagation on the SNF consensus at every (knn, t) of ``grid``.

    One diffusion per knn gives the consensus at all its t (snf_path). On a GPU (device "auto" or
    "cuda") the diffusion and AP run in torch and Triton, with the same result as on the CPU bit for
    bit; on the CPU the values of knn run in parallel. Returns one dict per setting, in the order of
    ``grid``: knn, t, C, pred, exemplars, converged, n_iter.
    """
    from joblib import Parallel, delayed
    if device == "auto":
        device = "cuda" if gpu_available() else "cpu"
    ap = dict(damping=damping, convergence_iter=convergence_iter, max_iter=max_iter, random_state=random_state)
    by_knn = {}
    for k, t in grid:
        by_knn.setdefault(int(k), []).append(int(t))
    if len(views) == 1:                                    # no fusion: the same AP at every setting
        r = (affinity_propagation_gpu([views[0]], device=device, **ap) if str(device).startswith("cuda")
             else [affinity_propagation_cpu(views[0], **ap)])[0]
        done = {k: {t: r for t in ts} for k, ts in by_knn.items()}
    elif str(device).startswith("cuda"):
        # the consensus of several knn are gathered into one GPU batch of affinity propagation, as many
        # as the memory holds (all 99 settings at once on Top-1); the result of each is unchanged
        n = np.asarray(views[0]).shape[0]
        cap = gpu_batch_size(n, convergence_iter)
        done = {k: {} for k in by_knn}
        pending = []

        def flush():
            for (k, t), r in zip([kt for kt, _ in pending], affinity_propagation_gpu([S for _, S in pending], device=device, **ap)):
                done[k][t] = r
            pending.clear()
        for k, ts in by_knn.items():
            path = snf_path(views, k, ts, device)
            pending.extend(((k, t), path[t]) for t in ts)
            if len(pending) >= cap:
                flush()
        if pending:
            flush()
    else:
        out = Parallel(n_jobs=min(n_jobs, len(by_knn)), max_nbytes="1M")(
            delayed(_sweep_knn)(views, k, ts, ap, None) for k, ts in by_knn.items())
        done = dict(zip(by_knn, out))
    res = []
    for k, t in grid:
        lab, ex, conv, n_iter = done[int(k)][int(t)]
        res.append(dict(knn=int(k), t=int(t), C=int(len(np.unique(lab))), pred=np.asarray(lab).astype(np.int16),
                        exemplars=np.asarray(ex, dtype=int), converged=bool(conv), n_iter=int(n_iter)))
    return res


def run(Q, fields=ALL_FIELDS, knn: int = 7, t: int = 10, fusion: str = "balanced"):
    """Full fusion + AP for one (knn, t). Returns dict(S, labels, exemplars, converged)."""
    if fusion == "balanced":
        S = consensus(scale_views(Q, fields), knn, t)
    elif fusion == "hierarchical":
        S = hierarchical_snf(Q, knn, t)
    else:
        raise ValueError(fusion)
    labels, ex, conv = affinity_propagation(S)
    return {"S": S, "labels": labels, "exemplars": ex, "converged": conv}


# --------------------------------------------------------------------------- #
# Legacy: three stacked SNF stages (kept for the fusion-design comparison)
# --------------------------------------------------------------------------- #
def hierarchical_snf(Q, knn: int, t: int) -> np.ndarray:
    """SNF(phi,psi,J) per scale -> SNF over scales per source -> rank -> SNF over sources."""
    scales = list(Q["rho"].keys())
    srcs = []
    for fl in SOURCES.values():
        per_scale = []
        for s in scales:
            v = [rank_affinity(Q[f][s]) for f in fl]
            per_scale.append(snf(v, knn, t) if len(v) > 1 else v[0])
        srcs.append(snf(per_scale, knn, t))
    ranked = []
    for A in srcs:
        r = rankdata(A, method="average").reshape(A.shape) / A.size
        np.fill_diagonal(r, 1.0)
        ranked.append(r.astype(np.float32))
    return snf(ranked, knn, t)


# --------------------------------------------------------------------------- #
# Reading the taxonomy: consensus over the plateau of the resolution map
# --------------------------------------------------------------------------- #
def plateau_consensus(results, n_runs: int):
    """Label-free reading of the taxonomy from a sweep over (knn, t).

    ``results`` holds one dict per setting with keys ``C`` (number of clusters),
    ``pred`` (AP labels) and ``exemplars`` (AP exemplar indices). The plateau is the
    set of settings whose C equals the modal C*. The taxonomy is the consensus
    partition of the plateau: co-association (fraction of plateau settings in which
    two runs share a cluster), average linkage, cut at C* clusters. The exemplar of a
    consensus cluster is the member most often chosen as AP exemplar on the plateau
    (ties: highest mean co-association with its cluster). Stability is the mean ARI
    between each plateau partition and the consensus.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score

    Cs = np.array([r["C"] for r in results])
    vals, cnt = np.unique(Cs, return_counts=True)
    c_star = int(vals[cnt.argmax()])
    plateau = [r for r in results if r["C"] == c_star]
    co = np.zeros((n_runs, n_runs), dtype=np.float32)
    ex_count = np.zeros(n_runs)
    for r in plateau:
        p = np.asarray(r["pred"])
        co += (p[:, None] == p[None, :])
        if r.get("exemplars") is not None:
            ex_count[np.asarray(r["exemplars"], dtype=int)] += 1
    co /= len(plateau)
    Z = linkage(squareform(1.0 - co, checks=False), "average")
    labels = fcluster(Z, c_star, "maxclust") - 1
    exemplars = []
    for c in range(labels.max() + 1):
        mem = np.where(labels == c)[0]
        score = ex_count[mem] + 1e-3 * co[np.ix_(mem, mem)].mean(1)
        exemplars.append(int(mem[np.argmax(score)]))
    stab = [adjusted_rand_score(labels, np.asarray(r["pred"])) for r in plateau]
    return {"C": c_star, "n_plateau": len(plateau), "labels": labels, "exemplars": np.array(exemplars),
            "exemplar_frequency": np.array([ex_count[e] / len(plateau) for e in exemplars]),
            "stability": float(np.mean(stab)), "stability_min": float(np.min(stab)), "coassociation": co}
