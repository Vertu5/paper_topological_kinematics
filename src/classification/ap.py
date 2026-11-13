"""
ap.py - affinity propagation on the CPU (scikit-learn) and on the GPU (same arithmetic)
=======================================================================================
Affinity propagation (Frey and Dueck, 2007) on a precomputed similarity, with the median similarity
as preference, as scikit-learn computes it (sklearn.cluster.AffinityPropagation). The GPU version
runs a batch of similarity matrices at once and repeats the float64 operations of scikit-learn in
the same order, so that it returns the same exemplars, labels and numbers of iterations:
  * the tie-breaking noise comes from the same generator (numpy RandomState) and is added the same way;
  * every message update is elementwise, a row maximum or a column sum; scikit-learn adds a column
    row after row, and so do the Triton kernels (kernels_gpu.py), without fused multiply-add;
  * the clusters are read from the exemplars by the code of scikit-learn, on the CPU (_labels).

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np
from sklearn.cluster import AffinityPropagation

from .affinity import net_affinity

_NOISE = {}


def affinity_propagation_cpu(S: np.ndarray, damping: float = 0.9, max_iter: int = 1000,
                             convergence_iter: int = 100, random_state: int = 0):
    """scikit-learn on net_affinity(S). Returns (labels, exemplars, converged, n_iter)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ap = AffinityPropagation(affinity="precomputed", damping=damping, max_iter=max_iter,
                                 convergence_iter=convergence_iter, random_state=random_state).fit(net_affinity(S))
    converged = not any("converge" in str(c.message) for c in caught)
    ex = np.asarray([] if ap.cluster_centers_indices_ is None else ap.cluster_centers_indices_, dtype=int)
    return ap.labels_.astype(int), ex, converged, int(ap.n_iter_)


def _noise(n: int, random_state: int) -> np.ndarray:
    """The standard normal draws scikit-learn adds to the similarities (RandomState(random_state))."""
    key = (n, random_state)
    if key not in _NOISE:
        if len(_NOISE) > 12:
            _NOISE.clear()
        _NOISE[key] = np.random.RandomState(random_state).standard_normal(size=(n, n))
    return _NOISE[key]


def _labels(S: np.ndarray, E: np.ndarray, never_converged: bool):
    """Clusters from the exemplars E (boolean), as scikit-learn reads them after the message loop."""
    n = S.shape[0]
    I = np.flatnonzero(E)
    K = I.size
    if K == 0:
        return np.full(n, -1), np.array([], dtype=int), False
    c = np.argmax(S[:, I], axis=1)
    c[I] = np.arange(K)
    for k in range(K):
        ii = np.asarray(c == k).nonzero()[0]
        j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
        I[k] = ii[j]
    c = np.argmax(S[:, I], axis=1)
    c[I] = np.arange(K)
    labels = I[c]
    centers = np.unique(labels)
    return np.searchsorted(centers, labels).astype(int), centers.astype(int), not never_converged


def _prepare(S_np, random_state, device):
    """net_affinity, median preference on the diagonal, tie-breaking noise; float64 on the GPU."""
    import torch
    S = torch.as_tensor(np.asarray(S_np, dtype=np.float64), device=device)
    n = S.shape[0]
    S = S.clamp(min=0.0)
    S = 0.5 * (S + S.T)
    mx = S.max()
    S.fill_diagonal_(float(mx) if mx > 0 else 1.0)
    srt = torch.sort(S.reshape(-1)).values
    N = n * n
    pref = srt[N // 2] if N % 2 else (srt[N // 2 - 1] + srt[N // 2]) / 2.0
    del srt
    offdiag = S[~torch.eye(n, dtype=torch.bool, device=device)]
    if n == 1 or bool((offdiag == offdiag[0]).all()):
        return None                                        # degenerate case: left to scikit-learn
    S.diagonal().copy_(pref.expand(n))
    eps, tiny = np.finfo(np.float64).eps, np.finfo(np.float64).tiny
    Z = torch.as_tensor(_noise(n, random_state), device=device)
    S += (S * float(eps) + float(tiny) * 100) * Z
    return S


def gpu_batch_size(n: int, convergence_iter: int = 100, memory: float = 0.35, device: str = "cuda") -> int:
    """How many n x n problems one GPU batch of affinity propagation holds (S, A, R in float64)."""
    import torch
    free = torch.cuda.mem_get_info(device)[0]
    return int(max(1, memory * free // (3 * 8 * n * n + 16 * n * convergence_iter)))


def affinity_propagation_gpu(S_list: Sequence[np.ndarray], damping: float = 0.9, max_iter: int = 1000,
                             convergence_iter: int = 100, random_state: int = 0, device: str = "cuda",
                             memory: float = 0.35) -> List[tuple]:
    """Affinity propagation on each matrix of ``S_list`` (same size), on the GPU, as scikit-learn
    computes it. Returns one (labels, exemplars, converged, n_iter) per matrix."""
    import torch
    from .kernels_gpu import ap_iteration
    if len(S_list) == 0:
        return []
    n = np.asarray(S_list[0]).shape[0]
    B = min(len(S_list), gpu_batch_size(n, convergence_iter, memory, device))
    out = [None] * len(S_list)
    damp = torch.tensor([damping, 1 - damping], dtype=torch.float64, device=device)
    for b0 in range(0, len(S_list), B):
        ids, prepared = [], []
        for b in range(b0, min(b0 + B, len(S_list))):
            S = _prepare(S_list[b], random_state, device)
            if S is None:
                out[b] = affinity_propagation_cpu(S_list[b], damping, max_iter, convergence_iter, random_state)
            else:
                ids.append(b); prepared.append(S)
        if not ids:
            continue
        S = torch.stack(prepared); del prepared
        A = torch.zeros_like(S); R = torch.zeros_like(S)
        CS = torch.empty((len(ids), n), dtype=torch.float64, device=device)
        e = torch.zeros((len(ids), convergence_iter, n), dtype=torch.bool, device=device)
        se = torch.zeros((len(ids), n), dtype=torch.int32, device=device)
        for it in range(max_iter):
            ap_iteration(S, A, R, CS, damp)
            E = (A.diagonal(dim1=1, dim2=2) + R.diagonal(dim1=1, dim2=2)) > 0
            slot = it % convergence_iter
            se += E.int() - e[:, slot].int()
            e[:, slot] = E
            last = it == max_iter - 1
            if it >= convergence_iter or last:
                done = (((se == convergence_iter) | (se == 0)).all(1) & (E.sum(1) > 0)) if it >= convergence_iter \
                    else torch.zeros(len(ids), dtype=torch.bool, device=device)
                stop = done | last
                if bool(stop.any()):
                    for j in torch.nonzero(stop).ravel().tolist():
                        conv = bool(done[j])
                        out[ids[j]] = _labels(S[j].cpu().numpy(), E[j].cpu().numpy(), not conv) + (it + 1,)
                    keep = ~stop
                    if not bool(keep.any()):
                        break
                    S, A, R, e, se = S[keep], A[keep], R[keep], e[keep], se[keep]
                    CS = CS[: int(keep.sum())]
                    ids = [i for i, k in zip(ids, keep.tolist()) if k]
        del S, A, R, CS, e, se
    return out


def gpu_available() -> bool:
    """A CUDA device with torch, and triton for the kernels."""
    import importlib.util
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
        return False
    import torch
    return torch.cuda.is_available()
