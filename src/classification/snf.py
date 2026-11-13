"""
snf.py - similarity network fusion of the scale views
======================================================
Similarity Network Fusion (Wang et al., 2014). Each view P_v is diffused through its own local kernel
K_v and the mean of the other views,

    P_v  <-  K_v ( sum_{u != v} P_u / (m - 1) ) K_v^T,

then symmetrised and normalised to unit row sums; after t iterations the fused matrix is the
symmetrised mean of the views. The local kernel keeps, in each row, the knn largest affinities (the
recording itself included), normalised to sum to one.

The state after t iterations does not depend on how many iterations follow, so one diffusion gives
the fused matrix at every t of a sweep (snf_path).

The CPU (numpy, scipy) and GPU (torch, Triton) versions compute in float64 and add in the same order,
and return the same matrices bit for bit:
  * a product with a local kernel adds the knn terms of a row in increasing column order (scipy's
    CSR product, and the kernel local_product);
  * a row sum is added element after element (numpy's sum along axis 0 of the transpose, and
    torch.cumsum along the rows of the transpose); for the symmetric matrices of the iterations the
    transpose is the matrix itself.

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.sparse import csr_matrix


def knn_kernel(a: np.ndarray, knn: int):
    """Local kernel of an affinity matrix: in each row the knn largest entries (the diagonal
    included), in increasing column order, divided by their sum. Returns (columns, weights), both
    of shape (n, knn)."""
    a = np.asarray(a, dtype=np.float64)
    k = min(int(knn), a.shape[0] - 1)
    cols = np.sort(np.argpartition(-a, k, axis=1)[:, :k], axis=1)
    vals = np.take_along_axis(a, cols, axis=1)
    total = vals.sum(axis=1)
    total[total == 0] = 1.0
    return cols, vals / total[:, None]


def _row_sums(a: np.ndarray) -> np.ndarray:
    """Row sums added element after element (column sums of the transpose, which numpy adds row
    after row)."""
    return np.ascontiguousarray(a.T).sum(axis=0)


def _snf_path_cpu(affs, kernels, ts):
    m = len(affs)
    P = [a / _row_sums(a)[:, None] for a in affs]
    K = [csr_matrix((w.ravel(), c.ravel(), np.arange(0, c.size + 1, c.shape[1])), shape=a.shape)
         for (c, w), a in zip(kernels, affs)]
    out = {}
    for it in range(1, max(ts) + 1):
        total = P[0] + P[1]
        for p in P[2:]:
            total += p
        Q = [K[v] @ np.ascontiguousarray((K[v] @ ((total - P[v]) / (m - 1))).T) for v in range(m)]
        for v in range(m):
            p = np.add(Q[v].T, Q[v], order="C")            # symmetric, stored by rows
            p *= 0.5
            p /= np.abs(p).sum(axis=0)[:, None]            # row sums = column sums, added row after row
            P[v] = p
        if it in ts:
            Pc = sum(P) / m
            out[it] = 0.5 * np.add(Pc, Pc.T, order="C")
    return out


def _snf_path_gpu(affs, kernels, ts, device):
    import torch
    from .kernels_gpu import local_product
    m = len(affs)
    P = []
    for a in affs:
        a = torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)
        P.append(a / torch.cumsum(a.T.contiguous(), dim=0)[-1][:, None])
    K = [(torch.as_tensor(c, device=device), torch.as_tensor(w, device=device)) for c, w in kernels]
    # torch divides by a Python number through its reciprocal; a tensor divisor gives the true quotient
    m1, mm = (torch.tensor(float(x), dtype=torch.float64, device=device) for x in (m - 1, m))
    out = {}
    for it in range(1, max(ts) + 1):
        total = P[0] + P[1]
        for p in P[2:]:
            total += p
        Q = []
        for v in range(m):
            M = local_product((total - P[v]) / m1, *K[v])
            Q.append(local_product(M.T.contiguous(), *K[v]))
            del M
        for v in range(m):
            p = (Q[v].T + Q[v]).contiguous()
            p *= 0.5
            p /= torch.cumsum(p.abs(), dim=0)[-1][:, None]
            P[v] = p
        del Q
        if it in ts:
            Pc = P[0] + P[1]                               # sum(P) / m, in the order of the CPU code
            for p in P[2:]:
                Pc += p
            Pc /= mm
            out[it] = (0.5 * (Pc + Pc.T).contiguous()).cpu().numpy()
    return out


def snf_path(affs: Sequence[np.ndarray], knn: int, ts: Sequence[int], device: str | None = None) -> Dict[int, np.ndarray]:
    """Fused matrices after each number of iterations in ``ts``, from one diffusion.

    ``device`` None or "cpu" runs numpy and scipy; "cuda" runs torch and Triton (same result bit
    for bit). A single view is returned as it is, for every t.
    """
    affs = [np.asarray(a, dtype=np.float64) for a in affs]
    ts = sorted(set(int(t) for t in ts))
    if len(affs) == 1:
        return {t: affs[0].copy() for t in ts}
    kernels = [knn_kernel(a, knn) for a in affs]
    if device is not None and str(device).startswith("cuda"):
        return _snf_path_gpu(affs, kernels, ts, device)
    return _snf_path_cpu(affs, kernels, ts)


def snf(affs: Sequence[np.ndarray], knn: int = 20, t: int = 8, device: str | None = None) -> np.ndarray:
    """Fuse a list of affinity matrices by cross-diffusion; (n, n) symmetric float64 matrix."""
    return snf_path(affs, knn, [t], device)[int(t)]


def snf_dense(affs: Sequence[np.ndarray], knn: int = 20, t: int = 8) -> np.ndarray:
    """Dense reference of :func:`snf` (same maths, products as dense matrices). For tests only."""
    affs = [np.asarray(a, dtype=np.float64) for a in affs]
    m = len(affs)
    P = [a / a.sum(1, keepdims=True) for a in affs]
    S = []
    for a in affs:
        c, w = knn_kernel(a, knn)
        s = np.zeros_like(a)
        np.put_along_axis(s, c, w, axis=1)
        S.append(s)
    for _ in range(t):
        total = sum(P)
        Pn = [S[v] @ ((total - P[v]) / (m - 1)) @ S[v].T for v in range(m)]
        P = [0.5 * (p + p.T) for p in Pn]
        P = [p / np.abs(p).sum(1, keepdims=True) for p in P]
    Pc = sum(P) / m
    return 0.5 * (Pc + Pc.T)


