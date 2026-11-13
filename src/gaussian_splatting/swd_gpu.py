"""
swd_gpu.py - the sliced-Wasserstein distance between all pairs of diagrams, on the GPU
=======================================================================================
The twin of distance.swd (order one): the same numbers bit for bit, for every pair of a list of diagrams
at once (stage 3 of the pipeline).

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations

import numpy as np
import torch


def _numpy_sum_last(x):
    """Sum over the last axis in the order numpy uses (its pairwise summation, starting from 0)."""
    n = x.shape[-1]
    if n < 8:
        r = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        for i in range(n):
            r = r + x[..., i]
        return r
    if n <= 128:
        r = [x[..., j] for j in range(8)]
        i = 8
        while i < n - n % 8:
            for j in range(8):
                r[j] = r[j] + x[..., i + j]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for i in range(n - n % 8, n):
            res = res + x[..., i]
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _numpy_sum_last(x[..., :n2]) + _numpy_sum_last(x[..., n2:])


def _pairs(diagrams, a_all, b_all, n_dirs, device, budget):
    """SWD of order one of the pairs (a_all[k], b_all[k]) of one list of diagrams, as a GPU tensor (float64)."""
    from .distance import sample_directions
    n = len(diagrams)
    dirs = torch.as_tensor(sample_directions(n_dirs), device=device, dtype=torch.float64)
    c, s = dirs[:, 0], dirs[:, 1]
    L = max(1, max(len(np.asarray(d).reshape(-1, 2)) for d in diagrams))
    P = np.full((n, L, 2), np.nan, dtype=np.float64)
    for k, d in enumerate(diagrams):
        d = np.asarray(d, dtype=np.float64).reshape(-1, 2)
        P[k, :len(d)] = d
    P = torch.as_tensor(P, device=device)
    proj = P[..., :1] * c + P[..., 1:] * s                             # (n, L, D)
    diag = (0.5 * (P[..., 0] + P[..., 1]))[..., None] * (c + s)          # (n, L, D)
    D = torch.tensor(float(n_dirs), dtype=torch.float64, device=device)  # a tensor divisor: true quotient
    batch = int(max(16, budget // (2 * L * n_dirs * 8)))
    out = torch.empty(len(a_all), dtype=torch.float64, device=device)
    for k0 in range(0, len(a_all), batch):
        a, b = a_all[k0:k0 + batch], b_all[k0:k0 + batch]
        X = torch.nan_to_num(torch.sort(torch.cat([proj[a], diag[b]], dim=1), dim=1).values, nan=0.0)
        Y = torch.nan_to_num(torch.sort(torch.cat([proj[b], diag[a]], dim=1), dim=1).values, nan=0.0)
        diff = (X - Y).abs()
        del X, Y
        per_dir = torch.cumsum(diff, dim=1)[:, -1]                      # points added one after the other
        out[k0:k0 + batch] = _numpy_sum_last(per_dir) / D
    return out


def swd_condensed_gpu(diagrams, n_dirs=25, device="cuda", budget=2e8):
    """SWD of order one between every pair (i < j) of a list of diagrams, as a condensed vector (float64).

    Same numbers, bit for bit, as the CPU routine distance.swd: the same directions (computed by
    numpy), the same augmentation and NaN padding, the same operations, and the sums in the same order
    (the points one after the other, as numpy sums an inner axis; the directions by numpy's pairwise
    summation). Each diagram is padded and projected once; the pairs are formed by indexing, in batches
    sized to `budget` bytes per array."""
    iu, ju = np.triu_indices(len(diagrams), 1)
    if len(iu) == 0:
        return np.zeros(0, dtype=np.float64)
    a = torch.as_tensor(iu, device=device)
    b = torch.as_tensor(ju, device=device)
    return _pairs(diagrams, a, b, n_dirs, device, budget).cpu().numpy()


def swd_field_condensed_gpu(diagram_sets, n_dirs=25, device="cuda", budget=2e8):
    """The distance of stage 3 for every pair (i < j): the sum over the six sub-diagrams, in their
    order, of the SWD of order one. diagram_sets[i] holds the six diagrams of run i. The pairs stay on
    the GPU and the sum is accumulated there; a sub-diagram empty in every run adds exactly zero and is
    skipped. Same numbers, bit for bit, as adding the six CPU results one after the other."""
    iu, ju = np.triu_indices(len(diagram_sets), 1)
    if len(iu) == 0:
        return np.zeros(0, dtype=np.float64)
    a = torch.as_tensor(iu, device=device)
    b = torch.as_tensor(ju, device=device)
    total = torch.zeros(len(iu), dtype=torch.float64, device=device)
    for s in range(len(diagram_sets[0])):
        diagrams = [d[s] for d in diagram_sets]
        if all(len(np.asarray(d).reshape(-1, 2)) == 0 for d in diagrams):
            continue
        total += _pairs(diagrams, a, b, n_dirs, device, budget)
    return total.cpu().numpy()
