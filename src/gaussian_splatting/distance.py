"""
distance.py - sliced-Wasserstein distance between persistence diagrams (CPU reference)
=====================================================================================
The sliced-Wasserstein distance of Carriere, Cuturi and Oudot (2017, Definition 3.1 and Algorithm 1), of
order one. Two diagrams D and D' are each augmented with the orthogonal projections onto the diagonal of
the points of the other, so that both hold the same number of points; on each direction theta the points
are projected, sorted and matched in order, and the absolute differences are summed. The distance is the
mean over the directions (paper, Eq. (sw)); the distance between two recordings on a field and a scale is
the sum over the six sub-diagrams (pipeline/stage3_distances.py).

The GPU version (swd_gpu.swd_condensed_gpu) gives the same numbers bit for bit.

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations

import numpy as np


def sample_directions(n_dirs: int) -> np.ndarray:
    """n_dirs distinct unit directions evenly spaced over a half-turn, as rows (cos t, sin t);
    -pi/2 and pi/2 give the same line, so the end point is excluded."""
    angles = np.linspace(-np.pi / 2, np.pi / 2, n_dirs, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def swd(A, B_list, directions, batch_size=64):
    """Sliced-Wasserstein distance of order one between diagram A and each diagram of B_list (float64).

    directions: (D, 2) array of unit vectors (cos t, sin t). The pairs are processed in batches;
    padding with NaN (sorted last, then set to zero) lets diagrams of different sizes share a batch."""
    D = directions.shape[0]
    cos_t, sin_t = directions[:, 0], directions[:, 1]
    diag_w = cos_t + sin_t
    A = np.asarray(A, dtype=np.float64).reshape(-1, 2)
    n_A = len(A)
    if n_A > 0:
        proj_A = A[:, 0:1] * cos_t[None, :] + A[:, 1:2] * sin_t[None, :]
        diag_A = (0.5 * (A[:, 0] + A[:, 1]))[:, None] * diag_w[None, :]
    else:
        proj_A = np.zeros((0, D))
        diag_A = np.zeros((0, D))
    out = np.zeros(len(B_list), dtype=np.float64)
    for start in range(0, len(B_list), batch_size):
        batch = B_list[start:start + batch_size]
        total = n_A + max([1] + [len(B) for B in batch])
        A_aug = np.full((len(batch), total, D), np.nan)
        B_aug = np.full((len(batch), total, D), np.nan)
        for b, B in enumerate(batch):
            B = np.asarray(B, dtype=np.float64).reshape(-1, 2)
            n_B = len(B)
            A_aug[b, :n_A] = proj_A
            B_aug[b, n_B:n_B + n_A] = diag_A                 # the points of A, seen from B's side
            if n_B:
                A_aug[b, n_A:n_A + n_B] = (0.5 * (B[:, 0] + B[:, 1]))[:, None] * diag_w[None, :]
                B_aug[b, :n_B] = B[:, 0:1] * cos_t[None, :] + B[:, 1:2] * sin_t[None, :]
        A_aug.sort(axis=1)
        B_aug.sort(axis=1)
        np.nan_to_num(A_aug, copy=False, nan=0.0)
        np.nan_to_num(B_aug, copy=False, nan=0.0)
        out[start:start + len(batch)] = np.abs(A_aug - B_aug).sum(axis=1).mean(axis=1)
    return out
