"""
affinity.py - from a distance matrix to an affinity
====================================================
Two conventions, used for different purposes:
  * the method of the paper replaces every distance matrix by the mutual ranks of its pairs (Obayashi
    and Kinoshita 2009), themselves replaced by their percentile ranks among all pairs (rank_distance),
    and turns a rank-normalised distance into the affinity 1 - D (rank_affinity);
  * the standard-SNF baseline uses the affinity of standard SNF (Wang et al. 2014), a Gaussian of the
    raw distance whose scale adapts to the distances of each pair to its nearest neighbours
    (snf_affinity).
net_affinity prepares a fused similarity for affinity propagation (non-negative, symmetric, the
largest similarity on the diagonal).

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations
import numpy as np
from scipy.stats import rankdata


# ------------------------------------------------------------------ #
# The affinity of standard SNF (baseline)
# ------------------------------------------------------------------ #
def snf_affinity(D: np.ndarray, knn: int = 20, mu: float = 0.5) -> np.ndarray:
    """The affinity of standard SNF (Wang et al. 2014, as its reference implementations compute it): a
    Gaussian of the distance, W_ij = N(D_ij; 0, mu eps_ij), whose scale adapts to the pair,
    eps_ij = (mean distance of i to its knn nearest neighbours + the same for j + D_ij) / 3."""
    from scipy.stats import norm
    D = np.asarray(D, dtype=np.float64)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    k = min(int(knn), D.shape[0] - 1)
    T = np.sort(D, axis=1)[:, 1:k + 1].mean(axis=1, keepdims=True) + np.spacing(1)
    eps = (T + T.T + D) / 3.0
    eps = np.where(eps <= np.spacing(1), np.spacing(1), eps)
    W = norm.pdf(D, loc=0.0, scale=mu * eps)
    W[np.isnan(W)] = 0.0
    return 0.5 * (W + W.T)


# ------------------------------------------------------------------ #
# Mutual and percentile ranks, the affinity 1 - D (the method of the paper)
# ------------------------------------------------------------------ #
def mutual_rank(D: np.ndarray) -> np.ndarray:
    """Mutual rank of every pair (Obayashi and Kinoshita 2009): sqrt(rank of j among the neighbours of i
    x rank of i among those of j), the nearest neighbour having rank 1 and ties their average rank; zero
    diagonal. A pair is close only when each recording is among the nearest of the other, whatever the
    density of its own neighbourhood."""
    D = np.asarray(D, dtype=np.float64)
    r = rankdata(D, axis=1) - 1.0                     # the recording itself, at distance 0, has rank 0
    M = np.sqrt(r * r.T)
    np.fill_diagonal(M, 0.0)
    return M


def rank_distance(D: np.ndarray) -> np.ndarray:
    """The rank normalisation of the method: the mutual rank of every pair, then its percentile rank
    among all pairs, so that every matrix has the same marginal distribution."""
    return quantile_norm(mutual_rank(D))


def quantile_norm(D: np.ndarray) -> np.ndarray:
    """Rank-transform the off-diagonal distances to ``[0, 1]`` (symmetric, zero diag)."""
    iu = np.triu_indices_from(D, k=1)
    r = rankdata(D[iu]) / len(D[iu])
    Dn = np.zeros_like(D)
    Dn[iu] = r
    Dn += Dn.T
    return Dn


def rank_affinity(D: np.ndarray) -> np.ndarray:
    """
    Affinity ``1 - D`` for an already rank-normalised distance.

    Build the SNF views from :func:`quantile_norm` outputs, then pass them
    through this (directly, or after averaging fields).
    """
    A = 1.0 - D
    np.fill_diagonal(A, 1.0)
    return np.clip(A, 0.0, 1.0)


# ------------------------------------------------------------------ #
# A fused similarity prepared for affinity propagation
# ------------------------------------------------------------------ #
def net_affinity(P: np.ndarray) -> np.ndarray:
    """Clean a (fused) similarity into a usable affinity: clip, symmetrise, set diagonal."""
    A = np.clip(P, 0.0, None)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, A.max() if A.max() > 0 else 1.0)
    return A


