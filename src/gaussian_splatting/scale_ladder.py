"""
scale_ladder.py - the scales sigma_1 < ... < sigma_M, from the data
====================================================================
  sigma_min = max(d_min / 2, eps_sigma): half the smallest distance between two agents, floored by the
              tracking precision;
  sigma_max = (D_s / 2.35) / max_factor: the merging width D_s / 2.35, at which footprints D_s apart
              stop being resolved (D_s the largest diameter of the group), divided by the cap factor
              that the pipeline sets without labels (pipeline/config.py SIGMA_MAX_FACTOR);
d_min and D_s are taken over every instant of every run, and the M scales are geometric between the two
bounds (paper, Appendix C).

Author: Olivier Vertu Ndinga Oba
"""

import numpy as np


def compute_scale_ladder(trajectories, M=4, epsilon_sigma=0.01, max_factor=10.0, n_instants=None):
    """M geometric scales from sigma_min = max(d_min / 2, epsilon_sigma) to sigma_max = (D_s / 2.35) / max_factor.

    d_min (smallest distance between two agents) and D_s (largest diameter of the group) are taken over
    every instant of every run, or over n_instants evenly spaced instants of each run if given."""
    d_min, D_s = np.inf, 0.0
    for traj in trajectories:
        pos = np.asarray(traj["pos"], dtype=np.float64)                       # (T, N, 2)
        T, N, _ = pos.shape
        if N < 2:
            continue
        if n_instants is not None:
            pos = pos[np.linspace(0, T - 1, min(T, n_instants), dtype=int)]
        i, j = np.triu_indices(N, 1)
        d = np.sqrt(((pos[:, i] - pos[:, j]) ** 2).sum(-1))                    # every pair, every instant
        d_min = min(d_min, float(d.min()))
        D_s = max(D_s, float(d.max()))
    sigma_min = max(d_min / 2.0, epsilon_sigma)
    sigma_max = (D_s / 2.35) / max_factor      # a tenth of the merging width (paper, Appendix C)
    if sigma_max <= sigma_min:                 # no ladder between the bounds: say so rather than invent one
        raise ValueError(f"sigma_max = {sigma_max:.4g} m is not above sigma_min = {sigma_min:.4g} m")
    return np.geomspace(sigma_min, sigma_max, M)
