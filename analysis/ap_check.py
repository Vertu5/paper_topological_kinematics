#!/usr/bin/env python3
"""
ap_check.py - does affinity propagation have the time to finish, on the largest dataset too?
============================================================================================
Affinity propagation stops when the exemplars have not changed for `convergence_iter` iterations,
within `max_iter`. A longer window, a higher ceiling, another damping in the stable range or another
tie-breaking noise must give the same partitions if the settings of the paper let it finish; the
shorter window of scikit-learn (15 iterations) shows what stopping early would change. On Top-1
and Top-10, every variant below is run at the 99 settings and compared with the paper's settings
(damping 0.9, window 100, at most 1000 iterations, noise 0), setting by setting (adjusted Rand index)
and through the plateau consensus. Label-free; GPU when available (same numbers as scikit-learn).

Usage: python analysis/ap_check.py
Output: results/epuck/paper/ap_check.pkl (resumable; reused only for the same distances)

Author: Olivier Vertu Ndinga Oba
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from classification import fusion as F  # noqa: E402
from pipeline.checkpoint import Store  # noqa: E402

PAPER = dict(damping=0.9, convergence_iter=100, max_iter=1000, random_state=0)
SHORT = ("window 15", dict(convergence_iter=15))      # the default window of scikit-learn, shorter than the paper's
VARIANTS = [SHORT, ("window 200", dict(convergence_iter=200)), ("window 400", dict(convergence_iter=400)),
            ("ceiling 5000", dict(max_iter=5000)), ("damping 0.8", dict(damping=0.8)),
            ("damping 0.95", dict(damping=0.95)), ("noise 1", dict(random_state=1)),
            ("noise 2", dict(random_state=2))]


def main():
    lab, rk, _ = RS.load_labels()
    D, _ = RS.load_distances()
    grid = list(product(RS.KNNS, RS.TS))
    out = Store(RS.OUT / "ap_check.pkl", RS.dist_source())
    for top in (1, 10):
        idx = RS.subset(top, rk)
        views = F.scale_views(F.rank_normalise(D, F.ALL_FIELDS, idx))
        ref = F.sweep(views, grid, **PAPER)
        ref_pc = F.plateau_consensus(ref, len(idx))
        out[(top, "paper")] = dict(iterations=[r["n_iter"] for r in ref], converged=sum(r["converged"] for r in ref),
                                   C=ref_pc["C"], plateau=ref_pc["n_plateau"], stability=ref_pc["stability"])
        print(f"Top-{top} paper settings: AP iterations median {np.median([r['n_iter'] for r in ref]):.0f}, "
              f"max {max(r['n_iter'] for r in ref)}, converged {out[(top, 'paper')]['converged']}/{len(ref)}", flush=True)
        for name, change in VARIANTS:
            if (top, name) in out:
                continue
            res = F.sweep(views, grid, **dict(PAPER, **change))
            pc = F.plateau_consensus(res, len(idx))
            same = [adjusted_rand_score(a["pred"], b["pred"]) for a, b in zip(ref, res)]
            out[(top, name)] = dict(settings_identical=int(sum(s > 0.999999 for s in same)), settings_ari_min=float(min(same)),
                                    consensus_ari=float(adjusted_rand_score(ref_pc["labels"], pc["labels"])),
                                    converged=sum(r["converged"] for r in res), iterations_max=max(r["n_iter"] for r in res))
            r = out[(top, name)]
            print(f"Top-{top} {name:12s}: identical partitions {r['settings_identical']}/{len(res)} "
                  f"(smallest ARI {r['settings_ari_min']:.3f}), consensus ARI {r['consensus_ari']:.3f}, "
                  f"converged {r['converged']}/{len(res)}, iterations max {r['iterations_max']}", flush=True)


if __name__ == "__main__":
    main()
