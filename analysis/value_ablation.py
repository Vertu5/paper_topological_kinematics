#!/usr/bin/env python3
"""
value_ablation.py - does the Jacobi field help the statistical reader?
======================================================================
The field ablation of run_sweep.py, restricted to the combinations that test J, on the distances of
the value reader (32 quantiles of each field at the seeds, 1-D Wasserstein; analysis/baselines.py):
(phi, psi), (phi, psi, J) and (rho, K, phi, psi). The five fields together are the "fieldstat"
entries of baselines_eval.pkl. Same rank normalisation, balanced weights per physical source present,
SNF across scales, affinity propagation and 99 settings; labels are used only for scoring.

Usage: python analysis/value_ablation.py [n_jobs]
Output: results/epuck/paper/value_ablation.pkl  {(combo, top): list of 99 results}
        results/epuck/paper/value_ablation_consensus.pkl  {(combo, top): taxonomy read from the plateau}

Author: Olivier Vertu Ndinga Oba
"""
import os
import sys
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from classification import fusion as F  # noqa: E402
from pipeline.checkpoint import Store, file_fingerprint, load_pickle  # noqa: E402

NJ = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = RS.OUT / "value_ablation.pkl"
CONS = RS.OUT / "value_ablation_consensus.pkl"
COMBOS = [("phi", "psi"), ("phi", "psi", "J"), ("rho", "K", "phi", "psi")]


def main():
    lab, rk, _ = RS.load_labels()
    D = load_pickle(RS.OUT / "baselines.pkl")["fieldstat_D"]
    out = Store(OUT, file_fingerprint(RS.OUT / "baselines.pkl"))      # resumable, for these distances only
    cons = Store(CONS, file_fingerprint(RS.OUT / "baselines.pkl"))     # the taxonomy read from each plateau
    for combo in COMBOS:
        for top in range(1, 11):
            if (combo, top) in out and (combo, top) in cons:
                continue
            idx = RS.subset(top, rk)
            views = F.scale_views(F.rank_normalise(D, combo, idx), combo)
            res = RS.sweep(views, lab[idx], list(product(RS.KNNS, RS.TS)), NJ)
            cons[(combo, top)] = RS.consensus_summary(res, lab[idx])
            for r in res:
                r.pop("pred"); r.pop("exemplars")
            out[(combo, top)] = res
            P = np.array([r["purity"] for r in res])
            print(f"values {'+'.join(combo):14s} Top-{top:2d} median purity={np.median(P):.3f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
