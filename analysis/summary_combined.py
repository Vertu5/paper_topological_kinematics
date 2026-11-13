#!/usr/bin/env python3
"""
summary_combined.py - do persistence diagrams and field-value distributions complement each other?
===================================================================================================
Same five fields, same balanced weights, same SNF and AP; each field's weight is shared equally between
its persistence distance and its value-distribution distance (both rank-normalised).
Output: results/epuck/paper/summary_combined.pkl  ({top: list of grid results with pred})

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
from classification.affinity import rank_affinity  # noqa: E402

NJ = int(os.environ.get("COMB_JOBS", "6"))
PATH = RS.OUT / "summary_combined.pkl"


def main():
    lab, rk, _ = RS.load_labels()
    D_swd, _ = RS.load_distances()
    D_val = load_pickle(RS.OUT / "baselines.pkl")["fieldstat_D"]
    out = Store(PATH, f"{RS.dist_source()}+{file_fingerprint(RS.OUT / 'baselines.pkl')}")
    w = F.source_weights(F.ALL_FIELDS)
    for top in range(1, 11):
        if top in out:
            continue
        idx = RS.subset(top, rk)
        Qp = F.rank_normalise(D_swd, F.ALL_FIELDS, idx)
        Qv = F.rank_normalise(D_val, F.ALL_FIELDS, idx)
        views = [rank_affinity(np.asarray(sum(w[f] * 0.5 * (Qp[f][s] + Qv[f][s]) for f in w), dtype=np.float32))
                 for s in Qp["rho"]]
        res = RS.sweep(views, lab[idx], list(product(RS.KNNS, RS.TS)), NJ)
        out[top] = res
        pc = F.plateau_consensus(res, len(idx))
        s = RS.score(lab[idx], pc["labels"])
        print(f"combined Top-{top:2d}: C*={pc['C']} ({pc['n_plateau']}/99) stability={pc['stability']:.3f} "
              f"purity={s['purity']:.3f} ARI={s['ari']:.3f}", flush=True)


if __name__ == "__main__":
    main()
