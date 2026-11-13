#!/usr/bin/env python3
"""
value_variants.py - how much of the separation do the fields carry by themselves?
==================================================================================
Cruder readings of the same five fields, all through exactly the paper fusion (balanced ranks,
SNF across scales, affinity propagation, 99 settings; labels only for scoring):
  coreset  : 32 quantiles of the field values at the robot-position seeds (the coreset) only
  extras   : 32 quantiles at the added seeds only (near-critical barycentres and J = 0 points)
  moments  : mean and standard deviation of the values at all seeds (2 numbers per field and scale)

Usage: python analysis/value_variants.py [n_jobs]
Output: results/epuck/paper/value_variants.pkl  (Top-1, Top-4, Top-10)

Author: Olivier Vertu Ndinga Oba
"""
import os
import pickle
import sys
import time
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from pipeline.checkpoint import Store  # noqa: E402
from classification import fusion as F  # noqa: E402
OUT = RS.OUT / "value_variants.pkl"
CACHE = RS.RES / "value_variants_D.pkl"          # distance matrices (large, not part of the results)

NJ = int(sys.argv[1]) if len(sys.argv) > 1 else 8
QS = np.linspace(0, 1, 32)
NCORE_FORMER = 300                               # coreset size of files written before the size was recorded


def summarise(ti, M):
    q_core = np.zeros((5, M, len(QS))); q_ext = np.zeros((5, M, len(QS))); mom = np.zeros((5, M, 2))
    n_ext = np.zeros(M)
    for si in range(M):
        with open(RS.RES / "stage1_alpha_fields" / f"traj{ti:04d}_s{si}.pkl", "rb") as fh:
            d = pickle.load(fh)
        nc = int(d.get("n_core", NCORE_FORMER))
        for fi, f in enumerate(F.ALL_FIELDS):
            v = np.asarray(d["fields"][f], dtype=np.float64)
            ok = np.isfinite(v)
            core = v[:nc][ok[:nc]]; ext = v[nc:][ok[nc:]]; allv = v[ok]
            q_core[fi, si] = np.quantile(core, QS) if len(core) else 0.0
            q_ext[fi, si] = np.quantile(ext, QS) if len(ext) else 0.0
            mom[fi, si] = (allv.mean(), allv.std()) if len(allv) else (0.0, 0.0)
        n_ext[si] = len(d["seeds"]) - nc
    return q_core, q_ext, mom, n_ext


def evaluate(D, y, idx):
    views = F.scale_views(F.rank_normalise(D, F.ALL_FIELDS, idx))
    return RS.sweep(views, y, list(product(RS.KNNS, RS.TS)), NJ)


def main():
    lab, rk, _ = RS.load_labels()
    _, ladder = RS.load_distances()
    M = len(ladder)
    res = Store(OUT, f"{RS.stage1_source()}+order one")  # resumable, for these stage-1 fields and this order only
    cache = Store(CACHE, f"{RS.stage1_source()}+order one")
    t0 = time.time()
    if "D" not in cache:
        S = Parallel(n_jobs=NJ)(delayed(summarise)(ti, M) for ti in range(len(lab)))
        qc = np.stack([s[0] for s in S]); qe = np.stack([s[1] for s in S]); mo = np.stack([s[2] for s in S])
        Z = (mo - mo.mean(0)) / np.maximum(mo.std(0), 1e-12)
        cache["D"] = dict(
            coreset={f: {si: cdist(qc[:, fi, si], qc[:, fi, si], "cityblock") / len(QS) for si in range(M)}
                     for fi, f in enumerate(F.ALL_FIELDS)},
            extras={f: {si: cdist(qe[:, fi, si], qe[:, fi, si], "cityblock") / len(QS) for si in range(M)}
                    for fi, f in enumerate(F.ALL_FIELDS)},
            moments={f: {si: cdist(Z[:, fi, si], Z[:, fi, si]) for si in range(M)} for fi, f in enumerate(F.ALL_FIELDS)})
        res["n_ext"] = np.stack([s[3] for s in S])
        print(f"summaries done [{time.time() - t0:.0f}s]", flush=True)
    for top in (1, 4, 10):
        idx = RS.subset(top, rk); y = lab[idx]
        for name in ("moments", "coreset", "extras"):
            if (name, top) in res:
                continue
            res[(name, top)] = evaluate(cache["D"][name], y, idx)
            P = np.array([r["purity"] for r in res[(name, top)]])
            print(f"{name:8s} Top-{top:2d} median purity {np.median(P):.3f} [{time.time() - t0:.0f}s]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
