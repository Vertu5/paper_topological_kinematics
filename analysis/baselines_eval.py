#!/usr/bin/env python3
"""
baselines_eval.py - score every baseline with the paper's clustering (current AP defaults)
=========================================================================================
Distances come from baselines.py (cached in baselines.pkl). For every Top-n and all 99
(knn, t) settings:
  order, crocker : one view each, diffused on its own kNN graph with the same SNF update
                   (two identical views), then Affinity Propagation
  fieldstat      : same seeds and same five fields, value distributions instead of
                   persistence diagrams, then exactly our fusion (4 scale views, SNF, AP)
  single views   : order, crocker, our fields per single scale and without diffusion,
                   AP directly (no k, t); scored also by homogeneity and completeness
Output: results/epuck/paper/baselines_eval.pkl

Author: Olivier Vertu Ndinga Oba
"""
import os
import sys
import time
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from classification import fusion as F  # noqa: E402
from pipeline.checkpoint import Store, file_fingerprint, load_pickle  # noqa: E402
from classification.affinity import rank_affinity, rank_distance  # noqa: E402

NJ = int(os.environ.get("BASELINE_JOBS", "16"))
PATH = RS.OUT / "baselines_eval.pkl"


def main():
    lab, rk, _ = RS.load_labels()
    base = load_pickle(RS.OUT / "baselines.pkl")
    D_swd, _ = RS.load_distances()
    out = Store(PATH, f"{RS.dist_source()}+{file_fingerprint(RS.OUT / 'baselines.pkl')}")
    t0 = time.time()
    for top in range(1, 11):
        idx = RS.subset(top, rk); y = lab[idx]
        if ("single", top) not in out or "hom" not in out[("single", top)]["nodiff"]:
            views = F.scale_views(F.rank_normalise(D_swd, F.ALL_FIELDS, idx))
            mats = [rank_affinity(rank_distance(base[name]["D"][np.ix_(idx, idx)]).astype(np.float32))
                    for name in ("order", "crocker")] + list(views) + [np.mean(views, axis=0)]
            names = ["order", "crocker"] + [f"scale{si + 1}" for si in range(len(views))] + ["nodiff"]
            res = {name: dict(RS.score(y, l_), converged=c_, **dict(zip(("hom", "comp"), homogeneity_completeness_v_measure(y, l_))))
                   for name, (l_, _, c_) in zip(names, F.ap_many(mats))}
            out[("single", top)] = res
            print(f"single Top-{top}: " + " | ".join(f"{k} C={v['C']} P={v['purity']:.3f}" for k, v in res.items()), flush=True)
        for name in ("fieldstat", "order", "crocker"):
            if (name, top) in out:
                continue
            if name == "fieldstat":
                views = F.scale_views(F.rank_normalise(base["fieldstat_D"], F.ALL_FIELDS, idx))
            else:
                A = rank_affinity(rank_distance(base[name]["D"][np.ix_(idx, idx)]).astype(np.float32))
                views = [A, A]
            res = RS.sweep(views, y, list(product(RS.KNNS, RS.TS)), NJ)
            out[(name, top)] = res
            C = np.array([r["C"] for r in res]); P = np.array([r["purity"] for r in res])
            v, c = np.unique(C, return_counts=True)
            print(f"{name:9s} Top-{top:2d}: modal C={v[c.argmax()]} ({c.max()}/99) plateau purity={np.median(P[C == v[c.argmax()]]):.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
