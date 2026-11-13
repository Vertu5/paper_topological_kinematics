#!/usr/bin/env python3
"""
merge_analysis.py - which XOR Aggregation runs would join the Shelter cluster, and why the mutual ranks
=========================================================================================================
Read with percentile ranks alone instead of the mutual ranks of the method, the taxonomies of Top-3 and
Top-4 place XOR Aggregation runs in the Shelter cluster (the method keeps them with their mission). This
script reads Top-3 and Top-4 that way, then, for these runs, the other XOR runs and the Shelter runs,
records the controllers involved, the mean distance to each group field by field under both kinds of
ranks, the missions of their KNN nearest neighbours in each scale view under both kinds of ranks (what the
diffusion links them to), the time after which fewer than a tenth of the robots still move, the mean
speed, and the final positions. Labels are used only to define the three groups after the taxonomy is read.
Output: results/epuck/paper/merge_analysis.pkl

Author: Olivier Vertu Ndinga Oba
"""
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics.cluster import contingency_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from pipeline.checkpoint import save_pickle  # noqa: E402
from classification import fusion as F  # noqa: E402
from classification.affinity import quantile_norm  # noqa: E402

DT = 0.1
MOVING = 0.01          # m/s: a robot slower than this is considered at rest
GROUPS = ("absorbed", "kept", "shelter")
KNN = 7                # the neighbourhood of the illustrations (Section 5)


def dynamics(P, V):
    speed = np.linalg.norm(V, axis=2)
    moving = (speed > MOVING).mean(1)
    above = np.where(moving > 0.1)[0]
    settle = (above[-1] + 1) * DT if len(above) else 0.0
    return moving, settle, float(speed.mean()), P[-1]


def main():
    lab, rk, _ = RS.load_labels()
    D, _ = RS.load_distances()
    meta = pickle.load(open(RS.RES / "meta.pkl", "rb"))["trajectories"]
    grid = list(product(RS.KNNS, RS.TS))
    out = {}
    for top in (3, 4):
        idx = RS.subset(top, rk)
        Qp = {f: {s: quantile_norm(np.asarray(D[f][s], dtype=np.float64)[np.ix_(idx, idx)]).astype(np.float32) for s in D[f]}
              for f in F.ALL_FIELDS}                         # percentile ranks alone
        y = lab[idx]
        p = F.plateau_consensus(F.sweep(F.scale_views(Qp), grid), len(idx))["labels"]
        home = contingency_matrix(y, p).argmax(1)
        members = {"absorbed": np.where((y == 4) & (p == home[3]))[0],
                   "kept": np.where((y == 4) & (p == home[4]))[0],
                   "shelter": np.where(y == 3)[0]}
        r = rk[idx]
        ctrl = {int(c) + 1: (int(((r == c) & np.isin(np.arange(len(y)), members["absorbed"])).sum()),
                             int(((r == c) & (y == 4)).sum())) for c in np.unique(r[y == 4])}
        Qm = F.rank_normalise(D, F.ALL_FIELDS, idx)              # the mutual ranks of the method
        fields, fields_mutual = {}, {}
        for Q, target in ((Qp, fields), (Qm, fields_mutual)):
            for f in F.ALL_FIELDS:
                d = np.mean([Q[f][s] for s in Q[f]], axis=0)
                target[f] = {(a, b): float(d[np.ix_(members[a], members[b])].mean())
                             for a in ("absorbed", "kept") for b in ("shelter", "kept")}
        dyn = {}
        for g, mem in members.items():
            res = [dynamics(np.asarray(meta[int(idx[i])]["pos"], float), np.asarray(meta[int(idx[i])]["vel"], float))
                   for i in mem]
            Lmin = min(len(x[0]) for x in res)
            dyn[g] = dict(moving=np.median([x[0][:Lmin] for x in res], axis=0),
                          settle=float(np.median([x[1] for x in res])),
                          speed=float(np.median([x[2] for x in res])),
                          final=np.concatenate([x[3] for x in res]))
        neighbours = {}
        for name, Q in (("percentile", Qp), ("mutual", Qm)):
            share = []
            for W in F.scale_views(Q):
                W = W.copy()
                np.fill_diagonal(W, -np.inf)
                nn = np.argsort(-W[members["absorbed"]], axis=1)[:, :KNN]
                share.append([(y[nn] == 3).mean(), (y[nn] == 4).mean()])
            neighbours[name] = dict(zip(("shelter", "xor"), np.mean(share, axis=0).tolist()))
        out[top] = dict(n=len(members["absorbed"]), controllers=ctrl, fields=fields, fields_mutual=fields_mutual,
                        neighbours=neighbours, dynamics=dyn, reading="percentile ranks alone")
        print(f"Top-{top}: {len(members['absorbed'])} XOR runs in the Shelter cluster; controllers {ctrl}; "
              f"Shelter among their {KNN} nearest neighbours: {100 * neighbours['percentile']['shelter']:.0f} % "
              f"(percentile), {100 * neighbours['mutual']['shelter']:.0f} % (mutual)")
        for g in GROUPS:
            print(f"   {g:8s} settle {dyn[g]['settle']:.0f} s, mean speed {dyn[g]['speed']:.3f} m/s")
        for f in F.ALL_FIELDS:
            v, w = fields[f], fields_mutual[f]
            print(f"   {f:4s} percentile: absorbed->Shelter {v[('absorbed', 'shelter')]:.2f}, ->other XOR {v[('absorbed', 'kept')]:.2f}"
                  f" | mutual: ->Shelter {w[('absorbed', 'shelter')]:.2f}, ->other XOR {w[('absorbed', 'kept')]:.2f}")
    save_pickle(out, RS.OUT / "merge_analysis.pkl")


if __name__ == "__main__":
    main()
