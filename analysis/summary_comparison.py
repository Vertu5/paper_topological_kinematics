#!/usr/bin/env python3
"""
summary_comparison.py - persistence diagrams vs field-value distributions, same fields and fusion
=================================================================================================
For every Top-n, the taxonomy of both summaries is built on its own plateau (same consensus rule) and compared
on: plateau size, stability, purity, homogeneity, completeness, ARI, alignment of the clusters with
the controllers inside each mission (median NMI), and monotonicity of C in t.
Output: printed table and results/epuck/paper/summary_comparison.pkl

Author: Olivier Vertu Ndinga Oba
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from classification.fusion import plateau_consensus  # noqa: E402
import run_sweep as RS  # noqa: E402
from pipeline.checkpoint import save_pickle  # noqa: E402

P = RS.OUT
G = pickle.load(open(P / "grid.pkl", "rb"))
B = pickle.load(open(P / "baselines_eval.pkl", "rb"))


def describe(results, y, ranks):
    pc = plateau_consensus(results, len(y))
    lab = pc["labels"]
    cm_p = RS.score(y, lab)
    h, c, _ = homogeneity_completeness_v_measure(y, lab)
    nmi_ctrl = np.median([normalized_mutual_info_score(ranks[y == m], lab[y == m]) for m in range(5)])
    grid = {(r["knn"], r["t"]): r["C"] for r in results}
    mono = sum(all(grid[(k, RS.TS[i + 1])] <= grid[(k, RS.TS[i])] for i in range(len(RS.TS) - 1)) for k in RS.KNNS)
    return dict(C=pc["C"], plateau=pc["n_plateau"], stability=pc["stability"], purity=cm_p["purity"], ari=cm_p["ari"],
                hom=h, comp=c, ctrl_nmi=nmi_ctrl, monotone=mono)


out = {}
print(f"{'set':>6} {'reader':>12} {'C*':>3} {'plat':>5} {'stab':>6} {'pur':>6} {'hom':>6} {'comp':>6} {'ARI':>6} {'ctrlNMI':>7} {'mono':>5}")
for top in range(1, 11):
    g = G[top]; y = g["labels"]; rk = g["ranks"]
    rows = {"persistence": describe(g["results"], y, rk)}
    if ("fieldstat", top) in B:
        rows["values"] = describe(B[("fieldstat", top)], y, rk)
    out[top] = rows
    for name, d in rows.items():
        print(f"Top-{top:<2} {name:>12} {d['C']:>3} {d['plateau']:>5} {d['stability']:6.3f} {d['purity']:6.3f} {d['hom']:6.3f} "
              f"{d['comp']:6.3f} {d['ari']:6.3f} {d['ctrl_nmi']:7.3f} {d['monotone']:>3}/9")
save_pickle(out, P / "summary_comparison.pkl")
