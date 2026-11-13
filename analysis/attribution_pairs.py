#!/usr/bin/env python3
"""
attribution_pairs.py - the attribution of Section 6.6 on every pair of missions
================================================================================
Runs analysis/attribution.py on the Top-1 exemplars of every pair of the five missions, each on the field that
separates the pair best (the bold entries of the pair separation figure, results/epuck/paper/pair_separation.pkl), and gathers what
the paper reads for both summaries: where the split of the distance falls, how stable it is under noise, and
whether its largest structures recur over the runs of the first controller of the pair.

Usage: python analysis/attribution_pairs.py
Output: results/epuck/paper/attribution_pairs/<a>_<b>_<field>.pkl (one run of attribution.py per pair)
        results/epuck/paper/attribution_pairs.pkl (the summary)

Author: Olivier Vertu Ndinga Oba
"""
import os
import pickle
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "epuck" / "paper"
OUT = PAPER / "attribution_pairs"
ROWS = {"rho": 0, "K": 1, "phi": 2, "psi": 3, "J": 4}     # rows of the separation table


def summary(A):
    """What the paper reads in one run of attribution.py."""
    T, sh = np.asarray(A["types_A"]), np.asarray(A["persistence_share_A"])
    ext = np.where(np.isin(T, ("min", "max")))[0]
    top2 = ext[np.argsort(sh[ext])[-2:]]
    kinds = tuple(sorted(str(t) for t in T[top2]))
    rp = A["reproducibility"]
    same = [r for r in rp["centres"]["per_run"] if r["types"] == kinds]
    return dict(field=A["field"], missions=A["missions"], distance=A["persistence_distance"],
                n_seeds=int((sh > 0).sum()), n_total=len(sh),
                critical=A["persistence_shares_A"]["critical"], values_critical=A["values_shares_A"]["critical"],
                top2_kinds=kinds, top2_share=float(sh[top2].sum()),
                stab2=(A["stability"][0.002]["persistence_same"], A["stability"][0.002]["values_same"]),
                stab5=(A["stability"][0.005]["persistence_same"], A["stability"][0.005]["values_same"]),
                runs=len(rp["runs"]), recur=len(same),
                recur_share=float(np.median([r["share"] for r in rp["centres"]["per_run"]])),
                overlap=(rp["persistence"]["mean_overlap"], rp["values"]["mean_overlap"]),
                half_mean=(rp["persistence"]["area_half"], rp["values"]["area_half"]),
                half_robots=(A["persistence_robot_area_half_A"], A["values_robot_area_half_A"]))


def main():
    PS = pickle.load(open(PAPER / "pair_separation.pkl", "rb"))
    R, pairs = PS["R"], [tuple(p) for p in PS["pairs"]]
    OUT.mkdir(exist_ok=True)
    out = {}
    for a, b in combinations(range(5), 2):
        j = pairs.index((a, b))
        field = max(ROWS, key=lambda f: R[ROWS[f], j])             # the field that separates the pair best
        f = OUT / f"{a}_{b}_{field}.pkl"
        if not f.exists():
            env = dict(os.environ, ATTR_PAIR=f"{a},{b}", ATTR_FIELD=field, ATTR_OUT=str(f))
            subprocess.run([sys.executable, str(ROOT / "analysis" / "attribution.py")], env=env, check=True)
        with open(f, "rb") as fh:
            out[(a, b)] = summary(pickle.load(fh))
        s = out[(a, b)]
        print(f"{s['missions'][0]:>8s} / {s['missions'][1]:<8s} {field}: {s['n_seeds']} seeds, critical {s['critical']:.2f} "
              f"(values {s['values_critical']:.2f}); two largest extrema {s['top2_kinds']} {s['top2_share']:.2f}; "
              f"2 mm {s['stab2'][0]:.2f}/{s['stab2'][1]:.2f}; recur {s['recur']}/{s['runs']}", flush=True)
    with open(PAPER / "attribution_pairs.pkl", "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
