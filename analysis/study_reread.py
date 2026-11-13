#!/usr/bin/env python3
"""
study_reread.py - the parameter study read again with the current method, from its stored distances
===================================================================================================
analysis/parameter_study.py varies each setting alone on Top-1: it computes the fields, the diagrams and
the distances of every version, then reads its taxonomy. When only the reading of the distances changes
(here the mutual ranks of the rank normalisation), the distances stay valid. This script reads every
version of the three plans again with the current fusion and affinity propagation, from the stored
distances, compares it with the same reference as before, and rewrites the summary of each plan; the
previous summary is kept as summary_<tag>.pkl. Nothing upstream is recomputed and no cache is touched.

Usage: python analysis/study_reread.py [tag of the previous reading, default "percentile"]

Author: Olivier Vertu Ndinga Oba
"""
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "src"), str(ROOT)]
import parameter_study as PS  # noqa: E402
import run_sweep as RS  # noqa: E402
from pipeline.checkpoint import code_fingerprint, load_pickle, save_pickle, save_text  # noqa: E402

PLANS = {"main": ("parameter_study", None), "extension": ("parameter_study_ext", PS.EXTENSION),
         "final": ("parameter_study_final", PS.FINAL)}


def reference(name, value, others, plan):
    """The configuration a version was compared with in the study, and its label."""
    if plan is None:                                                   # main plan
        if name in dict(PS.RESOLUTION):
            finest = (PS.FINEST.get(name, max))(dict(PS.RESOLUTION)[name])
            return dict(PS.PAPER, **{name: finest}), f"{name}={finest}"
        return dict(PS.PAPER), "paper"
    for n, values, ref_value, oth in plan:
        if n == name and dict(oth) == dict(others) and value in values:
            return dict(PS.PAPER, **oth, **{name: ref_value}), f"{name}={ref_value}"
    raise KeyError((name, value, others))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "percentile"
    lab, rk, _ = RS.load_labels()
    runs = [int(g) for g in RS.subset(1, rk)]
    y = lab[runs]
    lad = load_pickle(RS.RES / "scale_ladder.pkl")
    base = (np.asarray(lad["ladder"], float), float(lad["max_factor"]))
    fusion = code_fingerprint(*(ROOT / "src" / "classification" / f for f in ("affinity.py", "fusion.py", "snf.py", "ap.py")))
    for plan_name, (folder_name, plan) in PLANS.items():
        folder = ROOT / "results" / folder_name
        old = folder / "summary.pkl"
        kept = folder / f"summary_{tag}.pkl"
        if not kept.exists():                                          # the previous reading, kept once
            shutil.copy2(old, kept)
            shutil.copy2(folder / "summary.pkl.source", folder / f"summary_{tag}.pkl.source")
        rows_before = load_pickle(kept)
        cache, out = {}, {}

        def version(cfg):
            k = PS.key(cfg, tuple(PS.PAPER))
            if k not in cache:
                D = load_pickle(folder / "stage3" / f"{PS.key(cfg, PS.STAGE1 + ('alpha', 'n_dirs'))}.pkl")
                s1 = load_pickle(folder / "stage1" / f"{PS.key(cfg, PS.STAGE1)}.pkl")
                res, tax = PS.taxonomy(cfg, D, len(runs))
                cache[k] = dict(D=D, res=res, tax=tax, time1=float(s1["time"]), sigma=PS.ladder_of(cfg["ladder"], base),
                                seeds=float(np.mean([len(v[0]) for kk, v in s1.items() if kk != "time"])))
            return cache[k]
        for row_key in rows_before:
            name, value = row_key[0], row_key[1]
            others = dict(row_key[2]) if len(row_key) > 2 else {}
            ref_cfg, ref_name = reference(name, value, others, plan)
            v = version(dict(PS.PAPER, **others, **{name: value}))
            ref = version(ref_cfg)
            out[row_key] = dict(PS.compare(v, ref), C=v["tax"]["C"], plateau=v["tax"]["n_plateau"],
                                stability=v["tax"]["stability"], purity=float(RS.score(y, v["tax"]["labels"])["purity"]),
                                seeds=v["seeds"], stage1_s=v["time1"], reference=ref_name,
                                iterations=float(np.median([r["n_iter"] for r in v["res"]])),
                                converged=int(sum(r["converged"] for r in v["res"])))
            b, a = rows_before[row_key], out[row_key]
            print(f"{plan_name:9s} {str(row_key)[:60]:60s} C*={a['C']} plateau {b['plateau']:2d} -> {a['plateau']:2d}, "
                  f"stability {b['stability']:.3f} -> {a['stability']:.3f}, ARI {b['ari_consensus']:.3f} -> {a['ari_consensus']:.3f}",
                  flush=True)
        save_pickle(out, old)
        save_text(f"{RS.data_source()}+reread:{fusion}", folder / "summary.pkl.source")
    print("done", flush=True)


if __name__ == "__main__":
    main()
