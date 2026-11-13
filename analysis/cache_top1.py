#!/usr/bin/env python3
"""
cache_top1.py - the Top-1 inputs of the figures and notebooks
=============================================================
Two small files read by the figures, the notebooks and the attribution:
  * top1_tracks.pkl: positions and velocities of the 250 runs of Top-1 (one controller per mission),
    with their indices among the 2500 runs, missions and controllers;
  * reference_top1.pkl: the consensus S of Top-1 at the reference setting (knn = 7, t = 10), its
    affinity propagation (labels, exemplars) and the scale ladder.
Each is rebuilt only when its inputs (the recordings, the stage-3 distances) have changed.

Usage: python analysis/cache_top1.py

Author: Olivier Vertu Ndinga Oba
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from classification import fusion as F  # noqa: E402
from pipeline.checkpoint import load_pickle  # noqa: E402

KNN, T = 7, 10


def main():
    lab, rk, ctrl = RS.load_labels()
    idx = RS.subset(1, rk)
    if not RS.done("top1_tracks.pkl", RS.data_source()):
        trajs = load_pickle(RS.RES / "meta.pkl")["trajectories"]
        RS.save(dict(tracks=[dict(pos=trajs[g]["pos"], vel=trajs[g]["vel"]) for g in idx], idx=idx,
                     labels=lab[idx], controllers=ctrl[idx]), "top1_tracks.pkl", RS.data_source())
    if not RS.done("reference_top1.pkl"):
        D, ladder = RS.load_distances()
        S = F.consensus(F.scale_views(F.rank_normalise(D, F.ALL_FIELDS, idx)), KNN, T)
        labels, exemplars, _ = F.affinity_propagation(S)
        RS.save(dict(knn=KNN, t=T, S=S, labels=labels, exemplars=exemplars, idx=idx, y=lab[idx],
                     ladder=list(ladder)), "reference_top1.pkl")


if __name__ == "__main__":
    main()
