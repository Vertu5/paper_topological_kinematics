#!/usr/bin/env python3
"""
invariance_check.py - is the computed signature rotation invariant on the e-puck data?
====================================================================================
Five Top-1 runs (one per mission) and an exactly rotated copy of each (random angle about
the arena centre) go through stages 1-2 with the SAME code path (CPU). For every field and
scale we compare the sliced-Wasserstein distance between a run and its rotated copy with
the typical distance between two different runs of the same mission (from the stored data).
Output: results/invariance_check/report.pkl

Author: Olivier Vertu Ndinga Oba
"""
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
NJ = int(os.environ.get("INV_JOBS", "4"))
os.cpu_count = lambda: NJ
OUT = ROOT / "results" / "invariance_check"
FIELDS = ["rho", "K", "phi", "psi", "J"]


def swd(dA, dB, directions):
    from gaussian_splatting.distance import swd as sliced
    return float(sum(sliced(dA[s], [dB[s]], directions)[0] for s in range(6)))


def main():
    from pipeline.stage1_seeds import run_stage1
    from pipeline.stage2_persistence import run_stage2
    from gaussian_splatting.distance import sample_directions
    from pipeline.checkpoint import save_pickle
    import run_sweep as RS
    OUT.mkdir(parents=True, exist_ok=True)
    d = pickle.load(open(RS.OUT / "top1_tracks.pkl", "rb"))
    ladder = list(pickle.load(open(RS.RES / "scale_ladder.pkl", "rb"))["ladder"])
    picks = [39, 56, 109, 175, 216]                 # one Top-1 run per mission (consensus exemplars)
    rng = np.random.default_rng(7)
    c = np.array([1.231, 1.231], dtype=np.float64)
    trajs, angles = [], []
    for e in picks:
        p = d["tracks"][e]["pos"].astype(np.float64); v = d["tracks"][e]["vel"].astype(np.float64)
        a = rng.uniform(-np.pi, np.pi); angles.append(a)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        trajs.append({"pos": p.astype(np.float32), "vel": v.astype(np.float32)})
        trajs.append({"pos": ((p - c) @ R.T + c).astype(np.float32), "vel": (v @ R.T).astype(np.float32)})
    run_stage1(trajs, np.array(ladder), "cpu", OUT)
    run_stage2(len(trajs), np.array(ladder), OUT)
    directions = sample_directions(25)
    load = lambda root, ti, si: pickle.load(open(root / "stage2_alpha_dgms" / f"traj{ti:04d}_s{si}.pkl", "rb"))["dgms"]
    labels = d["labels"]; idx = d["idx"]
    report = {}
    for k, e in enumerate(picks):
        m = int(labels[e]); same = [int(idx[j]) for j in np.where(labels == m)[0] if j != e][:12]
        for si in range(len(ladder)):
            A = load(OUT, 2 * k, si); B = load(OUT, 2 * k + 1, si); S = load(RS.RES, int(idx[e]), si)
            for f in FIELDS:
                rot = swd(A[f], B[f], directions)
                path = swd(A[f], S[f], directions)
                within = np.median([swd(S[f], load(RS.RES, j, si)[f], directions) for j in same])
                report[(e, f, si)] = dict(rotation=rot, code_path=path, within=within)
        print(f"run {e} (mission {m}) done", flush=True)
    save_pickle(dict(report=report, picks=picks, angles=angles), OUT / "report.pkl")
    print("relative error = distance(run, rotated copy) / median distance between runs of the same mission")
    for f in FIELDS:
        rel = np.array([[report[(e, f, si)]["rotation"] / max(report[(e, f, si)]["within"], 1e-12)
                         for si in range(len(ladder))] for e in picks])
        print(f"  {f:4s} median over runs per scale: " + "  ".join(f"s{si + 1}={np.median(rel[:, si]):.3f}" for si in range(len(ladder)))
              + f"   max={rel.max():.3f}", flush=True)


if __name__ == "__main__":
    main()
