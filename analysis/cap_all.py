#!/usr/bin/env python3
"""
cap_all.py - another top of the ladder on all the runs, read with percentile and with mutual ranks
==================================================================================================
The top of the ladder is chosen on Top-1 as the cap with the largest plateau (Appendix C). Read with
mutual ranks instead of percentile ranks (analysis/fusion_explore.py), the largest plateau of Top-1
moves from k = 7 to k = 10. This script computes the fields, the diagrams and the distances of all the
runs on the ladder capped at sigma_max = D_s/(2.35 k), in results/epuck_cap{k}/ (the recordings are
shared with results/epuck), then reads every Top-n with both kinds of ranks as the paper reads it: the
same fusion and affinity propagation over the 99 settings, the taxonomy from the plateau.

Usage: python analysis/cap_all.py [k, default 10]
Output: results/epuck_cap{k}/ (stages 1-3, resumable) and results/epuck_cap{k}/cap_check.pkl

Author: Olivier Vertu Ndinga Oba
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "src"), str(ROOT)]
import fusion_explore as X  # noqa: E402
import run_sweep as RS  # noqa: E402
from classification import fusion as F  # noqa: E402
from gaussian_splatting.scale_ladder import compute_scale_ladder  # noqa: E402
from pipeline.checkpoint import Store, file_fingerprint, load_pickle, save_pickle  # noqa: E402
from pipeline.config import EPS_SIGMA, LADDER_INSTANTS, M_SCALES, RESULTS_DIR  # noqa: E402
from pipeline.io import load_trajectories  # noqa: E402
from pipeline.stage1_seeds import run_stage1  # noqa: E402
from pipeline.stage2_persistence import run_stage2  # noqa: E402
from pipeline.stage3_distances import run_stage3  # noqa: E402

RANKS = {"rank": "percentile ranks", "mutual": "mutual ranks"}


def main():
    k = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    root = RESULTS_DIR.parent / f"epuck_cap{k:g}"
    root.mkdir(exist_ok=True)
    for name in ("meta.pkl", "meta.pkl.sha1", "labels.npz"):               # the same recordings
        if not (root / name).exists():
            (root / name).symlink_to(RESULTS_DIR / name)
    trajectories, _, _ = load_trajectories(root)
    path = root / "scale_ladder.pkl"
    if path.exists():
        ladder = load_pickle(path)["ladder"]
    else:
        ladder = compute_scale_ladder(trajectories, M=M_SCALES, epsilon_sigma=EPS_SIGMA, max_factor=k,
                                      n_instants=LADDER_INSTANTS)
        save_pickle(dict(ladder=ladder, M=M_SCALES, eps_sigma=EPS_SIGMA, max_factor=k, n_instants=LADDER_INSTANTS), path)
    print(f"cap k = {k:g}: ladder {np.round(np.asarray(ladder) * 100, 2).tolist()} cm", flush=True)
    L = len(trajectories)
    run_stage1(trajectories, ladder, "gpu", root, data_fingerprint=file_fingerprint(root / "meta.pkl"))
    run_stage2(L, ladder, root)
    run_stage3(L, ladder, "gpu", root)
    del trajectories
    dist = root / "stage3_distances" / "dist_multiscale.pkl"
    D = load_pickle(dist)["dist_matrices"]
    lab, rk, _ = RS.load_labels()
    out = Store(root / "cap_check.pkl", file_fingerprint(dist))
    for top in range(1, 11):
        idx = RS.subset(top, rk)
        y = lab[idx]
        Dsub = {f: {s: np.asarray(D[f][s], dtype=np.float64)[np.ix_(idx, idx)] for s in D[f]} for f in F.ALL_FIELDS}
        for aff in RANKS:
            if (aff, top) in out:
                continue
            res = F.sweep(X.build_views(Dsub, aff, "balanced", "snf4"), X.GRID)   # on the GPU when available
            out[(aff, top)] = r = X.summary(res, y)
            print(f"cap {k:g} Top-{top:2d} {RANKS[aff]:16s} C*={r['C']:2d} ({r['plateau']:2d}/99) stab {r['stability']:.3f} | "
                  f"purity {r['purity']:.3f} ARI {r['ari']:.3f} wrong {r['misassigned']:3d} "
                  f"(XOR in Shelter {r['xor_in_shelter']:3d})", flush=True)


if __name__ == "__main__":
    main()
