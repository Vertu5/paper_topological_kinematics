#!/usr/bin/env python3
"""
mirror_check.py - how far is a run from its mirror image, for each reader?
==========================================================================
Under a mirror reflection of the whole scene, the seeds and the alpha complex are reflected, the density,
the kinetic intensity, the divergence potential and the coupling keep their values, and the vorticity
potential changes sign (checked numerically on the closed forms). The mirror image of a run therefore has
the diagrams of -psi on the same complex. For every Top-1 run and scale, this script compares psi with -psi:
  persistence : sliced-Wasserstein distance between the six extended diagrams of psi and of -psi
  values      : distance between the 32 quantiles of psi and of -psi (the value reader of the baselines)
each relative to the median distance between two runs of the same mission for that reader.
Output: results/epuck/paper/mirror_check.pkl

Author: Olivier Vertu Ndinga Oba
"""
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from pipeline.checkpoint import save_pickle  # noqa: E402
from gaussian_splatting.alpha import AlphaSimplexTree  # noqa: E402
from gaussian_splatting.persistence import PersistenceComputer  # noqa: E402
from gaussian_splatting.distance import sample_directions, swd  # noqa: E402

QS = np.linspace(0.0, 1.0, 32)
DIRS = sample_directions(25)


def one(g, si, sigma):
    with open(RS.RES / "stage1_alpha_fields" / f"traj{g:04d}_s{si}.pkl", "rb") as fh:
        d = pickle.load(fh)
    psi = np.asarray(d["fields"]["psi"], np.float64).ravel()
    pc = PersistenceComputer(AlphaSimplexTree(np.asarray(d["seeds"], np.float64), alpha_sq_max=(3 * sigma) ** 2))
    A, B = pc.compute_field(psi), pc.compute_field(-psi)
    sw = float(sum(swd(A[s], [B[s]], DIRS)[0] for s in range(6)))
    v = psi[np.isfinite(psi)]
    q = np.abs(np.quantile(v, QS) - np.quantile(-v, QS)).mean()
    return sw, float(q)


def main():
    lab, rk, _ = RS.load_labels()
    idx = RS.subset(1, rk); y = lab[idx]
    D, ladder = RS.load_distances()
    V = pickle.load(open(RS.OUT / "baselines.pkl", "rb"))["fieldstat_D"]
    out = {}
    for si, sigma in enumerate(ladder):
        res = Parallel(n_jobs=int(os.environ.get("MIRROR_JOBS", "6")))(delayed(one)(int(g), si, float(sigma)) for g in idx)
        sw = np.array([r[0] for r in res]); q = np.array([r[1] for r in res])
        within = {}
        for name, M in (("persistence", np.asarray(D["psi"][si])[np.ix_(idx, idx)]),
                        ("values", np.asarray(V["psi"][si])[np.ix_(idx, idx)])):
            within[name] = float(np.median(np.concatenate([M[np.ix_(y == m, y == m)][np.triu_indices((y == m).sum(), 1)]
                                                           for m in range(5)])))
        out[si] = dict(persistence=sw / within["persistence"], values=q / within["values"], labels=y)
        print(f"sigma_{si + 1}: mirror distance / within-mission median   persistence: median {np.median(sw / within['persistence']):.2f} "
              f"(max {np.max(sw / within['persistence']):.2f})   values: median {np.median(q / within['values']):.2f} "
              f"(max {np.max(q / within['values']):.2f})", flush=True)
        for m in range(5):
            print(f"      mission {m}: persistence {np.median(sw[y == m]) / within['persistence']:.2f}   "
                  f"values {np.median(q[y == m]) / within['values']:.2f}", flush=True)
    save_pickle(out, RS.OUT / "mirror_check.pkl")


if __name__ == "__main__":
    main()
