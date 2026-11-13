#!/usr/bin/env python3
"""
run_sweep.py - single entry point for every clustering result of the paper
============================================================================
Input  : results/epuck/stage3_distances/dist_multiscale.pkl (20 SWD matrices,
         5 fields x 4 scales, 2500 runs; produced by run_pipeline.py stages 1-3)
Method : src/classification/fusion.py (balanced rank fusion + cross-scale SNF + AP)
Output : results/epuck/paper/<experiment>.pkl

Experiments
  grid        Top-1..Top-10, all five fields, 99 (knn, t) settings    -> grid.pkl
  ablation    11 field combinations x Top-1..Top-10 x 99 settings     -> ablation.pkl
  extended    Top-1, all fields, extended grid (knn<=50, t<=100)      -> extended_top1.pkl
  designs     Top-1, 99 settings, alternative fusion designs          -> designs_top1.pkl
  standard    Top-1, knn=5, t in {10,20,30,50,100}: ours vs heat-kernel SNF -> standard_snf.pkl
  taxonomy    plateau consensus (C*, partition, exemplars, stability) from grid.pkl -> taxonomy.pkl

Usage
  python run_sweep.py grid [--tops 1 2 ...] [--n-jobs 10]

Author: Olivier Vertu Ndinga Oba
"""
import argparse
import os
import pickle
import sys
import time
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from classification import fusion as F  # noqa: E402
from classification.affinity import snf_affinity  # noqa: E402
from classification.ap import gpu_available  # noqa: E402
from classification.snf import snf  # noqa: E402
from pipeline.checkpoint import Store, file_fingerprint, load_pickle, save_pickle, save_text, stage_fingerprint  # noqa: E402
from pipeline.config import BEHAVIORS, DATA_DIR, RESULTS_DIR, TOP_CONTROLLERS  # noqa: E402
from pipeline.ranking import ranked_controllers  # noqa: E402

RES = RESULTS_DIR
OUT = RES / "paper"
KNNS = [3, 5, 7, 9, 11, 13, 15, 20, 25]
TS = [1, 2, 3, 5, 7, 9, 10, 12, 15, 20, 25]
KNNS_EXT = [3, 5, 7, 9, 11, 13, 15, 20, 25, 30, 40, 50]
TS_EXT = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 70, 100]
COMBOS = [("rho",), ("K",), ("phi",), ("psi",), ("J",), ("rho", "K"), ("phi", "psi"),
          ("phi", "psi", "J"), ("K", "phi", "psi", "J"), ("rho", "K", "phi", "psi"), ("rho", "K", "phi", "psi", "J")]


# --------------------------------------------------------------------------- io
def load_labels():
    """labels (mission index), ranks (controller rank by mean score), controller ids; cached as npz."""
    cache = RES / "labels.npz"
    if not cache.exists():
        with open(RES / "meta.pkl", "rb") as fh:
            meta = pickle.load(fh)
        lab = np.asarray(meta["labels"], dtype=int)
        st = np.asarray(meta["ranks"], dtype=int)
        ctrl = np.array([TOP_CONTROLLERS[BEHAVIORS[m]][r] for m, r in zip(lab, st)], dtype=int)
        ranking = ranked_controllers(ROOT / DATA_DIR)
        rk = np.array([ranking[BEHAVIORS[m]].index(int(c)) for m, c in zip(lab, ctrl)], dtype=int)
        np.savez(cache, labels=lab, ranks=rk, controllers=ctrl)
    z = np.load(cache)
    return z["labels"], z["ranks"], z["controllers"]


DIST = RES / "stage3_distances" / "dist_multiscale.pkl"


def load_distances():
    d = load_pickle(DIST)
    return d["dist_matrices"], list(d["ladder"])


def dist_source():
    """Fingerprint of the stage-3 distances: a saved result is reused only if computed from them."""
    return file_fingerprint(DIST)


def stage1_source():
    """Fingerprint of the complete stage-1 folder (fields at the seeds)."""
    return stage_fingerprint(RES / "stage1_alpha_fields")


def data_source():
    """Fingerprint of the recordings."""
    return file_fingerprint(RES / "meta.pkl")


def subset(top, ranks):
    return np.where(ranks < top)[0]


# --------------------------------------------------------------------------- scoring
def score(y, p):
    cm = contingency_matrix(y, p)
    return dict(C=int(len(np.unique(p))), purity=float(cm.max(0).sum() / cm.sum()),
                ari=float(adjusted_rand_score(y, p)), nmi=float(normalized_mutual_info_score(y, p)))


def sweep(views, y, grid, n_jobs=9, **ap):
    """AP taxonomy at every (knn, t) of grid (GPU when available, identical to the CPU), scored against y."""
    res = F.sweep(views, grid, n_jobs=n_jobs, **ap)
    for r in res:
        r.update(score(y, r["pred"]))
    return res


def device_name():
    """The processor that runs the clustering (the GPU when available)."""
    if gpu_available():
        import torch
        return torch.cuda.get_device_name(0)
    return "CPU"


def save(obj, name, source=None):
    """Save a result atomically, with the fingerprint of the inputs it was computed from."""
    save_pickle(obj, OUT / name)
    save_text(dist_source() if source is None else source, OUT / (name + ".source"))
    print(f"saved {OUT / name}", flush=True)


def done(name, source=None):
    """True if the result is saved and was computed from the current inputs."""
    src = OUT / (name + ".source")
    return (OUT / name).exists() and src.exists() and src.read_text().strip() == (dist_source() if source is None else source)


# --------------------------------------------------------------------------- experiments
def exp_grid(D, lab, rk, ctrl, tops, n_jobs, fields=F.ALL_FIELDS, verbose=True):
    """The 99 settings on every Top-n; resumable Top-n by Top-n (grid.pkl)."""
    out = Store(OUT / "grid.pkl", dist_source())
    for top in tops:
        if top in out:
            continue
        t0 = time.time()
        idx = subset(top, rk)
        views = F.scale_views(F.rank_normalise(D, fields, idx), fields)
        res = sweep(views, lab[idx], list(product(KNNS, TS)), n_jobs)
        out[top] = dict(idx=idx, labels=lab[idx], controllers=ctrl[idx], ranks=rk[idx], results=res,
                        seconds=time.time() - t0, device=device_name())
        if verbose:
            C = np.array([r["C"] for r in res]); P = np.array([r["purity"] for r in res])
            vals, cnt = np.unique(C, return_counts=True)
            print(f"Top-{top:2d} [{'+'.join(fields)}] modal C={vals[cnt.argmax()]} ({cnt.max()}/99) "
                  f"median purity={np.median(P):.3f} [{time.time() - t0:.0f}s]", flush=True)
    return out.data


def consensus_summary(res, y):
    """The taxonomy read from the plateau of a sweep, as the paper reads it, scored against y."""
    pc = F.plateau_consensus(res, len(y))
    return dict(score(y, pc["labels"]), C=pc["C"], plateau=pc["n_plateau"], stability=pc["stability"],
                labels=np.asarray(pc["labels"], dtype=np.int16))


def exp_ablation(D, lab, rk, ctrl, tops, n_jobs):
    """Eleven field combinations on every Top-n: the 99 settings (ablation.pkl) and the taxonomy read
    from their plateau (ablation_consensus.pkl); resumable."""
    out = Store(OUT / "ablation.pkl", dist_source())
    cons = Store(OUT / "ablation_consensus.pkl", dist_source())
    for top in tops:
        todo = [c for c in COMBOS if (c, top) not in out or (c, top) not in cons]
        if not todo:
            continue
        idx = subset(top, rk)
        Q = F.rank_normalise(D, F.ALL_FIELDS, idx)
        for combo in todo:
            res = sweep(F.scale_views(Q, combo), lab[idx], list(product(KNNS, TS)), n_jobs)
            cons[(combo, top)] = consensus_summary(res, lab[idx])
            for r in res:
                r.pop("pred"); r.pop("exemplars")
            out[(combo, top)] = res
            P = np.array([r["purity"] for r in res])
            print(f"ablation {'+'.join(combo):18s} Top-{top:2d} median purity={np.median(P):.3f}", flush=True)
    return out.data


def exp_extended(D, lab, rk, n_jobs):
    idx = subset(1, rk)
    Q = F.rank_normalise(D, F.ALL_FIELDS, idx)
    views = F.scale_views(Q)
    res = sweep(views, lab[idx], list(product(KNNS_EXT, TS_EXT)), n_jobs)
    return dict(idx=idx, labels=lab[idx], results=res, knns=KNNS_EXT, ts=TS_EXT)


def exp_standard(D, lab, rk):
    """Rank fusion (ours) against the affinity of standard SNF on the raw distances (Top-1, knn=5)."""
    idx = subset(1, rk)
    y = lab[idx]
    Q = F.rank_normalise(D, F.ALL_FIELDS, idx)
    views = F.scale_views(Q)
    raw = {f: {s: np.asarray(D[f][s], dtype=np.float64)[np.ix_(idx, idx)] for s in D[f]} for f in F.ALL_FIELDS}
    out = []
    ts = [10, 20, 30, 50, 100]
    ours_all = sweep(views, y, [(5, t) for t in ts])
    for t, ours in zip(ts, ours_all):
        # standard SNF on the raw distances: the affinity of Wang et al. (2014) per field, SNF over the
        # fields at each scale, then SNF over the scales (no rank step)
        per_scale = []
        for s in raw["rho"]:
            per_scale.append(snf([snf_affinity(raw[f][s], 5) for f in F.ALL_FIELDS], 5, t))
        S = snf(per_scale, 5, t)
        lab_std, _, conv = F.affinity_propagation(S)
        std = score(y, lab_std)
        out.append(dict(t=t, ours={k: ours[k] for k in ("C", "purity", "ari", "nmi")}, standard=std))
        print(f"t={t:3d} ours C={ours['C']} P={ours['purity']:.3f} | standard C={std['C']} P={std['purity']:.3f}", flush=True)
    return out


def exp_taxonomy():
    """Label-free reading of the taxonomy on every Top-n: consensus over the plateau of the grid."""
    from sklearn.metrics import homogeneity_completeness_v_measure
    if not done("grid.pkl"):
        raise RuntimeError("grid.pkl is missing or was computed from other distances: run `run_sweep.py grid` first")
    G = load_pickle(OUT / "grid.pkl")
    out = {}
    for top, g in G.items():
        pc = F.plateau_consensus(g["results"], len(g["labels"]))
        s = score(g["labels"], pc["labels"])
        h, c, _ = homogeneity_completeness_v_measure(g["labels"], pc["labels"])
        pc.update(s, homogeneity=float(h), completeness=float(c), idx=g["idx"],
                  misassigned=int(round((1 - s["purity"]) * len(g["labels"]))))
        if top != 1:
            pc.pop("coassociation")
        out[top] = pc
        print(f"Top-{top:2d}: C*={pc['C']} plateau={pc['n_plateau']}/99 stability={pc['stability']:.3f} "
              f"purity={s['purity']:.3f} ARI={s['ari']:.3f} hom={h:.3f} comp={c:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["grid", "ablation", "extended", "standard", "taxonomy", "all"])
    ap.add_argument("--tops", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--n-jobs", type=int, default=10)
    a = ap.parse_args()
    lab, rk, ctrl = load_labels()
    D, ladder = load_distances()
    print(f"{len(lab)} runs, scales {np.round(ladder, 4).tolist()}", flush=True)
    if a.experiment in ("grid", "all"):
        exp_grid(D, lab, rk, ctrl, a.tops, a.n_jobs)
    if a.experiment in ("extended", "all") and not done("extended_top1.pkl"):
        save(exp_extended(D, lab, rk, a.n_jobs), "extended_top1.pkl")
    if a.experiment in ("standard", "all") and not done("standard_snf.pkl"):
        save(exp_standard(D, lab, rk), "standard_snf.pkl")
    if a.experiment in ("taxonomy", "all"):
        save(exp_taxonomy(), "taxonomy.pkl")
    if a.experiment in ("ablation", "all"):
        exp_ablation(D, lab, rk, ctrl, a.tops, a.n_jobs)


if __name__ == "__main__":
    main()
