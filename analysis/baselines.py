#!/usr/bin/env python3
"""
baselines.py - alternative representations evaluated under the paper's protocol
===============================================================================
Every representation gives run-to-run distances; all of them go through the same
rank affinity and the same Affinity Propagation (median preference, damping 0.9),
so only the representation changes. Labels are used only for scoring.

  order      order parameters (Vicsek polarisation, milling, speed, spread; mean and
             standard deviation over the run), standardised, Euclidean distance
  crocker    topology of agent positions through time (CROCKER-style Betti-0/Betti-1
             curves of the alpha complex of the 20 positions at 60 instants), L2
  fieldstat  SAME seeds and SAME five fields, but the persistence diagrams are replaced
             by the distribution of field values (32 quantiles, 1-D Wasserstein of order one, as the diagrams);
             then exactly our fusion (balanced ranks, SNF across scales, AP)
  scale<m>   our five fields at a single scale m, no diffusion (one view, AP)
  nodiff     our five fields, balanced rank average over the four scales, no diffusion

Output: results/epuck/paper/baselines.pkl

Author: Olivier Vertu Ndinga Oba
"""
import os
import pickle
import sys
import time
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from pipeline.checkpoint import Store  # noqa: E402
from classification import fusion as F  # noqa: E402
from classification.affinity import rank_affinity, rank_distance  # noqa: E402

OUT = RS.OUT / "baselines.pkl"
NJ = int(os.environ.get("BASELINE_JOBS", "6"))
TOPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# ----------------------------------------------------------------- representations
def order_features(pos, vel):
    sp = np.linalg.norm(vel, axis=2)
    ok = sp > 1e-6
    u = np.where(ok[..., None], vel / np.maximum(sp, 1e-12)[..., None], 0.0)
    P = np.linalg.norm(u.sum(1), axis=1) / np.maximum(ok.sum(1), 1)
    c = pos.mean(1, keepdims=True)
    r = pos - c
    rn = np.linalg.norm(r, axis=2)
    cross = (r[..., 0] * u[..., 1] - r[..., 1] * u[..., 0]) / np.maximum(rn, 1e-12)
    M = np.abs(cross.sum(1)) / np.maximum(ok.sum(1), 1)
    S = sp.mean(1)
    R = rn.mean(1)
    f = [P, M, S, R]
    return np.array([g.mean() for g in f] + [g.std() for g in f])


EPS = np.linspace(0.0, 0.6, 25)  # alpha radii in metres


def crocker(pos, n_frames=60):
    import gudhi
    T = pos.shape[0]
    idx = np.linspace(0, T - 1, n_frames).astype(int)
    out = np.zeros((n_frames, 2, len(EPS)))
    for a, t in enumerate(idx):
        st = gudhi.AlphaComplex(points=pos[t].astype(np.float64)).create_simplex_tree()
        st.persistence()
        for dim in (0, 1):
            iv = st.persistence_intervals_in_dimension(dim)
            if len(iv) == 0:
                continue
            b = np.sqrt(np.maximum(iv[:, 0], 0))
            d = np.sqrt(np.where(np.isinf(iv[:, 1]), 1e12, iv[:, 1]))
            out[a, dim] = ((b[:, None] <= EPS[None, :]) & (d[:, None] > EPS[None, :])).sum(0)
    return out.ravel()


def _batch(fn, trajs):
    return [fn(t["pos"].astype(np.float64), t["vel"].astype(np.float64)) if fn is order_features else fn(t["pos"]) for t in trajs]


QS = np.linspace(0.0, 1.0, 32)


def field_quantiles(ti, ladder_n):
    out = np.zeros((5, ladder_n, len(QS)))
    for si in range(ladder_n):
        with open(RS.RES / "stage1_alpha_fields" / f"traj{ti:04d}_s{si}.pkl", "rb") as fh:
            d = pickle.load(fh)
        for fi, f in enumerate(F.ALL_FIELDS):
            v = np.asarray(d["fields"][f], dtype=np.float64)
            v = v[np.isfinite(v)]
            out[fi, si] = np.quantile(v, QS) if len(v) else 0.0
    return out


# ----------------------------------------------------------------- evaluation
def main():
    lab, rk, ctrl = RS.load_labels()
    # resumable; reused only for the same recordings, stage-1 fields and distances
    res = Store(OUT, f"{RS.data_source()}+{RS.stage1_source()}+{RS.dist_source()}+values of order one")
    t0 = time.time()

    # --- order parameters and crocker need the tracks
    if "order" not in res or "crocker" not in res:
        with open(RS.RES / "meta.pkl", "rb") as fh:
            trajs = pickle.load(fh)["trajectories"]
        chunks = np.array_split(np.arange(len(trajs)), NJ * 4)
        if "order" not in res:
            feats = np.vstack(sum(Parallel(n_jobs=NJ)(delayed(_batch)(order_features, [trajs[i] for i in c]) for c in chunks), []))
            Z = (feats - feats.mean(0)) / np.maximum(feats.std(0), 1e-12)
            res["order"] = {"features": feats, "D": cdist(Z, Z)}
            print(f"order parameters done [{time.time() - t0:.0f}s]", flush=True)
        if "crocker" not in res:
            cro = np.vstack(sum(Parallel(n_jobs=NJ)(delayed(_batch)(crocker, [trajs[i] for i in c]) for c in chunks), []))
            res["crocker"] = {"D": cdist(cro, cro)}
            print(f"crocker done [{time.time() - t0:.0f}s]", flush=True)
        del trajs

    # --- field-value distributions on the same seeds
    D_swd, ladder = RS.load_distances()
    M = len(ladder)
    if "fieldstat_D" not in res:
        q = np.stack(Parallel(n_jobs=NJ)(delayed(field_quantiles)(ti, M) for ti in range(len(lab))))
        res["fieldstat_D"] = {f: {si: cdist(q[:, fi, si], q[:, fi, si], "cityblock") / len(QS) for si in range(M)}
                              for fi, f in enumerate(F.ALL_FIELDS)}
        print(f"field quantiles done [{time.time() - t0:.0f}s]", flush=True)

    # --- scoring: single-view baselines on every Top-n
    for top in TOPS:
        key = ("single", top)
        if key in res:
            continue
        idx = RS.subset(top, rk)
        y = lab[idx]
        views = F.scale_views(F.rank_normalise(D_swd, F.ALL_FIELDS, idx))
        mats = [rank_affinity(rank_distance(res[name]["D"][np.ix_(idx, idx)]).astype(np.float32))
                for name in ("order", "crocker")] + list(views) + [np.mean(views, axis=0)]
        names = ["order", "crocker"] + [f"scale{si + 1}" for si in range(len(views))] + ["nodiff"]
        out = {name: dict(RS.score(y, l_), converged=c_) for name, (l_, _, c_) in zip(names, F.ap_many(mats))}
        res[key] = out
        print(f"Top-{top}: " + " | ".join(f"{k} C={v['C']} P={v['purity']:.3f} ARI={v['ari']:.3f}" for k, v in out.items()), flush=True)

    # --- field statistics through our full fusion: Top-1 and Top-10, all 99 settings
    for top in (1, 10):
        key = ("fieldstat_grid", top)
        if key in res:
            continue
        idx = RS.subset(top, rk)
        Q = F.rank_normalise(res["fieldstat_D"], F.ALL_FIELDS, idx)
        views = F.scale_views(Q)
        grid = RS.sweep(views, lab[idx], list(product(RS.KNNS, RS.TS)), NJ)
        res[key] = grid
        C = np.array([r["C"] for r in grid]); P = np.array([r["purity"] for r in grid])
        v, c = np.unique(C, return_counts=True)
        print(f"fieldstat grid Top-{top}: modal C={v[c.argmax()]} ({c.max()}/99) plateau median purity={np.median(P[C == v[c.argmax()]]):.3f} [{time.time() - t0:.0f}s]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
