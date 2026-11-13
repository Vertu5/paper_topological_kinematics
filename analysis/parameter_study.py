#!/usr/bin/env python3
"""
parameter_study.py - does any constant of the chain still change the result?
============================================================================
Each constant is varied alone, the others keeping the values of the paper (pipeline/config.py), on
the 250 runs of Top-1, through the whole chain: seeds and fields, extended persistence, distances, the
99 settings of the fusion and affinity propagation, and the plateau consensus.

Two kinds of constants are distinguished.
  * Resolution constants (coreset size, fraction of flat triangles, seeds on J = 0, bisection steps,
    evaluation window, duplicate tolerance, number of directions) approximate an exact object; the
    reference of each is its finest value, and the question is the smallest value that no longer
    changes the result.
  * Design choices (start of the farthest-point sampling, alpha radius, scale ladder, settings of
    affinity propagation) are compared with the values of the paper.

Every comparison is label-free:
  * distances: Spearman rank correlation between the distance matrices of the two versions, for every
    field and scale (the fusion only uses ranks), smallest and median over the twenty;
  * taxonomy: adjusted Rand index between the two consensus partitions, and the mean over the 99
    settings of the index between the partitions of the same setting.
The purity against the missions is printed for information only.

Resumable: every version is saved as soon as it is computed (results/parameter_study/).
Usage: nohup nice -n 19 python analysis/parameter_study.py > results/parameter_study/log.txt 2>&1 &

Author: Olivier Vertu Ndinga Oba
"""
import os
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from classification import fusion as F  # noqa: E402
from gaussian_splatting.alpha import AlphaSimplexTree  # noqa: E402
from gaussian_splatting.fields_cpu import AVAILABLE_FIELDS  # noqa: E402
from gaussian_splatting.persistence import N_SUB_DIAGRAMS, PersistenceComputer  # noqa: E402
from gaussian_splatting.seeds_gridfree import (coreset_orders_gpu, coresets_gpu, delaunay_structure,  # noqa: E402
                                               prefix_for_radius, seeds_from_coresets)
from pipeline import config as C  # noqa: E402
from pipeline.checkpoint import Store, code_fingerprint, load_pickle, save_pickle  # noqa: E402

OUT = Path(os.environ.get("STUDY_OUT", ROOT / "results" / "parameter_study"))
N_RUNS = int(os.environ.get("STUDY_RUNS", "0"))            # a smaller set of runs, for a quick trial only
NJ = int(os.environ.get("STUDY_JOBS", "16"))
PAPER = dict(n_core=C.N_CORESET, cover=None, start=0, frac=C.FRAC_LOW_GRAD, n_jacobi=C.N_JACOBI, n_bis=C.N_BIS,
             window=C.WINDOW, dedup=C.DEDUP_TOL, alpha=C.C_ALPHA, n_dirs=C.N_DIRS, ladder="paper",
             damping=0.9, conv=100, max_iter=1000, seed=0)
RESOLUTION = [("n_core", [50, 100, 200, 300, 600, 1000, 2000, 4000]), ("frac", [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]),
              ("n_jacobi", [0, 25, 50, 100, 150, 300, 600]), ("n_bis", [4, 6, 8, 12, 16, 20]),
              ("window", [3.0, 4.0, 5.0, 6.0, 8.0]), ("dedup", [1e-2, 1e-3, 1e-4, 1e-5]),
              ("n_dirs", [10, 25, 50, 100, 200, 400])]
FINEST = {"dedup": min}                          # the finest value of the others is the largest
DESIGN = [("start", [0, 1, 777777]), ("alpha", [2.0, 2.5, 3.0, 4.0]),
          ("ladder", ["paper", "M3", "M5", "M6", "uncapped4", "uncapped10", "uncapped20"]),
          ("damping", [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]), ("conv", [50, 100, 200, 400]),
          ("max_iter", [1000, 5000]), ("seed", [0, 1, 2, 3, 4])]
STAGE1 = ("n_core", "cover", "start", "frac", "n_jacobi", "n_bis", "window", "dedup", "ladder")
# the extension: the adaptive coreset (no position farther than cover * sigma from the coreset), the
# ladder up to D_s / 2.35 with more and more scales, and the cap on the J = 0 seeds with a fine coreset;
# each entry: (parameter, values, reference value, values of the other parameters)
EXTENSION = [("ladder", ["cap2", "cap3", "cap5", "cap20"], "paper", {}),
             ("ladder", ["uncapped4", "uncapped10", "uncapped20", "uncapped40"], "uncapped40", {}),
             ("cover", [2.0, 1.0, 0.5], 0.5, {}),
             ("n_jacobi", [150, 100000], 100000, {"cover": 0.5})]
# the final settings (adaptive coreset, cover 1/2): the top of the ladder chosen without labels (the cap
# with the largest plateau of the resolution map), and the resolution settings checked again on this domain
FINAL = [("ladder", ["cap3", "cap5", "cap7", "cap10", "cap14", "cap20"], "cap10", {"cover": 0.5}),
         ("cover", [1.0, 0.5, 1 / 3], 1 / 3, {}),
         ("frac", [0.05, 0.2], 0.2, {"cover": 0.5}),
         ("n_bis", [12, 20], 20, {"cover": 0.5}),
         ("window", [5.0, 8.0], 8.0, {"cover": 0.5}),
         ("n_dirs", [25, 100], 100, {"cover": 0.5})]


def ladder_of(name, base):
    """The scales of a ladder variant. base = (ladder of the pipeline, its cap factor): the ladder runs from
    sigma_min to (D_s / 2.35) / factor, so that D_s / 2.35 = ladder[-1] * factor."""
    ladder, factor = base
    lo, hi = float(ladder[0]), float(ladder[-1])
    merge = hi * factor                                              # D_s / 2.35
    if name == "paper":
        return np.asarray(ladder, float)
    if name.startswith("M"):
        return np.geomspace(lo, hi, int(name[1:]))
    if name.startswith("cap"):                                       # 4 scales up to (D_s / 2.35) / k
        return np.geomspace(lo, merge / float(name[3:]), 4)
    return np.geomspace(lo, merge, int(name[len("uncapped"):]))         # up to D_s / 2.35


def key(cfg, names):
    return "_".join(f"{n}={cfg[n]}" for n in names if not (n == "cover" and cfg[n] is None))


def stage1(cfg, runs, trajs, base):
    path = OUT / "stage1" / f"{key(cfg, STAGE1)}.pkl"
    if path.exists():
        return load_pickle(path)
    from gaussian_splatting.fields_gpu import TorchFieldComputer
    comp = TorchFieldComputer(fields=AVAILABLE_FIELDS, kappa=cfg["window"])
    comp_j = TorchFieldComputer(fields=("J",), kappa=1e6)
    groups = defaultdict(list)
    for g in runs:
        groups[trajs[g]["pos"].shape].append(g)
    out, t0 = {}, time.time()
    for group in groups.values():
        for b0 in range(0, len(group), C.BATCH_SEEDS):
            batch = group[b0:b0 + C.BATCH_SEEDS]
            pos = [trajs[g]["pos"] for g in batch]
            vel = [trajs[g]["vel"] for g in batch]
            if cfg["cover"] is None:
                fixed = coresets_gpu(pos, cfg["n_core"], cfg["start"])
                structs = [delaunay_structure(c) for c in fixed]
            else:
                orders = coreset_orders_gpu(pos, C.N_CORESET_MAX, cfg["start"],
                                            stop_radius=cfg["cover"] * float(min(ladder_of(cfg["ladder"], base))))
            for si, sig in enumerate(ladder_of(cfg["ladder"], base)):
                if cfg["cover"] is None:
                    cores = fixed
                else:
                    cores = [o[0][:prefix_for_radius(o[1], cfg["cover"] * float(sig))] for o in orders]
                    structs = [delaunay_structure(c) for c in cores]
                res = seeds_from_coresets(pos, vel, cores, comp, float(sig), computer_J=comp_j, frac_low_grad=cfg["frac"],
                                          n_jacobi=cfg["n_jacobi"], n_bis=cfg["n_bis"], dedup_tol=cfg["dedup"],
                                          structures=structs)
                for g, r in zip(batch, res):
                    out[(g, si)] = r
    out["time"] = time.time() - t0
    save_pickle(out, path)
    return out


def _diagrams(seeds, fields, sigma, alpha):
    pc = PersistenceComputer(AlphaSimplexTree(np.asarray(seeds, float), alpha_sq_max=(alpha * sigma) ** 2))
    return pc.compute_all({f: np.asarray(fields[f], float).ravel() for f in AVAILABLE_FIELDS})


def stage2(cfg, runs, s1, base):
    path = OUT / "stage2" / f"{key(cfg, STAGE1 + ('alpha',))}.pkl"
    if path.exists():
        return load_pickle(path)
    sig = ladder_of(cfg["ladder"], base)
    items = [(g, si) for g in runs for si in range(len(sig))]
    res = Parallel(n_jobs=NJ, batch_size=16)(delayed(_diagrams)(*s1[it], float(sig[it[1]]), cfg["alpha"]) for it in items)
    out = dict(zip(items, res))
    save_pickle(out, path)
    return out


def stage3(cfg, runs, s2, base):
    path = OUT / "stage3" / f"{key(cfg, STAGE1 + ('alpha', 'n_dirs'))}.pkl"
    if path.exists():
        return load_pickle(path)
    from gaussian_splatting.swd_gpu import swd_condensed_gpu
    D = {f: {} for f in AVAILABLE_FIELDS}
    for si in range(len(ladder_of(cfg["ladder"], base))):
        for f in AVAILABLE_FIELDS:
            dg = [s2[(g, si)][f] for g in runs]
            D[f][si] = sum(swd_condensed_gpu([d[s] for d in dg], n_dirs=cfg["n_dirs"])
                           for s in range(N_SUB_DIAGRAMS))                                 # condensed
    save_pickle(D, path)
    return D


def square(D, n):
    iu = np.triu_indices(n, 1)
    out = {}
    for f, per in D.items():
        out[f] = {}
        for si, v in per.items():
            M = np.zeros((n, n))
            M[iu] = v
            out[f][si] = M + M.T
    return out


def taxonomy(cfg, D, n):
    views = F.scale_views(F.rank_normalise(square(D, n), AVAILABLE_FIELDS))
    res = F.sweep(views, list(product(RS.KNNS, RS.TS)), damping=cfg["damping"], convergence_iter=cfg["conv"],
                  max_iter=cfg["max_iter"], random_state=cfg["seed"])
    return res, F.plateau_consensus(res, n)


def compare(a, b):
    """Label-free agreement of two versions (distances when they share the scales, and taxonomies)."""
    row = {}
    if np.array_equal(a["sigma"], b["sigma"]):
        rho = [spearmanr(a["D"][f][si], b["D"][f][si]).correlation for f in AVAILABLE_FIELDS for si in a["D"][f]]
        row.update(spearman_min=float(np.min(rho)), spearman_median=float(np.median(rho)))
    row["ari_consensus"] = float(adjusted_rand_score(a["tax"]["labels"], b["tax"]["labels"]))
    row["ari_settings"] = float(np.mean([adjusted_rand_score(x["pred"], y["pred"]) for x, y in zip(a["res"], b["res"])]))
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab, rk, _ = RS.load_labels()
    runs = [int(g) for g in RS.subset(1, rk)]
    if N_RUNS:
        runs = [g for m in range(5) for g in [r for r in runs if lab[r] == m][:N_RUNS // 5]]
    y = lab[runs]
    _lad = load_pickle(RS.RES / "scale_ladder.pkl")
    base = (np.asarray(_lad["ladder"], float), float(_lad.get("max_factor", 10.0)))
    src = ROOT / "src" / "gaussian_splatting"
    code = code_fingerprint(*(src / f for f in ("fields_gpu.py", "seeds_gridfree.py", "alpha.py", "persistence.py",
                                                 "swd_gpu.py", "distance.py")))
    summary = Store(OUT / "summary.pkl", f"{RS.data_source()}+{code}")
    for sub in ("stage1", "stage2", "stage3"):                    # cached versions belong to this code only
        stamp = OUT / sub / "CODE"
        if stamp.exists() and stamp.read_text() != code:
            for p in (OUT / sub).glob("*.pkl"):
                p.unlink()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(code)
    trajs = None
    versions = {}

    def version(cfg):
        nonlocal trajs
        k = key(cfg, tuple(PAPER))
        if k in versions:
            return versions[k]
        need1 = not (OUT / "stage1" / f"{key(cfg, STAGE1)}.pkl").exists()
        if need1 and trajs is None:
            trajs = load_pickle(RS.RES / "meta.pkl")["trajectories"]
        t0 = time.time()
        s1 = stage1(cfg, runs, trajs, base)
        s2 = stage2(cfg, runs, s1, base)
        D = stage3(cfg, runs, s2, base)
        res, tax = taxonomy(cfg, D, len(runs))
        seeds = float(np.mean([len(v[0]) for kk, v in s1.items() if kk != "time"]))
        versions[k] = dict(D=D, res=res, tax=tax, seeds=seeds, time1=float(s1["time"]), wall=time.time() - t0,
                           sigma=ladder_of(cfg["ladder"], base))
        return versions[k]

    def report(name, value, v, ref, ref_name):
        row = dict(compare(v, ref), C=v["tax"]["C"], plateau=v["tax"]["n_plateau"], stability=v["tax"]["stability"],
                   purity=float(RS.score(y, v["tax"]["labels"])["purity"]), seeds=v["seeds"], stage1_s=v["time1"],
                   reference=ref_name, iterations=float(np.median([r["n_iter"] for r in v["res"]])),
                   converged=int(sum(r["converged"] for r in v["res"])))
        summary[name if isinstance(name, tuple) else (name, value)] = row
        sp = f"Spearman min {row['spearman_min']:.4f} med {row['spearman_median']:.4f}" if "spearman_min" in row else "(other scales)"
        label = name[0] if isinstance(name, tuple) else name
        print(f"{label:9s} {str(value):11s} vs {ref_name:12s} {sp} | ARI consensus {row['ari_consensus']:.3f} "
              f"settings {row['ari_settings']:.3f} | C*={row['C']} ({row['plateau']}/99) stab {row['stability']:.3f} "
              f"| seeds {row['seeds']:.0f}, stage 1 {row['stage1_s']:.0f}s, AP it {row['iterations']:.0f} "
              f"conv {row['converged']}/99 [purity {row['purity']:.3f}]", flush=True)

    paper = version(dict(PAPER))
    plan = {"extension": EXTENSION, "final": FINAL}.get(os.environ.get("STUDY_PLAN", ""))
    if plan is not None:
        for name, values, ref_value, others in plan:
            base_cfg = dict(PAPER, **others)
            ref = version(dict(base_cfg, **{name: ref_value}))
            for v in values:
                if (name, v, tuple(sorted(others.items()))) not in summary:
                    report((name, v, tuple(sorted(others.items()))), v, version(dict(base_cfg, **{name: v})), ref,
                           f"{name}={ref_value}")
        print("done", flush=True)
        return
    for name, values in RESOLUTION:
        finest = (FINEST.get(name, max))(values)
        ref = version(dict(PAPER, **{name: finest}))
        for v in values:
            if (name, v) not in summary:
                report(name, v, version(dict(PAPER, **{name: v})), ref, f"{name}={finest}")
    for name, values in DESIGN:
        for v in values:
            if (name, v) not in summary:
                report(name, v, version(dict(PAPER, **{name: v})), paper, "paper")
    print("done", flush=True)


if __name__ == "__main__":
    main()
