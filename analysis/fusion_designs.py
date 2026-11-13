#!/usr/bin/env python3
"""
fusion_designs.py - compare fusion architectures on Top-1 (99 settings each)
==========================================================================
Same rank-normalised SWD matrices, same Affinity Propagation; only the way the
20 (field, scale) views are combined changes.

  none         balanced rank average over fields and scales, no diffusion
  balanced     OURS: balanced rank average of the fields per scale, SNF across scales
  flux_rerank  previous version: flux potentials averaged and re-ranked before balancing
  snf12        one SNF over 12 views (3 sources x 4 scales)
  snf20        one SNF over all 20 field x scale views
  hierarchical three stacked SNF stages (flux fields, scales, sources)

Output: results/epuck/paper/designs_top1.pkl and a printed summary table.

Author: Olivier Vertu Ndinga Oba
"""
import os
import sys
from collections import Counter
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from classification import fusion as F  # noqa: E402
from classification.affinity import quantile_norm, rank_affinity  # noqa: E402
from classification.snf import snf  # noqa: E402
from pipeline.checkpoint import load_pickle  # noqa: E402

MISS = ["gate", "forag", "homing", "shelter", "xor"]


def _flux(Q, s):
    return np.mean([Q[f][s] for f in ("phi", "psi", "J")], axis=0)


def design(name, Q, knn, t):
    sc = list(Q["rho"].keys())
    if name == "balanced":
        return F.consensus(F.scale_views(Q), knn, t)
    if name == "none":
        d = np.mean([sum(w * Q[f][s] for f, w in F.source_weights(F.ALL_FIELDS).items()) for s in sc], axis=0)
        return rank_affinity(d.astype(np.float32))
    if name == "flux_rerank":
        v = [rank_affinity(((quantile_norm(Q["rho"][s]) + quantile_norm(Q["K"][s]) + quantile_norm(_flux(Q, s))) / 3).astype(np.float32)) for s in sc]
        return snf(v, knn, t)
    if name == "snf12":
        v = []
        for s in sc:
            v += [rank_affinity(Q["rho"][s]), rank_affinity(Q["K"][s]), rank_affinity(quantile_norm(_flux(Q, s)).astype(np.float32))]
        return snf(v, knn, t)
    if name == "snf20":
        return snf([rank_affinity(Q[f][s]) for f in F.ALL_FIELDS for s in sc], knn, t)
    if name == "hierarchical":
        return F.hierarchical_snf(Q, knn, t)
    raise ValueError(name)


def pattern(y, p):
    from sklearn.metrics.cluster import contingency_matrix
    cm = contingency_matrix(y, p)
    home = cm.argmax(1)
    groups = {}
    for m, c in enumerate(home):
        groups.setdefault(c, []).append(MISS[m])
    clean = all(cm[m, home[m]] >= 0.95 * cm[m].sum() for m in range(len(home))) and len(np.unique(p)) == len(groups)
    return "|".join(sorted("+".join(g) for g in groups.values())), clean


def one(name, Q, y, knn, t):
    lab, ex, conv = F.affinity_propagation(design(name, Q, knn, t))
    r = RS.score(y, lab)
    r["pattern"], r["clean"] = pattern(y, lab)
    r.update(design=name, knn=knn, t=t, converged=conv)
    return r


NAMES = ["none", "balanced", "flux_rerank", "snf12", "snf20", "hierarchical"]


def main():
    lab, rk, _ = RS.load_labels()
    D, _ = RS.load_distances()
    idx = RS.subset(1, rk)
    Q = F.rank_normalise(D, F.ALL_FIELDS, idx)
    y = lab[idx]
    if RS.done("designs_top1.pkl"):
        res = load_pickle(RS.OUT / "designs_top1.pkl")
    else:
        jobs = [(n, k, t) for n in NAMES for k, t in product(RS.KNNS, RS.TS)]
        res = Parallel(n_jobs=int(os.environ.get("DESIGN_JOBS", "16")), max_nbytes="1M")(delayed(one)(n, Q, y, k, t) for n, k, t in jobs)
        RS.save(res, "designs_top1.pkl")
    print(f"{'design':13s} {'P=1&ARI=1':>9s} {'C=5':>4s} {'C<=2':>5s} {'C4 clean':>8s} {'C3 clean':>8s} {'monotone':>8s} "
          f"{'median P':>8s}  most frequent clean merges at C=4 and C=3")
    summary = {}
    for n in NAMES:
        R = [r for r in res if r["design"] == n]
        C = np.array([r["C"] for r in R]); P = np.array([r["purity"] for r in R]); A = np.array([r["ari"] for r in R])
        g = {(r["knn"], r["t"]): r["C"] for r in R}
        mono = sum(all(g[(k, RS.TS[i + 1])] <= g[(k, RS.TS[i])] for i in range(len(RS.TS) - 1)) for k in RS.KNNS)
        clean = {c: Counter(r["pattern"] for r in R if r["C"] == c and r["clean"]) for c in (4, 3)}   # whole missions merged
        summary[n] = dict(perfect=int(np.sum((P > .9999) & (A > .9999))), c5=int(np.sum(C == 5)), collapse=int(np.sum(C <= 2)),
                          c4_clean=dict(clean[4]), c3_clean=dict(clean[3]), monotone=int(mono), median_purity=float(np.median(P)))
        s = summary[n]
        modal = "  ".join(clean[c].most_common(1)[0][0] if clean[c] else "-" for c in (4, 3))
        print(f"{n:13s} {s['perfect']:9d} {s['c5']:4d} {s['collapse']:5d} {sum(clean[4].values()):8d} {sum(clean[3].values()):8d} "
              f"{s['monotone']:6d}/9 {s['median_purity']:8.3f}  {modal}")
    # without diffusion every setting gives the same partition: how it divides each mission (one controller on Top-1)
    from sklearn.metrics.cluster import contingency_matrix
    pred = F.affinity_propagation(design("none", Q, RS.KNNS[0], RS.TS[0]))[0]
    summary["none"]["clusters_per_mission"] = (contingency_matrix(y, pred) > 0).sum(1).tolist()
    print("no diffusion: clusters per mission", summary["none"]["clusters_per_mission"])
    RS.save(summary, "designs_top1_summary.pkl")


if __name__ == "__main__":
    main()
