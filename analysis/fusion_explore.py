#!/usr/bin/env python3
"""
fusion_explore.py - other ways to combine the views and to cluster them, on the same distances
==============================================================================================
The method turns the twenty distance matrices (five fields x four scales) into a taxonomy in a fixed
way: mutual then percentile ranks, an equal weight per physical source, SNF across the four scales,
affinity propagation, and the taxonomy read from the plateau of the 99 settings. Each variant below changes
one of these steps and keeps the others; every variant is read in the same way (plateau, consensus).
Reported per dataset: what the reading sees without labels (C*, plateau, stability) and the agreement
with the missions (purity, ARI, homogeneity, completeness, misassigned runs, XOR Aggregation runs in a
Shelter cluster). The labels only score; nothing is chosen with them here.

One change at a time from the method:
  affinity of each view   mutual then percentile rank (method) | percentile rank alone | probit rank |
                          self-tuning kernel
  fields at one scale     equal weight per source (method) | per field | rank confidence (each field
                          weighted by the spectral gap of its neighbour graph) | the same gap shared
                          within each source (equal gaps give the method's weights) | weighted
                          geometric mean | maximum
  scales                  SNF across the four scales (method) | mean, no diffusion | diffusion of the
                          mean | SNF across the twenty field-and-scale views
  clustering              AP, median preference (method) | AP, minimum preference | spectral, C from
                          the eigengap | Louvain modularity | average linkage cut at the largest gap |
                          HDBSCAN

Usage: python analysis/fusion_explore.py [tops, default 1,2,3,4] [n_jobs, default 8] [variants]
       python analysis/fusion_explore.py report
Output: results/epuck/paper/fusion_explore.pkl (resumable, for the current distances only; a job run
        alongside writes to the file named by FUSION_EXPLORE_OUT, and the report reads them all)

Author: Olivier Vertu Ndinga Oba
"""
import os
import sys
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import run_sweep as RS  # noqa: E402
from classification import fusion as F  # noqa: E402
from classification.affinity import net_affinity, quantile_norm, rank_affinity, rank_distance  # noqa: E402
from classification.snf import snf_path  # noqa: E402
from pipeline.checkpoint import Store  # noqa: E402

GRID = list(product(RS.KNNS, RS.TS))
OUT_NAME = os.environ.get("FUSION_EXPLORE_OUT", "fusion_explore.pkl")     # one file per concurrent job
FIELDS = F.ALL_FIELDS
SOURCE_SIZE = {f: len(fl) for fl in F.SOURCES.values() for f in fl}        # number of fields of each field's source
VARIANTS = {          # name: (affinity of each view, fields at one scale, scales, clustering[, reader])
    "method": ("mutual", "balanced", "snf4", "ap"),
    "aff-percentile": ("rank", "balanced", "snf4", "ap"),          # the percentile ranks alone (former method)
    "aff-probit": ("probit", "balanced", "snf4", "ap"),
    "aff-selftuning": ("selftune", "balanced", "snf4", "ap"),
    "fields-equal": ("mutual", "equal", "snf4", "ap"),
    "fields-confidence": ("mutual", "confidence", "snf4", "ap"),
    "fields-sourceconf": ("mutual", "source-confidence", "snf4", "ap"),
    "fields-geomean": ("mutual", "geomean", "snf4", "ap"),
    "fields-max": ("mutual", "max", "snf4", "ap"),
    "scales-mean": ("mutual", "balanced", "mean", "ap"),
    "scales-meandiff": ("mutual", "balanced", "meandiff", "ap"),
    "scales-snf20": ("mutual", "balanced", "snf20", "ap"),
    "clu-ap-minpref": ("mutual", "balanced", "snf4", "ap-minpref"),
    "clu-spectral": ("mutual", "balanced", "snf4", "spectral"),
    "clu-louvain": ("mutual", "balanced", "snf4", "louvain"),
    "clu-agglo": ("mutual", "balanced", "snf4", "agglo"),
    "clu-hdbscan": ("mutual", "balanced", "snf4", "hdbscan"),
    # the same question on the other reader of the fields: the distribution of their values
    "values-method": ("mutual", "balanced", "snf4", "ap", "values"),
    "values-percentile": ("rank", "balanced", "snf4", "ap", "values"),
}
FIRST_ROUND = [n for n, v in VARIANTS.items() if len(v) == 4]          # every variant on the diagrams


# --------------------------------------------------------------------------- affinity of one view
def probit_rank(d):
    """Percentile rank mapped through the normal quantile function, rescaled to [0, 1]."""
    q = quantile_norm(d)
    iu = np.triu_indices_from(d, 1)
    z = norm.ppf(q[iu] - 0.5 / len(iu[0]))
    out = np.zeros_like(d)
    out[iu] = (z - z.min()) / (z.max() - z.min())
    return out + out.T


def self_tuning(d, k=7):
    """1 - exp(-d^2 / (s_i s_j)), s_i the distance to the k-th neighbour (Zelnik-Manor and Perona)."""
    s = np.sort(d, axis=1)[:, k]
    a = np.exp(-d ** 2 / np.maximum(np.outer(s, s), 1e-300))
    out = 1.0 - a
    np.fill_diagonal(out, 0.0)
    return out


def spectral_gap_confidence(d, k=7):
    """The rank-confidence weight of the earlier code: largest gap of the normalised Laplacian of
    the k-neighbour heat-kernel graph of one field (no label)."""
    n = len(d)
    w = np.exp(-d / (d.std() + 1e-12))
    nn = np.argsort(d, axis=1)[:, 1:k + 1]
    mask = np.zeros((n, n), dtype=bool)
    mask[np.repeat(np.arange(n), k), nn.ravel()] = True
    w = np.where(mask | mask.T, w, 0.0)
    w = 0.5 * (w + w.T)
    deg = np.maximum(w.sum(1), 1e-12)
    lap = np.eye(n) - w / np.sqrt(np.outer(deg, deg))
    ev = np.sort(np.linalg.eigvalsh(lap))
    return float(np.diff(ev)[1:].max())


TRANSFORM = {"rank": quantile_norm, "probit": probit_rank, "mutual": rank_distance, "selftune": self_tuning}


def scale_views(Dsub, aff, comb):
    """One view per scale, built from the five fields as the method does, with one step changed."""
    w = F.source_weights(FIELDS)
    views = []
    for s in sorted(Dsub[FIELDS[0]]):
        T = {f: TRANSFORM[aff](Dsub[f][s]).astype(np.float32) for f in FIELDS}    # as rank_normalise stores them
        if comb == "balanced":
            d = sum(w[f] * T[f] for f in FIELDS)
        elif comb == "equal":
            d = sum(T[f] for f in FIELDS) / len(FIELDS)
        elif comb == "confidence":
            c = {f: spectral_gap_confidence(Dsub[f][s]) for f in FIELDS}
            d = sum(c[f] * T[f] for f in FIELDS) / sum(c.values())
        elif comb == "source-confidence":                            # the gap shared among the fields of its source
            c = {f: spectral_gap_confidence(Dsub[f][s]) / SOURCE_SIZE[f] for f in FIELDS}
            d = sum(c[f] * T[f] for f in FIELDS) / sum(c.values())
        elif comb == "geomean":
            d = 1.0 - np.prod([np.clip(1.0 - T[f], 1e-12, 1.0) ** w[f] for f in FIELDS], axis=0)
        elif comb == "max":
            d = 1.0 - np.max([1.0 - T[f] for f in FIELDS], axis=0)
        else:
            raise ValueError(comb)
        np.fill_diagonal(d, 0.0)
        views.append(rank_affinity(np.asarray(d, dtype=np.float32)))
    return views


def build_views(Dsub, aff, comb, scales):
    if scales == "snf20":
        return [rank_affinity(TRANSFORM[aff](Dsub[f][s]).astype(np.float32)) for f in FIELDS for s in sorted(Dsub[f])]
    views = scale_views(Dsub, aff, comb)
    if scales == "snf4":
        return views
    mean = (sum(views) / len(views)).astype(np.float32)
    return [mean] if scales == "mean" else [mean, mean]            # [W, W]: SNF diffuses W through itself


# --------------------------------------------------------------------------- clustering of a consensus
def cluster(S, how, knn):
    S = net_affinity(np.asarray(S, dtype=np.float64))
    n = len(S)
    if how == "ap-minpref":
        from sklearn.cluster import AffinityPropagation
        off = S[~np.eye(n, dtype=bool)]
        return AffinityPropagation(affinity="precomputed", damping=0.9, max_iter=1000, convergence_iter=100,
                                   preference=float(off.min()), random_state=0).fit(S).labels_
    if how == "spectral":
        from sklearn.cluster import SpectralClustering
        deg = np.maximum(S.sum(1), 1e-12)
        ev = np.sort(np.linalg.eigvalsh(np.eye(n) - S / np.sqrt(np.outer(deg, deg))))[:21]
        c = int(np.argmax(np.diff(ev)[1:]) + 2)                      # C in 2..20, at the largest eigengap
        return SpectralClustering(n_clusters=c, affinity="precomputed", assign_labels="cluster_qr",
                                  random_state=0).fit(S).labels_
    if how == "louvain":
        import networkx as nx
        nn = np.argsort(-S, axis=1)[:, 1:knn + 1]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in nn[i]:
                g.add_edge(i, int(j), weight=float(S[i, j]))
        lab = np.empty(n, dtype=int)
        for c, members in enumerate(nx.community.louvain_communities(g, weight="weight", seed=0)):
            lab[list(members)] = c
        return lab
    if how == "agglo":
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        Z = linkage(squareform(1.0 - S, checks=False), "average")
        h = Z[-20:, 2]                                               # the last merges, from 20 clusters to 1
        c = 20 - int(np.argmax(np.diff(h)))                          # cut before the largest jump
        return fcluster(Z, c, "maxclust") - 1
    if how == "hdbscan":
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(metric="precomputed", min_cluster_size=5).fit(1.0 - S).labels_
        return np.where(lab < 0, lab.max() + 1, lab)                 # noise counted as one cluster
    raise ValueError(how)


def _cluster_knn(views, knn, how):
    path = snf_path(views, knn, RS.TS, None)
    return knn, {t: cluster(path[t], how, knn) for t in RS.TS}


def sweep(views, clu, n_jobs):
    """The 99 settings, each clustered by `clu`; same output as fusion.sweep."""
    if clu == "ap":
        return F.sweep(views, GRID, device=os.environ.get("FUSION_EXPLORE_DEVICE", "auto"), n_jobs=n_jobs)
    done = dict(Parallel(n_jobs=n_jobs, max_nbytes="1M")(delayed(_cluster_knn)(views, k, clu) for k in RS.KNNS))
    return [dict(knn=k, t=t, C=int(len(np.unique(done[k][t]))), pred=np.asarray(done[k][t]).astype(np.int16),
                 exemplars=None) for k, t in GRID]


# --------------------------------------------------------------------------- reading and scores
def summary(res, y):
    pc = F.plateau_consensus(res, len(y))
    lab = pc["labels"]
    cm = contingency_matrix(y, lab)
    dom = cm.argmax(0)
    h, c, _ = homogeneity_completeness_v_measure(y, lab)
    sc = RS.score(y, lab)
    per = [RS.score(y, r["pred"])["purity"] for r in res]
    return dict(C=pc["C"], plateau=pc["n_plateau"], stability=pc["stability"], purity=sc["purity"], ari=sc["ari"],
                hom=h, comp=c, misassigned=int(len(y) - cm.max(0).sum()),
                xor_in_shelter=int(sum(1 for i in range(len(y)) if y[i] == 4 and dom[lab[i]] == 3)),
                median_purity=float(np.median(per)), labels=np.asarray(lab, dtype=np.int16))


def report():
    """One table per dataset, and the means over the datasets done for every variant: first what the
    reading sees without labels (plateau, stability), then the agreement with the missions."""
    from pipeline.checkpoint import load_pickle
    R = {}
    for f in sorted(RS.OUT.glob("fusion_explore*.pkl")):                 # every job, for the current distances
        if (f.parent / (f.name + ".source")).read_text().strip() == RS.dist_source():
            R.update(load_pickle(f))
    G = load_pickle(RS.OUT / "grid.pkl")                            # the method itself, where no job ran it yet
    for t in {t for _, t in R}:
        if ("method", t) not in R and t in G:
            R[("method", t)] = summary(G[t]["results"], G[t]["labels"])
    tops = sorted({t for _, t in R})
    names = [n for n in VARIANTS if any((n, t) in R for t in tops)]
    head = f"{'variant':18s} {'C*':>3s} {'plat':>4s} {'stab':>5s} | {'pur':>5s} {'ARI':>5s} {'hom':>5s} {'comp':>5s} {'wrong':>5s} {'xor>sh':>6s}"
    for t in tops:
        print(f"\nTop-{t}\n{head}")
        for n in names:
            if (n, t) in R:
                r = R[(n, t)]
                print(f"{n:18s} {r['C']:3d} {r['plateau']:4d} {r['stability']:5.3f} | {r['purity']:5.3f} {r['ari']:5.3f} "
                      f"{r['hom']:5.3f} {r['comp']:5.3f} {r['misassigned']:5d} {r['xor_in_shelter']:6d}")
    common = [t for t in tops if all((n, t) in R for n in names)]
    if common:
        print(f"\nmeans over Top-{','.join(map(str, common))}\n{head}")
        for n in names:
            m = {k: np.mean([R[(n, t)][k] for t in common]) for k in ("plateau", "stability", "purity", "ari", "hom", "comp")}
            w = sum(R[(n, t)]["misassigned"] for t in common); x = sum(R[(n, t)]["xor_in_shelter"] for t in common)
            print(f"{n:18s} {'':3s} {m['plateau']:4.0f} {m['stability']:5.3f} | {m['purity']:5.3f} {m['ari']:5.3f} "
                  f"{m['hom']:5.3f} {m['comp']:5.3f} {w:5d} {x:6d}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        return report()
    tops = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3,4").split(",")]
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else FIRST_ROUND
    lab, rk, _ = RS.load_labels()
    D = {"persistence": RS.load_distances()[0]}
    if any(len(VARIANTS[n]) > 4 for n in names):                    # the value reader (analysis/baselines.py)
        from pipeline.checkpoint import load_pickle
        D["values"] = load_pickle(RS.OUT / "baselines.pkl")["fieldstat_D"]
    out = Store(RS.OUT / OUT_NAME, RS.dist_source())

    def done_elsewhere():
        """the entries that a job run alongside (FUSION_EXPLORE_OUT) has already written for these distances"""
        from pipeline.checkpoint import load_pickle
        keys = set()
        for f in RS.OUT.glob("fusion_explore*.pkl"):
            src = f.parent / (f.name + ".source")
            if f.name != OUT_NAME and src.exists() and src.read_text().strip() == RS.dist_source():
                keys |= set(load_pickle(f))
        return keys
    for top in tops:
        idx = RS.subset(top, rk)
        y = lab[idx]
        Dsub = {r: {f: {s: np.asarray(Dr[f][s], dtype=np.float64)[np.ix_(idx, idx)] for s in Dr[f]} for f in FIELDS}
                for r, Dr in D.items()}
        for name in names:
            if (name, top) in out or (name, top) in done_elsewhere():
                continue
            aff, comb, scales, clu, *reader = VARIANTS[name]
            res = sweep(build_views(Dsub[reader[0] if reader else "persistence"], aff, comb, scales), clu, n_jobs)
            out[(name, top)] = r = summary(res, y)
            print(f"Top-{top:2d} {name:18s} C*={r['C']:2d} ({r['plateau']:2d}/99) stab {r['stability']:.3f} | "
                  f"purity {r['purity']:.3f} ARI {r['ari']:.3f} hom {r['hom']:.3f} comp {r['comp']:.3f} "
                  f"wrong {r['misassigned']:3d} (XOR in Shelter {r['xor_in_shelter']:3d}) | median purity {r['median_purity']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
