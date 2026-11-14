#!/usr/bin/env python3
"""
make_numbers.py - write manuscript/numbers.tex and manuscript/tables/*.tex from the results.
Every number quoted in the paper comes from here; a missing result simply leaves its macro
undefined, which prints in red in the PDF.

Author: Olivier Vertu Ndinga Oba
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from pipeline import config as CFG0  # noqa: E402
import run_sweep as _RS  # noqa: E402
P = ROOT / "results" / "epuck" / "paper"
MS = ROOT / "manuscript"
TAB = MS / "tables"; TAB.mkdir(parents=True, exist_ok=True)
KNNS = [3, 5, 7, 9, 11, 13, 15, 20, 25]
TS = [1, 2, 3, 5, 7, 9, 10, 12, 15, 20, 25]
MNAME = ["Directional Gate", "Foraging", "Homing", "Shelter", "XOR Aggregation"]
WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen twenty").split()     # counts written in words in the text
nums = {}


def load(name, root=P):
    f = root / name
    return pickle.load(open(f, "rb")) if f.exists() else None



def full_width(table):
    """A tabular spread over the whole text width, its columns separated evenly, so that every table is aligned with the
    text like the figures."""
    table = re.sub(r"\\begin\{tabular\}\{@\{\}", r"\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}", table, count=1)
    return table.replace(r"\end{tabular}", r"\end{tabular*}")


def full_width_shaded(table):
    """Spread a table across \\linewidth using tabularx (column 1 = X) so \\methodrow shading is continuous without
    @{\\extracolsep{\\fill}} white vertical gaps."""
    table = re.sub(
        r"\\begin\{tabular\}\{@\{\}(?:(?:>\{[^}]*\})?p\{[^}]*\}|l)",
        r"\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}X",
        table,
        count=1,
    )
    return table.replace(r"\end{tabular}", r"\end{tabularx}")


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}"


# the indent of an item under a group heading in a table; written in the cell itself (the table environment of the
# journal class, through threeparttable, cancels a column hook), so that a wrapped item stays under its first line
ITEM = r"\hangindent=1em\hangafter=1\quad "


def put(k, v):
    nums[k] = v


CLAIMS = []      # the statements of the text that the numbers alone do not carry, checked on the results


def claim(text, ok):
    CLAIMS.append((text, bool(ok)))


# ------------------------------------------------------------------ data and ladder
lad = load("scale_ladder.pkl", ROOT / "results" / "epuck")
if lad is not None:
    for i, s in enumerate(lad["ladder"]):
        put(f"sigma{i + 1}-cm", f"{100 * s:.1f}")
    _lr = np.asarray(lad["ladder"], dtype=float)
    put("ladder-ratio", f"{(_lr[-1] / _lr[0]) ** (1 / (len(_lr) - 1)):.2f}")      # each width over the previous one
    claim("the widths of the ladder grow geometrically (one ratio between consecutive widths)",
          np.allclose(_lr[1:] / _lr[:-1], (_lr[-1] / _lr[0]) ** (1 / (len(_lr) - 1)), rtol=1e-9))
_labels = ROOT / "results" / "epuck" / "labels.npz"
if _labels.exists():
    _n = len(np.load(_labels)["labels"])
    put("n-runs", str(_n)); put("n-pairs-million", f"{_n * (_n - 1) / 2e6:.1f}")

# ------------------------------------------------------------------ cost of the stages (measured, from scratch)
_tm = ROOT / "results" / "epuck" / "timing.json"
if _tm.exists():
    import json
    tm = json.loads(_tm.read_text())
    for _stage, _key in (("splat", "fields"), ("eph", "diagrams"), ("swd", "distances")):
        if tm.get(_stage, {}).get("from_scratch"):
            put(f"cost-{_key}-min", f"{tm[_stage]['seconds'] / 60:.0f}")

# ------------------------------------------------------------------ the tail of the footprint, by quadrature
sys.path.insert(0, str(MS))
from verify_propositions import tail_fraction_quadrature  # noqa: E402
_m, _e = f"{tail_fraction_quadrature(CFG0.WINDOW)[1]:.0e}".split("e")
put("tail-quad-err", f"{_m}\\times10^{{{int(_e)}}}")
import baselines as _BL  # noqa: E402   the quantiles of the value reader
put("n-quantiles", str(len(_BL.QS)))

# ------------------------------------------------------------------ grid and taxonomy
G = load("grid.pkl"); T = load("taxonomy.pkl")  # noqa
if G is not None:
    for top, g in G.items():
        R = g["results"]; C = np.array([r["C"] for r in R]); Pu = np.array([r["purity"] for r in R]); A = np.array([r["ari"] for r in R])
        put(f"top{top}-perfect", str(int(((Pu > .9999) & (A > .9999)).sum())))
        put(f"top{top}-c5", str(int((C == 5).sum())))
        put(f"top{top}-cmax", str(int(C.max())))
        _r = R[int(np.argmax(C))]
        put(f"top{top}-cmax-at", f"$\\knn={_r['knn']}$, $t={_r['t']}$")
        put(f"top{top}-nonconv", str(sum(not r["converged"] for r in R)))
        # the sweep setting by setting, beside the consensus of the plateau: settings with no misassigned run (purity one),
        # those among them on the plateau (C = C*), and the fewest, median and most misassigned runs over the settings
        _N = len(g["labels"]); _err = np.array([round((1 - p) * _N) for p in Pu]); _free = _err == 0
        _cs = T[top]["C"] if T is not None else None
        put(f"top{top}-errfree", str(int(_free.sum()))); put(f"top{top}-errfree-c5", str(int((_free & (C == 5)).sum())))
        put(f"top{top}-errfree-more", str(int((_free & (C > 5)).sum())))
        put(f"top{top}-errfree-plateau", str(int((_free & (C == _cs)).sum())) if _cs is not None else "0")
        put(f"top{top}-err-min", str(int(_err.min()))); put(f"top{top}-err-med", f"{np.median(_err):.0f}")
        put(f"top{top}-err-max", str(int(_err.max())))
        put(f"top{top}-min-pur", f3(float(Pu.min())))
        put(f"top{top}-err-le2", str(int((_err <= 2).sum())))
        row = sorted([r for r in R if r["t"] == 10], key=lambda r: r["knn"])
        put(f"top{top}-t10-pmin", f3(min(r["purity"] for r in row)))
        _pmin = min(r["purity"] for r in row)
        put(f"top{top}-t10-phrase", "purity is one for every neighborhood size" if _pmin > 1 - 1e-12
            else f"purity stays at or above {_pmin:.3f} for every neighborhood size")
    _all_res = [r for g in G.values() for r in g["results"]]
    _all_ge995 = sum(r["purity"] >= 0.995 for r in _all_res)
    put("all-settings-n", str(len(_all_res)))
    put("all-pur-ge995", str(_all_ge995))
    put("all-pur-ge995-pct", f"{100 * _all_ge995 / len(_all_res):.1f}")
    if 10 in G:
        _r10_k3 = sorted([r for r in G[10]["results"] if r["knn"] == 3], key=lambda r: r["t"])
        _w10 = min(_r10_k3, key=lambda r: r["purity"])
        put("top10-worst-knn", str(_w10["knn"]))
        put("top10-worst-t", str(_w10["t"]))
        put("top10-worst-pur", f3(_w10["purity"]))
        put("top10-worst-err", str(int(round((1 - _w10["purity"]) * len(G[10]["labels"])))))
        put("top10-k3-t10plus-pmin", f3(min(r["purity"] for r in _r10_k3 if r["t"] >= 10)))
    # from which dataset on no setting is error-free, and how few runs the best setting then misassigns
    _none = [t_ for t_ in sorted(G) if not any(round((1 - r_["purity"]) * len(G[t_]["labels"])) == 0 for r_ in G[t_]["results"])]
    if _none:
        put("errfree-none-from", str(_none[0]))
        _best = [min(round((1 - r_["purity"]) * len(G[t_]["labels"])) for r_ in G[t_]["results"]) for t_ in _none]
        put("errfree-none-best-lo", str(min(_best))); put("errfree-none-best-hi", str(max(_best)))
        put("errfree-none-best-phrase", "a single run" if max(_best) == 1 else
            (f"{min(_best)} runs" if min(_best) == max(_best) else f"{min(_best)} to {max(_best)} runs"))
    claim("error-free settings exist on the first datasets and on none from one dataset on",
          (_none == list(range(_none[0], max(G) + 1))) if _none else True)
    # monotone columns over all datasets
    nonmono = 0
    for top, g in G.items():
        grid = {(r["knn"], r["t"]): r["C"] for r in g["results"]}
        for k in KNNS:
            seq = [grid[(k, t)] for t in TS]
            nonmono += any(seq[i + 1] > seq[i] for i in range(len(seq) - 1))
    put("nonmono-columns", str(nonmono))
    put("nonmono-phrase", "in any of the $90$ columns" if nonmono == 0 else f"except in {nonmono} of the $90$ columns")
    for top in (1, 10):                    # cost of the clustering, per setting (the sweep of one dataset / 99)
        if top in G and "seconds" in G[top]:
            put(f"top{top}-sec-per-setting", f"{G[top]['seconds'] / len(G[top]['results']):.2g}")
            put("grid-device", G[top]["device"].replace("NVIDIA GeForce ", ""))
    _nc = sum(not r["converged"] for g in G.values() for r in g["results"])
    put("all-nonconv", str(_nc))
    _ns = sum(len(g["results"]) for g in G.values())
    put("conv-phrase", f"in all {_ns} settings" if _nc == 0 else f"in all but {_nc} of the {_ns} settings")
    # residual errors on Top-3 and Top-4: XOR runs absorbed by the Shelter cluster
    if T is not None:
        _mis = _xs = _df = _runs = _exact = 0
        for _top in range(1, 11):
            _y, _cl = G[_top]["labels"], T[_top]["labels"]
            _c = contingency_matrix(_y, _cl); _dom = _c.argmax(0)            # dominant mission of each cluster
            _wrong = [(int(_y[i]), int(_dom[_cl[i]])) for i in range(len(_y)) if _dom[_cl[i]] != _y[i]]
            _x = sum(1 for a_, b_ in _wrong if (a_, b_) == (4, 3))           # XOR Aggregation runs in a Shelter cluster
            _df += sum(1 for a_, b_ in _wrong if (a_, b_) == (0, 1))         # Directional Gate runs in a Foraging cluster
            _mis += len(_wrong); _xs += _x; _runs += len(_y); _exact += not _wrong
            put(f"top{_top}-xor-in-shelter", str(_x)); put(f"top{_top}-misassigned", str(int(T[_top]["misassigned"])))
        put("all-misassigned", str(_mis)); put("all-xor-in-shelter", str(_xs)); put("all-dg-in-foraging", str(_df))
        put("all-misassigned-words", WORDS[_mis] if _mis < len(WORDS) else str(_mis))
        put("all-runs", f"{_runs:,}".replace(",", "\\,")); put("n-exact-datasets", WORDS[_exact])
        claim("every misassigned run is a Directional Gate run with Foraging or an XOR Aggregation run with Shelter",
              _mis == _df + _xs)
        claim("every misassigned run of Top-1 is an XOR Aggregation run placed with Shelter",
              int(T[1]["misassigned"]) == int(nums["top1-xor-in-shelter"]))
        _last = (_none[0] - 1) if _none else max(G)          # the last dataset with an error-free setting
        _ef = [int(nums[f"top{t_}-errfree"]) for t_ in range(2, _last + 1)]
        put("errfree-pooled-last", str(_last))
        put("errfree-pooled-min", str(min(_ef))); put("errfree-pooled-max", str(max(_ef)))
        claim(f"from Top-2 to Top-{_last} some settings are error-free", all(e > 0 for e in _ef))
        # the counts of the text in words, with the grammar of their number
        _nruns = lambda n, what: f"{WORDS[n]} {what} run{'' if n == 1 else 's'}"
        put("all-errors-phrase", f"{_nruns(_df, 'Directional Gate')} placed with Foraging and "
                                 f"{_nruns(_xs, 'XOR Aggregation')} placed with Shelter")
        _n10 = int(T[10]["misassigned"])
        put("top10-misassigned-phrase", "no run is misassigned" if _n10 == 0 else
            ("only one is misassigned" if _n10 == 1 else f"{_n10} are misassigned"))
    t10 = {top: sorted([r for r in g["results"] if r["t"] == 10], key=lambda r: r["knn"]) for top, g in G.items()}
    put("t10-pooled-pmin-k15", f3(min(r["purity"] for top in range(5, 11) for r in t10[top] if r["knn"] <= 15)))
    put("t10-pooled-pmin", f3(min(r["purity"] for top in range(5, 11) for r in t10[top])))
    put("t10-all-pmin", f3(min(r["purity"] for top in t10 for r in t10[top])))
    _mono = sum(all(a["C"] >= b["C"] for a, b in zip(t10[top], t10[top][1:])) for top in t10)
    put("t10-monotone", str(_mono))
    put("t10-monotone-phrase", "for every dataset" if _mono == len(t10) else f"for {_mono} of the {len(t10)} datasets")
    # appendix table: C and purity at t = 10
    lines = [r"\begin{tabular}{@{}l" + "c" * len(KNNS) + r"@{\hspace{1.2em}}c@{}}", r"\toprule",
             r" & \multicolumn{" + str(len(KNNS)) + r"}{c}{Number of clusters $C$ at the neighborhood size $\knn$} & Lowest\\",
             r"\cmidrule(lr){2-" + str(len(KNNS) + 1) + "}",
             r"Dataset & " + " & ".join(str(k) for k in KNNS) + r" & purity\\", r"\midrule"]
    for top in sorted(t10):
        cs = T[top]["C"] if T is not None else None
        cells = [(f"\\textbf{{{r['C']}}}" if r["C"] == cs else str(r["C"])) for r in t10[top]]
        lines.append(f"Top-{top} & " + " & ".join(cells) + f" & {min(r['purity'] for r in t10[top]):.3f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t10.tex").write_text(full_width("\n".join(lines)) + "\n")
if T is not None:
    put("top1-purity", f3(T[1]["purity"])); put("top1-ari", f3(T[1]["ari"]))
    put("top1-plateau", str(T[1]["n_plateau"])); put("top1-stability", f3(T[1]["stability"]))
    put("min-plateau-purity", f3(min(T[t]["purity"] for t in T)))
    put("min-stability", f3(min(T[t]["stability"] for t in T)))
    put("max-stability", f3(max(T[t]["stability"] for t in T)))
    put("top10-cstar", str(T[10]["C"])); put("top10-purity", f3(T[10]["purity"]))
    put("top10-hom", f3(T[10]["homogeneity"])); put("top10-comp", f3(T[10]["completeness"]))
    put("top10-correct", str(len(T[10]["labels"]) - int(T[10]["misassigned"])))
    put("pooled-min-purity", f3(min(T[t]["purity"] for t in range(2, 11))))
    _n1 = len(T[1]["labels"]); _m1 = int(T[1]["misassigned"])
    put("top1-n", str(_n1)); put("top1-correct", str(_n1 - _m1)); put("top1-misassigned", str(_m1))
    claim("Top-1: the plateau lies at C* = 5", T[1]["C"] == 5)
    claim("Top-1: the consensus places every run in the cluster of its mission", _m1 == 0)
    if G is not None:                    # the settings that misassign runs on Top-1, and how many runs each
        _bad1 = [e for e in (round((1 - r["purity"]) * _n1) for r in G[1]["results"]) if e > 0]
        claim("Top-1: a single setting misassigns runs, and it misassigns one run", _bad1 == [1])
        claim("Top-1: every setting with more than five clusters is pure",
              all(r["purity"] > 1 - 1e-9 for r in G[1]["results"] if r["C"] > 5))
    if _m1 == 1:                         # the exception, and how often the plateau settings agree with it
        _y1, _c1 = G[1]["labels"], T[1]["labels"]
        _dom1 = contingency_matrix(_y1, _c1).argmax(0)
        _i = next(i for i in range(_n1) if _dom1[_c1[i]] != _y1[i])
        _R1 = [r for r in G[1]["results"] if r["C"] == T[1]["C"]]
        _away = np.mean([r["pred"][_i] != np.bincount(r["pred"][_y1 == _y1[_i]]).argmax() for r in _R1])
        put("top1-exception-mission", MNAME[int(_y1[_i])]); put("top1-exception-joined", MNAME[int(_dom1[_c1[_i]])])
        put("top1-exception-share", f"{100 * _away:.0f}")
        claim("the exception of Top-1 is an XOR Aggregation run placed with Shelter", (_y1[_i], _dom1[_c1[_i]]) == (4, 3))
    claim("the plateau rises with pooled controllers, up to its largest value on Top-10", T[10]["C"] == max(T[t]["C"] for t in T))
    claim("Top-10: almost no cluster mixes missions (consensus purity at least 0.98) and missions split (C* > 5)",
          T[10]["purity"] >= 0.98 and T[10]["C"] > 5)
    put("pooled-min-hom", f3(min(T[t]["homogeneity"] for t in range(2, 11))))
    # the ARI once the clusters that share a dominant mission are merged: what remains of its decrease is mixing
    _am = []
    for _t in sorted(T):
        _yy, _pp = np.asarray(G[_t]["labels"]), np.asarray(T[_t]["labels"])
        _dd = contingency_matrix(_yy, _pp).argmax(0)                    # dominant mission of each cluster
        _am.append(adjusted_rand_score(_yy, _dd[np.searchsorted(np.unique(_pp), _pp)]))
    put("ari-merged-min", f3(min(_am)))
    claim("merging the clusters that share a dominant mission raises the ARI to at least 0.99 on every dataset", min(_am) >= 0.99)
    claim("on every pooled dataset the homogeneity stays at least 0.99, above the completeness",
          all(T[t]["homogeneity"] >= 0.99 and T[t]["homogeneity"] >= T[t]["completeness"] for t in range(2, 11)))
    # one row per dataset; the misassigned runs are given out of the runs of the dataset, which replaces a column of sizes
    lines = [r"\begin{tabular}{@{}l@{\hspace{0.6em}}ccc@{\hspace{0.9em}}cccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Without labels} & \multicolumn{6}{c}{Against the five missions}\\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-10}",
             r"Dataset & $\Cstar$ & Modal & Stability & Purity & Misassigned & Error-free & Hom. & Compl. & ARI\\",
             r"\midrule"]
    for top in sorted(T):
        t = T[top]
        _N = len(t["labels"])
        _e = np.array([round((1 - r["purity"]) * _N) for r in G[top]["results"]]); _c = np.array([r["C"] for r in G[top]["results"]])
        lines.append(f"Top-{top} & {t['C']} & {t['n_plateau']}/{len(G[top]['results'])} & {t['stability']:.3f} & "
                     f"{t['purity']:.3f} & {t['misassigned']}/{_N} & {int((_e == 0).sum())} ({int(((_e == 0) & (_c == t['C'])).sum())}) & "
                     f"{t['homogeneity']:.3f} & {t['completeness']:.3f} & {t['ari']:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "taxonomy.tex").write_text(full_width("\n".join(lines)) + "\n")
# ------------------------------------------------------------------ affinity propagation has the time to finish
AC = load("ap_check.pkl")
if AC is not None:
    short = "window 15"
    longer = [k for k in AC if k[1] not in ("paper", short)]
    for top in (1, 10):
        if (top, "paper") in AC:
            it = AC[(top, "paper")]["iterations"]
            put(f"ap-top{top}-conv", str(AC[(top, "paper")]["converged"])); put(f"ap-top{top}-n", str(len(it)))
            put(f"ap-top{top}-iter-median", f"{np.median(it):.0f}"); put(f"ap-top{top}-iter-max", str(max(it)))
            claim(f"Top-{top}: with the paper's settings affinity propagation converges at every setting",
                  AC[(top, "paper")]["converged"] == len(it))
        if (top, short) in AC:
            put(f"ap-short-top{top}-same", str(AC[(top, short)]["settings_identical"]))
            put(f"ap-short-top{top}-cons", f3(AC[(top, short)]["consensus_ari"]))
    finish = [k for k in longer if k[1].startswith(("window", "ceiling"))]       # more time to finish
    other = [k for k in longer if k not in finish]                                # another damping or noise
    n_ = lambda k: len(AC[(k[0], "paper")]["iterations"])
    if finish:
        put("ap-finish-same-min", str(min(AC[k]["settings_identical"] for k in finish)))
        claim("affinity propagation has the time to finish: a longer window or a higher ceiling leaves every partition unchanged",
              all(AC[k]["settings_identical"] == n_(k) for k in finish))
    if other:
        put("ap-other-changed-max", str(max(n_(k) - AC[k]["settings_identical"] for k in other)))
        put("ap-other-cons-min", f3(min(AC[k]["consensus_ari"] for k in other)))
        claim("another damping or noise leaves the taxonomy built on the plateau nearly unchanged (ARI at least 0.99)",
              all(AC[k]["consensus_ari"] >= 0.99 for k in other))
    for top in (1, 10):
        if (top, short) in AC and (top, "paper") in AC:
            put(f"ap-short-top{top}-changed", str(n_((top, short)) - AC[(top, short)]["settings_identical"]))
    claim("a shorter convergence window can stop on a transient set of exemplars (the window of scikit-learn changes "
          "at least one setting)", any(AC[(top, short)]["settings_identical"] < len(AC[(top, "paper")]["iterations"])
                                       for top in (1, 10) if (top, short) in AC and (top, "paper") in AC))

# ------------------------------------------------------------------ how the exemplars are selected (Top-1 plateau)
if G is not None and T is not None:
    from collections import Counter
    _t = T[1]; _lab = np.asarray(_t["labels"])
    _plateau = [r for r in G[1]["results"] if r["C"] == _t["C"]]
    _cands, _freq, _ties = [], [], 0
    for _e in _t["exemplars"]:
        _members = set(np.flatnonzero(_lab == _lab[_e]).tolist())
        _cnt = Counter(int(x) for r in _plateau for x in r["exemplars"] if int(x) in _members)
        _cands.append(len(_cnt)); _freq.append(_cnt[int(_e)] / len(_plateau))
        _best = max(_cnt.values())
        _ties += sum(1 for v in _cnt.values() if v == _best) > 1
        claim(f"the exemplar of each Top-1 cluster is a run that affinity propagation chooses most often on the plateau "
              f"(run {int(_e)})", _cnt[int(_e)] == _best)
    put("ex-top1-cands-min", str(min(_cands))); put("ex-top1-cands-max", str(max(_cands)))
    put("ex-top1-freq-min", f"{100 * min(_freq):.0f}"); put("ex-top1-freq-max", f"{100 * max(_freq):.0f}")
    put("ex-top1-ties", WORDS[_ties])

# ------------------------------------------------------------------ the exemplars described in the text (Figure 10)
TR = load("top1_tracks.pkl")
if TR is not None and T is not None:
    _ex = {int(TR["labels"][e]): int(e) for e in T[1]["exemplars"]}

    def gathered(m, r=0.35):
        """fraction of the robots of the exemplar of mission m that end within r of their median final position"""
        fin = TR["tracks"][_ex[m]]["pos"][-1]
        return float(np.mean(np.hypot(*(fin - np.median(fin, axis=0)).T) < r))
    if len(_ex) == 5:
        claim("the Homing exemplar ends in a single dense spot (90 % of its robots or more)", gathered(2) >= 0.9)
        from scipy.cluster.vq import kmeans2 as _km
        _fin = TR["tracks"][_ex[4]]["pos"][-1]                       # final positions of the XOR Aggregation exemplar
        _cen, _lb = _km(_fin, 2, minit="++", seed=0)
        _sz = sorted((int(s_) for s_ in np.bincount(_lb, minlength=2)), reverse=True)
        put("ex-xor-split", f"{_sz[0]} and {_sz[1]}")
        claim("the XOR Aggregation exemplar splits its robots between two groups at least 0.6 m apart, each holding at least a third "
              "of them", min(_sz) >= len(_fin) / 3 and float(np.hypot(*(_cen[0] - _cen[1]))) > 0.6)

# ------------------------------------------------------------------ extended grid (fusion sequence)
E = load("extended_top1.pkl")
if E is not None:
    y = E["labels"]
    grid = {(r["knn"], r["t"]): r["C"] for r in E["results"]}
    put("ext-monotone", str(sum(all(grid[(k, E['ts'][i + 1])] <= grid[(k, E['ts'][i])] for i in range(len(E['ts']) - 1)) for k in E["knns"])))
    put("ext-ncols", str(len(E["knns"])))
    claim("in the extended grid the number of clusters never increases with the diffusion length",
          int(nums["ext-monotone"]) == len(E["knns"]))
    _beyond = [(k, t) for (k, t) in grid if not (k in KNNS and t in TS)]
    put("ext-five-beyond", str(sum(grid[kt] == 5 for kt in _beyond))); put("ext-beyond", str(len(_beyond)))
    claim("beyond the main grid the five missions still hold at most settings", 2 * sum(grid[kt] == 5 for kt in _beyond) > len(_beyond))
    claim("whole missions merge only where the neighborhoods reach 30 runs or the diffusion 40 iterations",
          all(k >= 30 or t >= 40 for (k, t), c in grid.items() if c < 5))

    def merged(pred, y=y):
        """the groups of missions whose runs fall mostly in one same cluster"""
        home = {}
        for m in np.unique(y):
            home.setdefault(int(np.bincount(pred[y == m]).argmax()), []).append(int(m))
        return sorted(tuple(g) for g in home.values() if len(g) > 1)
    seqs = {}                                         # knn -> [(merged groups, result)] each time the groups change
    for k in E["knns"]:
        seq = []
        for r in sorted((r for r in E["results"] if r["knn"] == k), key=lambda r: r["t"]):
            g = merged(r["pred"])
            if g and (not seq or seq[-1][0] != g):
                seq.append((g, r))
        seqs[k] = seq
    if G is not None and 1 in G:
        claim("no setting of the main grid joins two missions on Top-1",
              not any(merged(r["pred"], G[1]["labels"]) for r in G[1]["results"]))
    kmerge = [k for k in E["knns"] if seqs[k]]        # the neighbourhoods at which two missions merge
    DGF, SX = [(0, 1)], [(3, 4)]                      # Directional Gate with Foraging; Shelter with XOR Aggregation
    put("ext-n", str(len(E["results"])))
    _c3 = [r for r in E["results"] if r["C"] == 3]
    put("merge-c3-n", str(len(_c3)))
    if _c3:
        put("merge-c3-purity", f3(min(r["purity"] for r in _c3)))
    if kmerge:
        put("merge-knn-min", str(min(kmerge)))
        claim("no neighbourhood below the smallest merging one joins two missions up to t = ext-t-max, and every larger one does",
              kmerge == [k for k in E["knns"] if k >= min(kmerge)])
        claim("a merge joins Directional Gate with Foraging, Shelter with XOR Aggregation, or both, and never another pair",
              all(g in (DGF, SX, DGF + SX) for k in kmerge for g, _ in seqs[k]))
        def _kind(pred):
            """three clusters: Directional Gate with Foraging, Shelter with XOR Aggregation, Homing alone. 'exact' when every run
            lies with its mission, 'xor-in-homing' when a single XOR Aggregation run lies with Homing, else 'other'"""
            cm = contingency_matrix(y, pred)
            col = {m: int(cm[m].argmax()) for m in range(5)}              # the cluster of each mission: its majority cluster
            if not (col[0] == col[1] and col[3] == col[4] and len({col[0], col[3], col[2]}) == 3):
                return "other"
            stray = int(cm.sum() - sum(cm[m, col[m]] for m in range(5)))   # runs outside the cluster of their mission
            if stray == 0:
                return "exact"
            return "xor-in-homing" if stray == 1 and cm[4, col[2]] == 1 else "other"
        _kinds = [_kind(np.asarray(r["pred"])) for r in _c3]
        put("merge-c3-xor-homing", str(_kinds.count("xor-in-homing")))
        claim("every setting at C = 3 gathers exactly Directional Gate with Foraging, Shelter with XOR Aggregation, and Homing alone",
              bool(_c3) and all(k_ == "exact" for k_ in _kinds))
        claim("every setting at C = 4 merges one of the two pairs",
              all(merged(r["pred"]) in (DGF, SX) for r in E["results"] if r["C"] == 4))
    joins = [r for r in E["results"] if any(2 in g for g in merged(r["pred"]))]       # Homing with another mission
    claim("Homing never shares a cluster with another mission over the whole extended grid", not joins)
    put("ext-knn-max", str(max(E["knns"]))); put("ext-t-max", str(max(E["ts"])))

# ------------------------------------------------------------------ standard SNF
S = load("standard_snf.pkl")
if S is not None:
    _st = [(r["t"], r["standard"]["C"], r["standard"]["purity"]) for r in S]       # standard SNF as t grows
    put("std-t-first", str(_st[0][0])); put("std-C-first", str(_st[0][1]))
    put("std-t-max", str(_st[-1][0]))
    _mg = [t_ for t_, c_, p_ in _st if c_ < 5]                                    # where two missions share a cluster
    if _mg:
        put("std-merge-from", str(min(_mg))); put("std-merge-C", str(min(c_ for t_, c_, p_ in _st if t_ >= min(_mg))))
        put("std-merge-purity", f3(min(p_ for t_, c_, p_ in _st if t_ >= min(_mg))))
    claim("standard SNF merges two missions from some diffusion length on, up to the longest tested",
          bool(_mg) and all(c_ < 5 for t_, c_, p_ in _st if t_ >= min(_mg)))
    _oc = [(r["t"], r["ours"]["C"]) for r in S]                  # rank fusion: the first t from which C no longer changes
    _tf = min(t_ for t_, _ in _oc if all(c2 == _oc[-1][1] for t2, c2 in _oc if t2 >= t_))
    put("std-ours-t-from", str(_tf)); put("std-ours-C", str(_oc[-1][1]))
    claim("from some number of iterations on, rank fusion keeps its number of clusters as the diffusion lengthens, "
          "and standard SNF lets it drift", len({r["standard"]["C"] for r in S if r["t"] >= _tf}) > 1)

# ------------------------------------------------------------------ rotation invariance (e-puck)
INV = load("report.pkl", ROOT / "results" / "invariance_check")
if INV is not None:
    rep = INV["report"]; picks = INV["picks"]
    FIELDS_ = ["rho", "K", "phi", "psi", "J"]
    sym = {"rho": r"density $\rho_\sigma$", "K": r"kinetic intensity $K_\sigma$", "phi": r"divergence potential $\varphi_\sigma$",
           "psi": r"vorticity potential $\psi_\sigma$", "J": r"coupling $J_\sigma$"}
    worst = max(rep[(e, f, si)]["rotation"] / max(rep[(e, f, si)]["within"], 1e-12)
                for f in FIELDS_ for si in range(4) for e in picks)
    _ex = int(np.floor(np.log10(worst)))            # below a thousandth, as a power of ten
    put("inv-worst", f"{worst:.3f}" if worst >= 1e-3 else f"${worst / 10 ** _ex:.0f}\\times10^{{{_ex}}}$")
    rel = {(f, si): max(rep[(e, f, si)]["rotation"] / max(rep[(e, f, si)]["within"], 1e-12) for e in picks)
           for f in FIELDS_ for si in range(4)}
    other = max(v for (f, si), v in rel.items() if f != "J")
    put("inv-other-max", f"{other:.4f}" if other >= 5e-5 else "0.0001")
    claim("rotation: density, kinetic intensity and potentials recovered to rounding (below 0.001 of a within-mission "
          "distance); the coupling field differs most at the finest scale", other < 1e-3 and max(rel, key=rel.get) == ("J", 0))

# ------------------------------------------------------------------ pair separation (Top-1)
PS = load("pair_separation.pkl")
if PS is not None:
    R, pairs = PS["R"], [tuple(p) for p in PS["pairs"]]
    j_sx = pairs.index((3, 4)); j_gf = pairs.index((0, 1))
    put("sep-sx-best", f"{R[:, j_sx].max():.2f}"); put("sep-gf-best", f"{R[:, j_gf].max():.2f}")
    # the pairs and the fields as the text explains them (rows rho, K, phi, psi, J; a higher ratio separates more)
    j = {pr: pairs.index(pr) for pr in pairs}
    above = lambda jj: [f for f in range(5) if R[f, jj] > np.median(R[f])]        # fields above the median pair
    put("sep-others-min", f"{min(R[:, jj].max() for jj in range(len(pairs)) if jj not in (j_gf, j_sx)):.2f}")
    claim("the two pairs that merge, Directional Gate with Foraging and Shelter with XOR Aggregation, are the two that no field "
          "separates well: their best ratios are the two smallest",
          sorted(int(i) for i in np.argsort(R.max(0))[:2]) == sorted([j_gf, j_sx]))
    claim("Directional Gate and Foraging lie below the median pair in all five fields, and the vorticity potential separates them best",
          above(j_gf) == [] and int(np.argmax(R[:, j_gf])) == 3)
    claim("Shelter and XOR Aggregation also lie below the median pair in all five fields, and the divergence potential "
          "separates them best", above(j_sx) == [] and int(np.argmax(R[:, j_sx])) == 2)
    _best = lambda a_, b_: int(np.argmax(R[:, pairs.index((a_, b_))]))
    claim("Homing is separated from Directional Gate and Foraging best by the divergence potential, and from Shelter and "
          "XOR Aggregation best by the kinetic intensity",
          _best(0, 2) == 2 and _best(1, 2) == 2 and _best(2, 3) == 1 and _best(2, 4) == 1)

# ------------------------------------------------------------------ pooled controllers
if G is not None and T is not None:
    from sklearn.metrics import normalized_mutual_info_score as _nmi

    def alone(g, pred):
        """the controllers whose runs make up one cluster exactly: all of them, and only them"""
        key = g["labels"] * 100 + g["ranks"]
        out = []
        for c in np.unique(pred):
            ks = np.unique(key[pred == c])
            if len(ks) == 1 and (pred == c).sum() == (key == ks[0]).sum():
                out.append(int(ks[0]))
        return out
    iso = next(((n, k) for n in sorted(T) if n > 1 for k in alone(G[n], T[n]["labels"])), None)
    claim("on some pooled dataset, one consensus cluster is exactly the 50 runs of one controller", iso is not None)
    if iso is not None:
        n_, k_ = iso
        R_ = [r for r in G[n_]["results"] if r["C"] == T[n_]["C"]]
        hits = sum(k_ in alone(G[n_], r["pred"]) for r in R_)
        put("iso-top", str(n_)); put("iso-mission", MNAME[k_ // 100])
        put("iso-phrase", f"all {len(R_)} modal settings" if hits == len(R_) else f"{hits} of the {len(R_)} modal settings")
    g10 = G[10]; cons = T[10]["labels"]
    vals = []
    for mi in range(5):
        s_ = g10["labels"] == mi
        vals.append(_nmi(g10["ranks"][s_], cons[s_]))
    put("top10-nmi-min", f"{min(vals):.3f}"); put("top10-nmi-max", f"{max(vals):.3f}")
    # how the consensus clusters divide the missions and the controllers (Section 5.1 and the discussion)
    def per_mission(n):
        y_, c_ = G[n]["labels"], np.asarray(T[n]["labels"])
        dom = [int(np.bincount(y_[c_ == c]).argmax()) for c in np.unique(c_)]
        return [dom.count(m) for m in range(5)]
    _pm = {n: per_mission(n) for n in sorted(T)}
    claim("XOR Aggregation is the only mission divided on Top-2, and on every pooled dataset it is divided into at least "
          "as many clusters as any other mission", _pm[2] == [1, 1, 1, 1, 2] and all(_pm[n][4] == max(_pm[n]) for n in _pm if n > 1))
    claim("the modal count is the same on Top-8, Top-9 and Top-10", T[8]["C"] == T[9]["C"] == T[10]["C"])
    _key = G[10]["labels"] * 100 + G[10]["ranks"]; _c10 = np.asarray(T[10]["labels"])
    put("top10-ctrl-split", str(sum(len(np.unique(_c10[_key == k])) > 1 for k in np.unique(_key))))
    put("top10-ctrl", str(len(np.unique(_key))))
    _multi = sum(len(np.unique(_key[_c10 == c])) > 1 for c in np.unique(_c10))
    claim("on Top-10 most consensus clusters gather the runs of several controllers", _multi > len(np.unique(_c10)) / 2)
    _pmin = min(r["purity"] for n in G for r in G[n]["results"] if r["knn"] in KNNS and r["t"] in TS)
    put("maps-purity-min", f"{_pmin:.3f}")
    claim("on every resolution map the purity stays close to one at every setting (at least 0.98)", _pmin >= 0.98)
    put("top10-nmi-min-mission", MNAME[int(np.argmin(vals))]); put("top10-nmi-max-mission", MNAME[int(np.argmax(vals))])

# ------------------------------------------------------------------ fusion designs (Top-1)
DS = load("designs_top1.pkl")
if DS is not None:
    names = [("balanced", "Balanced views, SNF across scales (the method)"),
             ("none", "No diffusion"),
             ("flux_rerank", "Flux re-ranked before balancing"),
             ("snf12", "One SNF, 12 views, one per moment and scale"),
             ("snf20", "One SNF, 20 field views"),
             ("hierarchical", "Three stacked SNF stages")]
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.1cm}ccccc@{}}", r"\toprule",
             r"Fusion & Modal & Perfect & $C\le2$ & \makecell{Median\\purity} & Monotone\\", r"\midrule"]
    for key, label in names:
        R = [r for r in DS if r["design"] == key]
        if not R:
            continue
        C = np.array([r["C"] for r in R]); Pu = np.array([r["purity"] for r in R]); A = np.array([r["ari"] for r in R])
        v, c = np.unique(C, return_counts=True)
        g = {(r["knn"], r["t"]): r["C"] for r in R}
        mono = sum(all(g[(k, TS[i + 1])] <= g[(k, TS[i])] for i in range(len(TS) - 1)) for k in KNNS)
        plateau = f"$C^\\star={int(v[c.argmax()])}$ ({c.max()})" if key != "none" else f"$C={int(C[0])}$"
        lines.append(("\\methodrow " if key == "balanced" else "") +
                     f"{label} & {plateau} & {int(((Pu > .9999) & (A > .9999)).sum())} & {int((C <= 2).sum())} & "
                     f"{np.median(Pu):.3f} & {mono}/9\\\\")
        put(f"design-{key.replace('_', '')}-perfect", str(int(((Pu > .9999) & (A > .9999)).sum())))
        put(f"design-{key.replace('_', '')}-mono", str(mono))
        put(f"design-{key.replace('_', '')}-medpur", f"{np.median(Pu):.3f}")
        put(f"design-{key.replace('_', '')}-plateau", str(int(c.max())))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "designs.tex").write_text(full_width_shaded("\n".join(lines)) + "\n")
    _dk = [k.replace("_", "") for k, _ in names if k != "none" and f"design-{k.replace('_', '')}-plateau" in nums]
    _p = lambda k: int(nums[f"design-{k}-plateau"]); _m = lambda k: float(nums[f"design-{k}-medpur"])
    _pf = lambda k: int(nums[f"design-{k}-perfect"])
    _top, _low = ["balanced", "fluxrerank", "snf12"], ["snf20", "hierarchical"]
    if all(k in _dk for k in _top + _low):
        put("design-low-plateau-max", str(max(_p(k) for k in _low)))
        put("design-low-medpur-max", f3(max(_m(k) for k in _low)))
        _order = sorted(_dk, key=_p, reverse=True)
        put("design-top-medpur-min", f3(min(_m(k) for k in _top)))
        claim("among the architectures that diffuse, the two balanced fusions and the single SNF over twelve source views have the "
              "three largest plateaus, each with a median purity above 0.99, and the two others keep a median purity below 0.9",
              set(_order[:3]) == set(_top) and min(_m(k) for k in _top) > 0.99 and max(_m(k) for k in _low) < 0.9)
    DSS = load("designs_top1_summary.pkl")
    if DSS is not None and "clusters_per_mission" in DSS.get("none", {}):
        put("design-none-C", str(int(DS[[r["design"] for r in DS].index("none")]["C"])))
        claim("without diffusion affinity propagation splits every mission of Top-1, one controller each, into several groups",
              min(DSS["none"]["clusters_per_mission"]) >= 2)

# ------------------------------------------------------------------ variants of the fusion, one change at a time (all datasets)
FX = {}
for _f in sorted(P.glob("fusion_explore*.pkl")):                    # every job run on the current distances
    _src = _f.parent / (_f.name + ".source")
    if _src.exists() and _src.read_text().strip() == _RS.dist_source():
        FX.update(load(_f.name))
VARIANTS = [(None, [("the method", "method")]),
            ("affinity of each view", [("percentile ranks alone", "aff-percentile"), ("probit ranks", "aff-probit"),
                                       ("self-tuning kernel", "aff-selftuning")]),
            ("fields at one scale", [("equal weight per field", "fields-equal"), ("spectral gap, per field", "fields-confidence"),
                                     ("spectral gap, shared within each moment", "fields-sourceconf"),
                                     ("weighted geometric mean", "fields-geomean"), ("maximum", "fields-max")]),
            ("scales", [("mean, no diffusion", "scales-mean"), ("diffusion of the mean", "scales-meandiff"),
                        ("SNF across the twenty views", "scales-snf20")]),
            ("clustering of the consensus", [("AP, minimum preference", "clu-ap-minpref"), ("spectral, eigengap", "clu-spectral"),
                                             ("Louvain modularity", "clu-louvain"), ("average linkage, largest gap", "clu-agglo"),
                                             ("HDBSCAN", "clu-hdbscan")])]
_fx_keys = [k for _, rows in VARIANTS for _, k in rows]
if FX and all((k, t) in FX for k in _fx_keys for t in range(1, 11)):
    _mean = lambda k, m: float(np.mean([FX[(k, t)][m] for t in range(1, 11)]))
    _wrong = lambda k: int(sum(FX[(k, t)]["misassigned"] for t in range(1, 11)))
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Without labels} & \multicolumn{3}{c}{Against the missions}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
             r"Variant & Modal & Stability & Purity & ARI & Misassigned\\", r"\midrule"]
    for group, rows in VARIANTS:
        if group:
            lines += [r"\addlinespace", r"\multicolumn{6}{@{}l}{\emph{" + group[0].upper() + group[1:] + r"}}\\"]
        for label, k in rows:
            pad = r"\quad " if group else ""
            lines.append(("" if group else "\\methodrow ") + f"{pad}{label[0].upper() + label[1:]} & {_mean(k, 'plateau'):.1f} & {_mean(k, 'stability'):.3f} & "
                         f"{_mean(k, 'purity'):.3f} & {_mean(k, 'ari'):.3f} & {_wrong(k)}\\\\")
            put(f"fx-{k.replace('-', '')}-wrong", str(_wrong(k)))
            put(f"fx-{k.replace('-', '')}-plateau", f"{_mean(k, 'plateau'):.1f}")
            put(f"fx-{k.replace('-', '')}-stability", f3(_mean(k, 'stability')))
            put(f"fx-{k.replace('-', '')}-purity", f3(_mean(k, 'purity')))
            put(f"fx-{k.replace('-', '')}-xs", str(int(sum(FX[(k, t)]["xor_in_shelter"] for t in range(1, 11)))))
            put(f"fx-{k.replace('-', '')}-ari", f3(_mean(k, 'ari')))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "variants.tex").write_text(full_width_shaded("\n".join(lines)) + "\n")
    _all = range(1, 11)
    for _k, _nm in (("clu-ap-minpref", "apmin"), ("clu-louvain", "louvain")):     # the two clusterings closest to the method
        put(f"fx-{_nm}-same", WORDS[sum(FX[(_k, t)]["C"] == FX[("method", t)]["C"] for t in _all)])
        put(f"fx-{_nm}-ari-min", f3(min(adjusted_rand_score(FX[(_k, t)]["labels"], FX[("method", t)]["labels"]) for t in _all)))
    claim("AP with the minimum preference finds the C* of the method on every dataset, at an ARI of at least 0.97 from it",
          nums["fx-apmin-same"] == "ten" and float(nums["fx-apmin-ari-min"]) >= 0.97)
    claim("Louvain modularity finds the C* of the method on most datasets, at an ARI of at least 0.85 from it",
          WORDS.index(nums["fx-louvain-same"]) > 5 and float(nums["fx-louvain-ari-min"]) >= 0.85)
    _coll = ("clu-spectral", "clu-agglo", "clu-hdbscan")
    put("fx-collapse-cmin", str(min(FX[(k, t)]["C"] for k in _coll for t in _all)))
    put("fx-collapse-cmax", str(max(FX[(k, t)]["C"] for k in _coll for t in _all)))
    put("fx-collapse-pmax", f3(max(FX[(k, t)]["purity"] for k in _coll for t in _all)))
    claim("the clusterings that choose the number of groups by an eigengap, a gap of the tree or density merge missions on "
          "every dataset (C* below five)", all(FX[(k, t)]["C"] < 5 for k in _coll for t in _all))
    put("fx-mean-cmin", str(min(FX[("scales-mean", t)]["C"] for t in _all)))
    put("fx-mean-cmax", str(max(FX[("scales-mean", t)]["C"] for t in _all)))
    claim("averaging the scale views without diffusion returns one partition at every setting (plateau of 99 on every "
          "dataset), and its number of groups grows with the number of controllers",
          all(FX[("scales-mean", t)]["plateau"] == 99 for t in _all)
          and all(FX[("scales-mean", a)]["C"] < FX[("scales-mean", b)]["C"] for a, b in zip(_all, _all[1:])))
    _hi = [k for k in _fx_keys if round(_mean(k, "ari"), 3) > round(_mean("method", "ari"), 3)]    # higher in the table
    claim("every variant with a higher mean ARI than the method in the table finds fewer groups on average and misassigns "
          "more runs", bool(_hi) and all(_mean(k, "C") < _mean("method", "C") and _wrong(k) > _wrong("method") for k in _hi))
    claim("the plateau does not rank the variants by their errors: the maximum over the fields and the SNF over twenty views "
          "have larger mean plateaus than the method and misassign more runs",
          all(_mean(k, "plateau") > _mean("method", "plateau") and _wrong(k) > _wrong("method") for k in ("fields-max", "scales-snf20")))
    claim("most runs that percentile ranks alone misassign are XOR Aggregation runs placed with Shelter",
          2 * int(nums["fx-affpercentile-xs"]) > int(nums["fx-affpercentile-wrong"]))
    if T is not None:
        claim("the exploration reproduces the taxonomy of the method on every dataset (same partition)",
              all(adjusted_rand_score(FX[("method", t)]["labels"], T[t]["labels"]) > 1 - 1e-12 for t in range(1, 11)))

# ------------------------------------------------------------------ the elementary flows (synthetic, Figure fig:flows)
sys.path.insert(0, str(ROOT / "analysis"))
import illustrations as _IL  # noqa: E402
_flat = []
for _P, _off in ((_IL._swarm_radial(inward=True), "psi"), (_IL._swarm_radial(inward=False), "psi"), (_IL._swarm_mill(), "phi")):
    _f = _IL._fields(_P, 0.07, 1.0, 1.0, n=140)
    _flat.append(np.nanmax(np.abs(_f[_off])) / max(np.nanmax(np.abs(_f["phi"])), np.nanmax(np.abs(_f["psi"]))))
put("flows-flat-pct", f"{int(np.ceil(100 * max(_flat)))}")
claim("in the elementary flows, the potential a motion does not produce stays below 2 % of the scale of its column",
      max(_flat) < 0.02)

# ------------------------------------------------------------------ the coupling example (synthetic, Figure fig:potentials)
# the same two groups, together or apart, through the grid-free pipeline: how much each diagram changes, relative to
# its size; the diagram of J is empty when the sink and the rotation center coincide
_cd = _IL.coupling_distances()
put("cpl-phi-pct", f"{round(100 * _cd['phi']['relative'])}")
put("cpl-psi-pct", f"{round(100 * _cd['psi']['relative'])}")
claim("in the coupling example, the diagrams of the two potentials change by less than half their size, while the "
      "diagram of the coupling is empty when the sink and the rotation center coincide and present when they are apart",
      max(_cd["phi"]["relative"], _cd["psi"]["relative"]) < 0.5
      and _cd["J"]["size_together"] < 1e-3 * _cd["J"]["size_apart"] and _cd["J"]["size_apart"] > 0)

# ------------------------------------------------------------------ the partition of Legarda Herranz et al. on Top-1
# Their Fig. 4c (Research Square preprint, doi 10.21203/rs.3.rs-8672197/v1, transcribed by hand): for each of their five
# clusters C1..C5 (columns), the percentage of its runs from each mission (rows For, DiG, XOR, Hom, Shl); every column
# sums to 100. Their text calls the diagonal the accuracy of each mission. With 50 runs per mission (the 250 runs of
# Top-1), the counts behind these percentages can be recovered: every table of counts whose percentages lie within one
# point of the figure is enumerated, column by column.
LEGARDA_PCT = np.array([[100, 4, 0, 0, 0],
                        [0, 96, 0, 0, 0],
                        [0, 0, 87, 15, 2],
                        [0, 0, 6, 80, 0],
                        [0, 0, 7, 5, 98]])
LEGARDA_ORDER = [1, 0, 4, 2, 3]          # their rows For, DiG, XOR, Hom, Shl as our missions (0 D-Gate ... 4 XOR)


def tables_from_percentages(pct, per_row=50, tol=1.0):
    """Every table of counts (rows summing to per_row) whose column percentages lie within tol of pct."""
    from itertools import product
    n_rows, n_cols = pct.shape
    candidates = []
    for j in range(n_cols):
        rows = np.flatnonzero(pct[:, j])
        col = []
        for n in range(1, per_row * n_rows + 1):
            ranges = [[c for c in range(0, per_row + 1) if abs(100 * c / n - pct[i, j]) <= tol] for i in rows]
            for counts in product(*ranges):
                if sum(counts) == n:
                    v = np.zeros(n_rows, int); v[rows] = counts
                    col.append(v)
        candidates.append(col)
    out = []
    last = np.array([np.flatnonzero(pct[i]).max() for i in range(n_rows)])     # the last column a row can fill

    def extend(j, partial, cols):                    # columns in turn: no row beyond per_row, each row full at its last column
        if j == n_cols:
            out.append(np.stack(cols, 1))
            return
        for v in candidates[j]:
            s = partial + v
            if (s <= per_row).all() and (s[last == j] == per_row).all():
                extend(j + 1, s, cols + [v])
    extend(0, np.zeros(n_rows, int), [])
    return out


_lt = tables_from_percentages(LEGARDA_PCT)
put("legarda-n-tables", str(len(_lt)))
claim("a single table of counts reproduces the confusion matrix of Legarda Herranz et al. to within one point",
      len(_lt) == 1)
if len(_lt) == 1:
    _M = _lt[0]
    _y = np.repeat(np.arange(5), _M.sum(1)); _p = np.concatenate([np.repeat(np.arange(5), _M[i]) for i in range(5)])
    LEGARDA = dict(purity=_M.max(0).sum() / _M.sum(), ari=adjusted_rand_score(_y, _p), wrong=int(_M.sum() - _M.max(0).sum()))
    put("legarda-purity", f3(LEGARDA["purity"])); put("legarda-ari", f3(LEGARDA["ari"])); put("legarda-wrong", str(LEGARDA["wrong"]))
    _names = ["Directional Gate", "Foraging", "Homing", "Shelter", "XOR Aggregation"]
    _diag = {LEGARDA_ORDER[i]: int(LEGARDA_PCT[i, i]) for i in range(5)}
    put("legarda-pct-phrase", ", ".join(f"${_diag[m]}\\%$ for {_names[m]}" for m in sorted(_diag, key=lambda m: -_diag[m])[:-1])
        + f" and ${_diag[min(_diag, key=_diag.get)]}\\%$ for {_names[min(_diag, key=_diag.get)]}")

# ------------------------------------------------------------------ baselines
BE = load("baselines_eval.pkl")
if BE is not None and G is not None and T is not None:
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    from classification.fusion import plateau_consensus

    def read(results, y):
        pc = plateau_consensus(results, len(y))
        cm = contingency_matrix(y, pc["labels"])
        h, c, _ = homogeneity_completeness_v_measure(y, pc["labels"])
        free = np.array([r["purity"] > 1 - 1e-9 for r in results])           # settings in which no run is misassigned
        on_plateau = np.array([r["C"] == pc["C"] for r in results])
        return dict(C=pc["C"], plateau=pc["n_plateau"], purity=cm.max(0).sum() / cm.sum(),
                    ari=adjusted_rand_score(y, pc["labels"]), hom=h, comp=c, stab=pc["stability"], cm=cm,
                    wrong=int(cm.sum() - cm.max(0).sum()), free=int(free.sum()), free_plateau=int((free & on_plateau).sum()))
    rows = []
    for top in (1, 10):
        y = G[top]["labels"]
        rows.append((top, "ours", read(G[top]["results"], y)))
        for name in ("fieldstat", "crocker", "order"):
            if (name, top) in BE:
                rows.append((top, name, read(BE[(name, top)], y)))
    CB = load("summary_combined.pkl")
    if CB is not None:
        for top in (1, 10):
            if top in CB:
                rows.append((top, "combined", read(CB[top], G[top]["labels"])))
    names = {"ours": "Persistence diagrams", "fieldstat": "Value distributions", "combined": "Diagrams and values",
             "crocker": "CROCKER curves of positions", "order": "Order parameters"}
    for t_, n_, v_ in rows:                                  # the macros of the text, one per description and dataset
        k_ = f"top{t_}"
        put(f"base-{n_}-{k_}-free", str(v_["free"])); put(f"base-{n_}-{k_}-free-plateau", str(v_["free_plateau"]))
        put(f"base-{n_}-{k_}-purity", f"{v_['purity']:.3f}"); put(f"base-{n_}-{k_}-ari", f"{v_['ari']:.3f}")
        put(f"base-{n_}-{k_}-hom", f"{v_['hom']:.3f}"); put(f"base-{n_}-{k_}-comp", f"{v_['comp']:.3f}")
        put(f"base-{n_}-{k_}-plateau", str(v_["plateau"])); put(f"base-{n_}-{k_}-cstar", str(v_["C"]))
        put(f"base-{n_}-{k_}-stab", f"{v_['stab']:.3f}"); put(f"base-{n_}-{k_}-wrong", str(v_["wrong"]))
    _bv = {(t_, n_): v_ for t_, n_, v_ in rows}
    if (1, "order") in _bv and (1, "crocker") in _bv:
        claim("on Top-1 no setting is error-free with the order parameters or the CROCKER curves",
              _bv[(1, "order")]["free"] == 0 and _bv[(1, "crocker")]["free"] == 0)
    if (10, "combined") in _bv:
        claim("on Top-10 the consensus of the diagrams and values together misassigns no run, and some of its settings are error-free",
              _bv[(10, "combined")]["wrong"] == 0 and _bv[(10, "combined")]["free"] > 0)
    single = {}
    if ("single", 1) in BE and ("single", 10) in BE:
        single = {top_: BE[("single", top_)] for top_ in (1, 10)}
        for key in ("scale1", "scale4", "nodiff"):
            a_ = single[1][key]
            put(f"base-{key}-top1-c", str(a_["C"])); put(f"base-{key}-top1-ari", f"{a_['ari']:.3f}")
            put(f"base-{key}-top1-purity", f"{a_['purity']:.3f}")
        claim("without diffusion, the finest scale alone and the coarsest scale alone each reach a purity of at least 0.97 on Top-1",
              BE[("single", 1)]["scale1"]["purity"] >= 0.97 and BE[("single", 1)]["scale4"]["purity"] >= 0.97)
        _s = [BE[("single", 1)][k] for k in ("scale1", "scale4", "nodiff")]          # the three rows of the table
        put("base-single-top1-cmin", str(min(r["C"] for r in _s))); put("base-single-top1-cmax", str(max(r["C"] for r in _s)))
        put("base-single-top1-arimax", f"{max(r['ari'] for r in _s):.3f}")
        claim("without diffusion Top-1 splits into more than five groups, each nearly pure (purity >= 0.95)",
              min(r["C"] for r in _s) > 5 and all(r["purity"] >= 0.95 for r in _s))
    # two panels with the same columns, Top-1 above Top-10: the consensus of the modal region (C*, purity, homogeneity,
    # completeness, ARI) and, beside it, the settings of the sweep in which no run is misassigned
    by = {(t_, n_): v_ for t_, n_, v_ in rows}
    groups = (("Our fields; same fusion, clustering and modal consensus", ("ours", "fieldstat", "combined")),
              ("Other descriptions; same fusion, clustering and modal consensus", ("crocker", "order")))
    def baseline_table(panels):
      lines = [r"\begin{tabular}{@{}l@{\hspace{0.8em}}cccccc@{}}", r"\toprule",
               r"Representation & $\Cstar$ & Purity & Hom. & Compl. & ARI & \makecell{Error-free\\(modal)}\\",
               r"\midrule"]
      for top_, panel in panels:
          if top_ == 10:
              lines.append(r"\addlinespace[1em]")
          lines.append(r"\multicolumn{7}{@{}l}{" + panel + r"}\\")
          for title, members in groups:
              lines += [r"\addlinespace[0.35em]", r"\multicolumn{7}{@{}l}{\emph{" + title + r"}}\\"]
              for n_ in members:
                  if (top_, n_) not in by:
                      continue
                  v_ = by[(top_, n_)]
                  lines.append(("\\methodrow " if n_ == "ours" else "") + f"\\quad {names[n_]} & {v_['C']} ({v_['plateau']}) & {v_['purity']:.3f} & "
                               f"{v_['hom']:.3f} & {v_['comp']:.3f} & {v_['ari']:.3f} & {v_['free']} ({v_['free_plateau']})\\\\")
          if single:
              lines += [r"\addlinespace[0.35em]", r"\multicolumn{7}{@{}l}{\emph{Our fields without diffusion; one partition}}\\"]
              for key, lab_ in (("scale1", "Finest scale only"), ("scale4", "Coarsest scale only"), ("nodiff", "Scales averaged")):
                  a_ = single[top_][key]
                  lines.append(f"\\quad {lab_} & $C={a_['C']}$ & {a_['purity']:.3f} & {a_['hom']:.3f} & {a_['comp']:.3f} & {a_['ari']:.3f} & --\\\\"
                               if "hom" in a_ else f"\\quad {lab_} & $C={a_['C']}$ & {a_['purity']:.3f} & -- & -- & {a_['ari']:.3f} & --\\\\")
          if top_ == 1:
              lines += [r"\addlinespace[0.35em]", r"\multicolumn{7}{@{}l}{\emph{Reported result, different protocol}}\\",
                        r"\quad Potential persistence$^{*}$ & 5 (fixed) & " + f"{LEGARDA['purity']:.3f}" + r" & -- & -- & "
                        + f"{LEGARDA['ari']:.3f}" + r" & --\\"]
      lines += [r"\bottomrule", r"\end{tabular}"]
      return full_width_shaded("\n".join(lines)) + "\n"
    # the paper shows the Top-1 panel, which its text discusses; the Supplementary Information both panels
    (TAB / "baselines_top1.tex").write_text(baseline_table(((1, r"\textbf{Top-1}, 250 runs"),)))
    (TAB / "baselines.tex").write_text(baseline_table(((1, r"\textbf{Top-1}, 250 runs"), (10, r"\textbf{Top-10}, 2500 runs"))))
    _b = {(t_, n_): r for t_, n_, r in rows}
    if all(k in _b for k in ((1, "ours"), (1, "order"), (10, "order"), (1, "crocker"))):
        def misplaced(cm):
            """(mission of the run, mission that dominates its cluster) for every misassigned run"""
            dom = cm.argmax(0)
            return {(m, int(dom[c])) for m in range(cm.shape[0]) for c in range(cm.shape[1]) if cm[m, c] and m != dom[c]}
        _o = _b[(1, "order")]["cm"]
        claim("on Top-1 order parameters split XOR Aggregation in two and mix Directional Gate with Foraging, "
              "on a smaller plateau than ours", (_o[4] >= 10).sum() >= 2 and {(0, 1), (1, 0)} <= misplaced(_o)
              and _b[(1, "order")]["plateau"] < _b[(1, "ours")]["plateau"])
        claim("order parameters scatter Top-10 into more clusters, with a lower purity than ours",
              _b[(10, "order")]["C"] > _b[(10, "ours")]["C"] and _b[(10, "order")]["purity"] < _b[(10, "ours")]["purity"])
        _cr = _b[(1, "crocker")]["cm"]; _crd = _cr.argmax(0)
        put("base-crocker-top1-fdg", str(int(sum(_cr[1, c] for c in range(_cr.shape[1]) if _crd[c] == 0))))   # Foraging runs with Directional Gate
        claim("on Top-1 CROCKER curves put every Foraging run in a cluster dominated by Directional Gate",
              int(sum(_cr[1, c] for c in range(_cr.shape[1]) if _crd[c] == 0)) == int(_cr[1].sum()))
        _c10 = _b[(10, "crocker")]["cm"]
        claim("on Top-10 CROCKER curves put most runs of Directional Gate and of Foraging into one cluster",
              any(_c10[0, c] > 0.5 * _c10[0].sum() and _c10[1, c] > 0.5 * _c10[1].sum() for c in range(_c10.shape[1])))

# ------------------------------------------------------------------ persistence against value distributions
RC = load("summary_comparison.pkl")
if RC is not None:
    tops = [t for t in sorted(RC) if "values" in RC[t]]
    P_, V_ = {t: RC[t]["persistence"] for t in tops}, {t: RC[t]["values"] for t in tops}
    put("rc-n", str(len(tops)))
    put("rc-purity-min", f3(min(min(P_[t]["purity"], V_[t]["purity"]) for t in tops)))
    _dari = {t: V_[t]["ari"] - P_[t]["ari"] for t in tops}
    put("rc-values-ari-higher", str(sum(d > 1e-9 for d in _dari.values())))
    put("rc-pers-ari-higher", str(sum(d < -1e-9 for d in _dari.values())))
    _tv, _tp = max(tops, key=_dari.get), min(tops, key=_dari.get)     # the largest lead of each summary
    put("rc-values-best-top", str(_tv)); put("rc-values-best-ari", f3(V_[_tv]["ari"])); put("rc-pers-at-values-best", f3(P_[_tv]["ari"]))
    put("rc-pers-best-top", str(_tp)); put("rc-pers-best-ari", f3(P_[_tp]["ari"])); put("rc-values-at-pers-best", f3(V_[_tp]["ari"]))
    put("rc-pers-plateau-larger", str(sum(P_[t]["plateau"] > V_[t]["plateau"] for t in tops)))
    put("rc-values-plateau-larger", str(sum(V_[t]["plateau"] > P_[t]["plateau"] for t in tops)))
    claim("summarised by their values, the fields recover the five missions of Top-1 without exception (ARI 1)",
          V_[1]["ari"] > 0.999999)
    claim("neither summary dominates: each has the higher ARI on at least one dataset",
          any(d > 1e-9 for d in _dari.values()) and any(d < -1e-9 for d in _dari.values()))
    claim("where the value distributions have the higher ARI, they find as many groups as the persistence diagrams or fewer, "
          "and their homogeneity is never higher", all(V_[t]["C"] <= P_[t]["C"] for t in tops if V_[t]["ari"] > P_[t]["ari"] + 1e-9)
          and all(V_[t]["hom"] <= P_[t]["hom"] + 1e-9 for t in tops))
    CB = load("summary_combined.pkl")
    if CB is not None and all(t in CB for t in tops) and G is not None:
        from classification.fusion import plateau_consensus
        _cb = {}
        for t in tops:
            y = G[t]["labels"]; pc = plateau_consensus(CB[t], len(y)); cm = contingency_matrix(y, pc["labels"])
            _cb[t] = dict(purity=cm.max(0).sum() / cm.sum(), ari=adjusted_rand_score(y, pc["labels"]), plateau=pc["n_plateau"])
        put("comb-ari-below", str(sum(_cb[t]["ari"] < max(P_[t]["ari"], V_[t]["ari"]) - 1e-9 for t in tops)))
        put("comb-plateau-larger", str(sum(_cb[t]["plateau"] > P_[t]["plateau"] for t in tops)))
        claim("combining both summaries keeps or raises the purity on every dataset",
              all(_cb[t]["purity"] >= max(P_[t]["purity"], V_[t]["purity"]) - 1e-9 for t in tops))
        claim("the two summaries together have an ARI below that of the better summary on some datasets, and a plateau never "
              "larger than that of the persistence diagrams", any(_cb[t]["ari"] < max(P_[t]["ari"], V_[t]["ari"]) - 1e-9 for t in tops)
              and not any(_cb[t]["plateau"] > P_[t]["plateau"] for t in tops))

# ------------------------------------------------------------------ attribution: the distance split over the critical points
AT = load("attribution.pkl")
if AT is not None:
    pc = lambda x: f"{100 * x:.0f}\\,\\%"
    ck = AT["check"]
    claim("the differentiable chain reproduces the stored fields to their storage precision and the diagrams and distance exactly",
          all(ck[k] < 1e-6 for k in ck if k.startswith("field_")) and all(ck[k] < 1e-12 for k in ck if k.startswith(("diagrams_", "value_match")) and k.endswith(("_vs_stored", "_A", "_B")))
          and all(ck[k] for k in ck if k.endswith(("sizes_equal", "unchanged"))) and ck["persistence_distance_vs_stored"] < 1e-12
          and ck["value_distance_vs_stored"] < 1e-12)
    claim("the split of the distance over the points of the diagrams sums to the distance", ck["split_sums_to_distance"] < 1e-12)
    # the share of the distance carried by each seed of the Directional Gate exemplar, for each summary
    gP, gV, T_ = np.asarray(AT["persistence_share_A"]), np.asarray(AT["values_share_A"]), AT["types_A"]
    ext = np.where(np.isin(T_, ("min", "max")))[0]
    top2 = ext[np.argsort(gP[ext])[-2:]]
    put("attr-nseeds-total", str(len(gP)))
    put("attr-nseeds-pers", str(int((gP > 0).sum())))
    put("attr-nseeds-val", str(int((gV > 0).sum())))
    put("attr-pers-critical", pc(AT["persistence_shares_A"]["critical"]))
    put("attr-val-ordinary", pc(1 - AT["values_shares_A"]["critical"]))
    put("attr-grad-critical", pc(AT["persistence_grad_shares_A"]["critical"]))
    put("attr-centres-exemplar", pc(gP[top2].sum() / gP.sum()))
    put("attr-share-saddles", pc(gP[T_ == "saddle"].sum() / gP.sum()))
    claim("on the exemplar, the two extrema with the largest shares are a maximum and a minimum, and they carry the two "
          "largest shares of all the seeds", sorted(str(t) for t in T_[top2]) == ["max", "min"]
          and set(np.argsort(gP)[-2:].tolist()) == set(top2.tolist()))
    claim("all of the topological share, and all of the topological gradient, fall on critical points",
          AT["persistence_shares_A"]["critical"] > 1 - 1e-9 and AT["persistence_grad_shares_A"]["critical"] > 1 - 1e-9)
    # the envelope theorem holds at the critical points inside the domain: the share of the distance carried
    # there, on the complex of the Directional Gate exemplar rebuilt as the attribution builds it
    import attribution as _attr
    from collections import Counter as _Counter
    _cx = _attr.pairing(np.asarray(AT["seeds_A"]), np.asarray(AT["F_A"], dtype=np.float64), AT["sigma"])[2]
    _ntri = _Counter(tuple(sorted(e)) for tr in _cx["tris"] for e in ((tr[0], tr[1]), (tr[1], tr[2]), (tr[0], tr[2])))
    _bd = np.zeros(len(gP), bool)
    for _ea, _eb in _cx["edges"]:
        if _ntri[tuple(sorted((_ea, _eb)))] < 2:                 # an edge of fewer than two triangles: the boundary
            _bd[_ea] = _bd[_eb] = True
    _crit = np.isin(T_, ("min", "max", "saddle"))
    put("attr-interior-share", pc(gP[_crit & ~_bd].sum() / gP.sum()))
    claim("the topological share falls on critical points only, inside the domain or on its boundary",
          gP[~_crit].sum() <= 1e-9 * gP.sum())
    _tr = load("top1_tracks.pkl")
    if _tr is not None:
        _e = int(AT["run_A"][0])
        _pos, _vel = (np.asarray(_tr["tracks"][_e][k_], dtype=np.float64) for k_ in ("pos", "vel"))
        _Pf = np.concatenate([_pos, _pos[-1:] + _attr.DT * _vel[-1:]], 0)          # positions at T + 1 instants
        _v = np.diff(_Pf, axis=0) / _attr.DT
        _u = _v / np.maximum(np.linalg.norm(_v, axis=2), 1e-12)[..., None]
        _turn = np.zeros(_Pf.shape[:2]); _turn[1:-1] = np.linalg.norm(_u[1:] - _u[:-1], axis=2)   # change of direction
        _Mt = _attr.robot_map(_Pf, _turn)
        _c = (np.arange(_Mt.shape[0]) + 0.5) * _attr.ARENA / _Mt.shape[0]
        _at = lambda M: np.array([_c[np.unravel_index(np.argmax(M), M.shape)[1]], _c[np.unravel_index(np.argmax(M), M.shape)[0]]])
        claim("carried back to the robots, both splits are largest where the robots change direction most (within two sigma)",
              all(np.linalg.norm(_at(AT[f"{r_}_robot_map_A"]) - _at(_Mt)) < 2 * AT["sigma"] for r_ in ("persistence", "values")))
    put("attr-robot-half-pers", pc(AT["persistence_robot_area_half_A"]))
    put("attr-robot-half-val", pc(AT["values_robot_area_half_A"]))
    put("attr-time-top-pers", pc(AT["persistence_top_decile_time_A"]))
    st = AT["stability"][0.002]
    put("attr-stab-field-change", pc(st["field_change"]))
    put("attr-stab-pers", f"{st['persistence_same']:.2f}")
    put("attr-stab-val", f"{st['values_same']:.2f}")
    s5 = AT["stability"][0.005]
    put("attr-stab5-pers", f"{s5['persistence_same']:.2f}"); put("attr-stab5-val", f"{s5['values_same']:.2f}")
    claim("at every noise tested the topological attribution keeps more of its weight on the same seeds than the value "
          "attribution", all(r["persistence_same"] > r["values_same"] for r in AT["stability"].values()))
    rp = AT["reproducibility"]
    put("attr-runs", str(len(rp["runs"])))
    put("attr-mean-half-pers", pc(rp["persistence"]["area_half"]))
    put("attr-mean-half-val", pc(rp["values"]["area_half"]))
    put("attr-overlap-pers", f"{rp['persistence']['mean_overlap']:.2f}")
    put("attr-overlap-val", f"{rp['values']['mean_overlap']:.2f}")
    claim("two topological maps overlap less on average than two value maps",
          rp["persistence"]["mean_overlap"] < rp["values"]["mean_overlap"])
    ce = rp["centres"]
    put("attr-centres-runs", str(ce["n_max_min"]))
    put("attr-centres-share", pc(ce["share_median"]))
    put("attr-centres-spread", f"{100 * np.mean(list(ce['spread_m'].values())):.0f}\\,cm")
    put("attr-centres-phrase", "in every one of the" if ce["n_max_min"] == len(rp["runs"]) else f"in {ce['n_max_min']} of the")
    claim("over the runs of the controller, the two extrema with the largest shares are a maximum and a minimum in most runs",
          2 * ce["n_max_min"] > len(rp["runs"]))

# ------------------------------------------------------------------ the split on every pair of missions (Top-1 exemplars)
APS = load("attribution_pairs.pkl")
if APS is not None and AT is not None:
    _rows = [(a_, b_, s_) for (a_, b_), s_ in sorted(APS.items())]
    _S = [r_[2] for r_ in _rows]
    put("attrp-n", WORDS[len(_S)])
    put("attrp-val-critical-max", pc(max(s_["values_critical"] for s_ in _S)))
    put("attrp-top2-min", pc(min(s_["top2_share"] for s_ in _S))); put("attrp-top2-max", pc(max(s_["top2_share"] for s_ in _S)))
    put("attrp-stab-pers-min", f"{min(s_['stab2'][0] for s_ in _S):.2f}"); put("attrp-stab-pers-max", f"{max(s_['stab2'][0] for s_ in _S):.2f}")
    put("attrp-stab-val-min", f"{min(s_['stab2'][1] for s_ in _S):.2f}"); put("attrp-stab-val-max", f"{max(s_['stab2'][1] for s_ in _S):.2f}")
    put("attrp-recur-min", str(min(s_["recur"] for s_ in _S)))
    put("attrp-pairs-50", WORDS[sum(s_["recur"] == s_["runs"] for s_ in _S)])
    _pds = {(a_, b_): load(f"attribution_pairs/{a_}_{b_}_{s_['field']}.pkl") for a_, b_, s_ in _rows}
    if all(v_ is not None for v_ in _pds.values()):
        _fc_h = [round(100 * _pds[(a_, b_)]["stability"][0.002]["field_change"])
                 for a_, b_, s_ in _rows if a_ == 2 and s_["field"] == "K"]
        put("attrp-homing-k-change", f"{max(_fc_h)}\\,\\%")
        put("attrp-sx-blur", f"{100 * _pds[(3, 4)]['stability'][0.002]['persistence_blur']:.0f}\\,\\%")
        _exc = [(a_, b_, [tuple(map(str, r_["types"])) for r_ in _pds[(a_, b_)]["reproducibility"]["centres"]["per_run"]
                          if tuple(map(str, r_["types"])) != ("max", "min")])
                for a_, b_, s_ in _rows if s_["recur"] < s_["runs"]]
        claim("only Directional Gate against XOR Aggregation drops below 50/50, because one run places its two largest "
              "weights on two density maxima",
              len(_exc) == 1 and _exc[0][:2] == (0, 4) and _exc[0][2] == [("max", "max")])
    claim("on every pair of missions all of the topological split lies on critical points",
          all(s_["critical"] > 1 - 1e-9 for s_ in _S))
    claim("on every pair the two extrema with the largest shares are a maximum and a minimum",
          all(s_["top2_kinds"] == ("max", "min") for s_ in _S))
    claim("on every pair they are again a maximum and a minimum in all the runs of the first controller but at most one",
          all(s_["recur"] >= s_["runs"] - 1 for s_ in _S))
    claim("on every pair the topological split keeps more of its weight on the same seeds under a 2 mm noise than the value split",
          all(s_["stab2"][0] > s_["stab2"][1] for s_ in _S))
    _pot = [s_ for s_ in _S if s_["field"] in ("phi", "psi", "J")]; _sca = [s_ for s_ in _S if s_["field"] in ("rho", "K")]
    claim("the pairs split on a potential or on the coupling give a larger share to their two largest extrema than those "
          "split on the density or the kinetic intensity",
          min(s_["top2_share"] for s_ in _pot) > max(s_["top2_share"] for s_ in _sca))
    claim("the split of the pairs on the density or the kinetic intensity spreads over more critical points than on the "
          "potentials", min(s_["n_seeds"] for s_ in _sca) > max(s_["n_seeds"] for s_ in _S if s_["field"] in ("phi", "psi")))
    _pn = {"rho": r"$\rho_\sigma$", "K": r"$K_\sigma$", "phi": r"$\varphi_\sigma$", "psi": r"$\psi_\sigma$",
           "J": r"$J_\sigma$"}
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Two largest extrema} & Kept under $2$\,mm\\",
             r"\cmidrule(lr){3-4}",
             r"Pair & Field & Share & Same in runs & Persistence / values\\", r"\midrule"]
    for first in sorted({r_[0] for r_ in _rows}):                 # rows grouped under the first mission of the pair
        lines += ([r"\addlinespace"] if first else []) + [r"\multicolumn{5}{@{}l}{\emph{" + MNAME[first] + r" against}}\\"]
        for a_, b_, s_ in (r_ for r_ in _rows if r_[0] == first):
            lines.append(f"\\quad {MNAME[b_]} & {_pn[s_['field']]} & {100 * s_['top2_share']:.0f}\\,\\% & "
                         f"{s_['recur']}/{s_['runs']} & {s_['stab2'][0]:.2f} / {s_['stab2'][1]:.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "attribution_pairs.tex").write_text(full_width("\n".join(lines)) + "\n")

# ------------------------------------------------------------------ XOR runs that percentile ranks alone would place with Shelter
MA = load("merge_analysis.pkl")
if MA is not None:
    for top in (3, 4):
        m = MA[top]; dyn = m["dynamics"]; fl = m["fields"]
        put(f"merge-top{top}-n", str(m["n"]))
        for g, key in (("absorbed", "abs"), ("kept", "kept"), ("shelter", "shel")):
            put(f"merge-top{top}-settle-{key}", f"{dyn[g]['settle']:.0f}")
            put(f"merge-top{top}-speed-{key}", f"{dyn[g]['speed']:.3f}")
        for f in ("phi", "psi"):
            put(f"merge-top{top}-{f}-shel", f"{fl[f][('absorbed', 'shelter')]:.2f}")
            put(f"merge-top{top}-{f}-kept", f"{fl[f][('absorbed', 'kept')]:.2f}")
        if "neighbours" in m:                          # what the diffusion links them to, under each kind of rank
            nb = m["neighbours"]
            put(f"merge-top{top}-nn-pct", f"{100 * nb['percentile']['shelter']:.0f}")
            put(f"merge-top{top}-nn-mut", f"{100 * nb['mutual']['shelter']:.0f}")
            claim(f"Top-{top}: the mutual ranks remove more than half of the Shelter runs from their nearest neighbours",
                  nb["mutual"]["shelter"] < nb["percentile"]["shelter"] / 2)
        ctrl = m["controllers"]
        best, worst = min(ctrl), max(ctrl)
        put(f"merge-top{top}-best", f"{ctrl[best][0]} of the {ctrl[best][1]}")
        put(f"merge-top{top}-worst", f"{ctrl[worst][0]} of {ctrl[worst][1]}")
        if top == 3 and G is not None:
            claim("Top-3: with percentile ranks alone, every XOR Aggregation run lies in the Shelter cluster",
                  m["n"] == int((np.asarray(G[3]["labels"]) == 4).sum()))
        if T is not None:
            claim(f"Top-{top}: the method places no XOR Aggregation run with Shelter", int(nums[f"top{top}-xor-in-shelter"]) == 0)
        if top == 4:
            claim("Top-4: every XOR Aggregation controller supplies runs to the Shelter cluster",
                  len(ctrl) == 4 and all(v_[0] > 0 for v_ in ctrl.values()))
            claim("Top-4: the XOR runs in the Shelter cluster settle later than the other XOR runs",
                  dyn["absorbed"]["settle"] > dyn["kept"]["settle"])
            claim("Top-4: the best XOR controller supplies more of the runs in the Shelter cluster than any other",
                  ctrl[best][0] == max(v_[0] for v_ in ctrl.values()))
    d4, f4 = MA[4]["dynamics"], MA[4]["fields"]
    claim("Top-4: Shelter robots keep moving for the whole run (median settling time at the end of the 180 s)",
          d4["shelter"]["settle"] >= 179)
    _sa, _sk = d4["absorbed"]["settle"], d4["kept"]["settle"]
    put("merge-top4-settle-phrase",
        f"In the median run more than a tenth of their robots still move at the end, as in Shelter, whereas the other "
        f"XOR runs come to rest after {_sk:.0f}\\,s" if _sa >= 179 else
        f"The median time after which fewer than a tenth of them still move is {_sa:.0f}\\,s, against {_sk:.0f}\\,s "
        f"for the other XOR runs and {d4['shelter']['settle']:.0f}\\,s, the whole run, for Shelter")
    claim("Top-4: they move about twice as fast as the other XOR runs (ratio 1.5 to 2.5)",
          1.5 <= d4["absorbed"]["speed"] / d4["kept"]["speed"] <= 2.5)
    claim("Top-4: averaged over the recording, every field of these runs is closer to Shelter than to the other XOR runs",
          all(f4[f][("absorbed", "shelter")] < f4[f][("absorbed", "kept")] for f in ("rho", "K", "phi", "psi", "J")))
    claim("Top-4: their divergence potential is closer to that of Shelter than to that of the other XOR runs",
          f4["phi"][("absorbed", "shelter")] < f4["phi"][("absorbed", "kept")])

# ------------------------------------------------------------------ field ablation
AB = load("ablation.pkl")
if AB is not None:
    from scipy.stats import wilcoxon
    def vec(combo, metric="purity"):
        out = []
        for top in range(1, 11):
            if (combo, top) not in AB:
                return None
            out += [r[metric] for r in sorted(AB[(combo, top)], key=lambda r: (r["knn"], r["t"]))]
        return np.array(out)
    full = ("rho", "K", "phi", "psi", "J")
    put("abl-n-combos", WORDS[len({c for c, _ in AB})])
    for combo, key in ((("rho",), "rho"), (("K",), "K"), (("phi",), "phi"), (("psi",), "psi"), (("J",), "J"),
                       (("rho", "K"), "rhoK"), (("phi", "psi"), "phipsi"), (("phi", "psi", "J"), "phipsiJ"),
                       (("K", "phi", "psi", "J"), "KphipsiJ"), (("rho", "K", "phi", "psi"), "noJ"), (full, "all")):
        v = vec(combo)
        if v is not None:
            put(f"abl-{key}-median", f"{np.median(v):.3f}"); put(f"abl-{key}-q1", f"{np.quantile(v, .25):.3f}")
            put(f"abl-{key}-errfree", str(int((v > 1 - 1e-9).sum())))
            va = vec(combo, "ari")
            put(f"abl-{key}-ari-median", f"{np.median(va):.3f}")
    _st = {c: (float(np.median(vec(c))), int((vec(c) > 1 - 1e-9).sum())) for c in {c for c, _ in AB} if vec(c) is not None}
    if full in _st:
        claim("the five fields together have the highest median purity and the most settings with no misassigned run of all "
              "combinations", all(_st[c][0] <= _st[full][0] and _st[c][1] < _st[full][1] for c in _st if c != full))
    # what the text reads in the ablation: the best and the worst single field, and the tests below
    _single = {f: float(np.median(vec((f,)))) for f in ("rho", "K", "phi", "psi", "J") if vec((f,)) is not None}
    if len(_single) == 5:
        _name = {"rho": "the density", "K": "the kinetic intensity", "phi": "the divergence potential",
                 "psi": "the vorticity potential", "J": "the coupling field"}
        _b, _w = max(_single, key=_single.get), min(_single, key=_single.get)
        put("abl-best-field", _name[_b]); put("abl-best-median", f3(_single[_b]))
        put("abl-worst-field", _name[_w]); put("abl-worst-median", f3(_single[_w]))
        claim("each field alone separates the missions at most settings (median purity at least 0.8)", _single[_w] >= 0.8)
    for a, b, key in ((("phi", "psi"), ("phi", "psi", "J"), "addJ"), (("rho", "K", "phi", "psi"), full, "addJall"),
                      (("K", "phi", "psi", "J"), full, "addrho"),
                      (("rho", "K"), full, "addflux")):
        va, vb = vec(a), vec(b)
        if va is None or vb is None:
            continue
        d = vb - va
        wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        stat, pval = wilcoxon(vb, va, zero_method="wilcox")
        put(f"abl-{key}-wins", str(wins)); put(f"abl-{key}-losses", str(losses)); put(f"abl-{key}-n", str(len(d)))
        exp = int(np.floor(np.log10(max(pval, 1e-300))))
        put(f"abl-{key}-p", f"10^{{{exp + 1}}}" if pval > 0 else "10^{-300}")
        claim(f"adding {key[3:]} raises the purity in more settings than it lowers (Wilcoxon p < 1e-10)",
              wins > losses and pval < 1e-10)

# ------------------------------------------------------------------ the taxonomy built on the plateau, per combination
ABC = load("ablation_consensus.pkl")
if ABC is not None and G is not None:
    def _wrong(combo):
        """misassigned runs over the ten datasets, and how many are XOR Aggregation runs in a Shelter cluster"""
        w = xs = 0
        for t in range(1, 11):
            y, lab_ = G[t]["labels"], ABC[(combo, t)]["labels"]
            cm = contingency_matrix(y, lab_); dom = cm.argmax(0)
            w += int(cm.sum() - cm.max(0).sum()); xs += int(sum(1 for i in range(len(y)) if y[i] == 4 and dom[lab_[i]] == 3))
        return w, xs
    _five, _noj = ("rho", "K", "phi", "psi", "J"), ("rho", "K", "phi", "psi")
    if all((c, t) in ABC for c in (_five, _noj) for t in range(1, 11)):
        (w5, x5), (w4, x4) = _wrong(_five), _wrong(_noj)
        put("abl-cons-all-wrong", str(w5)); put("abl-cons-noJ-wrong", str(w4)); put("abl-cons-all-xs", str(x5))
        _gf = {t: int(sum(c_[0, k] for k in range(c_.shape[1]) if c_.argmax(0)[k] == 1))      # Directional Gate in Foraging
               for t in range(1, 11) for c_ in [contingency_matrix(G[t]["labels"], ABC[(_five, t)]["labels"])]}
        put("abl-cons-all-gf", str(sum(_gf.values()))); put("abl-cons-gf-from", str(min(t for t in _gf if _gf[t])))
        _pm = lambda c: float(np.mean([ABC[(c, t)]["plateau"] for t in range(1, 11)]))
        _sm = lambda c: float(np.mean([ABC[(c, t)]["stability"] for t in range(1, 11)]))
        put("abl-plateau-all", f"{_pm(_five):.1f}"); put("abl-plateau-noJ", f"{_pm(_noj):.1f}")
        put("abl-stab-all", f3(_sm(_five))); put("abl-stab-noJ", f3(_sm(_noj)))
        claim("with the coupling the taxonomy misassigns fewer runs over the ten datasets", w5 < w4)
        claim("with the five fields the errors are chiefly Directional Gate runs placed with Foraging (over half), the others "
              "XOR Aggregation runs placed with Shelter",
              2 * sum(_gf.values()) > w5 and sum(_gf.values()) + x5 == w5)
        _gf4 = sum(int(sum(c_[0, k] for k in range(c_.shape[1]) if c_.argmax(0)[k] == 1))      # the same without the coupling
                   for t in range(1, 11) for c_ in [contingency_matrix(G[t]["labels"], ABC[(_noj, t)]["labels"])])
        put("abl-cons-noJ-gf", str(_gf4))
        claim("as many Directional Gate runs lie with Foraging with and without the coupling", _gf4 == sum(_gf.values()))
        others = [c for c, t in ABC if t == 1 and c not in (_five, _noj)]
        claim("every combination that leaves out a field other than the coupling misassigns more runs than the five fields",
              all(_wrong(c)[0] > w5 for c in others))
        # each dataset contains the previous one, so that the behaviours found on one are still present in the next
        _c5 = [ABC[(_five, t)]["C"] for t in range(1, 11)]; _c4 = [ABC[(_noj, t)]["C"] for t in range(1, 11)]
        _drops = [(t + 2, _c4[t], _c4[t + 1]) for t in range(9) if _c4[t + 1] < _c4[t]]
        claim("with the coupling C* never falls from one dataset to the next, and without it it does",
              all(b_ >= a_ for a_, b_ in zip(_c5, _c5[1:])) and bool(_drops))
        if _drops:
            put("abl-noJ-drops", " and ".join(f"from {a_} to {b_} on Top-{t_}" for t_, a_, b_ in _drops))
            _in = min(float((cm_.max(1) / cm_.sum(1)).min()) for t_, _, _ in _drops
                      for cm_ in [contingency_matrix(ABC[(_five, t_)]["labels"], ABC[(_noj, t_)]["labels"])])
            claim("where C* falls without the coupling, every cluster of the five fields lies, for at least 95 % of its runs, in one "
                  "cluster of the four fields", _in >= 0.95)
        if AB is not None:
            import figlib as _FL
            _iv = lambda c, m: _FL.median_interval([np.array([r[m] for r in AB[(c, t)]]) for t in range(1, 11)])
            # the settings in which no run is misassigned, beside the median purity: the median hides them
            for _nm, _c in (("all", _five), ("noJ", _noj)):
                put(f"abl-{_nm}-errfree", str(sum(int(sum(r["purity"] > 1 - 1e-9 for r in AB[(_c, t)])) for t in range(1, 11))))
                put(f"abl-top1-{_nm}-errfree", str(int(sum(r["purity"] > 1 - 1e-9 for r in AB[(_c, 1)]))))
                put(f"abl-{_nm}-median", f3(float(np.median(np.concatenate([[r["purity"] for r in AB[(_c, t)]] for t in range(1, 11)])))))
            _ef = lambda c: sum(int(sum(r["purity"] > 1 - 1e-9 for r in AB[(c, t)])) for t in range(1, 11))
            claim("with and without the coupling the notches of the median ARI overlap",
                  max(_iv(_five, "ari")[0], _iv(_noj, "ari")[0]) <= min(_iv(_five, "ari")[1], _iv(_noj, "ari")[1]))
            claim("the five fields have more settings with no misassigned run than the four fields without the coupling, over the "
                  "ten datasets and on Top-1", _ef(_five) > _ef(_noj)
                  and sum(r["purity"] > 1 - 1e-9 for r in AB[(_five, 1)]) > sum(r["purity"] > 1 - 1e-9 for r in AB[(_noj, 1)]))
            claim("the coupling alone has a lower median purity than the vorticity potential alone, with notches apart",
                  _iv(("J",), "purity")[1] < _iv(("psi",), "purity")[0])

# ------------------------------------------------------------------ does J help the value distributions too?
VA = load("value_ablation.pkl"); BE_ = load("baselines_eval.pkl")
if VA is not None and BE_ is not None:
    from scipy.stats import wilcoxon
    full = ("rho", "K", "phi", "psi", "J")

    def vvec(combo, metric="purity"):
        """the value distributions over the 99 settings of the ten datasets (the five fields: baselines_eval)"""
        out = []
        for top in range(1, 11):
            res = BE_.get(("fieldstat", top)) if combo == full else VA.get((combo, top))
            if res is None:
                return None
            out += [r[metric] for r in sorted(res, key=lambda r: (r["knn"], r["t"]))]
        return np.array(out)
    for a_, b_, key in ((("phi", "psi"), ("phi", "psi", "J"), "addJ"), (("rho", "K", "phi", "psi"), full, "addJall")):
        for metric in ("purity", "ari"):
            va, vb = vvec(a_, metric), vvec(b_, metric)
            if va is None or vb is None:
                continue
            d = vb - va
            tag = f"vab-{key}" + ("" if metric == "purity" else "-ari")
            put(f"{tag}-wins", str(int((d > 1e-12).sum()))); put(f"{tag}-losses", str(int((d < -1e-12).sum())))
            put(f"{tag}-n", str(len(d)))
            pval = wilcoxon(vb, va, zero_method="wilcox").pvalue if (abs(d) > 1e-12).any() else 1.0
            put(f"{tag}-p", f"10^{{{int(np.floor(np.log10(max(pval, 1e-300)))) + 1}}}")
            if metric == "purity":
                _w, _l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
                claim(f"value distributions: J {'helps the two potentials' if key == 'addJ' else 'does not help the five fields together'}",
                      _w > _l if key == "addJ" else _w < _l)

# ------------------------------------------------------------------ parameter study (label-free)
def _study(name):
    f = ROOT / "results" / name / "summary.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def runs_moved(ari, n=250, c=5):
    """How many runs must change cluster, among c balanced clusters of n runs, to give this ARI."""
    y = np.repeat(np.arange(c), n // c)
    return min(range(n // c), key=lambda k: abs(adjusted_rand_score(y, np.r_[np.ones(k, int), y[k:]]) - ari))


RESOLUTION_SETTINGS = ("cover", "frac", "n_bis", "n_dirs", "window", "n_jacobi")     # rows of Table study, resolution
STM, STE, STF = _study("parameter_study"), _study("parameter_study_ext"), _study("parameter_study_final")
if STF is not None:
    caps = {int(k[1][3:]): r for k, r in STF.items() if k[0] == "ladder" and str(k[1]).startswith("cap")}
    if caps:
        sys.path[:0] = [str(ROOT), str(ROOT / "src")]
        from pipeline import config as _C
        kc = int(round(float(_C.SIGMA_MAX_FACTOR)))                     # the cap of the method
        put("cap-k", str(kc))
        # how far the resolution settings alone move the plateau, on the domain of the method (seeds within sigma/2)
        _res = [r["plateau"] for st in (STF, STE or {}) for k, r in st.items() if k[0] in RESOLUTION_SETTINGS]
        lo, hi = min(_res), max(_res)
        put("res-plateau-min", str(lo)); put("res-plateau-max", str(hi))
        ks = sorted(caps)
        level = [k for k in ks if lo <= caps[k]["plateau"] <= hi]       # the caps at the level of the largest plateau
        claim("the plateau grows with k up to the cap of the method, and from it on stays within the range that the resolution "
              "settings alone produce", all(caps[a]["plateau"] < caps[b]["plateau"] for a, b in zip(ks, ks[1:]) if b <= kc)
              and level == [k for k in ks if k >= kc])
        claim("the cap of the method is the first of this level, the one that reaches the largest width, and its most stable",
              bool(level) and kc == level[0] and all(caps[kc]["stability"] >= caps[k]["stability"] for k in level))
        put("cap-plateau-min", str(min(r["plateau"] for r in caps.values())))
        put("cap-plateau-max", str(max(r["plateau"] for r in caps.values())))
        put("cap-best-stability", f3(caps[kc]["stability"]))
        put("cap-level-stab-other", f3(max(caps[k]["stability"] for k in level if k != kc)))
        for k, r in caps.items():
            put(f"cap{k}-plateau", str(r["plateau"])); put(f"cap{k}-stability", f3(r["stability"]))
        same = sorted(k for k, r in caps.items() if r["ari_consensus"] > 0.999999)
        if same:
            ds = 2.35 * float(lad.get("max_factor", 10.0)) * float(lad["ladder"][-1]) if lad is not None else None   # D_s
            put("cap-same-kmin", str(min(same))); put("cap-same-kmax", str(max(same)))
            if ds is not None:
                put("merge-width-cm", f"{100 * ds / 2.35:.0f}")
                put("cap-same-cmmin", f"{100 * ds / (2.35 * max(same)):.0f}"); put("cap-same-cmmax", f"{100 * ds / (2.35 * min(same)):.0f}")
        _apart = [k for k in ks if k not in same]
        if _apart:
            put("cap-apart-k", str(_apart[0])); put("cap-apart-runs", WORDS[runs_moved(caps[_apart[0]]["ari_consensus"])])
        claim("every cap tested gives the same taxonomy, except the widest, which moves one run",
              _apart == ks[:1] and runs_moved(caps[ks[0]]["ari_consensus"]) == 1)
if STF is not None:
    _c = [r for k, r in STF.items() if k[0] == "cover" and abs(float(k[1]) - 0.5) < 1e-9 and "spearman_min" in r]
    if _c:
        put("cover-third-spearman", f3(_c[0]["spearman_min"]))
if STE is not None:
    unc = {k[1]: r for k, r in STE.items() if k[0] == "ladder" and str(k[1]).startswith("uncapped")}
    if unc:
        put("uncapped-plateau-min", str(min(r["plateau"] for r in unc.values())))
        put("uncapped-plateau-max", str(max(r["plateau"] for r in unc.values())))
        put("uncapped-stability", f3(float(np.median([r["stability"] for r in unc.values()]))))
        put("uncapped-mmax", str(max(int(k[len("uncapped"):]) for k in unc)))
        put("uncapped-same", "yes" if all(r["ari_consensus"] > 0.999999 for r in unc.values()) else "no")
        claim("with no cap at all, and with more scales between the same bounds, the taxonomy is the same",
              all(r["ari_consensus"] > 0.999999 for r in unc.values()))
    _m = {k[1]: r for k, r in (STM or {}).items() if k[0] == "ladder" and str(k[1]).startswith("M")}
    if _m:
        claim("for three to six scales up to the same cap the taxonomy is the same", all(r["ari_consensus"] > 0.999999 for r in _m.values()))
# compact table of the study (Appendix B): one row per setting
def _rows_of(st, name, others=None):
    """{value: row} of one setting in a summary (others: the fixed settings of the extension plans)."""
    out = {}
    for key, r in (st or {}).items():
        if key[0] != name:
            continue
        if others is not None and (len(key) < 3 or dict(key[2]) != others):
            continue
        out[key[1]] = r
    return out


def _cell(rows, value, field="spearman_min"):
    r = rows.get(value)
    return f3(r[field]) if r is not None and field in r else "--"


_study_ari, _study_C = [], set()      # every taxonomy compared in the table, for the text


def _worst(rows, field="ari_consensus"):
    v = [r[field] for r in rows.values() if field in r]
    _study_ari.extend(v); _study_C.update(r["C"] for r in rows.values() if "C" in r)
    return f3(min(v)) if v else "--"


_unc_main = [r for k, r in (STM or {}).items() if k[0] == "ladder" and str(k[1]).startswith("uncapped")]
if _unc_main:                            # no cap at all against the cap of the method, on the fixed subset of positions
    put("uncapped-runs-moved", WORDS[runs_moved(min(r["ari_consensus"] for r in _unc_main))])
    claim("with no cap at all, the taxonomy moves one run from that of the capped ladder on the same subset of positions",
          runs_moved(min(r["ari_consensus"] for r in _unc_main)) == 1)
_study_rows = []
_half = {"cover": 0.5}
cov = _rows_of(STF, "cover", {})
if cov:                                  # sigma/3 to sigma in the final plan, 2 sigma in the extension
    cov_all = {**cov, **{("e", k): v for k, v in _rows_of(STE, "cover", {}).items()}}
    _study_rows.append(("covering radius of the seeds", "resolution", r"$2\sigma$ to $\sigma/3$", r"$\sigma/2$",
                        _cell(cov, 0.5), _worst(cov_all)))
nco = _rows_of(STM, "n_core")
if nco:
    _study_rows.append(("fixed subset of positions", "resolution", "50 to 4000", "--", _cell(nco, 300), _worst(nco)))
for name, label, rng, method, ref_main in (
        ("frac", "flat triangles per field", r"1 to 100\,\%", f"{100 * CFG0.FRAC_LOW_GRAD:g}\\,\\%", CFG0.FRAC_LOW_GRAD),
        ("n_bis", "bisection steps", "4 to 20", str(CFG0.N_BIS), CFG0.N_BIS),
        ("window", "window of $\\rho$ and $K$", r"$3\sigma$ to $8\sigma$", f"${CFG0.WINDOW:g}\\sigma$", CFG0.WINDOW),
        ("n_dirs", "sliced-Wasserstein directions", "10 to 400", str(CFG0.N_DIRS), CFG0.N_DIRS)):
    fin, main = _rows_of(STF, name, _half), _rows_of(STM, name)
    if fin or main:
        dist = _cell(fin, ref_main) if ref_main in fin else _cell(main, ref_main)
        _study_rows.append((label, "resolution", rng, method, dist, _worst({**main, **{("f", k): v for k, v in fin.items()}})))
nj_fin, nj_main = _rows_of(STE, "n_jacobi", _half), _rows_of(STM, "n_jacobi")
if nj_fin or nj_main:
    _study_rows.append(("seeds on $J_\\sigma=0$", "resolution", "0 to all", str(CFG0.N_JACOBI),
                        _cell(nj_fin, CFG0.N_JACOBI) if CFG0.N_JACOBI in nj_fin else _cell(nj_main, CFG0.N_JACOBI),
                        _worst({**nj_main, **{("e", k): v for k, v in nj_fin.items()}})))
dd = _rows_of(STM, "dedup")
if dd:
    _study_rows.append(("duplicate threshold", "resolution", r"$10^{-2}$ to $10^{-5}\sigma$", r"$10^{-3}\sigma$", _cell(dd, CFG0.DEDUP_TOL), _worst(dd)))
st_ = _rows_of(STM, "start")
if st_:
    _study_rows.append(("first point of the subset of positions", "design", "3 choices", "first position", "{:.3f}".format(min(r["spearman_min"] for r in st_.values())), _worst(st_)))
al = _rows_of(STM, "alpha")
if al:
    _study_rows.append(("alpha radius", "design", r"$2\sigma$ to $4\sigma$", f"${CFG0.C_ALPHA:g}\\sigma$", "{:.3f}".format(min(r["spearman_min"] for r in al.values())), _worst(al)))
ms = {k: v for k, v in _rows_of(STM, "ladder").items() if str(k).startswith("M")}
unc = {k: v for k, v in _rows_of(STE, "ladder", {}).items() if str(k).startswith("uncapped")}
if ms or unc:
    _study_rows.append(("number of scales $M$", "design", "3 to 6 (and 4 to 40 uncapped)", str(CFG0.M_SCALES), "--", _worst({**ms, **unc})))
cp = {k: v for k, v in _rows_of(STF, "ladder", _half).items() if str(k).startswith("cap")}
if cp:
    _study_rows.append(("cap $k$ of the ladder", "design", "3 to 20", f"{CFG0.SIGMA_MAX_FACTOR:g}", "--", _worst(cp)))
for name, label, rng, method in (("damping", "AP damping", "0.8 to 0.95", "0.9"), ("conv", "AP convergence window", "50 to 400", "100"),
                                 ("max_iter", "AP ceiling", "1000, 5000", "1000"), ("seed", "AP tie-breaking noise", "5 draws", "0")):
    rows = _rows_of(STM, name)
    if name == "damping":
        rows = {k: v for k, v in rows.items() if 0.8 <= k <= 0.95}
    if rows:
        _study_rows.append((label, "resolution", rng, method, "--", _worst(rows)))
if _study_ari:
    put("study-ari-min", f3(min(_study_ari))); put("study-runs-moved", WORDS[runs_moved(min(_study_ari))])
    claim("no value tested in the study of the settings changes the number of behaviours", len(_study_C) == 1)
if _study_rows:
    _rag = r">{\raggedright\arraybackslash}p"            # columns of text wrap within their width
    lines = [r"\begin{tabular}{@{}" + _rag + "{3.8cm}" + _rag + "{2.6cm}" + _rag + "{2.1cm}cc@{}}", r"\toprule",
             r"Setting & Tested & Method & Distances & Taxonomy\\", r"\midrule"]
    for kind, head in (("resolution", "Resolution settings"), ("design", "Design choices")):
        lines += ([r"\addlinespace"] if kind == "design" else []) + [r"\multicolumn{5}{@{}l}{\emph{" + head + r"}}\\"]
        lines += [ITEM + " & ".join(r[:1] + r[2:]) + r"\\" for r in _study_rows if r[1] == kind]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "study.tex").write_text(full_width("\n".join(lines)) + "\n")

# table of the study: every setting varied alone, compared label-free with its reference
_rows = []
for src, st in (("main", STM), ("extension", STE), ("final", STF)):
    if st is None:
        continue
    for key, r in st.items():
        name, value = key[0], key[1]
        _rows.append((src, name, value, r))
if _rows:
    lines = [r"\begin{tabular}{@{}llrcccc@{}}", r"\toprule",
             r"Setting & Value & Reference & Spearman (min) & ARI & Modal & Stability\\", r"\midrule"]
    for src, name, value, r in _rows:
        sp = f3(r["spearman_min"]) if "spearman_min" in r else "--"
        lines.append(f"{name.replace('_', ' ')} & {value} & {r['reference'].replace('_', ' ')} & {sp} & {f3(r['ari_consensus'])} & "
                     f"{r['plateau']}/99 & {f3(r['stability'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "parameter_study.tex").write_text(full_width("\n".join(lines)) + "\n")

# ------------------------------------------------------------------ constants of the method, and the seeds
from fractions import Fraction  # noqa: E402
put("n-jacobi", str(CFG0.N_JACOBI)); put("n-bis", str(CFG0.N_BIS)); put("n-dirs", str(CFG0.N_DIRS))
put("frac-pct", f"{100 * CFG0.FRAC_LOW_GRAD:g}"); put("window", f"{CFG0.WINDOW:g}"); put("c-alpha", f"{CFG0.C_ALPHA:g}")
put("m-scales", str(CFG0.M_SCALES)); put("n-coreset", str(CFG0.N_CORESET))
if CFG0.COVER is not None:
    fr = Fraction(CFG0.COVER).limit_denominator(16)
    put("cover", f"{fr.numerator}/{fr.denominator}" if fr.denominator != 1 else str(fr.numerator))
    put("cover-sigma", ("\\sigma_m" if fr.numerator == 1 else f"{fr.numerator}\\sigma_m") + (f"/{fr.denominator}" if fr.denominator != 1 else ""))
_s1 = ROOT / "results" / "epuck" / "stage1_alpha_fields"
if (_s1 / "MANIFEST.json").exists():
    _n = {}
    for f in _s1.glob("traj*_s*.pkl"):
        si = int(f.stem.split("_s")[-1])
        _n.setdefault(si, []).append(len(pickle.load(open(f, "rb"))["seeds"]))
    for si, v in sorted(_n.items()):
        put(f"seeds-s{si + 1}", f"{np.mean(v):.0f}")
        put(f"seeds-s{si + 1}-max", str(int(np.max(v))))

# ------------------------------------------------------------------ settings of the chain (Appendix B)
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from pipeline import config as CFG  # noqa: E402
import inspect  # noqa: E402
from classification import fusion as _F  # noqa: E402
_ap = {k: v.default for k, v in inspect.signature(_F.sweep).parameters.items() if k in ("damping", "convergence_iter", "max_iter")}


def _sci(x):
    """x in scientific notation, as math content (no dollar signs)."""
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return f"10^{{{e}}}" if abs(m - 1) < 1e-12 else f"{m:g}\\times10^{{{e}}}"


cap = "$\\sigma_{\\max}=D_s/" + f"{2.35 * CFG.SIGMA_MAX_FACTOR:g}" + "$"
inst = "every instant of every run" if CFG.LADDER_INSTANTS is None else f"${CFG.LADDER_INSTANTS}$ instants per run"
core = (f"${CFG.N_CORESET}$ agent positions" if CFG.COVER is None else
        f"no agent position farther than ${CFG.COVER:g}\\sigma_m$ (at most ${CFG.N_CORESET_MAX}$ points)")
rows = [
    ("Footprint", "evaluation window of $\\rho_\\sigma$ and $K_\\sigma$", f"${CFG.WINDOW:g}\\sigma$ (\\cref{{prop:tail}})", "resolution"),
    ("Ladder", "number of scales $M$", f"${CFG.M_SCALES}$", "resolution"),
    ("", "floor $\\epsilon_\\sigma$", f"${CFG.EPS_SIGMA:g}$\\,m, the tracking precision", "design"),
    ("", "cap", cap, "design"),
    ("", "estimation of $d_{\\min}$, $D_s$", inst, "design"),
    ("Seeds", "subset of the agent positions", core, "resolution"),
    ("", "near-critical seeds", f"${100 * CFG.FRAC_LOW_GRAD:g}\\%$ flattest triangles per field", "resolution"),
    ("", "coupling skeleton", f"$\\le{CFG.N_JACOBI}$ points, ${CFG.N_BIS}$ bisection steps", "resolution"),
    ("", "duplicates", f"one seed per cell of side ${_sci(CFG.DEDUP_TOL)}\\,\\sigma$", "resolution"),
    ("Complex", "alpha complex truncation", f"squared radius $({CFG.C_ALPHA:g}\\sigma_m)^2$", "design"),
    ("Persistence", "filtration; diagrams", "lower-star, extended; six", "design"),
    ("Distance", "sliced Wasserstein", f"${CFG.N_DIRS}$ distinct directions in $[-\\pi/2,\\pi/2)$", "resolution"),
    ("", "order", "one (absolute differences), summed over the six diagrams", "design"),
    ("Fusion", "ranks; weights", "mutual, then percentile; $\\tfrac13\\rho$, $\\tfrac13K$, $\\tfrac19\\varphi$, $\\tfrac19\\psi$, $\\tfrac19J$", "design"),
    ("", "SNF", f"across the ${CFG.M_SCALES}$ scale views, $(\\knn,t)$ swept over ${len(_RS.KNNS) * len(_RS.TS)}$ settings", "design"),
    ("AP", "preference", r"median of the entries of $S^\star$ (diagonal set to the largest)", "design"),
    ("", "damping; convergence", f"${_ap['damping']:g}$; ${_ap['convergence_iter']}$ unchanged iterations, $\\le{_ap['max_iter']}$", "resolution"),
    ("Taxonomy", "modal consensus", "co-association, average linkage cut at $\\Cstar$", "design"),
]
lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.9cm}>{\raggedright\arraybackslash}p{5.5cm}l@{}}", r"\toprule",
         r"Setting & Value & Kind\\", r"\midrule"]
for a, b, c, d in rows:                                  # one block per stage of the method
    if a:
        lines += ([r"\addlinespace"] if len(lines) > 4 else []) + [r"\multicolumn{3}{@{}l}{\emph{" + a + r"}}\\"]
    lines.append(f"{ITEM}{b} & {c} & {d}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "settings.tex").write_text(full_width("\n".join(lines)) + "\n")

# ------------------------------------------------------------------ the resolution map over a window that grows
# with the dataset: its stable states, their transitions and the law of the transition to the three families
# (analysis/stable_states.py --scaled --t-fixed; the states are named by stable_states.state_of)
_SWF = {n: load(f"stable_states_scaled_top{n}.pkl") for n in range(1, 11)}
if all(v is not None for v in _SWF.values()):
    sys.path.insert(0, str(ROOT / "analysis"))
    import stable_states as _SS
    from pipeline.config import BEHAVIORS as _BEH
    _G = load("grid.pkl")
    _same99 = all(np.array_equal(np.asarray(r["pred"]),
                                 np.asarray(next(q for q in _SWF[n]["results"] if (q["knn"], q["t"]) == (r["knn"], r["t"]))["pred"]))
                  for n in range(1, 11) for r in _G[n]["results"])
    claim("every map of the wide window returns the partition of the main grid at its 99 settings", _same99)
    put("sw-t-max", str(max(_SWF[1]["ts"]))); put("sw-knn-top1-max", str(max(_SWF[1]["knns"])))
    put("sw-knn-top10-max", str(max(_SWF[10]["knns"])))
    put("sw-n-settings", f"{len(_SS.on_map(_SWF[2])):,}".replace(",", "\\,"))
    put("sw-n-settings-top1", f"{len(_SS.on_map(_SWF[1])):,}".replace(",", "\\,"))
    _ST = {n: [(_r["knn"], _r["t"], _SS.state_of(_r["pred"], _SWF[n]["labels"], _BEH)) for _r in _SS.on_map(_SWF[n])]
           for n in range(1, 11)}
    _cnt = {n: {st: sum(x[2] == st for x in _ST[n]) for st in _SS.STATES} for n in range(1, 11)}
    _fmt = lambda v: f"{v:,}".replace(",", "\\,")
    put("sw-top1-missions", _fmt(_cnt[1]["missions"])); put("sw-top1-families", _fmt(_cnt[1]["families"]))
    put("sw-top1-shelter-xor", _fmt(_cnt[1]["shelter_xor"]))
    _m17 = [_cnt[n]["missions"] for n in range(1, 8)]
    put("sw-missions-min-1to7", _fmt(min(_m17))); put("sw-missions-max-1to7", _fmt(max(_m17)))
    for n in (8, 9, 10):
        put(f"sw-missions-top{n}", _fmt(_cnt[n]["missions"]))
    _fp = [_cnt[n]["families"] for n in range(2, 11)]
    put("sw-families-min-pooled", _fmt(min(_fp))); put("sw-families-max-pooled", _fmt(max(_fp)))
    put("sw-partial-top10", _fmt(_cnt[10]["partial"]))
    claim("every map passes through the five missions, Shelter + XOR and the three families",
          all(_cnt[n]["missions"] > 0 and _cnt[n]["shelter_xor"] > 0 and _cnt[n]["families"] > 0 for n in range(1, 11)))
    claim("the two-family state appears on Top-9 and Top-10 only",
          all((_cnt[n]["two_families"] > 0) == (n in (9, 10)) for n in range(1, 11)))
    put("sw-two-top9", _fmt(_cnt[9]["two_families"])); put("sw-two-top10", _fmt(_cnt[10]["two_families"]))
    _mg = {n: sum(_SS.state_of(r["pred"], _G[n]["labels"], _BEH) == "finer" for r in _G[n]["results"]) for n in range(2, 11)}
    put("sw-maingrid-finer-min", str(min(_mg.values()))); put("sw-maingrid-finer-max", str(max(_mg.values())))
    claim("on every pooled dataset most of the main grid reads a state finer than the missions", min(_mg.values()) > 49)
    # Top-1: purity 1, partitions equal to their neighbors (label-free), and the runs that cross before a merge
    _d1 = _SWF[1]; _y1 = np.asarray(_d1["labels"]); _by1 = {(r["knn"], r["t"]): np.asarray(r["pred"]) for r in _d1["results"]}
    _pure = sum(all(len(np.unique(_y1[p_ == c])) == 1 for c in np.unique(p_)) for p_ in (_by1[(k, t)] for k, t, _ in _ST[1]))
    put("sw-top1-pure", _fmt(_pure))

    def _same_partition(a, b):
        return adjusted_rand_score(a, b) == 1.0
    _kk, _tt = _d1["knns"], _d1["ts"]
    _stable = [all(_same_partition(_by1[(k, t)], _by1[nb]) for nb in ((_kk[i + 1], t) if i + 1 < len(_kk) else None,
                                                                         (k, _tt[j + 1]) if j + 1 < len(_tt) else None) if nb)
               for i, k in enumerate(_kk) for j, t in enumerate(_tt)]
    put("sw-top1-same-pct", f"{100 * np.mean(_stable):.0f}")
    _cross, _pairs = 0, set()
    for k, t, st in _ST[1]:
        p_ = _by1[(k, t)]
        main = {m: np.bincount(p_[_y1 == m]).argmax() for m in np.unique(_y1)}
        if len(set(main.values())) < 5:
            continue
        maj = {c: np.bincount(_y1[p_ == c], minlength=5).argmax() for c in np.unique(p_)}
        out_ = [i for i in range(len(_y1)) if p_[i] != main[_y1[i]] and maj[p_[i]] != _y1[i]]
        _cross = max(_cross, len(out_))
        _pairs |= {(_SS.ABBR[_BEH[_y1[i]]], _SS.ABBR[_BEH[maj[p_[i]]]]) for i in out_}
    put("sw-top1-cross-max", str(_cross))
    claim("on Top-1 the runs that cross before a merge are XOR runs with Shelter and D-Gate runs with Foraging",
          _pairs <= {("X", "S"), ("G", "F")} and _cross > 0)
    # Homing: shares a cluster with another mission only on Top-9 and Top-10, only with D-Gate and Foraging, at the
    # end of the window
    _H = _BEH.index("homing") if isinstance(_BEH, list) else [i for i, b in enumerate(_BEH) if b == "homing"][0]
    _hom = {}
    for n in range(1, 11):
        yn = np.asarray(_SWF[n]["labels"]); kmax = max(_SWF[n]["knns"]); hits = []
        for r in _SS.on_map(_SWF[n]):
            p_ = np.asarray(r["pred"])
            for c in np.unique(p_[yn == _H]):
                sel = p_ == c
                ys, ct = np.unique(yn[sel], return_counts=True)
                part = {_SS.ABBR[_BEH[m]] for m, v in zip(ys, ct) if m != _H and v >= 0.1 * sel.sum()}
                if part and (ct[list(ys).index(_H)] >= 0.1 * sel.sum()):
                    hits.append((r["knn"] / kmax, frozenset(part)))
                    break
        _hom[n] = hits
    put("sw-homing-free-max", str(max(n for n in range(1, 11) if all(not _hom[m] for m in range(1, n + 1)))))
    put("sw-homing-top9", _fmt(len(_hom[9]))); put("sw-homing-top10", _fmt(len(_hom[10])))
    claim("Homing shares a cluster with another mission on Top-9 and Top-10 only, and then only with D-Gate and Foraging",
          all(not _hom[n] for n in range(1, 9)) and all(h[1] == {"G", "F"} for n in (9, 10) for h in _hom[n])
          and _hom[9] and _hom[10])
    put("sw-homing-knn-pct", f"{100 * min(h[0] for n in (9, 10) for h in _hom[n]):.0f}")
    # the law of the transition to the three families, fitted on all six regular pooled datasets (Top-2 to Top-7)
    _FIT = (2, 3, 4, 5, 6, 7)
    _tr = {n: _SS.transition(_SWF[n], _BEH) for n in range(1, 11)}
    _c, _a = _SS.power_law(np.vstack([_tr[n] for n in _FIT]))
    _per = {n: _SS.power_law(_tr[n]) for n in range(1, 11)}
    put("law-c", f"{_c:.0f}"); put("law-alpha", f"{_a:.2f}")
    put("law-fit-alpha-min", f"{min(_per[n][1] for n in _FIT):.2f}"); put("law-fit-alpha-max", f"{max(_per[n][1] for n in _FIT):.2f}")
    put("law-drift-alpha-min", f"{min(_per[n][1] for n in (8, 9, 10)):.2f}")
    put("law-drift-alpha-max", f"{max(_per[n][1] for n in (8, 9, 10)):.2f}")
    put("law-top1-c", f"{_per[1][0]:.0f}"); put("law-top1-alpha", f"{_per[1][1]:.2f}")
    claim("all regular pooled datasets (Top-2 to Top-7) stay close to a 0.55 exponent (0.53 to 0.59)",
          all(0.53 <= _per[n][1] <= 0.59 for n in _FIT))
    claim("Top-8, Top-9 and Top-10 have steeper exponents than every regular pooled dataset (Top-2 to Top-7)",
          all(_per[n][1] > max(_per[m][1] for m in _FIT) for n in (8, 9, 10)))
    claim("Top-1 follows a shallower law than the pooled datasets", _per[1][1] < min(_per[m][1] for m in range(2, 11)))
    # the drift: the weakest controllers sit with another mission
    _d10 = _SWF[10]; _y10 = np.asarray(_d10["labels"]); _r10 = np.asarray(_d10["ranks"])

    def _sits_with(m_name, rank):
        m = [i for i, b in enumerate(_BEH) if b == m_name][0]
        sel = (_y10 == m) & (_r10 == rank)
        w = {}
        for r in _SS.on_map(_d10):
            p_ = np.asarray(r["pred"])
            c = np.bincount(p_[sel]).argmax()
            maj = np.bincount(_y10[p_ == c], minlength=5).argmax()
            if maj != m:
                w[_SS.ABBR[_BEH[maj]]] = w.get(_SS.ABBR[_BEH[maj]], 0) + 1
        return w
    _s10, _x10 = _sits_with("shelter", 9), _sits_with("xor_aggregation", 9)
    put("drift-s10-nav", _fmt(_s10.get("G", 0) + _s10.get("F", 0))); put("drift-x10-shelter", _fmt(_x10.get("S", 0)))
    claim("the tenth Shelter controller sits with D-Gate or Foraging at most settings of Top-10",
          _s10.get("G", 0) + _s10.get("F", 0) > 0.5 * len(_SS.on_map(_d10)) and set(_s10) <= {"G", "F"})
    # its robots keep moving: the mean speed of each run (the order parameters of the baselines, m/s)
    _bl = load("baselines.pkl")
    if _bl is not None and _labels.exists():
        _z = np.load(_labels); _lab_all, _rk_all = _z["labels"], _z["ranks"]
        _spd = np.asarray(_bl["order"]["features"])[:, 2]
        _ix = {b: i for i, b in enumerate(_BEH)}
        _sh = [float(np.median(_spd[(_lab_all == _ix["shelter"]) & (_rk_all == r)])) for r in range(10)]
        _nav = float(np.median(_spd[np.isin(_lab_all, [_ix["directional_gate"], _ix["foraging"]])]))
        put("drift-s10-speed", f"{_sh[9]:.3f}"); put("drift-shelter-speed-min", f"{min(_sh[:9]):.3f}")
        put("drift-shelter-speed-max", f"{max(_sh[:9]):.3f}"); put("drift-nav-speed", f"{_nav:.3f}")
        claim("the robots of the tenth Shelter controller move faster than those of the nine others, slower than the traffic",
              max(_sh[:9]) < _sh[9] < _nav)

out = "% generated by analysis/make_numbers.py - do not edit\n" + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in sorted(nums.items()))
(MS / "numbers.tex").write_text(out)
print(f"{len(nums)} numbers written; tables: {sorted(p.name for p in TAB.glob('*.tex'))}")
claim("order parameters and CROCKER curves leave no setting of Top-1 without a misassigned run",
      nums.get("base-order-top1-free") == "0" and nums.get("base-crocker-top1-free") == "0")

for _text, _ok in CLAIMS:
    if not _ok:
        print(f"CLAIM FAILS: {_text}")
print(f"claims of the text checked on the results: {sum(ok for _, ok in CLAIMS)} of {len(CLAIMS)} hold")
