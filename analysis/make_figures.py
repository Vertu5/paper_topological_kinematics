#!/usr/bin/env python3
"""
make_figures.py - every figure of the paper, one function each.
Run `python analysis/make_figures.py all` or call the functions from the notebook.
All inputs are cached results; nothing here recomputes distances.

Author: Olivier Vertu Ndinga Oba
"""
import functools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as L  # noqa: E402
from sklearn.metrics.cluster import contingency_matrix  # noqa: E402

KNNS = [3, 5, 7, 9, 11, 13, 15, 20, 25]
TS = [1, 2, 3, 5, 7, 9, 10, 12, 15, 20, 25]
PURE_SEQ = L.LinearSegmentedColormap.from_list("pure_seq", ["#e7f0fb", "#bed7f4", "#7db3ea", "#3b84d6", "#16509c"])


def _grid():
    return L.load_pickle(L.PAPER / "grid.pkl")


def _mats(results, knns=KNNS, ts=TS):
    C = np.full((len(ts), len(knns)), np.nan); P = C.copy(); A = C.copy()
    for r in results:
        if r["knn"] in knns and r["t"] in ts:
            i, j = ts.index(r["t"]), knns.index(r["knn"])
            C[i, j], P[i, j], A[i, j] = r["C"], r["purity"], r["ari"]
    return C, P, A


# --------------------------------------------------------------------------- resolution map
def _outline_region(ax, mask, lw_out=1.6, lw_in=0.8):
    """Outline the union of the True cells of ``mask`` (image coordinates, origin lower), merging
    contiguous collinear edges so corners join cleanly without overlapping end-caps."""
    from matplotlib.collections import LineCollection
    n_r, n_c = mask.shape
    segs = []
    for i in range(n_r + 1):
        start = None
        for j in range(n_c):
            edge = (bool(mask[i, j]) if i < n_r else False) != (bool(mask[i - 1, j]) if i > 0 else False)
            if edge and start is None:
                start = j - 0.5
            elif not edge and start is not None:
                segs.append([(start, i - 0.5), (j - 0.5, i - 0.5)])
                start = None
        if start is not None:
            segs.append([(start, i - 0.5), (n_c - 0.5, i - 0.5)])
    for j in range(n_c + 1):
        start = None
        for i in range(n_r):
            edge = (bool(mask[i, j]) if j < n_c else False) != (bool(mask[i, j - 1]) if j > 0 else False)
            if edge and start is None:
                start = i - 0.5
            elif not edge and start is not None:
                segs.append([(j - 0.5, start), (j - 0.5, i - 0.5)])
                start = None
        if start is not None:
            segs.append([(j - 0.5, start), (j - 0.5, n_r - 0.5)])
    ax.add_collection(LineCollection(segs, colors=L.INK, linewidths=lw_out, capstyle="projecting", clip_on=False, zorder=4))
    ax.add_collection(LineCollection(segs, colors="white", linewidths=lw_in, capstyle="projecting", clip_on=False, zorder=5))


def resolution_panel(ax, results, title, show_x=True, show_y=True, fs=8.5, modal_bump=0.0, cstar_fs=8.5,
                     family=None, lw_out=1.6, lw_in=0.8, title_pad=4.0):
    """One local resolution map; family='Liberation Sans Narrow' keeps two-digit counts legible in compact cells."""
    C, P, _ = _mats(results)
    vals, cnt = np.unique(C[np.isfinite(C)], return_counts=True)
    modal = vals[cnt.argmax()]
    im = ax.imshow(P, origin="lower", aspect="auto", cmap=PURE_SEQ, vmin=0.0, vmax=1.0, interpolation="nearest")
    for i in range(len(TS)):
        for j in range(len(KNNS)):
            c = int(C[i, j])
            dark = P[i, j] > 0.60
            is_m = c == modal
            ax.text(j, i, str(c), ha="center", va="center_baseline", fontsize=fs + (modal_bump if is_m else 0),
                    zorder=6, color="white" if dark else L.INK, fontweight="bold" if is_m else "normal",
                    **({"fontfamily": family} if family else {}))
    _outline_region(ax, C == modal, lw_out=lw_out, lw_in=lw_in)
    ax.set_xticks(range(len(KNNS))); ax.set_yticks(range(len(TS)))
    ax.set_xticklabels(KNNS if show_x else [], fontweight="bold")
    ax.set_yticklabels(TS if show_y else [], fontweight="bold")
    if show_x:
        ax.set_xlabel(r"neighborhood size $k_{\mathrm{nn}}$", labelpad=2.5)
    if show_y:
        ax.set_ylabel("diffusion iterations $t$", labelpad=2.0)
    ax.set_title(title, loc="left", pad=title_pad)
    ax.set_title(f"$C^\\star\\!=\\!{int(modal)}$ ({cnt.max()}/99)", loc="right", fontsize=cstar_fs, fontweight="normal",
                 pad=title_pad)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0, pad=2.0)
    ax.set_xlim(-0.5, len(KNNS) - 0.5); ax.set_ylim(-0.5, len(TS) - 0.5)
    return im


def fig_resolution():
    """Local resolution maps of the ten datasets, three per row, at the full text width."""
    G = _grid()
    L.style()
    plt.rcParams.update({"axes.titlesize": 9.5, "axes.labelsize": 8.5})
    W, H = L.width(), 6.50
    x_left, x_right_pad, col_gap = 0.42, 0.055, 0.075
    panel_w = (W - x_left - x_right_pad - 2 * col_gap) / 3.0
    top_m, gap_tight, gap_mid, bot_m = 0.175, 0.225, 0.405, 0.340
    panel_h = (H - top_m - 2 * gap_tight - gap_mid - bot_m) / 4.0
    row_y = [H - top_m - panel_h]
    row_y.append(row_y[-1] - gap_tight - panel_h)
    row_y.append(row_y[-1] - gap_tight - panel_h)
    row_y.append(row_y[-1] - gap_mid - panel_h)
    fig = plt.figure(figsize=(W, H))
    im = None
    for k, top in enumerate(range(1, 11)):
        r, c = divmod(k, 3)
        x0 = x_left + c * (panel_w + col_gap)
        ax = fig.add_axes([x0 / W, row_y[r] / H, panel_w / W, panel_h / H])
        im = resolution_panel(ax, G[top]["results"], f"Top-{top}", show_x=k >= 7, show_y=k % 3 == 0,
                              fs=8.5, modal_bump=0.0, cstar_fs=8.5, family="Liberation Sans Narrow",
                              lw_out=1.55, lw_in=0.78, title_pad=3.5)
    cb_x0 = x_left + 1.12 * (panel_w + col_gap)
    cb_x1 = x_left + 3 * panel_w + 2 * col_gap - 0.04
    cax = fig.add_axes([cb_x0 / W, (row_y[3] + 0.50 * panel_h) / H, (cb_x1 - cb_x0) / W, 0.175 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      format=FormatStrFormatter("%.1f"))
    L.bold_colorbar(cb, "purity against the five missions", size=9)
    L.save(fig, "fig_resolution_map")


def fig_resolution_compact(tops=(1, 10)):
    """The local resolution maps of the first and the last dataset side by side."""
    G = _grid()
    L.style()
    plt.rcParams.update({"axes.titlesize": 9.5, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, len(tops), figsize=(L.width(), 3.0))
    for k, (ax, top) in enumerate(zip(axes, tops)):
        im = resolution_panel(ax, G[top]["results"], f"Top-{top}", show_y=k == 0, fs=8.5, modal_bump=0.5,
                              cstar_fs=9, lw_out=2.0, lw_in=1.0)
    cax = fig.add_axes([0.25, 0.05, 0.55, 0.033])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      format=FormatStrFormatter("%.1f"))
    L.bold_colorbar(cb, "purity against the five missions", size=9)
    fig.subplots_adjust(left=0.09, right=1.0, bottom=0.29, top=0.925, wspace=0.08)
    L.save(fig, "fig_resolution_compact")



# --------------------------------------------------------------------------- fusion sequence
MISS = ["gate", "forag", "homing", "shelter", "xor"]


def _pattern(y, p):
    cm = contingency_matrix(y, p)
    home = cm.argmax(1)
    groups = {}
    for m, c in enumerate(home):
        groups.setdefault(c, []).append(m)
    return tuple(sorted(tuple(g) for g in groups.values()))


def _pairs_at(pats, c=3):
    """The pairs of missions that share a cluster at C = c (the most frequent grouping of the missions), as
    [(C, mission a, mission b)]. The figure draws which missions merge, not the order in which they do."""
    if not pats.get(c):
        return []
    target = [tuple(g) for g in pats[c].most_common(1)[0][0]]
    if any(len(g) > 2 for g in target):
        print(f"WARNING fig_fusion_sequence: the grouping at C={c} holds more than two missions in a cluster ({target})")
    return [(c, (g[0],), (g[1],)) for g in target if len(g) == 2]


def _split_label(a, b):
    """The fields in which two missions are farther apart than the median pair (pair separation, Top-1)."""
    R, pairs = pair_separation()
    j = pairs.index((min(a, b), max(a, b)))
    sym = ["$\\rho$", "$K$", "$\\varphi$", "$\\psi$", "$J$"]
    split = [sym[f] for f in range(5) if R[f, j] > np.median(R[f])]
    if not split:
        return "close in\nall fields"
    return f"split only\nby {split[0]}" if len(split) == 1 else "split by\n" + ", ".join(split)


def fig_fusion_sequence():
    E = L.load_pickle(L.PAPER / "extended_top1.pkl")
    y, R, knns, ts = E["labels"], E["results"], E["knns"], E["ts"]
    C, P, _ = _mats(R, knns, ts)
    L.style()
    fig = plt.figure(figsize=(L.width(), 2.85))
    ax = fig.add_axes([0.092, 0.165, 0.487, 0.7175])
    levels = {5: "#efeeea", 4: "#f6b896", 3: "#eb6834", 2: "#a83a10"}
    cols = [levels.get(c, "#cde2fb") for c in range(1, 13)]
    cmap = ListedColormap(cols); norm = BoundaryNorm(np.arange(0.5, 13.5), cmap.N)
    ax.imshow(C, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    for i in range(len(ts)):
        for j in range(len(knns)):
            ax.text(j, i, int(C[i, j]), ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if C[i, j] <= 4 else "normal", color="white" if C[i, j] <= 3 else L.INK)
    ax.set_xticks(range(len(knns))); ax.set_xticklabels(knns, fontweight="bold")
    ax.set_yticks(range(len(ts))); ax.set_yticklabels(ts, fontweight="bold")
    ax.set_xlabel(r"neighborhood size $k_{\mathrm{nn}}$"); ax.set_ylabel("diffusion iterations $t$")
    ax.axvline(knns.index(max(KNNS)) + 0.5, color=L.INK2, lw=0.8, ls="--")       # the main grid: bottom left
    ax.axhline(ts.index(max(TS)) + 0.5, color=L.INK2, lw=0.8, ls="--")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Top-1: number of behaviors $C$", loc="left")
    pats = {}
    for r in R:
        if r["C"] in (2, 3, 4):
            pats.setdefault(r["C"], Counter())[_pattern(y, r["pred"])] += 1
    merges = _pairs_at(pats)
    # dendrogram of the observed merges (leaves on top, merges going down); the first merged groups on the left
    ax2 = fig.add_axes([0.60, 0.139, 0.40, 0.725]); ax2.axis("off")
    top_y, bottom = 3.4, 0.55
    height = {4: 2.4, 3: 1.3, 2: 0.85}
    members, merge_c = {}, {}                                    # each merged group: its two parts, the C it appears at
    roots = [(m,) for m in range(5)]
    for c, a, b in merges:
        g = tuple(sorted(a + b))
        members[g], merge_c[g] = (a, b), c
        roots = [r for r in roots if r not in (a, b)] + [g]
    earliest = lambda g: max([c for c, a, b in merges if set(a + b) <= set(g)], default=0)
    roots.sort(key=lambda g: (len(g) == 1, -earliest(g)))

    def leaves(g):
        return [g[0]] if len(g) == 1 else leaves(members[g][0]) + leaves(members[g][1])
    x_of, x = {}, 0.0
    for g in roots:
        for m in leaves(g):
            x_of[(m,)] = x; x += 1.4
        x += 0.4

    def draw(g):
        """Draw the subtree of group g; return its x and the height at which it ends (its own merge)."""
        if len(g) == 1:
            m = g[0]
            ax2.add_patch(Rectangle((x_of[g] - 0.35, top_y), 0.7, 0.18, color=L.MCOL[m], lw=0))
            ax2.text(x_of[g] - 0.12, top_y + 0.28, L.MSHORT[m], ha="left", va="bottom", fontsize=9, fontweight="bold",
                     rotation=30, rotation_mode="anchor")
            return x_of[g], top_y
        a, b = members[g]
        (xa, ya), (xb, yb) = draw(a), draw(b)
        yj = height[merge_c[g]]
        ax2.plot([xa, xa, xb, xb], [ya, yj, yj, yb], color=L.INK, lw=1.4, solid_joinstyle="miter")
        xm = 0.5 * (xa + xb)
        return xm, yj
    for g in roots:
        xg, yg = draw(g)
        ax2.plot([xg, xg], [yg, bottom], color=L.INK, lw=1.4)
        label = _split_label(*g) if len(g) == 2 else ("stays\napart" if len(g) == 1 else "")
        ax2.text(xg, bottom - 0.2, label, ha="center", va="top", fontsize=8.5)
    ax2.set_xlim(-0.8, x - 0.4 + 1.15); ax2.set_ylim(-0.7, 4.4)
    ax2.set_title("missions merged at $C=3$", loc="left")
    L.save(fig, "fig_fusion_sequence")
    return pats


# --------------------------------------------------------------------------- fields x missions
FCOL_TITLE = {"rho": "density\n$\\rho_\\sigma$", "K": "kinetic intensity\n$K_\\sigma$",
              "phi": "divergence potential\n$\\varphi_\\sigma$", "psi": "vorticity potential\n$\\psi_\\sigma$",
              "J": "Jacobi coupling\n$J_\\sigma$"}
FROW = {"rho": "density\n$\\rho_\\sigma$", "K": "kinetic\nintensity\n$K_\\sigma$",
        "phi": "divergence\npotential\n$\\varphi_\\sigma$", "psi": "vorticity\npotential\n$\\psi_\\sigma$",
        "J": "Jacobi\ncoupling\n$J_\\sigma$"}
FSYM = {"rho": "$\\rho_\\sigma$", "K": "$K_\\sigma$", "phi": "$\\varphi_\\sigma$", "psi": "$\\psi_\\sigma$", "J": "$J_\\sigma$"}


def draw_tracks(ax, pos, color, every=4):
    T, N, _ = pos.shape
    for i in range(N):
        ax.plot(pos[::every, i, 0], pos[::every, i, 1], color=color, lw=0.35, alpha=0.55)
    ax.scatter(pos[-1, :, 0], pos[-1, :, 1], s=6, color=L.INK, zorder=3, lw=0)
    ax.set_xlim(0, L.ARENA); ax.set_ylim(0, L.ARENA); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def fig_fields(si=2, name="fig_fields", exemplars=None):
    cache = L.load_pickle(L.PAPER / "display_fields.pkl")
    ref = L.load_pickle(L.PAPER / "reference_top1.pkl")
    d = L.load_pickle(L.PAPER / "top1_tracks.pkl")
    y = ref["y"]
    ex = list(exemplars) if exemplars is not None else list(L.load_pickle(L.PAPER / "taxonomy.pkl")[1]["exemplars"])
    order = sorted(ex, key=lambda e: int(y[e]))
    L.style()
    fig, axes = plt.subplots(5, 6, figsize=(L.width(), 6.1))
    for r, e in enumerate(order):
        m = int(y[e])
        draw_tracks(axes[r, 0], d["tracks"][e]["pos"], L.MCOL[m])
        axes[r, 0].text(-0.08, 0.5, L.MSHORT[m], transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=11, fontweight="bold")
        axes[r, 0].add_patch(Rectangle((-0.06, 0.02), 0.035, 0.96, transform=axes[r, 0].transAxes,
                                       color=L.MCOL[m], clip_on=False, lw=0))
        F = cache[("ladder", int(e), si)]
        for c, f in enumerate(L.FIELDS):
            L.show_field(axes[r, c + 1], F[f], f)
    axes[0, 0].set_title("tracks\n\n", fontsize=10, fontweight="bold")
    for c, f in enumerate(L.FIELDS):
        head = FROW[f]
        if head.count("\n") < 2:                      # pad to three lines so all headers align
            head = head.replace("\n", "\n\n", 1)
        axes[0, c + 1].set_title(head, fontsize=10, fontweight="bold", linespacing=1.15)
    fig.subplots_adjust(left=0.05, right=1.0, bottom=0.0, top=0.885, wspace=0.04, hspace=0.04)
    L.save(fig, name)


# --------------------------------------------------------------------------- pipeline walk-through
DGM_NAMES = ["Ord$_0$", "Ord$_1$", "Rel$_1$", "Rel$_2$", "Ext$_0$", "Ext$_1$"]
DGM_MARK = ["o", "s", "^", "v", "D", "P"]


def _seed_families(seeds, J, pos):
    from scipy.spatial import cKDTree
    core = cKDTree(pos.reshape(-1, 2).astype(np.float64)).query(seeds)[0] < 1e-5
    rel = np.abs(J) / np.nanmax(np.abs(J))
    skel = (~core) & (rel < 1e-3)
    bary = (~core) & (~skel)
    return core, bary, skel


# --------------------------------------------------------------------------- appendix: pair separation
@functools.lru_cache(maxsize=1)
def pair_separation():
    """Separation ratio per field and mission pair on Top-1: mean inter-mission rank distance over
    the mean of the two intra-mission ones (rank distances averaged over the four scales). Computed once
    per session (the merge order and the pair separation figures share it)."""
    import run_sweep as RS
    from classification import fusion as F
    lab, rk, _ = RS.load_labels(); D, _ = RS.load_distances()
    idx = RS.subset(1, rk); y = lab[idx]
    Q = F.rank_normalise(D, F.ALL_FIELDS, idx)
    pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    R = np.zeros((len(F.ALL_FIELDS), len(pairs)))
    for fi, f in enumerate(F.ALL_FIELDS):
        d = np.mean([Q[f][s] for s in Q[f]], axis=0)
        intra = [d[np.ix_(y == m, y == m)][np.triu_indices((y == m).sum(), 1)].mean() for m in range(5)]
        for pi, (a, b) in enumerate(pairs):
            R[fi, pi] = d[np.ix_(y == a, y == b)].mean() / (0.5 * (intra[a] + intra[b]))
    return R, pairs


def fig_pair_separation():
    R, pairs = pair_separation()
    import pickle as _pk
    _pk.dump({"R": R, "pairs": pairs, "fields": ["rho", "K", "phi", "psi", "J"]}, open(L.PAPER / "pair_separation.pkl", "wb"))
    order = np.argsort(R.max(0))           # hardest pairs (no good separator) first
    R = R[:, order]; pairs = [pairs[i] for i in order]
    L.style()
    # one row per pair of missions, one column per field: the pair names stay horizontal and legible
    T = R.T
    fig, ax = plt.subplots(figsize=(L.width(), 2.1))
    im = ax.imshow(T, cmap=L.SEQ, vmin=1.0, vmax=float(np.ceil(R.max())), aspect="auto")
    best = np.argmax(T, axis=1)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            v = T[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if best[i] == j else "normal", color="white" if v > 3.5 else L.INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(["$\\rho$", "$K$", "$\\varphi$", "$\\psi$", "$J$"], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{L.MSHORT[a]} / {L.MSHORT[b]}" for a, b in pairs], fontsize=8.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    L.bold_colorbar(fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02), "separation ratio")
    fig.tight_layout()
    L.save(fig, "fig_pair_separation")
    return R, pairs


# --------------------------------------------------------------------------- appendix: scale gallery
def mission_exemplar(m):
    """Top-1 position of the consensus exemplar of the cluster of mission m (0 D-Gate ... 4 XOR)."""
    y = L.load_pickle(L.PAPER / "top1_tracks.pkl")["labels"]
    return int([x for x in L.load_pickle(L.PAPER / "taxonomy.pkl")[1]["exemplars"] if int(y[x]) == m][0])


def fig_scales(e=None, fields=None, name="fig_scales"):
    """Fields of the Shelter exemplar at the four scales; fields=("rho", "psi") gives the compact strip
    of the main text (0.86 of the text width), all five fields the appendix version (0.70)."""
    e = mission_exemplar(3) if e is None else e                       # the Shelter exemplar
    fields = list(L.FIELDS if fields is None else fields)
    cache = L.load_pickle(L.PAPER / "display_fields.pkl")
    ladder = L.load_pickle(L.PAPER / "reference_top1.pkl")["ladder"]
    L.style()
    compact = len(fields) < 5
    W = L.width(0.86 if compact else 0.70)
    left, gap, head = 0.95, 0.03, (0.27 if compact else 0.42)        # inches: row labels, gap, column titles
    side = (W - left - 3 * gap) / 4
    H = len(fields) * side + (len(fields) - 1) * gap + head
    fig = plt.figure(figsize=(W, H))
    for si in range(4):
        F_ = cache[("ladder", e, si)]
        for fi, f in enumerate(fields):
            ax = fig.add_axes([(left + si * (side + gap)) / W, (H - head - (fi + 1) * side - fi * gap) / H, side / W, side / H])
            L.show_field(ax, F_[f], f)
            if si == 0:
                ax.set_ylabel(FROW[f], fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center", labelpad=6)
            if fi == 0:
                value = f"{100 * ladder[si]:.1f} cm"
                ax.set_title(f"$\\sigma_{si + 1}$ = {value}" if compact else f"$\\sigma_{si + 1}$\n{value}",
                             fontsize=9.5, fontweight="bold", linespacing=1.15)
    L.save(fig, name)


def fig_uncapped(e=None):
    e = mission_exemplar(2) if e is None else e                       # the Homing exemplar
    cache = L.load_pickle(L.PAPER / "display_fields.pkl")
    L.style()
    fig, axes = plt.subplots(2, 6, figsize=(L.width(), 2.15))
    for si in range(6):
        F_ = cache[("uncapped", e, si)]; sig = cache[("uncapped_sigma", si)]
        L.show_field(axes[0, si], F_["rho"], "rho"); L.show_field(axes[1, si], F_["psi"], "psi")
        axes[0, si].set_title(f"$\\sigma$ = {100 * sig:.0f} cm" if sig >= 0.1 else f"$\\sigma$ = {100 * sig:.1f} cm", fontsize=9.5)
    for r_, lab_ in ((0, "density\n$\\rho_\\sigma$"), (1, "vorticity\npotential\n$\\psi_\\sigma$")):
        axes[r_, 0].set_ylabel(lab_, fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center", labelpad=6)
    fig.subplots_adjust(left=0.13, right=1.0, bottom=0.0, top=0.9, wspace=0.03, hspace=0.04)
    L.save(fig, "figA_uncapped")


# --------------------------------------------------------------------------- pipeline walk-through
def _inch_figure(H):
    """A figure at the full text width whose axes, texts and arrows are placed in inches. The steps run in two rows,
    (a) to (b) from left to right, then down to (c) on the right, then (c) to (d) from right to left."""
    from matplotlib.patches import FancyArrowPatch
    W = L.width()
    fig = plt.figure(figsize=(W, H))

    def ax_at(x0, y0, w, h):
        a = fig.add_axes([x0 / W, y0 / H, w / W, h / H]); a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values():
            s_.set_visible(False)
        return a

    def head(x0, y0, letter, text):
        fig.text(x0 / W, y0 / H, f"({letter}) {text}", fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    def arrow(p0, p1, text, at, ha="center", va="bottom"):
        fig.add_artist(FancyArrowPatch((p0[0] / W, p0[1] / H), (p1[0] / W, p1[1] / H), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=11, lw=1.3, color=L.INK))
        fig.text(at[0] / W, at[1] / H, text, ha=ha, va=va, fontsize=8.5, fontweight="bold", linespacing=1.0)
    return fig, W, ax_at, head, arrow


def _shelter_exemplar(tax, y):
    return int([x for x in tax["exemplars"] if int(y[x]) == 3][0])


def fig_pipeline_one(e=None, si=2):
    """One recording followed through the method: tracks, fields, grid-free domain, diagrams (Figure 'one')."""
    import gudhi
    from matplotlib.collections import LineCollection
    tax = L.load_pickle(L.PAPER / "taxonomy.pkl")[1]
    d = L.load_pickle(L.PAPER / "top1_tracks.pkl")
    cache = L.load_pickle(L.PAPER / "display_fields.pkl")
    idx = d["idx"]; y = d["labels"]
    e = _shelter_exemplar(tax, y) if e is None else e
    g_e, m_e = int(idx[e]), int(y[e])
    st1 = L.load_pickle(L.RES / "stage1_alpha_fields" / f"traj{g_e:04d}_s{si}.pkl")
    st2 = L.load_pickle(L.RES / "stage2_alpha_dgms" / f"traj{g_e:04d}_s{si}.pkl")
    sig = float(L.load_pickle(L.PAPER / "reference_top1.pkl")["ladder"][si])
    if ("ladder", e, si) not in cache:
        raise KeyError(f"display fields missing for run {e}")
    L.style()
    cell, gap = 0.80, 0.03                               # the five fields, three per row
    bw, bh = 3 * cell + 2 * gap, 2 * cell + gap
    sq = 1.55                                            # tracks, domain and diagrams
    leg = 0.50                                           # legends under the second row
    H = 0.22 + bh + 0.55 + sq + leg
    fig, W, ax_at, head, arrow = _inch_figure(H)
    top1 = H - 0.22; bot1 = top1 - bh
    top2 = bot1 - 0.55; bot2 = top2 - sq
    # (a) tracks
    a = ax_at(0.0, bot1 + (bh - sq) / 2, sq, sq); draw_tracks(a, d["tracks"][e]["pos"], L.MCOL[m_e])
    head(0.0, top1 + 0.04, "a", "tracks")
    # (b) the five fields, three per row
    xb = W - bw
    F_ = cache[("ladder", e, si)]
    for k, f in enumerate(L.FIELDS):
        r_, c_ = divmod(k, 3)
        b = ax_at(xb + c_ * (cell + gap), top1 - (r_ + 1) * cell - r_ * gap, cell, cell)
        L.show_field(b, F_[f], f)
        b.text(0.04, 0.96, FSYM[f], transform=b.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold")
    head(xb, top1 + 0.04, "b", "five fields")
    # (c) grid-free domain, under the fields
    xc = W - sq
    c = ax_at(xc, bot2, sq, sq)
    seeds = st1["seeds"]
    core, bary, skel = _seed_families(seeds, np.asarray(st1["fields"]["J"]), d["tracks"][e]["pos"])
    stree = gudhi.AlphaComplex(points=seeds).create_simplex_tree(max_alpha_square=(3 * sig) ** 2)
    segs = [seeds[s_] for s_, _ in stree.get_skeleton(1) if len(s_) == 2]
    c.add_collection(LineCollection(segs, colors="#c3c2b7", linewidths=0.3, zorder=1))
    c.scatter(seeds[core, 0], seeds[core, 1], s=3.5, c=L.INK2, lw=0, zorder=2, label="robot positions")
    c.scatter(seeds[bary, 0], seeds[bary, 1], s=9, c=L.MCOL[1], lw=0, zorder=3, label="near-critical")
    c.scatter(seeds[skel, 0], seeds[skel, 1], s=7, c=L.MCOL[0], lw=0, zorder=3, label="$J_\\sigma=0$")
    c.set_xlim(0, L.ARENA); c.set_ylim(0, L.ARENA); c.set_aspect("equal")
    c.legend(loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=1, frameon=False, fontsize=8.5, markerscale=2.0,
             handletextpad=0.1, labelspacing=0.12, borderaxespad=0.0)
    head(xc, top2 + 0.04, "c", "grid-free domain")
    # (d) the diagrams of rho on the domain (square-root axes), on the left of the second row
    xd = 0.22
    dd = ax_at(xd, bot2, sq, sq)
    for s_ in ("left", "bottom"):
        dd.spines[s_].set_visible(True)
    dg = st2["dgms"]["rho"]
    allv = np.concatenate([np.asarray(x).ravel() for x in dg if len(x)])
    hi = np.sqrt(max(allv.max(), 1e-12)) * 1.05
    dd.plot([0, hi], [0, hi], color=L.GRID, lw=0.8)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#008300"]
    for k, x in enumerate(dg):
        x = np.sqrt(np.clip(np.asarray(x), 0, None))
        if len(x):
            dd.scatter(x[:, 0], x[:, 1], s=15, marker=DGM_MARK[k], c=cols[k], lw=0, label=DGM_NAMES[k], zorder=3)
    dd.set_xlim(0, hi); dd.set_ylim(0, hi)
    dd.set_xlabel("birth", fontsize=8.5, labelpad=1); dd.set_ylabel("death", fontsize=8.5, labelpad=1)
    dd.legend(loc="upper left", bbox_to_anchor=(0.0, -0.09), ncol=3, fontsize=8.5, frameon=False, handletextpad=0.05,
              borderaxespad=0.0, labelspacing=0.12, columnspacing=0.6)
    head(0.0, top2 + 0.04, "d", "diagrams of $\\rho_\\sigma$")
    ym = bot1 + bh / 2
    arrow((sq + 0.05, ym), (xb - 0.06, ym), "moments", ((sq + xb) / 2, ym + 0.06))
    xv = xc + sq / 2
    arrow((xv, bot1 - 0.04), (xv, top2 + 0.22), "seeds", (xv - 0.08, bot1 - 0.17), ha="right", va="center")
    y2 = bot2 + sq / 2
    arrow((xc - 0.06, y2), (xd + sq + 0.06, y2), "lower-\nstar", ((xc + xd + sq) / 2, y2 + 0.06))
    L.save(fig, "fig_pipeline_one")


def fig_pipeline_all(knn=7, t=10):
    """The whole Top-1 collection: distance matrices, scale views, consensus, exemplars (Figure 'all')."""
    import run_sweep as RS
    from classification import fusion as F
    tax = L.load_pickle(L.PAPER / "taxonomy.pkl")[1]
    d = L.load_pickle(L.PAPER / "top1_tracks.pkl")
    idx = d["idx"]; y = d["labels"]
    D, _ = RS.load_distances()
    Q = F.rank_normalise(D, F.ALL_FIELDS, idx)
    del D
    views = F.scale_views(Q)
    S = F.consensus(views, knn, t)
    cons = tax["labels"]; order = np.argsort(cons, kind="stable")
    L.style()
    cell, gap, lab = 0.31, 0.02, 0.30                   # (a): twenty matrices, fields in rows, scales in columns
    ah = 5 * cell + 4 * gap
    vc, vg = 0.80, 0.04                                 # (b): four views, two per row
    bw = 2 * vc + vg
    wc = 1.55                                           # (c): consensus
    tw, tg = 0.70, 0.05                                 # (d): exemplars, three per row
    H = 0.22 + 0.17 + ah + 0.55 + wc
    fig, W, ax_at, head, arrow = _inch_figure(H)
    top1 = H - 0.22; bot1 = top1 - 0.17 - ah
    top2 = bot1 - 0.55; bot2 = top2 - wc
    for fi, f in enumerate(L.FIELDS):
        for s_i in range(4):
            m = ax_at(lab + s_i * (cell + gap), top1 - 0.17 - (fi + 1) * cell - fi * gap, cell, cell)
            m.imshow(1.0 - Q[f][list(Q[f].keys())[s_i]], cmap=L.SEQ, interpolation="nearest", aspect="auto")
            if s_i == 0:
                m.set_ylabel(FSYM[f], fontsize=9.5, rotation=0, labelpad=11, va="center")
            if fi == 0:
                m.set_title(f"$\\sigma_{s_i + 1}$", fontsize=9, pad=1.5)
    head(0.0, top1 + 0.04, "a", "20 matrices")
    # (b) the four scale views
    xb = W - bw; yb = bot1 + (ah - bw) / 2
    for k in range(4):
        r_, c_ = divmod(k, 2)
        v = ax_at(xb + c_ * (vc + vg), yb + (1 - r_) * (vc + vg), vc, vc)
        v.imshow(views[k], cmap=L.SEQ, interpolation="nearest", aspect="auto")
        v.text(0.05, 0.95, f"$W_{k + 1}$", transform=v.transAxes, ha="left", va="top", fontsize=9,
               fontweight="bold", color="white")
    head(xb, top1 + 0.04, "b", "four views")
    # (c) the consensus, ordered by the discovered clusters; stars mark the exemplars
    xc = W - wc
    gS = ax_at(xc, bot2, wc, wc)
    Sv = S[np.ix_(order, order)]
    gS.imshow(Sv, cmap=L.SEQ, vmin=0, vmax=np.quantile(Sv, 0.995), interpolation="nearest", aspect="auto")
    for b_ in np.cumsum(np.bincount(cons))[:-1]:
        gS.axhline(b_ - 0.5, color="white", lw=0.9); gS.axvline(b_ - 0.5, color="white", lw=0.9)
    pos_of = {int(r): k for k, r in enumerate(order)}
    for ex in tax["exemplars"]:
        p_ = pos_of[int(ex)]
        gS.scatter([p_], [p_], marker="*", s=80, c="#eda100", edgecolors=L.INK, linewidths=0.5, zorder=4)
    head(xc, top2 + 0.04, "c", "consensus $S^\\star$")
    # (d) the exemplar of each behaviour, three per row, on the left of the second row
    ex_sorted = sorted(tax["exemplars"], key=lambda r: cons[r])
    n_row = 3
    for k, ex in enumerate(ex_sorted):
        r_, c_ = divmod(k, n_row)
        in_row = min(n_row, len(ex_sorted) - r_ * n_row)
        x0 = 0.05 + c_ * (tw + tg) + 0.5 * (n_row - in_row) * (tw + tg)        # a shorter last row is centred
        y0 = top2 - 0.16 - (r_ + 1) * tw - r_ * 0.2
        h_ = ax_at(x0, y0, tw, tw)
        draw_tracks(h_, d["tracks"][ex]["pos"], L.MCOL[int(y[ex])], every=8)
        fig.text((x0 + tw / 2) / W, (y0 + tw + 0.01) / H, f"{k + 1}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    head(0.0, top2 + 0.04, "d", f"$C={len(ex_sorted)}$ exemplars")
    ym = bot1 + ah / 2
    xa1 = lab + 4 * cell + 3 * gap
    arrow((xa1 + 0.06, ym), (xb - 0.06, ym), "ranks, balance", ((xa1 + xb) / 2, ym + 0.06))
    xv = xb + bw / 2
    arrow((xv, bot1 + (ah - bw) / 2 - 0.04), (xv, top2 + 0.22), "SNF", (xv - 0.08, bot1 - 0.12),
          ha="right", va="center")
    xd1 = 0.05 + n_row * tw + (n_row - 1) * tg
    y2 = bot2 + wc / 2
    arrow((xc - 0.06, y2), (xd1 + 0.08, y2), "AP", ((xc + xd1) / 2, y2 + 0.06))
    L.save(fig, "fig_pipeline_all")


def fig_method(si=2, knn=7, t=10):
    """The method in one figure, at the start of the methodology: one recording from its tracks to its diagrams (top
    row, left to right), then the whole of Top-1 from the distances to the behaviors (bottom row, right to left), and
    the attribution that runs back from the distance to the critical points and the robots (dashed). It gathers the
    overview and the two walk-throughs, fig_pipeline_one and fig_pipeline_all, in less space."""
    import gudhi
    import run_sweep as RS
    from classification import fusion as F
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch
    tax = L.load_pickle(L.PAPER / "taxonomy.pkl")[1]
    d = L.load_pickle(L.PAPER / "top1_tracks.pkl")
    cache = L.load_pickle(L.PAPER / "display_fields.pkl")
    idx = d["idx"]; y = d["labels"]
    e = _shelter_exemplar(tax, y)
    g_e = int(idx[e])
    st1 = L.load_pickle(L.RES / "stage1_alpha_fields" / f"traj{g_e:04d}_s{si}.pkl")
    st2 = L.load_pickle(L.RES / "stage2_alpha_dgms" / f"traj{g_e:04d}_s{si}.pkl")
    sig = float(L.load_pickle(L.PAPER / "reference_top1.pkl")["ladder"][si])
    D, _ = RS.load_distances()
    Q = F.rank_normalise(D, F.ALL_FIELDS, idx)
    Draw = {f: [np.asarray(D[f][k], dtype=np.float64)[np.ix_(idx, idx)] for k in Q[f]] for f in L.FIELDS}
    del D
    views = F.scale_views(Q)
    S = F.consensus(views, knn, t)
    cons = tax["labels"]; order = np.argsort(cons, kind="stable")
    L.style()
    W = L.width()
    sq, cell, gap = 0.80, 0.38, 0.025                   # top row: tracks, fields (three then two), domain, diagrams
    dw = 0.86                                            # width of the diagram panel, axis labels included
    mc, mg, lab = 0.158, 0.01, 0.17                      # bottom row: twenty matrices, fields in rows, scales in columns
    mh = 5 * mc + 4 * mg
    vc, vg = 0.38, 0.03                                  # four views, two per row
    wc = 0.80                                            # consensus
    tw, tg = 0.29, 0.04                                  # exemplars, three then two
    g1, g2, g3 = 0.30, 0.30, 0.38                        # gaps of the bottom row, under AP, SNF and ranks
    arch, hd, mid, note = 0.33, 0.17, 0.30, 0.02
    H = arch + hd + sq + mid + hd + 0.12 + mh + note
    fig = plt.figure(figsize=(W, H))

    def ax_at(x0, y0, w, h):
        a = fig.add_axes([x0 / W, y0 / H, w / W, h / H]); a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values():
            s_.set_visible(False)
        return a

    def head(x0, y0, text, ha="left"):
        fig.text(x0 / W, y0 / H, text, fontsize=9, fontweight="bold", ha=ha, va="bottom")

    def arrow(p0, p1, text=None, at=None, ha="center", va="bottom", color=L.INK, ls="-"):
        fig.add_artist(FancyArrowPatch((p0[0] / W, p0[1] / H), (p1[0] / W, p1[1] / H), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color, linestyle=ls,
                                       shrinkA=0, shrinkB=0))
        if text:
            fig.text(at[0] / W, at[1] / H, text, ha=ha, va=va, fontsize=8.5, fontweight="bold", linespacing=1.0,
                     color=color)

    # ---- top row: one recording, from its tracks to its diagrams
    top1 = H - arch - hd; bot1 = top1 - sq
    xa = 0.0
    a = ax_at(xa, bot1, sq, sq); draw_tracks(a, d["tracks"][e]["pos"], L.MCOL[int(y[e])], every=5)
    head(xa, top1 + 0.03, "(a) tracks")
    gtop = (W - (2 * sq + 3 * cell + 2 * gap + dw)) / 3                       # three equal gaps in the top row
    xb = xa + sq + gtop
    F_ = cache[("ladder", e, si)]
    for k, f in enumerate(L.FIELDS):
        r_, c_ = divmod(k, 3)
        x0 = xb + c_ * (cell + gap) + (0.5 * (cell + gap) if r_ == 1 else 0.0)   # the second row is centered
        b = ax_at(x0, top1 - (r_ + 1) * cell - r_ * gap, cell, cell)
        L.show_field(b, F_[f], f)
        b.text(0.04, 0.96, FSYM[f], transform=b.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold")
    bw = 3 * cell + 2 * gap
    head(xb, top1 + 0.03, "(b) five fields")
    xc = xb + bw + gtop
    c = ax_at(xc, bot1, sq, sq)
    seeds = st1["seeds"]
    core, bary, skel = _seed_families(seeds, np.asarray(st1["fields"]["J"]), d["tracks"][e]["pos"])
    stree = gudhi.AlphaComplex(points=seeds).create_simplex_tree(max_alpha_square=(3 * sig) ** 2)
    segs = [seeds[s_] for s_, _ in stree.get_skeleton(1) if len(s_) == 2]
    c.add_collection(LineCollection(segs, colors="#c3c2b7", linewidths=0.25, zorder=1))
    c.scatter(seeds[core, 0], seeds[core, 1], s=2.5, c=L.INK2, lw=0, zorder=2)
    c.scatter(seeds[bary, 0], seeds[bary, 1], s=6, c=L.MCOL[1], lw=0, zorder=3)
    c.scatter(seeds[skel, 0], seeds[skel, 1], s=5, c=L.MCOL[0], lw=0, zorder=3)
    c.set_xlim(0, L.ARENA); c.set_ylim(0, L.ARENA); c.set_aspect("equal")
    head(xc, top1 + 0.03, "(c) domain")
    xd = W - dw + 0.17
    dd = ax_at(xd, bot1 + 0.13, dw - 0.17, sq - 0.13)
    for s_ in ("left", "bottom"):
        dd.spines[s_].set_visible(True)
    dg = st2["dgms"]["rho"]
    allv = np.concatenate([np.asarray(x).ravel() for x in dg if len(x)])
    hi = np.sqrt(max(allv.max(), 1e-12)) * 1.05
    dd.plot([0, hi], [0, hi], color=L.GRID, lw=0.8)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#008300"]
    for k, x in enumerate(dg):
        x = np.sqrt(np.clip(np.asarray(x), 0, None))
        if len(x):
            dd.scatter(x[:, 0], x[:, 1], s=9, marker=DGM_MARK[k], c=cols[k], lw=0, zorder=3)
    dd.set_xlim(0, hi); dd.set_ylim(0, hi)
    dd.set_xlabel("birth", fontsize=8.5, labelpad=1); dd.set_ylabel("death", fontsize=8.5, labelpad=1)
    head(W - dw, top1 + 0.03, "(d) diagrams")
    ym = bot1 + sq / 2
    arrow((xa + sq + 0.05, ym), (xb - 0.07, ym))
    arrow((xb + bw + 0.05, ym), (xc - 0.07, ym))
    arrow((xc + sq + 0.05, ym), (xd - 0.25, ym))
    # ---- bottom row, from right to left: the twenty matrices, the four views, the consensus, the behaviors
    top2 = bot1 - mid - hd; bot2 = top2 - 0.12 - mh
    yv = bot2 + mh / 2
    xe = W - 4 * mc - 3 * mg
    for fi, f in enumerate(L.FIELDS):
        for s_i in range(4):
            m = ax_at(xe + s_i * (mc + mg), top2 - 0.12 - (fi + 1) * mc - fi * mg, mc, mc)
            Dm = Draw[f][s_i]                            # the raw distances, dark when close, one color range each
            m.imshow(Dm, cmap=L.SEQ.reversed(), vmin=0, vmax=np.quantile(Dm, 0.95), interpolation="nearest",
                     aspect="auto")
            if s_i == 0:
                fig.text((xe - 0.03) / W, (top2 - 0.12 - fi * (mc + mg) - mc / 2) / H, FSYM[f], ha="right",
                         va="center", fontsize=8.5)
            if fi == 0:
                fig.text((xe + s_i * (mc + mg) + mc / 2) / W, (top2 - 0.11) / H, f"$\\sigma_{s_i + 1}$",
                         ha="center", va="bottom", fontsize=8.5)
    head(W, top2 + 0.03, "(e) 20 distances", ha="right")
    vw = 2 * vc + vg
    xf = xe - lab - g3 - vw
    for k in range(4):
        r_, c_ = divmod(k, 2)
        v = ax_at(xf + c_ * (vc + vg), yv - vw / 2 + (1 - r_) * (vc + vg), vc, vc)
        v.imshow(views[k], cmap=L.SEQ, interpolation="nearest", aspect="auto")
        v.text(0.06, 0.94, f"$W_{k + 1}$", transform=v.transAxes, ha="left", va="top", fontsize=8.5,
               fontweight="bold", color="white")
    head(xf, top2 + 0.03, "(f) views")
    xg = xf - g2 - wc
    gS = ax_at(xg, yv - wc / 2, wc, wc)
    Sv = S[np.ix_(order, order)]
    gS.imshow(Sv, cmap=L.SEQ, vmin=0, vmax=np.quantile(Sv, 0.995), interpolation="nearest", aspect="auto")
    for b_ in np.cumsum(np.bincount(cons))[:-1]:
        gS.axhline(b_ - 0.5, color="white", lw=0.7); gS.axvline(b_ - 0.5, color="white", lw=0.7)
    pos_of = {int(r): k for k, r in enumerate(order)}
    for ex in tax["exemplars"]:
        p_ = pos_of[int(ex)]
        gS.scatter([p_], [p_], marker="*", s=45, c="#eda100", edgecolors=L.INK, linewidths=0.4, zorder=4)
    head(xg, top2 + 0.03, "(g) consensus $S^\\star$")
    ex_sorted = sorted(tax["exemplars"], key=lambda r: cons[r])
    xh1 = 3 * tw + 2 * tg
    xh0 = xg - g1 - xh1                                  # the block of exemplars ends a gap before the consensus
    hh = 2 * tw + 0.08                                   # two rows of exemplars, centered on the row
    for k, ex in enumerate(ex_sorted):
        r_, c_ = divmod(k, 3)
        x0 = xh0 + c_ * (tw + tg) + (0.5 * (tw + tg) if r_ == 1 else 0.0)
        y0 = yv + hh / 2 - (r_ + 1) * tw - r_ * 0.08
        h_ = ax_at(x0, y0, tw, tw)
        draw_tracks(h_, d["tracks"][ex]["pos"], L.MCOL[int(y[ex])], every=10)
        h_.collections[-1].set_sizes([2.5])
    head(xh0, top2 + 0.03, f"(h) $C={len(ex_sorted)}$ exemplars")
    arrow((xd + (dw - 0.17) / 2, bot1 - 0.03), (xd + (dw - 0.17) / 2, top2 + hd + 0.05), "distance",
          (xd - 0.04, (bot1 + top2 + 0.2) / 2), ha="right", va="center")
    arrow((xe - lab - 0.04, yv), (xf + vw + 0.07, yv), "ranks", ((xe - lab + xf + vw) / 2, yv + 0.04))
    arrow((xf - 0.05, yv), (xg + wc + 0.07, yv), "SNF", ((xf + xg + wc) / 2, yv + 0.04))
    arrow((xg - 0.05, yv), (xh0 + xh1 + 0.07, yv), "AP", ((xg + xh0 + xh1) / 2, yv + 0.04))
    # ---- the attribution, from the diagrams back to the critical points and the robots (a dashed bracket)
    PUR = "#5c3196"
    yl = top1 + hd + 0.16
    x_from, x_to = W - 0.05, xa + sq / 2
    fig.add_artist(plt.Line2D([x_from / W, x_from / W, x_to / W], [(top1 + hd + 0.03) / H, yl / H, yl / H],
                              transform=fig.transFigure, color=PUR, lw=1.1, ls=(0, (3, 2))))
    arrow((x_to, yl), (x_to, top1 + hd + 0.04), color=PUR)
    fig.text((x_from + x_to) / 2 / W, (yl + 0.03) / H, "attribution: back to the critical points and the robots",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=PUR)
    L.save(fig, "fig_method")


# --------------------------------------------------------------------------- the five missions
def _dodecagon(c=L.ARENA / 2, apothem=1.231):
    """The walls of the arena: a regular dodecagon of apothem 1.231 m, its sides facing the axes (the .argos files)."""
    r = apothem / np.cos(np.pi / 12)
    th = np.pi / 12 + np.arange(13) * np.pi / 6
    return np.column_stack([c + r * np.cos(th), c + r * np.sin(th)])


def _box(x0, x1, y0, y1, c=L.ARENA / 2):
    """A rectangle given in the frame of the simulator, as a polygon in the frame of the figures, where a position
    (x, y) of the simulator is drawn at (c - y, c + x) (pipeline/io.py)."""
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(c - yy, c + xx) for xx, yy in pts]


def fig_missions():
    """The arena of each mission, from the floor colors, walls and lights of the AutoMoDe loop functions and .argos
    files that simulate them (NEATvsEvo DirectionalGate, AggregationSingleSpot and SCAaroundLoopFunc, vanilla
    ForagingTwoSpots, complexity AggregationTwoSpotsXOR2), with the tracks of the exemplar run of each mission."""
    from matplotlib.patches import Polygon, Circle
    d = L.load_pickle(L.PAPER / "top1_tracks.pkl")
    y = d["labels"]
    c = L.ARENA / 2
    FLOOR, WALL = "#d3d2cc", "#3a3935"
    L.style()
    fig, axes = plt.subplots(1, 5, figsize=(L.width(), 1.32))
    dod = _dodecagon()

    def floor(ax, patch, color, clip):
        patch.set_facecolor(color); patch.set_edgecolor("none"); patch.set_zorder(1)
        ax.add_patch(patch); patch.set_clip_path(clip)

    def wall(ax, pts):
        ax.add_patch(Polygon(pts, closed=True, facecolor=WALL, edgecolor="none", zorder=4))

    def disk(ax, x, yy, r, clip):                       # a disk given in the frame of the simulator
        floor(ax, Circle((c - yy, c + x), r), "black", clip)

    def sun(ax, yy):                                    # the light, at (0, yy) in the simulator
        ax.scatter([c - yy], [c], s=40, marker="o", c="#f2c200", edgecolors=WALL, linewidths=0.6, zorder=5,
                   clip_on=False)

    for m, ax in enumerate(axes):
        arena = Polygon(dod, closed=True, facecolor=FLOOR, edgecolor="none", zorder=0)
        ax.add_patch(arena)
        if m == 0:                                      # Directional Gate: corridor, gate strip, two walls; light south
            floor(ax, Polygon(_box(-0.25, 0.25, 0.5 / 3, 1.4)), "black", arena)
            floor(ax, Polygon(_box(-0.25, 0.25, -0.5 / 3, 0.5 / 3)), "white", arena)
            for x_ in (-0.25, 0.25):
                wall(ax, _box(x_ - 0.0225, x_ + 0.0225, -0.5 / 3, 0.5 / 3 + 0.5 / 3))
            sun(ax, -1.5)
        elif m == 1:                                    # Foraging: two food sources, a white nest south of y = -0.6
            floor(ax, Polygon(_box(-1.4, 1.4, -1.4, -0.6)), "white", arena)
            for x_ in (0.75, -0.75):
                disk(ax, x_, 0.0, 0.15, arena)
            sun(ax, -1.25)
        elif m == 2:                                    # Homing: one black area; the robots start north of y = 0.6
            disk(ax, 0.0, -0.7, 0.3, arena)
            h_ = np.sqrt(1.2 ** 2 - 0.6 ** 2)
            ax.plot([c - 0.6, c - 0.6], [c - h_, c + h_], color=WALL, lw=0.8, ls=(0, (2.5, 1.5)), zorder=3)
        elif m == 3:                                    # Shelter: white shelter walled on three sides, two black areas
            floor(ax, Polygon(_box(-0.25, 0.25, -0.15, 0.15)), "white", arena)
            for x_ in (0.8, -0.8):
                disk(ax, x_, 0.0, 0.3, arena)
            wall(ax, _box(-0.2725, 0.2725, 0.15, 0.195))
            for x_ in (-0.2625, 0.2625):
                wall(ax, _box(x_ - 0.0225, x_ + 0.0225, -0.15, 0.195))
            sun(ax, -1.25)
        else:                                           # XOR Aggregation: two black areas
            for x_ in (0.5, -0.5):
                disk(ax, x_, 0.0, 0.3, arena)
        ax.add_patch(Polygon(dod, closed=True, facecolor="none", edgecolor=WALL, lw=1.2, zorder=4))
        e = mission_exemplar(m)
        pos = d["tracks"][e]["pos"]
        for i in range(pos.shape[1]):
            ax.plot(pos[::4, i, 0], pos[::4, i, 1], color=L.MCOL[m], lw=0.35, alpha=0.75, zorder=2)
        ax.scatter(pos[-1, :, 0], pos[-1, :, 1], s=6, color=L.INK, edgecolors="white", linewidths=0.35, zorder=6)
        ax.set_xlim(-0.06, L.ARENA + 0.36); ax.set_ylim(-0.06, L.ARENA + 0.06); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(L.MSHORT[m], fontsize=9.5, pad=2)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.86, wspace=0.03)
    L.save(fig, "fig_missions")


# --------------------------------------------------------------------------- field ablation
COMBO_LABEL = {("rho",): "$\\rho$", ("K",): "$K$", ("phi",): "$\\varphi$", ("psi",): "$\\psi$", ("J",): "$J$",
               ("rho", "K"): "$\\rho+K$", ("phi", "psi"): "$\\varphi+\\psi$", ("phi", "psi", "J"): "$\\varphi+\\psi+J$",
               ("K", "phi", "psi", "J"): "$K+\\varphi+\\psi+J$", ("rho", "K", "phi", "psi"): "$\\rho+K+\\varphi+\\psi$",
               ("rho", "K", "phi", "psi", "J"): "$\\rho+K+\\varphi+\\psi+J$"}


def fig_ablation():
    A = L.load_pickle(L.PAPER / "ablation.pkl")
    combos = [c for c in COMBO_LABEL if all((c, top) in A for top in range(1, 11))]
    per = {c: {m: [np.array([r[m] for r in A[(c, top)]]) for top in range(1, 11)] for m in ("purity", "ari")} for c in combos}
    vals = {c: {m: np.concatenate(per[c][m]) for m in per[c]} for c in combos}
    order = sorted(combos, key=lambda c: np.median(vals[c]["purity"]))
    L.style()
    fig, axes = plt.subplots(1, 2, figsize=(L.width(), 2.6), sharey=True)
    for ax, metric, title in ((axes[0], "purity", "purity"), (axes[1], "ari", "adjusted Rand index")):
        data = [vals[c][metric] for c in order]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.62, patch_artist=True, showfliers=True, notch=True,
                        conf_intervals=[L.median_interval(per[c][metric]) for c in order],
                        medianprops=dict(color=L.INK, lw=1.6),
                        whiskerprops=dict(color=L.INK2, lw=0.9), capprops=dict(color=L.INK2, lw=0.9),
                        flierprops=dict(marker="o", markersize=2, markerfacecolor=L.MUTED, markeredgecolor="none", alpha=0.5))
        for patch, c in zip(bp["boxes"], order):
            full = len(c) == 5
            patch.set_facecolor("#3987e5" if full else "#cde2fb"); patch.set_edgecolor("#184f95" if full else "#6da7ec")
        for k, c in enumerate(order):
            med = np.median(vals[c][metric])
            ax.text(1.03, k + 1, f"{med:.3f}", transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=8.5, fontweight="bold" if len(c) == 5 else "normal")
            if metric == "purity":       # beside the median, the settings in which no run is misassigned: the median hides them
                ax.text(1.40, k + 1, str(int((vals[c]["purity"] > 1 - 1e-9).sum())), transform=ax.get_yaxis_transform(),
                        va="center", ha="left", fontsize=8.5, fontweight="bold" if len(c) == 5 else "normal")
        if metric == "purity":
            ax.text(1.14, len(order) + 0.62, "median", transform=ax.get_yaxis_transform(), va="bottom", ha="center",
                    fontsize=8.5, fontweight="bold")
            ax.text(1.475, len(order) + 0.62, "error-\nfree", transform=ax.get_yaxis_transform(), va="bottom", ha="center",
                    fontsize=8.5, fontweight="bold", linespacing=0.95)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=L.GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(1, len(order) + 1)); axes[0].set_yticklabels([COMBO_LABEL[c] for c in order], fontsize=9.5)
    axes[0].set_xlim(0.45, 1.01); axes[1].set_xlim(-0.02, 1.01)
    for ax in axes:
        L.bold_ticks(ax, "x")
    fig.subplots_adjust(left=0.215, right=0.905, bottom=0.1, top=0.865, wspace=0.62)
    L.save(fig, "fig_ablation")


# --------------------------------------------------------------------------- appendix: pooled controllers
def fig_pooled(tops=(7, 10)):
    G = _grid(); T = L.load_pickle(L.PAPER / "taxonomy.pkl")
    L.style()
    fig, axes = plt.subplots(1, len(tops), figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[T[t]["C"] for t in tops]))
    for ax, top in zip(np.atleast_1d(axes), tops):
        y = G[top]["labels"]; cons = T[top]["labels"]
        cm = contingency_matrix(y, cons)
        order = np.lexsort((-cm.max(0), cm.argmax(0)))          # group clusters by dominant mission
        cm = cm[:, order]
        ax.imshow(cm, cmap=L.SEQ, vmin=0, vmax=cm.max(), aspect="auto")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > 0.55 * cm.max() else L.INK, fontweight="bold")
        ax.set_yticks(range(5)); ax.set_yticklabels(L.MSHORT if top == tops[0] else [], fontsize=9)
        ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels([str(k + 1) for k in range(cm.shape[1])], fontsize=8.5)
        ax.set_xlabel("discovered cluster")
        ax.set_title(f"Top-{top}: $C^\\star$ = {T[top]['C']}", loc="left")
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.17, top=0.88, wspace=0.08)
    L.save(fig, "figA_pooled")



# --------------------------------------------------------------------------- thumbnails of Figure 1
def fig_thumbs(si=2, knn=7, t=10):
    """Small pictures placed in the boxes of the overview (Figure 1): one per step of the chain."""
    import run_sweep as RS
    from classification import fusion as F
    tax = L.load_pickle(L.PAPER / "taxonomy.pkl")[1]
    d = L.load_pickle(L.PAPER / "top1_tracks.pkl")
    cache = L.load_pickle(L.PAPER / "display_fields.pkl")
    idx = d["idx"]; y = d["labels"]
    e = mission_exemplar(3)                                          # the Shelter exemplar
    g_e = int(idx[e])
    st2 = L.load_pickle(L.RES / "stage2_alpha_dgms" / f"traj{g_e:04d}_s{si}.pkl")
    D, _ = RS.load_distances()
    Q = F.rank_normalise(D, F.ALL_FIELDS, idx)
    del D
    S = F.consensus(F.scale_views(Q), knn, t)
    cons = tax["labels"]; order = np.argsort(cons, kind="stable")
    L.style()
    W, H = 0.80, 0.64

    def new(w=W, h=H):
        fig = plt.figure(figsize=(w, h))
        return fig

    def bare(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def out(fig, name):
        fig.savefig(L.FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.005, metadata=L.PDF_META)
        plt.close(fig)

    fig = new(); ax = fig.add_axes([0, 0, 1, 1])
    draw_tracks(ax, d["tracks"][e]["pos"], L.MCOL[3], every=6)
    ax.set_aspect("equal"); out(fig, "thumb_tracks")

    fig = new(); F_ = cache[("ladder", e, si)]
    for k, f in enumerate(L.FIELDS):
        r_, c_ = divmod(k, 3)
        ax = fig.add_axes([c_ / 3 + 0.005, 0.5 - r_ * 0.5 + 0.01, 1 / 3 - 0.01, 0.48]); L.show_field(ax, F_[f], f)
    out(fig, "thumb_fields")

    fig = new(); ax = fig.add_axes([0.08, 0.08, 0.84, 0.88])
    dg = st2["dgms"]["rho"]; allv = np.concatenate([np.asarray(x).ravel() for x in dg if len(x)])
    hi = np.sqrt(max(allv.max(), 1e-12)) * 1.05
    ax.plot([0, hi], [0, hi], color=L.MUTED, lw=0.6)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#008300"]
    for k, x in enumerate(dg):
        x = np.sqrt(np.clip(np.asarray(x), 0, None))
        if len(x):
            ax.scatter(x[:, 0], x[:, 1], s=9, marker=DGM_MARK[k], c=cols[k], lw=0)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    out(fig, "thumb_diagram")

    for name, M in (("thumb_distance", 1.0 - Q["rho"][si]), ("thumb_consensus", S)):
        fig = new(); ax = fig.add_axes([0.1, 0, 0.8, 1]); bare(ax)
        Mv = M[np.ix_(order, order)]
        ax.imshow(Mv, cmap=L.SEQ, vmin=0, vmax=np.quantile(Mv, 0.995), interpolation="nearest", aspect="equal")
        if name == "thumb_consensus":
            pos_of = {int(r): k for k, r in enumerate(order)}
            for ex in tax["exemplars"]:
                p = pos_of[int(ex)]
                ax.scatter([p], [p], marker="*", s=22, c="#eda100", edgecolors=L.INK, linewidths=0.3, zorder=4)
        out(fig, name)

    fig = new()
    ex_sorted = sorted(tax["exemplars"], key=lambda r: cons[r])
    for k, ex in enumerate(ex_sorted):
        r_, c_ = divmod(k, 3)
        ax = fig.add_axes([c_ / 3 + 0.01 + (1 / 6 if r_ == 1 else 0), 0.5 - r_ * 0.5 + 0.02, 1 / 3 - 0.02, 0.46])
        draw_tracks(ax, d["tracks"][ex]["pos"], L.MCOL[int(y[ex])], every=10)
        ax.set_aspect("equal")
    out(fig, "thumb_exemplars")
    print("saved thumbnails")


# --------------------------------------------------------------------------- attribution
ATTR_CMAP = LinearSegmentedColormap.from_list(
    "attr", ["#f7f3fb", "#e4d8f1", "#c7aee3", "#a383cf", "#7f57b6", "#5c3196", "#3b1260"])
CRIT_C, ORD_C = "#5c3196", "#1baf7a"
TMARK = {"max": "^", "min": "v", "saddle": "X"}


def _attr_seeds(ax, F, S, T, g):
    L.show_field(ax, F, "psi")
    ax.scatter(S[:, 0], S[:, 1], s=1.0, c=L.MUTED, lw=0, zorder=3)
    share = np.abs(g) / np.abs(g).sum()
    for v in np.where(share > 0)[0][np.argsort(share[share > 0])]:
        crit = T[v] in TMARK
        ax.scatter(S[v, 0], S[v, 1], s=10 + 650 * share[v], marker=TMARK.get(T[v], "o"),
                   c=CRIT_C if crit else ORD_C, edgecolors="white", linewidths=0.7, zorder=5)


def _attr_map(ax, M, domain, sqrt=False, tracks=None):
    """Attribution map, white outside the observed domain of the field (as in the field panels)."""
    n = M.shape[0]
    rows = np.clip(((np.arange(n) + 0.5) / n * domain.shape[0]).astype(int), 0, domain.shape[0] - 1)
    inside = domain[np.ix_(rows, rows)]
    vals = np.maximum(M, 0) / max(M[inside].max(), 1e-30)
    if sqrt:
        vals = np.sqrt(vals)
    im = ax.imshow(np.ma.masked_where(~inside, vals), origin="lower",
                   extent=(0, L.ARENA, 0, L.ARENA), cmap=ATTR_CMAP, vmin=0, vmax=1, interpolation="bilinear", zorder=1)
    if tracks is not None:
        for i in range(tracks.shape[1]):
            ax.plot(tracks[::6, i, 0], tracks[::6, i, 1], color="#6f7782", lw=0.25, alpha=0.35, zorder=2)
    ax.set_xlim(0, L.ARENA); ax.set_ylim(0, L.ARENA); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def fig_attribution():
    A = L.load_pickle(L.PAPER / "attribution.pkl")
    cache = L.load_pickle(L.PAPER / "display_fields.pkl")
    pos_A = L.load_pickle(L.PAPER / "top1_tracks.pkl")["tracks"][int(A["run_A"][0])]["pos"]
    F = cache[("ladder", int(A["run_A"][0]), int(A["scale"]))][A["field"]]
    S, T = A["seeds_A"], A["types_A"]
    L.style()
    # rows exactly as tall as the square maps, so that no empty band separates them
    fig, axes = plt.subplots(2, 3, figsize=(L.width(0.78), 3.2))
    rows = ("persistence", "values")
    for r, reader in enumerate(rows):
        _attr_seeds(axes[r, 0], F, S, T, A[f"{reader}_share_A"])
        _attr_map(axes[r, 1], A[f"{reader}_robot_map_A"], np.isfinite(F), sqrt=True, tracks=pos_A)
        im = _attr_map(axes[r, 2], A["reproducibility"][reader]["mean_map"], np.isfinite(F))
    for r, lab in enumerate(("extended\npersistence", "distribution\nof the values")):
        axes[r, 0].text(-0.07, 0.5, lab, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center",
                        fontsize=10, fontweight="bold", linespacing=1.1)
    for c, head in enumerate(("share on\nthe seeds", "back to the\nrobots (sqrt)", f"mean over\n{len(A['reproducibility']['runs'])} runs")):
        axes[0, c].set_title(head, fontsize=9.5, fontweight="bold")
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    for k, ax in enumerate(axes.ravel()):                 # in the empty corner, outside the observed area
        ax.text(0.01, 0.99, f"({'abcdef'[k]})", transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                fontweight="bold", path_effects=halo, zorder=9)
    name = {"max": "rotation\ncenter", "min": "rotation\ncenter"}
    sh, ext = A["persistence_share_A"], np.where(np.isin(T, ("min", "max")))[0]
    ends = [(S[v], str(T[v])) for v in ext[np.argsort(sh[ext])[-2:]]]      # the two extrema with the largest shares
    for k, (p, t) in enumerate(ends):
        other = ends[1 - k][0] if len(ends) == 2 else np.array([L.ARENA / 2, L.ARENA / 2])
        away = p - other if np.linalg.norm(p - other) > 1e-9 else np.array([1.0, 0.0])
        away = away / np.linalg.norm(away) + np.array([0.8, 0.0])   # away from the other point, and to the right, clear of
        off = 0.55 * away / np.linalg.norm(away)                     # the panel letter in the top-left corner
        q = np.array([np.clip(p[0] + off[0], 0.35, L.ARENA - 0.35), np.clip(p[1] + off[1], 0.35, L.ARENA - 0.45)])
        axes[0, 0].annotate(name[t], xy=p, xytext=q, fontsize=8.5, fontweight="bold", color=L.INK,
                            ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.8, color=L.INK),
                            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")], zorder=8)
    handles = [plt.Line2D([], [], ls="", marker=TMARK[t], ms=7, mfc=CRIT_C, mec="white", label=n_)
               for t, n_ in (("max", "maximum"), ("min", "minimum"), ("saddle", "saddle"))]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=7, mfc=ORD_C, mec="white", label="ordinary point"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, frameon=False, fontsize=8.5,
               handletextpad=0.2, columnspacing=1.0)
    cax = fig.add_axes([0.65, 0.087, 0.32, 0.027])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    L.bold_colorbar(cb, "attribution (max = 1)", size=8.5)
    fig.subplots_adjust(left=0.1, right=1.0, top=0.875, bottom=0.166, wspace=0.05, hspace=0.092)
    L.save(fig, "fig_attribution")


# --------------------------------------------------------------------------- XOR runs in the Shelter cluster
def fig_merge(top=4):
    from scipy.ndimage import uniform_filter1d
    M = L.load_pickle(L.PAPER / "merge_analysis.pkl")[top]["dynamics"]
    L.style()
    fig, axes = plt.subplots(1, 4, figsize=(L.width(), 2.3), gridspec_kw=dict(width_ratios=[1.6, 1, 1, 1]))
    style = {"absorbed": ("#8e2c64", "XOR runs with Shelter"), "kept": (L.MCOL[4], "other XOR runs"),
             "shelter": (L.MCOL[3], "Shelter runs")}
    ax = axes[0]
    for g in ("absorbed", "kept", "shelter"):
        col, name = style[g]
        m = uniform_filter1d(M[g]["moving"], 10)[5:]
        tt = (np.arange(len(m)) + 5) * 0.1
        ax.plot(tt, m, color=col, lw=1.8, label=name)
    ax.set_xlabel("time (s)"); ax.set_ylabel("robots moving")
    ax.set_ylim(-0.02, 1.1); ax.set_yticks([0, 0.5, 1])
    L.bold_ticks(ax)
    ax.set_title("(a) when the robots stop", fontsize=9.5)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=8.5, handlelength=1.6, columnspacing=1.6)
    for k, g in enumerate(("absorbed", "kept", "shelter")):
        ax = axes[k + 1]
        P = M[g]["final"]
        ax.hist2d(P[:, 0], P[:, 1], bins=36, range=[[0, L.ARENA], [0, L.ARENA]], cmap=L.SEQ)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[1].set_title("(b) XOR runs\nwith Shelter", fontsize=9.5, linespacing=1.0)
    axes[2].set_title("(c) other\nXOR runs", fontsize=9.5, linespacing=1.0)
    axes[3].set_title("(d) Shelter\nruns", fontsize=9.5, linespacing=1.0)
    fig.subplots_adjust(left=0.096, right=1.0, top=0.82, bottom=0.27, wspace=0.12)
    L.save(fig, "figA_merge")



# --------------------------------------------------------------------------- appendix: the top of the ladder
def fig_cap():
    """How decisive and how stable the reading of Top-1 is against the top of the ladder, label-free:
    plateau and stability for caps sigma_max = (D_s / 2.35) / k, on the final domain (no robot position
    farther than sigma/2 from a seed) and on the former fixed coreset of 300 positions (k = 1: no cap)."""
    import pickle as _pk

    def study(name):
        f = L.ROOT / "results" / name / "summary.pkl"
        return _pk.load(open(f, "rb")) if f.exists() else {}
    fin, ext, main = study("parameter_study_final"), study("parameter_study_ext"), study("parameter_study")
    lad = L.load_pickle(L.RES / "scale_ladder.pkl")
    merge = 100 * float(lad["ladder"][-1]) * float(lad.get("max_factor", 10.0))       # D_s / 2.35, cm
    final = {int(k[1][3:]): r for k, r in fin.items() if k[0] == "ladder" and str(k[1]).startswith("cap")}
    fixed = {int(k[1][3:]): r for k, r in ext.items() if k[0] == "ladder" and str(k[1]).startswith("cap")}
    if ("ladder", "paper") in main:
        fixed[10] = main[("ladder", "paper")]
    unc = [r for k, r in ext.items() if k[0] == "ladder" and k[1] == "uncapped4"]
    if unc:
        fixed[1] = unc[0]
    sys.path.insert(0, str(L.ROOT))
    from pipeline.config import SIGMA_MAX_FACTOR
    chosen = float(SIGMA_MAX_FACTOR)                                                 # the cap of the method
    same = sorted(k for k, r in final.items() if r["ari_consensus"] > 0.999999)
    L.style()
    fig, axes = plt.subplots(1, 2, figsize=(L.width(), 2.9))
    for ax, key, label in ((axes[0], "plateau", "modal region, settings of 99"),
                           (axes[1], "stability", "stability, mean ARI")):
        if same:
            ax.axvspan(merge / max(same), merge / min(same), color="#e8f0fb", zorder=0)
        for data, name, col, mk, ls in ((fixed, "fixed subset of 300 positions", L.MUTED, "s", "--"),
                                        (final, "seeds within $\\sigma/2$ (final)", "#184f95", "o", "-")):
            ks = sorted(data, reverse=True)
            x = [merge / k for k in ks]
            y = [data[k][key] for k in ks]
            ax.plot(x, y, ls=ls, color=col, lw=1.8, marker=mk, ms=6.5, label=name, zorder=3)
        ax.axvline(merge / chosen, color=L.INK, lw=1.0, ls=":", zorder=2)
        ax.set_xscale("log")
        ticks = sorted({merge / k for k in list(final) + list(fixed)})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.0f}" if t >= 10 else f"{t:.1f}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("top of the ladder $\\sigma_{\\max}$ (cm)")
        ax.set_title(label, loc="left")
        ax.grid(axis="y", color=L.GRID, lw=0.6)
        ax.set_axisbelow(True)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontweight("bold")
    # the plateaus obtained when the resolution settings are varied alone, on the domain of the method
    res = [r["plateau"] for st in (fin, ext) for k, r in st.items()
           if k[0] in ("cover", "frac", "n_bis", "n_dirs", "window", "n_jacobi")]
    if res:
        axes[0].axhspan(min(res), max(res), color="#dcdcdc", lw=0, zorder=1)
        axes[0].text(max(merge / k for k in list(final) + list(fixed)), min(res) - 0.6, "resolution settings",
                     ha="right", va="top", fontsize=8.5, color=L.INK)
    hi = axes[0].get_ylim()[1]
    axes[0].text(merge / chosen * 1.04, hi, "chosen", va="top", ha="left", fontsize=8.5)
    if same:                                   # in the free corner of (a), right of the cap of the method
        axes[0].text(merge / chosen * 1.06, 0.03, "same\ntaxonomy", ha="left", va="bottom",
                     fontsize=8.5, color="#184f95", transform=axes[0].get_xaxis_transform(), linespacing=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    L.panel_label(axes[0], "a"); L.panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.31, wspace=0.27)
    L.save(fig, "figA_cap")

# --------------------------------------------------------------------------- the stable states of the map
STATE_COLORS = {"finer": "#cde2fb", "missions": "#efeeea", "partial": "#dcdad3", "shelter_xor": "#f6b896",
                "families": "#eb6834", "two_families": "#8c2f0c"}


def _wide_maps():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import stable_states as SS
    from pipeline.config import BEHAVIORS
    maps = {n: L.load_pickle(L.PAPER / f"stable_states_scaled_top{n}.pkl") for n in range(1, 11)}
    return SS, BEHAVIORS, maps


def fig_states():
    """The stable states of the resolution map of every Top-n, over knn >= 3 up to 50 n and t from 1 to 50:
    the color gives the state a setting reads, a black line marks a transition (the state changes from one setting
    to the next), the dotted frame the local grid, the dashed line the edge of purity 1."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    SS, BEH, maps = _wide_maps()
    order = list(SS.STATES)
    L.style()
    fig, axes = plt.subplots(2, 5, figsize=(L.width(), 4.05), sharey=True)
    for ax, n in zip(axes.ravel(), range(1, 11)):
        d = maps[n]
        y, ts = np.asarray(d["labels"]), d["ts"]
        knns = [k for k in d["knns"] if k >= 3]
        by = {(r["knn"], r["t"]): np.asarray(r["pred"]) for r in d["results"]}
        S_ = np.array([[order.index(SS.state_of(by[(k, t)], y, BEH)) for k in knns] for t in ts])
        pure = np.array([[float(all(len(np.unique(y[by[(k, t)] == c])) == 1 for c in np.unique(by[(k, t)])))
                          for k in knns] for t in ts])
        left_edge = max(2.5, knns[0] - (knns[1] - knns[0]) / 2)
        ek = np.r_[left_edge, (np.array(knns[1:]) + np.array(knns[:-1])) / 2, knns[-1] + (knns[-1] - knns[-2]) / 2]
        et = np.r_[0.5, (np.array(ts[1:]) + np.array(ts[:-1])) / 2, ts[-1] + 0.5]
        ax.pcolormesh(ek, et, S_, cmap=ListedColormap([STATE_COLORS[s] for s in order]), vmin=-0.5,
                      vmax=len(order) - 0.5, shading="flat", rasterized=True)
        hatch = np.ma.masked_where(S_ != order.index("partial"), np.ones_like(S_, dtype=float))
        ax.pcolor(ek, et, hatch, hatch="/////", alpha=0.0, shading="flat")
        seg = [[(ek[j + 1], et[i]), (ek[j + 1], et[i + 1])] for i in range(len(ts)) for j in range(len(knns) - 1)
               if S_[i, j] != S_[i, j + 1]]
        seg += [[(ek[j], et[i + 1]), (ek[j + 1], et[i + 1])] for i in range(len(ts) - 1) for j in range(len(knns))
                if S_[i, j] != S_[i + 1, j]]
        ax.add_collection(LineCollection(seg, colors=L.INK, linewidths=0.55))
        if pure.any() and not pure.all():
            ax.contour(knns, ts, pure, levels=[0.5], colors="#184f95", linewidths=1.0, linestyles="--")
        ax.plot([ek[0], 25.5, 25.5], [25.5, 25.5, 0.5], color=L.INK, lw=0.9, ls=":")          # the local grid
        ax.set_xlim(ek[0], ek[-1]); ax.set_ylim(et[0], et[-1])
        ax.set_xticks([knns[0], knns[-1] // 2, knns[-1]]); ax.set_yticks([1, 25, 50])
        ax.set_title(f"Top-{n}", loc="left", fontsize=9, pad=3)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(length=0, pad=1.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("$t$", labelpad=1)
    for ax in axes[1]:
        ax.set_xlabel(r"$k_{\mathrm{nn}}$", labelpad=0)
    names = {"finer": "finer than the missions", "missions": "the five missions", "partial": "a mission partly merged",
             "shelter_xor": "Shelter + XOR", "families": "D-Gate + Foraging | Homing | Shelter + XOR",
             "two_families": "D-Gate + Foraging + Homing | Shelter + XOR"}
    patch = lambda s: Patch(facecolor=STATE_COLORS[s], hatch="/////" if s == "partial" else None,
                            edgecolor=L.INK if s == "partial" else STATE_COLORS[s], lw=0, label=names[s])
    h = [patch(s) for s in ("finer", "missions", "partial", "shelter_xor")]
    h.append(Line2D([], [], color="#184f95", lw=1.0, ls="--", label="edge of purity 1"))
    h += [patch(s) for s in ("families", "two_families")]
    h += [Line2D([], [], color=L.INK, lw=0.8, label="transition"), Line2D([], [], color=L.INK, lw=0.9, ls=":", label="local grid")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8.5, handlelength=1.3, columnspacing=1.4,
               handletextpad=0.4, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.95, bottom=0.33, wspace=0.25, hspace=0.42)
    L.save(fig, "fig_states")


def fig_law():
    """The law of the transition to the three families: (a) the knn/n at which it occurs against t, log-log, with the
    power law fitted on Top-2 to Top-7; (b) the exponent of every dataset against the range of the fitted ones."""
    SS, BEH, maps = _wide_maps()
    FIT = (2, 3, 4, 5, 6, 7)
    tr = {n: SS.transition(maps[n], BEH) for n in range(1, 11)}
    c, a = SS.power_law(np.vstack([tr[n] for n in FIT]))
    per = {n: SS.power_law(tr[n]) for n in range(1, 11)}
    lo, hi = min(per[n][1] for n in FIT), max(per[n][1] for n in FIT)
    group = {1: ("Top-1 (single controller)", L.INK, True)}
    for n in FIT:
        group[n] = ("fitted: Top-2 to Top-7", "#184f95", True)
    for n in range(8, 11):
        group[n] = ("with atypical controllers: Top-8 to Top-10", "#eb6834", False)
    L.style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(L.width(), 2.65), gridspec_kw=dict(width_ratios=[1.35, 1]))
    seen = set()
    for n in range(1, 11):
        name, col, filled = group[n]
        Q = tr[n]
        ax.plot(Q[:, 0], Q[:, 1], "o", ms=3.0, color=col, mfc=col if filled else "none", mew=0 if filled else 0.8,
                label=None if name in seen else name, zorder=3)
        seen.add(name)
    tt = np.geomspace(5, 50, 50)
    ax.plot(tt, c * tt ** -a, color=L.INK, lw=1.2, ls="--", zorder=4)
    ax.plot(tt, per[1][0] * tt ** -per[1][1], color=L.MUTED, lw=1.2, ls=":", zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([5, 10, 20, 50]); ax.set_xticklabels(["5", "10", "20", "50"])
    ax.set_yticks([10, 20, 50]); ax.set_yticklabels(["10", "20", "50"]); ax.minorticks_off()
    ax.set_xlim(4.5, 56); ax.set_ylim(9.5, 80)
    ax.set_xlabel("diffusion iterations $t$"); ax.set_ylabel(r"$k_{\mathrm{nn}}/n$ at the transition")
    ax.text(5.0, 10.4, f"$k_{{\\mathrm{{nn}}}}/n = {c:.0f}\\,t^{{-{a:.2f}}}$", fontsize=8.5)
    ax.text(14, 64, f"Top-1: ${per[1][0]:.0f}\\,t^{{-{per[1][1]:.2f}}}$", fontsize=8.5, color=L.INK2)
    ax.set_title("(a) transition to the three families", loc="left", fontsize=9)
    bx.axhspan(lo, hi, color="#e8f0fb", zorder=0)
    for n in range(2, 11):
        name, col, filled = group[n]
        bx.plot(n, per[n][1], "o", ms=5, color=col, mfc=col if filled else "none", mew=1.1, zorder=3)
    bx.set_xticks(range(2, 11)); bx.set_xlim(1.4, 10.6); bx.set_ylim(0.5, 0.71)
    bx.set_yticks([0.5, 0.55, 0.6, 0.65, 0.7])
    bx.set_xlabel("Top-$n$"); bx.set_ylabel(r"exponent of $t$")
    bx.text(1.65, lo - 0.006, "Top-2 to Top-7 range", fontsize=8.5, color="#184f95", va="top")
    bx.set_title("(b) exponent per dataset", loc="left", fontsize=9)
    for a_ in (ax, bx):
        for lab in a_.get_xticklabels() + a_.get_yticklabels():
            lab.set_fontweight("bold")
        a_.grid(color=L.GRID, lw=0.6); a_.set_axisbelow(True)
    _hs, _ls = ax.get_legend_handles_labels()
    _ord = [0, 2, 1]
    fig.legend([_hs[i] for i in _ord], [_ls[i] for i in _ord], loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.36, wspace=0.34)
    L.save(fig, "fig_law")


ALL = {
    "pipeline": lambda: (fig_pipeline_one(), fig_pipeline_all()),
    "fields": lambda: fig_fields(),
    "resolution": lambda: (fig_resolution(), fig_resolution_compact()),
    "fusion": lambda: fig_fusion_sequence(),
    "states": lambda: (fig_states(), fig_law()),
    "ablation": lambda: fig_ablation(),
    "pairs": lambda: fig_pair_separation(),
    "scales": lambda: (fig_scales(), fig_uncapped()),
    "cap": lambda: fig_cap(),
    "pooled": lambda: fig_pooled(),
    "attribution": lambda: fig_attribution(),
    "merge": lambda: fig_merge(),
    "thumbs": lambda: fig_thumbs(),
    "method": lambda: fig_method(),
    "missions": lambda: fig_missions(),
}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for name, fn in ALL.items():
        if "all" in which or name in which:
            try:
                fn()
            except FileNotFoundError as err:
                print(f"skip {name}: missing input ({err.filename})")
    print("sizes on the page:")
    L.report()
