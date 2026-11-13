#!/usr/bin/env python3
"""
illustrations.py - didactic figures of the method, on small synthetic configurations
====================================================================================
Each figure shows one step of Sections 3-4 on a toy swarm whose fields are computed with the same
closed forms as the pipeline (Gaussian footprint, velocity moments, potentials through kappa_sigma,
Jacobi coupling). The configurations are synthetic and serve only to make the equations visible.

Usage: python analysis/illustrations.py [moments] [potentials] [persistence]

Author: Olivier Vertu Ndinga Oba
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as L  # noqa: E402

ARROW = "#3b1260"


def _grid(lx, ly, n=220):
    xs = np.linspace(0, lx, n)
    ys = np.linspace(0, ly, int(round(n * ly / lx)))
    return np.meshgrid(xs, ys)


def _clean(ax, lx, ly):
    ax.set_xlim(0, lx); ax.set_ylim(0, ly); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- 0. the Gaussian footprint
def fig_footprint():
    """One agent becomes a smooth bump of width sigma; two bumps merge below 2.35 sigma; sigma acts as a lens."""
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    L.style()
    W, H = L.width(), 3.3
    box = lambda x0, x1, y0, y1: fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])   # inches
    fig = plt.figure(figsize=(W, H))
    # first row: one agent, its profile, two agents; second row: a group seen through two widths
    axs = [box(0.0, 1.35, 1.72, 3.07), box(1.78, 3.0, 2.2, 3.05), box(3.42, W, 2.2, 3.05),
           box(0.5, 2.05, 0.0, 1.55), box(W - 2.05, W - 0.5, 0.0, 1.55)]
    # (a) one agent: its footprint in the plane
    ax = axs[0]
    s_ = 1.0
    g = np.linspace(-5.6, 5.6, 300)
    X, Y = np.meshgrid(g, g)
    ax.imshow(np.exp(-(X ** 2 + Y ** 2) / 2), origin="lower", extent=(-5.6, 5.6, -5.6, 5.6), cmap=L.SEQ, vmin=0, vmax=1)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(s_ * np.cos(th), s_ * np.sin(th), color="white", lw=1.2, ls="--")
    ax.plot(5 * np.cos(th), 5 * np.sin(th), color=L.INK2, lw=0.9, ls=":")
    ax.scatter([0], [0], s=14, color=L.INK, zorder=5)
    ax.annotate("", xy=(s_, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="white", lw=1.0, mutation_scale=7))
    ax.text(0.5, 1.25, "$\\sigma$", color=L.INK, fontsize=10, ha="center", va="bottom", fontweight="bold", path_effects=halo)
    ax.text(0, -4.3, "window $5\\sigma$", color=L.INK2, fontsize=8.5, ha="center", va="center", path_effects=halo)
    ax.set_xlim(-5.6, 5.6); ax.set_ylim(-5.6, 5.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) one agent", fontsize=9.5)
    # (b) its profile
    ax = axs[1]
    d = np.linspace(0, 5.6, 300)
    ax.plot(d, np.exp(-d ** 2 / 2), color=L.INK, lw=1.6)
    ax.axvline(1, color=L.MCOL[0], lw=0.9, ls="--"); ax.axvline(5, color=L.INK2, lw=0.9, ls=":")
    ax.text(1.15, 1.0, "$\\sigma$", color=L.MCOL[0], fontsize=9.5, fontweight="bold", va="top")
    ax.text(4.85, 1.0, "$5\\sigma$", color=L.INK2, fontsize=9, ha="right", va="top")
    ax.text(4.85, 0.12, "$e^{-12.5}$\nleft out", color=L.INK2, fontsize=8.5, ha="right", va="bottom", linespacing=1.1)
    ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_xticklabels(["0", "", "", "", "", ""])
    ax.set_yticks([0, 1]); ax.set_ylim(-0.04, 1.08)
    L.bold_ticks(ax)
    ax.set_xlabel("distance\nto the agent", fontsize=8.5, labelpad=1, linespacing=1.0)
    ax.set_title("(b) its profile", fontsize=9.5)
    # (c) two agents: resolved or merged
    ax = axs[2]
    x = np.linspace(-4.5, 4.5, 400)
    for dd, col, lab in ((3.2, L.MCOL[0], "$d=3.2\\sigma$"), (2.0, L.MCOL[1], "$d=2\\sigma$")):
        f = np.exp(-(x - dd / 2) ** 2 / 2) + np.exp(-(x + dd / 2) ** 2 / 2)
        ax.plot(x, f, color=col, lw=1.6, label=lab)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2.05)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", handlelength=1.1, handletextpad=0.4, borderaxespad=0.05,
              labelspacing=0.2)
    ax.set_xlabel("unresolved when\n$d\\leq2.35\\,\\sigma$", fontsize=8.5, labelpad=2, linespacing=1.0)
    ax.set_title("(c) two agents", fontsize=9.5)
    # (d, e) a group through two lenses
    rng = np.random.default_rng(2)
    P = np.r_[rng.normal([0.32, 0.38], 0.05, (6, 2)), rng.normal([0.68, 0.62], 0.06, (7, 2))]
    gx = np.linspace(0, 1, 240)
    GX, GY = np.meshgrid(gx, gx)
    for ax, sig, title in ((axs[3], 0.03, "(d) small $\\sigma$"), (axs[4], 0.12, "(e) large $\\sigma$")):
        F = np.exp(-((GX[..., None] - P[:, 0]) ** 2 + (GY[..., None] - P[:, 1]) ** 2) / (2 * sig ** 2)).sum(-1)
        top = 1.3 if sig < 0.05 else F.max()                 # small sigma: scaled to one agent's bump
        ax.imshow(np.minimum(F / top, 1), origin="lower", extent=(0, 1, 0, 1), cmap=L.SEQ, vmin=0, vmax=1,
                  interpolation="bilinear")
        ax.scatter(P[:, 0], P[:, 1], s=1.5, color=L.INK, zorder=5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=9.5)
    L.save(fig, "fig_footprint")


# --------------------------------------------------------------------------- 1. from agents to fields
def fig_moments(sigma=0.065):
    """Gaussian footprints and the three velocity moments at one instant."""
    rest = np.array([[0.17, 0.24], [0.26, 0.21], [0.19, 0.33], [0.28, 0.31]])
    east = np.array([[0.40, 0.70], [0.55, 0.70], [0.70, 0.70]])
    west = np.array([[0.47, 0.74], [0.62, 0.74], [0.77, 0.74]])
    move = np.array([[0.86, 0.22], [0.95, 0.24], [0.90, 0.32]])
    X = np.vstack([rest, east, west, move])
    V = np.vstack([np.zeros_like(rest), np.tile([0.1, 0.0], (3, 1)), np.tile([-0.1, 0.0], (3, 1)),
                   np.tile([0.0, 0.1], (3, 1))])
    lx, ly = 1.15, 1.0
    gx, gy = _grid(lx, ly)
    O = np.exp(-((gx[..., None] - X[:, 0]) ** 2 + (gy[..., None] - X[:, 1]) ** 2) / (2 * sigma ** 2))
    M0 = O.sum(-1)
    M1 = np.stack([(O * V[:, 0]).sum(-1), (O * V[:, 1]).sum(-1)], -1)
    M2 = (O * (V ** 2).sum(1)).sum(-1)

    L.style()
    fig, axes = plt.subplots(1, 4, figsize=(L.width(), 2.05))
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 60)
    for x in X:
        ax.plot(x[0] + 2 * sigma * np.cos(th), x[1] + 2 * sigma * np.sin(th), color=L.MUTED, lw=0.5)
    ax.scatter(X[:, 0], X[:, 1], s=14, color=L.INK, zorder=5)
    ax.quiver(X[:, 0], X[:, 1], V[:, 0], V[:, 1], color=ARROW, angles="xy", scale_units="xy", scale=0.7,
              width=0.012, zorder=6)
    for txt, (x, y) in (("at rest", (0.22, 0.05)), ("crossing", (0.58, 0.92)), ("moving", (0.90, 0.05))):
        ax.text(x, y, txt, ha="center", va="center", fontsize=8.5, fontweight="bold", color=L.INK, linespacing=1.0)
    ax.set_title("(a) agents,\nvelocities", fontsize=9.5, linespacing=1.1)
    panels = ((M0, "(b) $M_0$\nwhere they are"), (np.linalg.norm(M1, axis=-1), "(c) $|\\vec{M}_1|$\nnet flux"),
              (M2, "(d) tr $\\mathbf{M}_2$\nagitation"))
    for ax, (F, title) in zip(axes[1:], panels):
        ax.imshow(F / F.max(), origin="lower", extent=(0, lx, 0, ly), cmap=L.SEQ, vmin=0, vmax=1,
                  interpolation="bilinear")
        ax.set_title(title, fontsize=9.5, linespacing=1.1)
    s = slice(None, None, 14)
    qx, qy, ux, uy = gx[s, s], gy[s, s], M1[s, s, 0], M1[s, s, 1]
    keep = np.hypot(ux, uy) > 0.15 * np.linalg.norm(M1, axis=-1).max()
    axes[2].quiver(qx[keep], qy[keep], ux[keep], uy[keep], color=ARROW, angles="xy", scale_units="xy",
                   scale=2.2, width=0.010)
    for ax in axes:
        _clean(ax, lx, ly)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.8, bottom=0.0, wspace=0.06)
    L.save(fig, "fig_moments")


# --------------------------------------------------------------------------- 2. potentials and coupling
def _fields(P, sigma, lx, ly, n=170, dt=0.1):
    from types import SimpleNamespace
    from gaussian_splatting.fields_cpu import FieldComputer
    pos = P[:-1].astype(np.float32)
    vel = ((P[1:] - P[:-1]) / dt).astype(np.float32)
    X, Y = _grid(lx, ly, n)
    grid = SimpleNamespace(X=X, Y=Y, shape=X.shape, sigma=float(sigma), n_vertices=X.size)
    return FieldComputer(fields=("rho", "phi", "psi", "J"), kappa=5.0, chunk_size=50).compute(pos, vel, grid)


def _mark_extrema(ax, F, key, lx, ly, frac=0.35):
    """Label the main extrema: sink (max of phi), source (min of phi), rotation centre (extremum of psi)."""
    import matplotlib.patheffects as pe
    ny, nx = F.shape
    G = np.nan_to_num(F, nan=0.0)
    m = np.nanmax(np.abs(F))
    marks = []
    if key == "phi":
        marks += [("sink", np.unravel_index(np.argmax(G), G.shape))] if G.max() > frac * m else []
        marks += [("source", np.unravel_index(np.argmin(G), G.shape))] if -G.min() > frac * m else []
    else:
        i = np.argmax(np.abs(G))
        marks.append(("rotation\ncenter", np.unravel_index(i, G.shape)))
    for label, (iy, ix) in marks:
        x, y = ix / (nx - 1) * lx, iy / (ny - 1) * ly
        ax.scatter([x], [y], s=26, marker="o", facecolor=L.INK, edgecolor="white", linewidth=0.8, zorder=6)
        ax.text(x, y - 0.11, label, ha="center", va="top", fontsize=8.5, fontweight="bold", color=L.INK, zorder=7,
                linespacing=1.0, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def _ring(n, centre, radii, turns=0.0, inward=None, T=300, phase=0.0):
    """n agents on evenly spaced angles around centre: circling (turns) or radial (inward)."""
    th0 = phase + (np.arange(n) * 2 * np.pi / n)[None]
    u = np.linspace(0, 1, T + 1)[:, None]
    if inward is None:
        R = np.asarray(radii)[np.arange(n) % len(radii)][None] * np.ones_like(u)
        th = th0 + 2 * np.pi * turns * u
    else:
        r_out, r_in = radii
        R = (r_out - (r_out - r_in) * u) if inward else (r_in + (r_out - r_in) * u)
        th = th0 * np.ones_like(u)
    return np.stack([centre[0] + R * np.cos(th), centre[1] + R * np.sin(th)], -1)


def coupling_pair(n=40, T=300):
    """The same two groups, n agents gathering and n agents milling, together or apart."""
    P, Q = (0.62, 0.5), (0.86, 0.5)
    gather = lambda c: _ring(n, c, (0.30, 0.04), inward=True, T=T)
    mill = lambda c: _ring(n, c, (0.15, 0.21), turns=1.5, phase=np.pi / n, T=T)
    together = np.concatenate([gather(P), mill(P)], axis=1)
    apart = np.concatenate([gather((0.36, 0.5)), mill(Q)], axis=1)
    return {"together": together, "apart": apart}


def coupling_distances(sigma=0.07, dt=0.1):
    """Run the pipeline (grid-free seeds, alpha complex, extended persistence, sliced Wasserstein) on the
    pair and return, per field, the distance between the two swarms relative to the size of their diagrams."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gaussian_splatting.seeds_gridfree import CPUPointComputer, gridfree_seeds_batch
    from pipeline import config as C
    from gaussian_splatting.alpha import AlphaSimplexTree
    from gaussian_splatting.persistence import PersistenceComputer
    from gaussian_splatting.distance import sample_directions, swd
    dirs = sample_directions(25)
    dg = {}
    for name, Pn in coupling_pair().items():
        pos = Pn[:-1].astype(np.float32); vel = ((Pn[1:] - Pn[:-1]) / dt).astype(np.float32)
        (seeds, fields), = gridfree_seeds_batch([pos], [vel], CPUPointComputer(fields=("rho", "K", "phi", "psi", "J")),
                                                sigma, cover=C.COVER, n_min=C.N_CORESET, frac_low_grad=C.FRAC_LOW_GRAD,
                                                n_jacobi=C.N_JACOBI, n_bis=C.N_BIS, dedup_tol=C.DEDUP_TOL, n_threads=1)
        ast = AlphaSimplexTree(seeds, alpha_sq_max=(3 * sigma) ** 2)
        dg[name] = PersistenceComputer(ast).compute_all({k: np.asarray(v).ravel() for k, v in fields.items()})
    empty = np.zeros((0, 2))
    sw = lambda A, B: float(sum(swd(A[s], [B[s]], dirs)[0] for s in range(6)))
    out = {}
    for f in ("rho", "K", "phi", "psi", "J"):
        a, b = dg["together"][f], dg["apart"][f]
        d = sw(a, b)
        size_a, size_b = sw(a, [empty] * 6), sw(b, [empty] * 6)
        size = 0.5 * (size_a + size_b)
        out[f] = dict(distance=d, size=size, relative=d / size, size_together=size_a, size_apart=size_b)
    return out


def fig_potentials(sigma=0.07):
    """Why the coupling J is needed: the same ingredients (a sink and a rotation centre), arranged
    differently. The diagrams of phi and psi change by a fifth and a third of their size (coupling_distances),
    the diagram of J completely: it is empty when the sink and the rotation centre coincide."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    lx, ly = 1.25, 1.0
    pair = coupling_pair()
    rows = (("sink and rotation\ncenter together", pair["together"]), ("sink and rotation\ncenter apart", pair["apart"]))
    F = [_fields(P, sigma, lx, ly, n=170) for _, P in rows]
    lim = {k: max(np.nanmax(np.abs(f[k])) for f in F) for k in ("phi", "psi", "J")}
    L.style()
    fig, axes = plt.subplots(2, 4, figsize=(L.width(), 3.1))
    heads = ("motion", "divergence\npotential $\\varphi_\\sigma$", "vorticity\npotential $\\psi_\\sigma$",
             "coupling\n$J_\\sigma$")
    for r, ((name, P), f) in enumerate(zip(rows, F)):
        ax = axes[r, 0]
        moving = np.where(np.ptp(P[:, :, 0], axis=0) + np.ptp(P[:, :, 1], axis=0) > 1e-9)[0]
        still = np.setdiff1d(np.arange(P.shape[1]), moving)
        for k in moving[::4]:
            ax.plot(P[:, k, 0], P[:, k, 1], color=L.INK2, lw=0.6, alpha=0.8)
            ax.annotate("", xy=P[-1, k], xytext=P[-6, k],
                        arrowprops=dict(arrowstyle="-|>", color=L.INK2, lw=0.6, mutation_scale=6))
        ax.scatter(P[0, still, 0], P[0, still, 1], s=7, color=L.MUTED, zorder=3)
        ax.text(-0.04, 0.5, name, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=9,
                fontweight="bold", linespacing=1.1)
        for c, key in enumerate(("phi", "psi", "J")):
            ax = axes[r, c + 1]
            ax.imshow(np.ma.masked_invalid(f[key]), origin="lower", extent=(0, lx, 0, ly), cmap=L.FCMAP[key],
                      vmin=-lim[key], vmax=lim[key], interpolation="bilinear")
            if key in ("phi", "psi"):
                _mark_extrema(ax, f[key], key, lx, ly)
        for ax in axes[r]:
            _clean(ax, lx, ly)
    for c, h in enumerate(heads):
        axes[0, c].set_title(h, fontsize=9.5, linespacing=1.1)
    fig.subplots_adjust(left=0.066, right=1.0, top=0.855, bottom=0.0, wspace=0.04, hspace=0.06)
    L.save(fig, "fig_potentials")


# --------------------------------------------------------------------------- 3. from a field to a distance
def fig_persistence(c=0.5):
    """Sublevel sets and diagram of a profile, then the sliced Wasserstein comparison of two diagrams."""
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    x = np.linspace(0, 1, 800)
    g = lambda m, w: np.exp(-((x - m) / w) ** 2)
    f = 0.66 - 0.56 * g(0.24, 0.10) - 0.36 * g(0.70, 0.085) + 0.36 * g(1.02, 0.13)
    loc_min = [k for k in range(1, len(x) - 1) if f[k] < f[k - 1] and f[k] < f[k + 1]]
    i1, i2 = sorted(loc_min, key=lambda k: f[k])[:2]                      # global minimum, second minimum
    lo, hi = sorted((i1, i2))
    isad = lo + int(np.argmax(f[lo:hi]))
    ord0, ext0 = (f[i2], f[isad]), (f[i1], f.max())

    L.style()
    W, H = L.width(), 3.2
    fig = plt.figure(figsize=(W, H))
    box = lambda x0, x1, y0, y1: fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])   # inches
    # first row: the sublevel sets of a profile and its diagram; second row: two diagrams and their matching
    axes = [box(0.0, 2.6, 1.95, 3.0), box(3.42, W - 0.05, 1.85, 3.0), box(0.22, 1.42, 0.27, 1.42),
            box(2.05, W - 0.05, 0.27, 1.42)]
    ax = axes[0]
    ax.plot(x, f, color=L.INK, lw=1.6)
    ax.axhline(c, color=L.MCOL[1], lw=1.0, ls="--")
    ax.fill_between(x, -0.02, 0.03, where=f <= c, color=L.MCOL[1], lw=0)
    ax.scatter([x[i1], x[i2], x[isad]], [f[i1], f[i2], f[isad]], s=18, color=L.INK, zorder=5)
    ax.text(x[i1] + 0.07, f[i1], "born", ha="left", va="center", fontsize=8.5)
    ax.text(x[i2], f[i2] - 0.07, "born", ha="center", va="top", fontsize=8.5)
    ax.text(x[isad], f[isad] + 0.05, "merge", ha="center", va="bottom", fontsize=8.5)
    ax.text(0.97, c - 0.05, "$F\\leq c$", ha="right", va="top", fontsize=8.5, color=L.MCOL[1], fontweight="bold",
            path_effects=halo)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.12); ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("(a) sublevel sets", fontsize=9.5)

    ax = axes[1]
    ax.plot([0, 1.1], [0, 1.1], color=L.MUTED, lw=0.8)
    ax.scatter(*ord0, s=28, color=L.MCOL[0], zorder=5)
    ax.scatter(*ext0, s=28, marker="D", color=L.MCOL[0], zorder=5)
    ax.text(ord0[0] + 0.06, ord0[1], "Ord$_0$", fontsize=9, va="center")
    ax.text(ext0[0] + 0.06, ext0[1], "Ext$_0$", fontsize=9, va="center")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.15); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("birth", fontsize=8.5, labelpad=2); ax.set_ylabel("death", fontsize=8.5, labelpad=2)
    ax.set_title("(b) diagram", fontsize=9.5)

    A = np.array([[0.15, 0.95], [0.40, 0.70], [0.55, 0.64]])
    B = np.array([[0.12, 0.85], [0.30, 0.74]])
    th = np.deg2rad(60); u = np.array([np.cos(th), np.sin(th)])
    pA, pB = np.c_[A.mean(1), A.mean(1)], np.c_[B.mean(1), B.mean(1)]      # diagonal projections
    ax = axes[2]
    ax.plot([0, 1.1], [0, 1.1], color=L.MUTED, lw=0.8)
    ax.scatter(A[:, 0], A[:, 1], s=26, color=L.MCOL[0], zorder=5)
    ax.scatter(B[:, 0], B[:, 1], s=32, marker="^", color=L.MCOL[1], zorder=5)
    ax.scatter(pB[:, 0], pB[:, 1], s=22, facecolor="white", edgecolor=L.MCOL[0], lw=1.0, zorder=6)
    ax.scatter(pA[:, 0], pA[:, 1], s=26, marker="^", facecolor="white", edgecolor=L.MCOL[1], lw=1.0, zorder=6)
    o = np.array([0.62, 0.08])
    ax.annotate("", xy=o + 0.42 * u, xytext=o, arrowprops=dict(arrowstyle="-|>", color=L.INK, lw=1.0))
    ax.text(*(o + 0.47 * u), "$\\theta$", fontsize=10, ha="center", va="bottom")
    ax.text(0.04, 1.08, "$D$", color=L.MCOL[0], fontsize=9, fontweight="bold", va="top")
    ax.text(0.20, 1.08, "$D'$", color=L.MCOL[1], fontsize=9, fontweight="bold", va="top")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.15); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("birth", fontsize=8.5, labelpad=2)
    ax.set_title("(c) two diagrams $D$, $D'$", fontsize=9.5)

    ax = axes[3]
    a = np.r_[A @ u, pB @ u]; fa = np.r_[np.zeros(len(A), bool), np.ones(len(pB), bool)]
    b = np.r_[B @ u, pA @ u]; fb = np.r_[np.zeros(len(B), bool), np.ones(len(pA), bool)]
    oa, ob = np.argsort(a), np.argsort(b)
    for ka, kb in zip(oa, ob):
        ax.plot([a[ka], b[kb]], [1, 0], color=L.MUTED, lw=0.8)
    for vals, flags, y, mk, col in ((a, fa, 1, "o", L.MCOL[0]), (b, fb, 0, "^", L.MCOL[1])):
        ax.scatter(vals[~flags], np.full((~flags).sum(), y), s=28, marker=mk, color=col, zorder=5)
        ax.scatter(vals[flags], np.full(flags.sum(), y), s=26, marker=mk, facecolor="white", edgecolor=col, lw=1.0, zorder=5)
    lo_, hi_ = min(a.min(), b.min()), max(a.max(), b.max())
    ax.text(lo_ - 0.08, 1, "$D$", ha="right", va="center", fontsize=9, color=L.MCOL[0], fontweight="bold")
    ax.text(lo_ - 0.08, 0, "$D'$", ha="right", va="center", fontsize=9, color=L.MCOL[1], fontweight="bold")
    ax.set_xlim(lo_ - 0.25, hi_ + 0.05); ax.set_ylim(-0.45, 1.45); ax.set_yticks([]); ax.set_xticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("projection on $\\theta$, sorted", fontsize=8.5, labelpad=2)
    ax.set_title("(d) projected, sorted, matched", fontsize=9.5)
    L.save(fig, "fig_persistence")


# --------------------------------------------------------------------------- 2a. elementary flows
def _swarm_radial(n=20, T=300, c=(0.5, 0.5), r_in=0.08, r_out=0.42, inward=True, seed=4):
    th = (np.arange(n) * 2 * np.pi / n)[None]                     # evenly spaced: no net rotation
    u = np.linspace(0, 1, T + 1)[:, None]
    R = (r_out - (r_out - r_in) * u) if inward else (r_in + (r_out - r_in) * u)
    return np.stack([c[0] + R * np.cos(th), c[1] + R * np.sin(th)], -1)


def _swarm_mill(n=20, T=300, c=(0.5, 0.5), r=0.25, turns=1.5, seed=5):
    th0 = (np.arange(n) * 2 * np.pi / n)[None]
    rad = r * (0.7 + 0.6 * (np.arange(n) % 4) / 3)[None]           # four rings, no net expansion
    th = th0 + 2 * np.pi * turns * np.linspace(0, 1, T + 1)[:, None]
    return np.stack([c[0] + rad * np.cos(th), c[1] + rad * np.sin(th)], -1)


def _mean_flow(P, X, Y, sigma, dt=0.1):
    """Time-averaged momentum density sum_t sum_i v_i O_i on a grid (for the arrows)."""
    pos, vel = P[:-1], (P[1:] - P[:-1]) / dt
    O = np.exp(-((X[None, ..., None] - pos[:, None, None, :, 0]) ** 2
                 + (Y[None, ..., None] - pos[:, None, None, :, 1]) ** 2) / (2 * sigma ** 2))
    return (O * vel[:, None, None, :, 0]).sum((0, 3)), (O * vel[:, None, None, :, 1]).sum((0, 3))


def _arrows(ax, P, sigma, lx, ly, n=11, color="#3a3a38"):
    X, Y = _grid(lx, ly, n)
    U, V = _mean_flow(P, X, Y, sigma)
    m = np.hypot(U, V)
    keep = m > 0.12 * m.max()
    ax.quiver(X[keep], Y[keep], U[keep] / m.max(), V[keep] / m.max(), color=color, angles="xy",
              scale_units="xy", scale=7.0, width=0.012, headwidth=4, headlength=4.5, zorder=4)


def fig_flows(sigma=0.07):
    """Three elementary motions (rows) and their two potentials (columns), with the averaged flow.

    Each row shares one colour scale for phi and psi, so that the potential a motion does not
    produce appears flat. With V = grad(phi) + perp-grad(psi), the flow climbs towards a maximum of phi
    (a sink) and turns around an extremum of psi (counter-clockwise: a minimum)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    lx = ly = 1.0
    cases = (("gathering", _swarm_radial(inward=True)), ("dispersing", _swarm_radial(inward=False)),
             ("milling", _swarm_mill()))
    F = [_fields(P, sigma, lx, ly, n=140) for _, P in cases]
    L.style()
    # one row per motion, one column per picture: the motion, then its two potentials; the colour bar on the right
    W = L.width(); lab_w, cb_w, gap = 0.78, 0.86, 0.05
    c = (W - lab_w - cb_w - 2 * gap) / 3
    H = 0.24 + 3 * c + 2 * gap
    fig = plt.figure(figsize=(W, H))
    axes = np.array([[fig.add_axes([(lab_w + j * (c + gap)) / W, (H - 0.24 - (i + 1) * c - i * gap) / H, c / W, c / H])
                      for j in range(3)] for i in range(3)])
    for r, ((name, P), f) in enumerate(zip(cases, F)):
        ax = axes[r, 0]
        for k in range(P.shape[1]):
            ax.plot(P[:, k, 0], P[:, k, 1], color=L.INK2, lw=0.6, alpha=0.8)
            ax.annotate("", xy=P[-1, k], xytext=P[-6, k],
                        arrowprops=dict(arrowstyle="-|>", color=L.INK2, lw=0.6, mutation_scale=6))
        ax.text(-0.06, 0.5, name, transform=ax.transAxes, ha="right", va="center", fontsize=9.5, fontweight="bold")
        m = max(np.nanmax(np.abs(f["phi"])), np.nanmax(np.abs(f["psi"])))
        for col, key in ((1, "phi"), (2, "psi")):
            ax = axes[r, col]
            im = ax.imshow(np.ma.masked_invalid(f[key] / m), origin="lower", extent=(0, lx, 0, ly),
                           cmap=L.FCMAP[key], vmin=-1, vmax=1, interpolation="bilinear")
            _arrows(ax, P, sigma, lx, ly)
            if np.nanmax(np.abs(f[key])) > 0.3 * m:
                _mark_extrema(ax, f[key], key, lx, ly)
        for ax in axes[r, :]:
            _clean(ax, lx, ly)
    for col, title in enumerate(("motion", "$\\varphi_\\sigma$", "$\\psi_\\sigma$")):
        axes[0, col].set_title(title, fontsize=10.5, fontweight="bold", pad=4)
    x_cb = lab_w + 3 * c + 2 * gap + 0.14
    cax = fig.add_axes([x_cb / W, (H - 0.24 - 3 * c - 2 * gap + 0.25) / H, 0.11 / W, (3 * c + 2 * gap - 0.5) / H])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["\u22121", "0", "+1"])
    L.bold_colorbar(cb, "potential / largest\nvalue in the row", size=8.5)
    L.save(fig, "fig_flows")


ALL = {"footprint": fig_footprint, "moments": fig_moments, "flows": fig_flows, "potentials": fig_potentials, "persistence": fig_persistence}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(ALL):
        ALL[name]()
    print("sizes on the page:")
    L.report()
