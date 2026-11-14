#!/usr/bin/env python3
"""
make_readme_visuals.py - animated educational GIFs for the repository README and notebooks
============================================================================================
Generates lightweight, publication-styled animated visualizations grounded directly in Paper 1's
closed-form equations and real Top-1..Top-10 pipeline outputs:
  1. assets/gaussian_splatting_demo.gif     - Two-robot Gaussian footprint & 2.35*sigma resolution limit
  2. assets/swarm_kinematic_fields.gif      - Live Top-1 exemplar trajectories unfolding into rho_sigma and psi_sigma
  3. assets/extended_persistence_demo.gif   - Sublevel/superlevel sweep -> Extended Persistence Diagram
  4. assets/phase_diagrams.gif              - Phase diagrams (k_nn/n, t) & transition scaling across Top-1..Top-10
  5. assets/attribution_to_robots.gif       - Differentiable chain from SW1 diagram -> 49 critical seeds -> robots

Usage:
    python analysis/make_readme_visuals.py                  # build all five GIFs
    python analysis/make_readme_visuals.py phase attribution  # build specific GIFs
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "src"), str(ROOT)]
import figlib as L  # noqa: E402

ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def _fig_to_pil(fig, dpi=105):
    """Render a matplotlib figure to a crisp RGB PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    im.load()
    buf.close()
    return im


def _save_gif(frames, out_path, fps=12, durations_ms=None, colors=256):
    """Quantize with a shared global palette to avoid flicker and keep file sizes compact for GitHub."""
    if not frames:
        raise ValueError("No frames to save")
    w, h = frames[0].size
    # Sample evenly across frames and downsample mosaic to build a fast representative 256-color palette
    step = max(1, len(frames) // 10)
    sample_frames = frames[::step][:10]
    sw, sh = max(1, w // 2), max(1, h // 2)
    sheet = Image.new("RGB", (sw, sh * len(sample_frames)))
    for idx, fr in enumerate(sample_frames):
        sheet.paste(fr.resize((sw, sh), Image.Resampling.BILINEAR), (0, idx * sh))
    palette_img = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    pal_frames = [fr.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG) for fr in frames]
    dur = durations_ms if durations_ms is not None else int(round(1000 / fps))
    pal_frames[0].save(
        out_path,
        save_all=True,
        append_images=pal_frames[1:],
        duration=dur,
        loop=0,
        optimize=True,
    )
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {out_path} ({len(frames)} frames, {w}x{h}, {mb:.2f} MB)")


# ---------------------------------------------------------------------------
# 1. Hero GIF: Gaussian Splatting Demo (two robots -> rho_sigma & 2.35*sigma limit)
# ---------------------------------------------------------------------------
def make_footprint_gif():
    """Two robots emitting Gaussian footprints O_i -> superposition & 1D resolution cross-section."""
    L.style()
    sigma = 0.12
    robot_r = 0.036
    arena = 1.0
    res = 221
    c_a = "#DC267F"
    c_b = "#2a78d6"
    c_sum = "#eb6834"
    rayleigh = 2.35482 * sigma  # 2*sqrt(2*ln(2))*sigma

    xs = np.linspace(0.0, arena, res)
    ys = np.linspace(0.0, arena, res)
    xg, yg = np.meshgrid(xs, ys)

    def gauss(cx, cy):
        return np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2.0 * sigma ** 2))

    # Smooth timeline: 68 frames
    n_frames = 68
    sep0, sep1 = 0.18, 0.68
    centre = 0.50

    def state(f):
        if f <= 13:
            t = f / 13.0
            alph = t * t * (3.0 - 2.0 * t)
            sep = sep0
            phase = 1
        elif f <= 23:
            alph, sep, phase = 1.0, sep0, 2
        elif f <= 51:
            t = (f - 24) / (51 - 24)
            s = t * t * (3.0 - 2.0 * t)
            alph, sep, phase = 1.0, sep0 + s * (sep1 - sep0), 3
        else:
            alph, sep, phase = 1.0, sep1, 4
        return centre - sep / 2.0, centre, centre + sep / 2.0, centre, alph, phase

    fig = plt.figure(figsize=(13.2, 7.8), facecolor="white")
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.10, 0.72],
        hspace=0.30, wspace=0.24,
        left=0.05, right=0.96, top=0.805, bottom=0.115,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[0, 2])
    ax_1d = fig.add_subplot(gs[1, :])

    blank = np.zeros((res, res))
    im_a = ax_a.imshow(blank, origin="lower", extent=[0, arena, 0, arena], cmap=L.SEQ, vmin=0, vmax=1.0)
    im_b = ax_b.imshow(blank, origin="lower", extent=[0, arena, 0, arena], cmap=L.SEQ, vmin=0, vmax=1.0)
    im_s = ax_s.imshow(blank, origin="lower", extent=[0, arena, 0, arena], cmap=L.SEQ, vmin=0, vmax=2.0)

    for ax, im, clab in (
        (ax_a, im_a, r"$O_A(\mathbf{r})$"),
        (ax_b, im_b, r"$O_B(\mathbf{r})$"),
        (ax_s, im_s, r"$O_A + O_B$"),
    ):
        cb = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.03, aspect=18)
        cb.set_ticks([0, 0.5, 1.0] if clab != r"$O_A + O_B$" else [0, 1.0, 2.0])
        L.bold_colorbar(cb, size=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#cfcec8"); sp.set_linewidth(0.8)

    ax_a.set_title(r"Robot A footprint  $O_A(\mathbf{r})$", fontsize=11, color=c_a, fontweight="bold", pad=8)
    ax_b.set_title(r"Robot B footprint  $O_B(\mathbf{r})$", fontsize=11, color=c_b, fontweight="bold", pad=8)
    ax_s.set_title(r"Superposed density  $M_0(\mathbf{r}) = O_A + O_B$", fontsize=11, color=L.INK, fontweight="bold", pad=8)

    fig.suptitle(
        r"Gaussian Footprint Intuition:  Two Robots $\rightarrow$ Continuous Density Field"
        + f"\n$\\sigma = {sigma:.2f}$ m   ·   $2.35\\,\\sigma = {rayleigh:.2f}$ m (FWHM threshold)   ·   $5\\sigma$ support cutoff",
        fontsize=13.5, fontweight="bold", y=0.965, color=L.INK,
    )
    fig.text(
        0.5, 0.870,
        r"$O_i(\mathbf{r}) = \exp\!\left(-\frac{\|\mathbf{r}-\mathbf{x}_i\|^2}{2\sigma^2}\right)$"
        r"       |       "
        r"$\rho_\sigma(\mathbf{r}) = \frac{1}{NT}\sum_{t,i} O_i(\mathbf{r},t)$",
        ha="center", va="center", fontsize=11, color=L.INK2,
    )

    patches = []
    for ax in (ax_a, ax_b, ax_s):
        dA = Circle((0, 0), robot_r, fc=c_a, ec="white", lw=1.6, zorder=10)
        dB = Circle((0, 0), robot_r, fc=c_b, ec="white", lw=1.6, zorder=10)
        sA = Circle((0, 0), sigma, fc="none", ec=c_a, lw=1.4, ls="--", zorder=9, alpha=0.8)
        sB = Circle((0, 0), sigma, fc="none", ec=c_b, lw=1.4, ls="--", zorder=9, alpha=0.8)
        for p in (dA, dB, sA, sB):
            ax.add_patch(p)
        patches.append((dA, dB, sA, sB))
    ax_a.axhline(centre, color="white", lw=0.7, ls=":", alpha=0.6)
    ax_b.axhline(centre, color="white", lw=0.7, ls=":", alpha=0.6)
    ax_s.axhline(centre, color="white", lw=0.7, ls=":", alpha=0.6)

    # 1D cross-section setup
    line_a, = ax_1d.plot([], [], color=c_a, lw=2.1, label=r"$O_A(x, y_0)$")
    line_b, = ax_1d.plot([], [], color=c_b, lw=2.1, label=r"$O_B(x, y_0)$")
    line_s, = ax_1d.plot([], [], color=c_sum, lw=2.8, label=r"$M_0 = O_A + O_B$")
    fill_ref = [None]
    pk_scatter = ax_1d.scatter([], [], s=65, fc="#0b0b0b", ec="white", lw=1.3, zorder=11, label="peaks of $M_0$")

    ax_1d.set_xlim(0.0, arena)
    ax_1d.set_ylim(-0.06, 2.18)
    ax_1d.set_xlabel("Cross-section position $x$ along $y_0 = 0.50$ m", fontsize=10.5, fontweight="bold")
    ax_1d.set_ylabel("Amplitude", fontsize=10.5, fontweight="bold")
    ax_1d.set_title("1-D cross-section along $y_0 = 0.50$ m (yellow = footprint overlap)", fontsize=11, pad=6)
    ax_1d.grid(True, alpha=0.22, color="#b5b4ae")
    ax_1d.legend(loc="upper right", fontsize=9.5, framealpha=0.95, ncol=2)

    status_txt = ax_1d.text(
        0.015, 0.90, "", transform=ax_1d.transAxes,
        ha="left", va="top", fontsize=10.5, fontweight="bold", color=L.INK,
        bbox=dict(facecolor="#f4f7fc", edgecolor="#9ec5f4", boxstyle="round,pad=0.35"),
    )
    regime_txt = ax_s.text(
        0.5, 0.06, "", transform=ax_s.transAxes,
        ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="white",
        bbox=dict(facecolor="#0b0b0b", edgecolor="none", boxstyle="round,pad=0.3", alpha=0.72),
    )
    phase_txt = fig.text(
        0.5, 0.024, "", ha="center", va="center", fontsize=11, fontweight="bold", color=L.INK,
        bbox=dict(facecolor="#f2f1ed", edgecolor="#c3c2b7", boxstyle="round,pad=0.40"),
    )

    phase_msgs = {
        1: "Phase 1 / 4  —  Gaussian kernels rising: each agent spreads smoothly over scale σ",
        2: "Phase 2 / 4  —  Close pair (d ≤ 2.35 σ): footprints fuse into ONE unresolved group peak",
        3: "Phase 3 / 4  —  Separating past 2.35 σ: saddle dips in the middle and individual peaks emerge",
        4: "Phase 4 / 4  —  Resolved pair (d > 2.35 σ): two distinct peaks — σ acts as the observation lens",
    }

    from scipy.signal import find_peaks
    mid = res // 2
    frames = []
    for f in range(n_frames):
        xA, yA, xB, yB, alph, phase = state(f)
        d = xB - xA
        ka = alph * gauss(xA, yA)
        kb = alph * gauss(xB, yB)
        ks = ka + kb

        im_a.set_data(ka)
        im_b.set_data(kb)
        im_s.set_data(ks)

        for (dA, dB, sA, sB) in patches:
            dA.center = (xA, yA); dB.center = (xB, yB)
            sA.center = (xA, yA); sB.center = (xB, yB)
            dA.set_alpha(max(0.25, alph))
            dB.set_alpha(max(0.25, alph))

        ya, yb, ys_ = ka[mid], kb[mid], ks[mid]
        line_a.set_data(xs, ya)
        line_b.set_data(xs, yb)
        line_s.set_data(xs, ys_)

        if fill_ref[0] is not None:
            fill_ref[0].remove()
        ov_mask = (ya > 0.015) & (yb > 0.015)
        ov_curve = np.where(ov_mask, np.minimum(ya, yb), 0.0)
        fill_ref[0] = ax_1d.fill_between(xs, 0.0, ov_curve, color="#FFD740", alpha=0.45, zorder=1)

        peaks_idx, _ = find_peaks(ys_, height=0.10, prominence=1e-4)
        if len(peaks_idx) == 0:
            peaks_idx = np.array([int(np.argmax(ys_))])
        pk_scatter.set_offsets(np.c_[xs[peaks_idx], ys_[peaks_idx]])

        n_peaks = int(len(peaks_idx))
        ov_pct = np.trapezoid(ov_curve, xs) / max(np.trapezoid(ys_, xs), 1e-9) * 100.0
        status_txt.set_text(
            f"d = {d:.2f} m  ({d / sigma:.2f} σ)   ·   overlap = {ov_pct:.0f}%   ·   peaks = {n_peaks}"
        )
        if d <= rayleigh:
            regime_txt.set_text("Unresolved (d ≤ 2.35 σ)" if n_peaks == 1 else "Merging (d ≤ 2.35 σ)")
        else:
            regime_txt.set_text("Resolved (2 peaks)")
        phase_txt.set_text(phase_msgs[phase])

        frames.append(_fig_to_pil(fig, dpi=105))

    plt.close(fig)
    _save_gif(frames, ASSETS / "gaussian_splatting_demo.gif", fps=12, colors=256)


# ---------------------------------------------------------------------------
# 2. Live ARGoS Swarms -> Cumulative Invariant Fields across 5 Missions
# ---------------------------------------------------------------------------
def make_swarm_fields_gif():
    """Real Top-1 exemplar runs unfolding over t = 0..120 s into density rho_sigma and vorticity psi_sigma."""
    L.style()

    d = L.load_pickle(L.PAPER / "top1_tracks.pkl")
    tax = L.load_pickle(L.PAPER / "taxonomy.pkl")[1]
    ladder = L.load_pickle(L.RES / "scale_ladder.pkl")["ladder"]
    sigma = float(ladder[2])  # sigma_3 = 7.18 cm — crisp group-level lens

    ex_by_mission = {int(d["labels"][e]): int(e) for e in tax["exemplars"]}
    missions = [ex_by_mission[m] for m in range(5)]
    n_t = d["tracks"][missions[0]]["pos"].shape[0]
    n_agents = d["tracks"][missions[0]]["pos"].shape[1]

    g_n = 88
    xs = np.linspace(0.0, L.ARENA, g_n)
    xg, yg = np.meshgrid(xs, xs)
    gx, gy = xg.ravel()[:, None], yg.ravel()[:, None]
    cx, cy = L.ARENA / 2.0, L.ARENA / 2.0
    arena_mask = ((xg - cx) ** 2 + (yg - cy) ** 2) > (1.28 ** 2)

    n_ckpts = 30
    edges = np.linspace(0, n_t, n_ckpts + 1, dtype=int)
    cum_rho = np.zeros((5, n_ckpts, g_n, g_n), dtype=np.float32)
    cum_psi = np.zeros((5, n_ckpts, g_n, g_n), dtype=np.float32)
    traj_pts = []

    for m_idx, run_i in enumerate(missions):
        pos = d["tracks"][run_i]["pos"].astype(np.float64)
        vel = d["tracks"][run_i]["vel"].astype(np.float64)
        vbar = float(np.mean(np.linalg.norm(vel, axis=-1))) + 1e-12
        traj_pts.append((pos, vel))

        r_acc = np.zeros(gx.shape[0], dtype=np.float64)
        p_acc = np.zeros(gx.shape[0], dtype=np.float64)
        for ci in range(n_ckpts):
            t0, t1 = int(edges[ci]), int(edges[ci + 1])
            p_blk = pos[t0:t1].reshape(-1, 2)
            v_blk = vel[t0:t1].reshape(-1, 2)
            dx = gx - p_blk[:, 0][None, :]
            dy = gy - p_blk[:, 1][None, :]
            r2 = dx * dx + dy * dy
            r_acc += np.exp(-r2 / (2.0 * sigma * sigma)).sum(axis=1)
            ks = np.where(r2 > 1e-12, (sigma * sigma / np.maximum(r2, 1e-12)) * (-np.expm1(-r2 / (2.0 * sigma * sigma))), 0.5)
            w_cross = v_blk[:, 0][None, :] * dy - v_blk[:, 1][None, :] * dx
            p_acc -= (w_cross * ks).sum(axis=1)
            steps_done = max(t1, 1)
            rr = (r_acc / (n_agents * steps_done)).reshape(g_n, g_n)
            pp = (p_acc / (n_agents * steps_done * sigma * vbar)).reshape(g_n, g_n)
            rr[arena_mask] = np.nan
            pp[arena_mask] = np.nan
            cum_rho[m_idx, ci] = rr
            cum_psi[m_idx, ci] = pp

    hold_end = 10
    total_frames = n_ckpts + hold_end

    rho_vmax = [float(np.nanquantile(cum_rho[m, -1], 0.995)) or 1.0 for m in range(5)]
    psi_vlim = [float(np.nanquantile(np.abs(cum_psi[m, -1]), 0.992)) or 1.0 for m in range(5)]
    th = np.linspace(0, 2 * np.pi, 160)
    ax_ring_x = cx + 1.236 * np.cos(th)
    ax_ring_y = cy + 1.236 * np.sin(th)

    cmap_seq = L.SEQ.copy(); cmap_seq.set_bad("white")
    cmap_div = L.DIV.copy(); cmap_div.set_bad("white")

    fig = plt.figure(figsize=(13.5, 8.2), facecolor="white")
    gs = fig.add_gridspec(3, 5, hspace=0.14, wspace=0.08, left=0.075, right=0.975, top=0.87, bottom=0.08)

    fig.suptitle(
        r"From Live Swarm Trajectories to Time-Averaged Invariant Fields (Top-1 Exemplars, $\sigma_3 = 7.18$ cm)",
        fontsize=13.5, fontweight="bold", y=0.96, color=L.INK,
    )
    row_titles = ["Trajectories\n& robots", r"Density  $\rho_\sigma$", r"Vorticity  $\psi_\sigma$"]

    track_lines, robot_scats, im_rhos, im_psis = [], [], [], []
    for m in range(5):
        ax0 = fig.add_subplot(gs[0, m])
        ax1 = fig.add_subplot(gs[1, m])
        ax2 = fig.add_subplot(gs[2, m])

        ax0.plot(ax_ring_x, ax_ring_y, color="#8a8984", lw=0.9, zorder=1)
        m_lines = []
        for a_i in range(n_agents):
            ln, = ax0.plot([], [], color=L.MCOL[m], lw=1.05, alpha=0.82, zorder=2)
            m_lines.append(ln)
        sc = ax0.scatter([], [], s=20, fc=L.INK, ec="white", lw=0.6, zorder=5)
        track_lines.append(m_lines)
        robot_scats.append(sc)
        ax0.set_title(L.MSHORT[m], fontsize=12, fontweight="bold", color=L.MCOL[m], pad=6)

        im_r = ax1.imshow(np.zeros((g_n, g_n)), origin="lower", extent=[0, L.ARENA, 0, L.ARENA],
                          cmap=cmap_seq, vmin=0.0, vmax=rho_vmax[m], interpolation="bilinear")
        ax1.plot(ax_ring_x, ax_ring_y, color="#8a8984", lw=0.7, alpha=0.75)
        im_rhos.append(im_r)

        im_p = ax2.imshow(np.zeros((g_n, g_n)), origin="lower", extent=[0, L.ARENA, 0, L.ARENA],
                          cmap=cmap_div, vmin=-psi_vlim[m], vmax=psi_vlim[m], interpolation="bilinear")
        ax2.plot(ax_ring_x, ax_ring_y, color="#8a8984", lw=0.7, alpha=0.75)
        im_psis.append(im_p)

        for r_i, ax in enumerate((ax0, ax1, ax2)):
            ax.set_xlim(0, L.ARENA); ax.set_ylim(0, L.ARENA); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if m == 0:
                ax.set_ylabel(row_titles[r_i], fontsize=11, fontweight="bold", labelpad=8, color=L.INK)

    foot_txt = fig.text(
        0.5, 0.025, "", ha="center", va="center", fontsize=11, fontweight="bold", color=L.INK,
        bbox=dict(facecolor="#f2f1ed", edgecolor="#c3c2b7", boxstyle="round,pad=0.40"),
    )

    frames = []
    for f in range(total_frames):
        ci = min(f, n_ckpts - 1)
        t_end = int(edges[ci + 1]) - 1
        secs = int(round(120.0 * (ci + 1) / n_ckpts))
        for m in range(5):
            pos, _ = traj_pts[m]
            t_start = max(0, t_end - 220) if f < n_ckpts else 0
            for a_i in range(n_agents):
                track_lines[m][a_i].set_data(pos[t_start:t_end + 1, a_i, 0], pos[t_start:t_end + 1, a_i, 1])
            robot_scats[m].set_offsets(pos[t_end])
            im_rhos[m].set_data(np.ma.masked_invalid(cum_rho[m, ci]))
            im_psis[m].set_data(np.ma.masked_invalid(cum_psi[m, ci]))

        if f < n_ckpts:
            foot_txt.set_text(
                f"Simulated time t = {secs:3d} / 120 s  —  Averaging kinematics builds stable density hotspots & counter-rotating vortices"
            )
        else:
            foot_txt.set_text(
                "Settled 120 s fields  —  Notice counter-rotating ψ lobes in D-Gate vs single vortex in Foraging & twin patches in XOR"
            )
        frames.append(_fig_to_pil(fig, dpi=88))

    plt.close(fig)
    _save_gif(frames, ASSETS / "swarm_kinematic_fields.gif", fps=8, colors=256)


# ---------------------------------------------------------------------------
# 3. Extended Persistence Intuition GIF (sublevel + superlevel -> (b, d) diagram)
# ---------------------------------------------------------------------------
def make_persistence_gif():
    """Sweep threshold c across a 1D multi-peak kinematic slice to populate the extended persistence diagram."""
    L.style()
    x = np.linspace(-1.0, 1.0, 320)
    # Two asymmetric peaks and a central saddle/valley
    f_x = 0.85 * np.exp(-((x + 0.42) / 0.22) ** 2) + 0.58 * np.exp(-((x - 0.44) / 0.20) ** 2) - 0.10 * x + 0.12
    f_min, f_max = float(f_x.min()), float(f_x.max())

    # Identify key critical heights analytically from discrete curve
    idx_left_pk = int(np.argmax(f_x[:160]))
    idx_right_pk = 160 + int(np.argmax(f_x[160:]))
    idx_mid_val = idx_left_pk + int(np.argmin(f_x[idx_left_pk:idx_right_pk]))
    h_p1 = float(f_x[idx_left_pk])
    h_p2 = float(f_x[idx_right_pk])
    h_sad = float(f_x[idx_mid_val])
    h_gmin = f_min

    # Superlevel sweep downward from f_max -> f_min reveals Rel1 (two peaks merging at saddle):
    # Main peak born at h_p1 -> paired with global min h_gmin as Ext0: (h_gmin, h_p1)
    # Secondary peak born at h_p2 -> dies/merges at saddle h_sad in Rel1: (h_p2, h_sad)
    c_up = np.linspace(f_min - 0.03, f_max + 0.04, 24)
    c_dn = np.linspace(f_max + 0.04, f_min - 0.03, 24)

    fig, (ax_f, ax_d) = plt.subplots(1, 2, figsize=(11.8, 5.25), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.75, bottom=0.16, wspace=0.25)

    fig.suptitle(
        "Extended Persistence:  Sweeping Threshold c Turns Extrema & Saddles into Diagram Points",
        fontsize=13, fontweight="bold", y=0.96, color=L.INK,
    )

    ax_f.plot(x, f_x, color=L.INK, lw=2.3, zorder=5)
    h_pk = ax_f.scatter([x[idx_left_pk], x[idx_right_pk]], [h_p1, h_p2], c=L.MCOL[1], s=55, zorder=8, label="peaks (maxima)")
    h_sd = ax_f.scatter([x[idx_mid_val]], [h_sad], c=L.MCOL[0], marker="x", s=70, lw=2.2, zorder=8, label="saddle (merge height)")
    thresh_line = ax_f.axhline(f_min, color=L.MCOL[1], lw=1.6, ls="--", zorder=6)
    fill_ref = [None]

    ax_f.set_xlim(-1.0, 1.0); ax_f.set_ylim(0.0, f_max + 0.14)
    ax_f.set_xlabel("Spatial coordinate along swarm slice", fontsize=10, fontweight="bold")
    ax_f.set_ylabel("Field value  f(x)", fontsize=10, fontweight="bold")
    ax_f.set_title("Field profile & active excursion set", fontsize=11, pad=7)
    ax_f.grid(True, alpha=0.20, color="#b5b4ae")

    # Right panel: Persistence Diagram (birth, death)
    lo, hi = 0.0, f_max + 0.14
    ax_d.plot([lo, hi], [lo, hi], color="#8a8984", lw=1.2, ls="-", zorder=1)
    ax_d.fill_between([lo, hi], [lo, hi], hi, color="#f6f9fe", zorder=0)
    ax_d.fill_between([lo, hi], lo, [lo, hi], color="#fef8f4", zorder=0)
    ax_d.text(0.75, 0.36, "superlevel (Rel1)\nbirth > death", transform=ax_d.transAxes,
              fontsize=8.8, color=L.MCOL[1], fontweight="bold", ha="center")
    ax_d.text(0.28, 0.78, "sublevel / Ext0\nbirth < death", transform=ax_d.transAxes,
              fontsize=8.8, color=L.MCOL[0], fontweight="bold", ha="center")

    sc_ext0 = ax_d.scatter([], [], s=110, marker="D", fc=L.MCOL[0], ec="white", lw=1.5, zorder=10, label=r"$\mathrm{Ext}_0$ (global span)")
    sc_ord0 = ax_d.scatter([], [], s=95, marker="o", fc=L.MCOL[2], ec="white", lw=1.4, zorder=10, label=r"$\mathrm{Ord}_0$ (saddle-well)")
    sc_rel1 = ax_d.scatter([], [], s=110, marker="s", fc=L.MCOL[1], ec="white", lw=1.5, zorder=10, label=r"$\mathrm{Rel}_1$ (peak-saddle)")
    sc_cur_c = ax_d.scatter([], [], s=45, c="#0b0b0b", zorder=11)

    ax_d.set_xlim(lo, hi); ax_d.set_ylim(lo, hi); ax_d.set_aspect("equal")
    ax_d.set_xlabel("Birth value  b", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Death value  d", fontsize=10, fontweight="bold")
    ax_d.set_title("Extended persistence diagram  (b, d)", fontsize=11, pad=7)
    ax_d.grid(True, alpha=0.20, color="#b5b4ae")

    fig.legend(
        handles=[h_pk, h_sd, sc_ext0, sc_ord0, sc_rel1],
        loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=5,
        fontsize=8.6, frameon=True, facecolor="#faf9f6", edgecolor="#d4d3ca",
        columnspacing=1.2, handletextpad=0.45,
    )

    banner = fig.text(
        0.5, 0.038, "", ha="center", va="center", fontsize=10.5, fontweight="bold", color=L.INK,
        bbox=dict(facecolor="#f2f1ed", edgecolor="#c3c2b7", boxstyle="round,pad=0.38"),
    )

    frames = []
    # Phase A: upward sublevel sweep {f <= c}
    for c in c_up:
        if fill_ref[0] is not None:
            fill_ref[0].remove()
        fill_ref[0] = ax_f.fill_between(x, 0, f_x, where=(f_x <= c), color=L.MCOL[0], alpha=0.28, zorder=2)
        thresh_line.set_ydata([c, c]); thresh_line.set_color(L.MCOL[0])
        sc_cur_c.set_offsets([[c, c]])

        # Reveal Ord0 minimum->boundary or central well when threshold crosses saddle
        ord0_pts = [[h_gmin, min(c, h_sad)]] if c > h_gmin else np.empty((0, 2))
        sc_ord0.set_offsets(ord0_pts)
        sc_rel1.set_offsets(np.empty((0, 2)))
        ext0_pts = [[h_gmin, min(c, h_p1)]] if c > h_gmin else np.empty((0, 2))
        sc_ext0.set_offsets(ext0_pts)

        banner.set_text(
            f"Upward sublevel sweep  {{f ≤ {c:.2f}}}:  wells grow from minima and track global span Ext0"
        )
        frames.append(_fig_to_pil(fig, dpi=105))

    # Phase B: downward superlevel sweep {f >= c}
    for c in c_dn:
        if fill_ref[0] is not None:
            fill_ref[0].remove()
        fill_ref[0] = ax_f.fill_between(x, 0, f_x, where=(f_x >= c), color=L.MCOL[1], alpha=0.28, zorder=2)
        thresh_line.set_ydata([c, c]); thresh_line.set_color(L.MCOL[1])
        sc_cur_c.set_offsets([[c, c]])

        sc_ord0.set_offsets([[h_gmin, h_sad]])
        sc_ext0.set_offsets([[h_gmin, h_p1]])
        if c <= h_p2:
            # Secondary peak born at h_p2, stretches toward saddle h_sad and stops there
            d_pt = max(c, h_sad)
            sc_rel1.set_offsets([[h_p2, d_pt]])
            msg = (
                f"Downward superlevel sweep  {{f ≥ {c:.2f}}}:  secondary peak at {h_p2:.2f} merges at saddle {h_sad:.2f} → Rel1 point!"
                if c <= h_sad
                else f"Downward superlevel sweep  {{f ≥ {c:.2f}}}:  two separate peaks appear above threshold"
            )
        else:
            sc_rel1.set_offsets(np.empty((0, 2)))
            msg = f"Downward superlevel sweep  {{f ≥ {c:.2f}}}:  highest peak emerges first at f = {h_p1:.2f}"
        banner.set_text(msg)
        frames.append(_fig_to_pil(fig, dpi=105))

    # Hold final diagram briefly (8 frames)
    for _ in range(8):
        frames.append(frames[-1])

    plt.close(fig)
    _save_gif(frames, ASSETS / "extended_persistence_demo.gif", fps=10, colors=256)


# ---------------------------------------------------------------------------
# 4. Phase Diagrams GIF (Top-1 -> Top-10 resolution phases & scaling law)
# ---------------------------------------------------------------------------
def make_phase_diagrams_gif():
    """Animate phase diagrams across Top-1..Top-10 in normalized (k_nn/n, t) space alongside scaling law."""
    import make_figures as MF
    from matplotlib import patheffects as PE
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    L.style()
    SS, BEH, maps = MF._wide_maps()
    tax = L.load_pickle(L.PAPER / "taxonomy.pkl")
    order = list(SS.STATES)
    state_cmap = ListedColormap([MF.STATE_COLORS[s] for s in order])

    # Precompute every Top-n phase grid (normalized x = k_nn / n), boundaries, purity=1 contour, transitions
    FIT = (2, 3, 4, 5, 6, 7)
    tr = {n: SS.transition(maps[n], BEH) for n in range(1, 11)}
    c_fit, a_fit = SS.power_law(np.vstack([tr[n] for n in FIT]))
    per = {n: SS.power_law(tr[n]) for n in range(1, 11)}
    lo_alpha = min(per[n][1] for n in FIT)
    hi_alpha = max(per[n][1] for n in FIT)

    panels = {}
    for n in range(1, 11):
        d = maps[n]
        y, ts = np.asarray(d["labels"]), d["ts"]
        knns = [k for k in d["knns"] if k >= 3]
        xk = np.array(knns, dtype=float) / n
        by = {(r["knn"], r["t"]): np.asarray(r["pred"]) for r in d["results"]}
        S_ = np.array([[order.index(SS.state_of(by[(k, t)], y, BEH)) for k in knns] for t in ts])
        pure = np.array([[float(all(len(np.unique(y[by[(k, t)] == c])) == 1 for c in np.unique(by[(k, t)])))
                          for k in knns] for t in ts])
        dx0 = (xk[1] - xk[0]) / 2.0
        left_e = max(0.5, xk[0] - dx0)
        ex = np.r_[left_e, (xk[1:] + xk[:-1]) / 2.0, xk[-1] + (xk[-1] - xk[-2]) / 2.0]
        et = np.r_[0.5, (np.array(ts[1:]) + np.array(ts[:-1])) / 2.0, ts[-1] + 0.5]
        seg = [[(ex[j + 1], et[i]), (ex[j + 1], et[i + 1])] for i in range(len(ts)) for j in range(len(xk) - 1)
               if S_[i, j] != S_[i, j + 1]]
        seg += [[(ex[j], et[i + 1]), (ex[j + 1], et[i + 1])] for i in range(len(ts) - 1) for j in range(len(xk))
                if S_[i, j] != S_[i + 1, j]]
        panels[n] = dict(knns=knns, xk=xk, ts=ts, S_=S_, pure=pure, ex=ex, et=et, seg=seg)

    fig = plt.figure(figsize=(12.6, 5.95), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.22, 1.0],
        height_ratios=[1.12, 1.0],
        hspace=0.36, wspace=0.25,
        left=0.06, right=0.97, top=0.815, bottom=0.195,
    )
    ax_p = fig.add_subplot(gs[:, 0])
    ax_law = fig.add_subplot(gs[0, 1])
    ax_exp = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        r"Phase Diagrams Across Resolutions & Dataset Sizes (Top-1 $\rightarrow$ Top-10)",
        fontsize=13.5, fontweight="bold", y=0.96, color=L.INK,
    )
    subtitle_txt = fig.text(0.5, 0.885, "", ha="center", va="center", fontsize=10.5, color=L.INK2, fontweight="bold")

    # Shared bottom legend for phase diagram
    pnames = {
        "finer": "finer than missions",
        "missions": "the five missions",
        "partial": "partly merged",
        "shelter_xor": "Shelter + XOR",
        "families": "D-Gate + Foraging | Homing | Shelter + XOR",
        "two_families": "D-Gate + Foraging + Homing | Shelter + XOR",
    }
    mk_patch = lambda s: Patch(facecolor=MF.STATE_COLORS[s], hatch="/////" if s == "partial" else None,
                               edgecolor=L.INK if s == "partial" else MF.STATE_COLORS[s], lw=0.4, label=pnames[s])
    leg_h = [mk_patch(s) for s in ("finer", "missions", "partial", "shelter_xor", "families", "two_families")]
    leg_h += [
        Line2D([], [], color=L.INK, lw=0.9, label="phase transition"),
        Line2D([], [], color="#184f95", lw=1.2, ls="--", label="edge of purity 1"),
        Line2D([], [], color=L.INK, lw=1.1, ls=":", label=r"local grid ($k_{\mathrm{nn}},t \leq 25$)"),
    ]
    fig.legend(
        handles=leg_h, loc="lower center", ncol=3, frameon=True,
        facecolor="#faf9f6", edgecolor="#d4d3ca", fontsize=8.4,
        bbox_to_anchor=(0.5, 0.015), handlelength=1.4, columnspacing=1.2, handletextpad=0.45,
    )

    tt_line = np.geomspace(5, 50, 60)

    def _color_of(n):
        if n == 1:
            return L.INK
        return "#184f95" if n in FIT else "#eb6834"

    frames = []
    for cur_n in range(1, 11):
        p = panels[cur_n]
        t_info = tax[cur_n]
        N_runs = len(t_info["idx"])
        C_star = int(t_info["C"])
        n_plat = int(t_info["n_plateau"])
        pur = float(t_info["purity"])
        ari = float(t_info["ari"])
        alpha_cur = per[cur_n][1]

        subtitle_txt.set_text(
            f"Top-{cur_n}:  {N_runs:,} runs ({5 * cur_n} controllers)   ·   "
            f"Modal plateau $C^\\star = {C_star}$ ({n_plat}/99)   ·   "
            f"Purity = {pur:.3f}   ·   ARI = {ari:.3f}"
        )

        # ---- Left panel: normalized phase diagram ----
        ax_p.clear()
        ax_p.pcolormesh(p["ex"], p["et"], p["S_"], cmap=state_cmap, vmin=-0.5, vmax=len(order) - 0.5,
                        shading="flat", rasterized=True)
        hatch = np.ma.masked_where(p["S_"] != order.index("partial"), np.ones_like(p["S_"], dtype=float))
        ax_p.pcolor(p["ex"], p["et"], hatch, hatch="/////", alpha=0.0, shading="flat")
        ax_p.add_collection(LineCollection(p["seg"], colors=L.INK, linewidths=0.7))
        if p["pure"].any() and not p["pure"].all():
            ax_p.contour(p["xk"], p["ts"], p["pure"], levels=[0.5], colors="#184f95", linewidths=1.3, linestyles="--")
        # Local grid box knn <= 25 -> x <= 25 / cur_n
        local_x = min(50.5, 25.5 / cur_n)
        if local_x > p["ex"][0]:
            ax_p.plot([p["ex"][0], local_x, local_x], [25.5, 25.5, p["et"][0]], color=L.INK, lw=1.25, ls=":")
        ax_p.set_xlim(0.5, 50.5)
        ax_p.set_ylim(0.5, 50.5)
        ax_p.set_xticks([1, 10, 20, 30, 40, 50])
        ax_p.set_yticks([1, 10, 25, 40, 50])
        ax_p.set_xlabel(r"Normalized neighborhood size $k_{\mathrm{nn}} / n$   ($k_{\mathrm{nn}} \in ["
                        + f"{p['knns'][0]}, {p['knns'][-1]}]$)", fontsize=9.8, fontweight="bold")
        ax_p.set_ylabel(r"Diffusion iterations $t$", fontsize=9.8, fontweight="bold")
        ax_p.set_title(f"(a) Phase diagram on Top-{cur_n}  (normalized width $k_{{\\mathrm{{nn}}}}/n$)",
                       loc="left", fontsize=10.5, fontweight="bold", pad=6)
        ax_p.tick_params(labelsize=8.5); L.bold_ticks(ax_p)
        for sp in ax_p.spines.values():
            sp.set_visible(True); sp.set_color("#c8c7c0"); sp.set_linewidth(0.8)

        # ---- Top-right panel: log-log scaling law ----
        ax_law.clear()
        for prev_n in range(1, cur_n):
            Q = tr[prev_n]
            ax_law.plot(Q[:, 0], Q[:, 1], "o", ms=2.2, color=_color_of(prev_n), alpha=0.22, zorder=2)
        Q_cur = tr[cur_n]
        cur_col = _color_of(cur_n)
        ax_law.plot(
            Q_cur[:, 0], Q_cur[:, 1], "o", ms=4.3, color=cur_col,
            mfc=cur_col if cur_n <= 7 else "white", mew=1.2, zorder=5,
            label=f"Top-{cur_n} transition ($\\alpha = {alpha_cur:.2f}$)",
        )
        ax_law.plot(tt_line, c_fit * tt_line ** (-a_fit), color=L.INK, lw=1.3, ls="--", zorder=4,
                    label=f"Top-2..7 law: ${c_fit:.0f}\\,t^{{-{a_fit:.2f}}}$")
        ax_law.set_xscale("log"); ax_law.set_yscale("log")
        ax_law.set_xticks([5, 10, 20, 50]); ax_law.set_xticklabels(["5", "10", "20", "50"])
        ax_law.set_yticks([10, 20, 50]); ax_law.set_yticklabels(["10", "20", "50"])
        ax_law.minorticks_off()
        ax_law.set_xlim(4.5, 56); ax_law.set_ylim(9.5, 78)
        ax_law.set_xlabel("diffusion $t$", fontsize=9.2, fontweight="bold")
        ax_law.set_ylabel(r"transition $k_{\mathrm{nn}}/n$", fontsize=9.2, fontweight="bold")
        ax_law.set_title("(b) Collapse onto one transition law", loc="left", fontsize=10.2, fontweight="bold", pad=5)
        ax_law.grid(color=L.GRID, lw=0.6); ax_law.set_axisbelow(True)
        ax_law.tick_params(labelsize=8.2); L.bold_ticks(ax_law)
        ax_law.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", edgecolor="#deddd6")

        # ---- Bottom-right panel: exponent & atypical shift ----
        ax_exp.clear()
        ax_exp.axhspan(lo_alpha, hi_alpha, color="#e8f0fb", zorder=0)
        ax_exp.text(2.15, lo_alpha - 0.006, "regular Top-2..7 band", fontsize=8.0, color="#184f95", va="top")
        for m in range(2, cur_n + 1):
            col = _color_of(m)
            is_cur = (m == cur_n)
            ax_exp.plot(
                m, per[m][1], "o",
                ms=7.8 if is_cur else 5.2,
                color=col,
                mfc=col if m <= 7 else ("white" if not is_cur else "#fde7dc"),
                mew=1.8 if is_cur else 1.2,
                zorder=5 if is_cur else 3,
            )
        if cur_n >= 2:
            ms = list(range(2, cur_n + 1))
            ax_exp.plot(ms, [per[m][1] for m in ms], color="#9b9a94", lw=1.0, zorder=2)
        if cur_n >= 8:
            ax_exp.text(9.0, 0.688, "atypical controllers\nsteepen boundary", ha="center", va="bottom",
                        fontsize=8.0, color="#c24918", fontweight="bold", linespacing=1.05,
                        path_effects=[PE.withStroke(linewidth=2.5, foreground="white")])
        ax_exp.set_xticks(range(2, 11)); ax_exp.set_xlim(1.5, 10.5); ax_exp.set_ylim(0.50, 0.725)
        ax_exp.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70])
        ax_exp.set_xlabel("Top-$n$ pooled dataset", fontsize=9.2, fontweight="bold")
        ax_exp.set_ylabel(r"exponent $\alpha_n$", fontsize=9.2, fontweight="bold")
        ax_exp.set_title("(c) Transition exponent across Top-$n$", loc="left", fontsize=10.2, fontweight="bold", pad=5)
        ax_exp.grid(color=L.GRID, lw=0.6); ax_exp.set_axisbelow(True)
        ax_exp.tick_params(labelsize=8.2); L.bold_ticks(ax_exp)

        n_hold = 12 if cur_n == 10 else 5
        fr = _fig_to_pil(fig, dpi=96)
        for _ in range(n_hold):
            frames.append(fr)

    plt.close(fig)
    _save_gif(frames, ASSETS / "phase_diagrams.gif", fps=8, colors=256)


# ---------------------------------------------------------------------------
# 5. Attribution-to-Robots GIF (SW1 -> 49 Critical Seeds -> Robot Trajectories)
# ---------------------------------------------------------------------------
def make_attribution_gif():
    """Animate backward chain from SW1 distance -> critical seeds -> time-resolved robot gradients."""
    import attribution as AT
    import make_figures as MF
    import torch
    from matplotlib import patheffects as PE
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from scipy.ndimage import gaussian_filter, uniform_filter1d

    L.style()
    A = L.load_pickle(L.PAPER / "attribution.pkl")
    cache = L.load_pickle(L.PAPER / "display_fields.pkl")
    tracks = L.load_pickle(L.PAPER / "top1_tracks.pkl")
    eA, gA = A["run_A"]
    eB, gB = A["run_B"]
    si = int(A["scale"])
    field = A["field"]
    pos = np.asarray(tracks["tracks"][eA]["pos"], np.float64)   # (1199, 20, 2)
    vel = np.asarray(tracks["tracks"][eA]["vel"], np.float64)
    P = np.concatenate([pos, pos[-1:] + AT.DT * vel[-1:]], axis=0)  # (1200, 20, 2)
    F_grid = cache[("ladder", eA, si)][field]
    domain = np.isfinite(F_grid)
    S = A["seeds_A"]
    T_types = A["types_A"]
    share = A["persistence_share_A"]

    # Diagrams of both compared runs (D-Gate vs Foraging in psi_sigma3)
    dgmA = [np.asarray(d, np.float64).reshape(-1, 2) for d in
            L.load_pickle(L.RES / "stage2_alpha_dgms" / f"traj{gA:04d}_s{si}.pkl")["dgms"][field]]
    dgmB = [np.asarray(d, np.float64).reshape(-1, 2) for d in
            L.load_pickle(L.RES / "stage2_alpha_dgms" / f"traj{gB:04d}_s{si}.pkl")["dgms"][field]]
    ptsA_all = np.vstack([d for d in dgmA if len(d)])
    ptsB_all = np.vstack([d for d in dgmB if len(d)])
    top_pts = A["points_A"][:14]  # largest diagram points driving SW1 difference

    # Exact differentiable per-(t, robot) attribution via chain rule (match ~1.3e-18 to attribution.pkl)
    PA_t = torch.tensor(P, dtype=AT.TD, device=AT.DEV)
    SA_t = torch.tensor(S, dtype=AT.TD, device=AT.DEV)
    flow = AT.robot_flow(PA_t, SA_t, float(A["sigma"]), share)  # (1200, 20), sums to 1.0
    if AT.DEV.type == "cuda":
        torch.cuda.empty_cache()

    # Smooth instantaneous flow per robot & precompute cumulative spatial maps across time checkpoints
    n_grid = 80
    rows_idx = np.clip(((np.arange(n_grid) + 0.5) / n_grid * domain.shape[0]).astype(int), 0, domain.shape[0] - 1)
    inside = domain[np.ix_(rows_idx, rows_idx)]
    flow_mid = flow[1:-1]                                         # (1198, 20), matching P[1:-1]
    pts_mid = P[1:-1]
    flow_smooth = uniform_filter1d(flow_mid, size=11, axis=0, mode="nearest")
    g_hi = float(np.percentile(flow_smooth, 97.5))

    x_all = pts_mid.reshape(-1, 2)
    H_all, _, _ = np.histogram2d(x_all[:, 1], x_all[:, 0], bins=n_grid,
                                 range=[[0, L.ARENA], [0, L.ARENA]], weights=flow_mid.ravel())
    H_all = gaussian_filter(H_all, 1.5) * inside
    h_all_max = max(float(H_all[inside].max()), 1e-30)

    def cum_map_scaled(t_end_idx):
        x = pts_mid[:t_end_idx].reshape(-1, 2)
        w = flow_mid[:t_end_idx].ravel()
        H, _, _ = np.histogram2d(x[:, 1], x[:, 0], bins=n_grid, range=[[0, L.ARENA], [0, L.ARENA]], weights=w)
        H = gaussian_filter(H, 1.5) * inside
        return np.sqrt(np.clip(H / h_all_max, 0.0, 1.0))

    # Build Figure (3 panels connected left-to-right by chain-rule arrows)
    fig = plt.figure(figsize=(13.1, 5.25), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.04], wspace=0.28,
                          left=0.055, right=0.975, top=0.77, bottom=0.205)
    ax_d = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[0, 2])

    fig.suptitle(
        r"Differentiable Attribution:  Extended Persistence $\rightarrow$ Critical Seeds $\rightarrow$ Robots",
        fontsize=13.5, fontweight="bold", y=0.955, color=L.INK,
    )
    fig.text(
        0.5, 0.872,
        f"D-Gate (run {gA}) vs Foraging (run {gB}) on vorticity $\\psi_{{\\sigma_3}}$ ($\\sigma_3 = 7.18$ cm)"
        f"   ·   $d_{{\\mathrm{{SW}}_1}} = {A['persistence_distance']:.4f}$"
        f"   ·   100% on 49 critical seeds",
        ha="center", va="center", fontsize=10.2, color=L.INK2,
    )

    # ---- Panel (a) static base: persistence diagrams ----
    lo, hi = -0.038, 0.042
    ax_d.plot([lo, hi], [lo, hi], color="#8a8984", lw=1.1, zorder=1)
    ax_d.scatter(ptsB_all[:, 0], ptsB_all[:, 1], s=26, marker="x", c=L.MCOL[1], lw=1.2, alpha=0.55,
                 zorder=2, label="Foraging")
    ax_d.scatter(ptsA_all[:, 0], ptsA_all[:, 1], s=28, marker="o", fc="#8ebcf0", ec=L.MCOL[0], lw=0.7, alpha=0.65,
                 zorder=3, label="D-Gate")
    top_xy = np.array([[p["birth"], p["death"]] for p in top_pts])
    top_sh = np.array([p["share"] for p in top_pts])
    top_sh_n = top_sh / max(top_sh.max(), 1e-12)
    sc_diag_hi = ax_d.scatter(top_xy[:, 0], top_xy[:, 1], s=0.0, fc=MF.CRIT_C, ec="white", lw=1.0, zorder=6)
    ann_ext0 = ax_d.annotate(
        r"$\mathrm{Ext}_0$" + "\nmin→max",
        xy=(top_xy[0, 0], top_xy[0, 1]), xytext=(-0.013, 0.032),
        fontsize=8.2, fontweight="bold", color=MF.CRIT_C, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", lw=1.0, color=MF.CRIT_C),
        alpha=0.0, zorder=8,
    )
    if ann_ext0.arrow_patch is not None:
        ann_ext0.arrow_patch.set_alpha(0.0)
    ax_d.set_xlim(lo, hi); ax_d.set_ylim(lo, hi); ax_d.set_aspect("equal")
    ax_d.set_xticks([-0.03, 0.0, 0.03]); ax_d.set_yticks([-0.03, 0.0, 0.03])
    ax_d.tick_params(labelsize=8.0); L.bold_ticks(ax_d)
    ax_d.set_xlabel("Birth value", fontsize=9.2, fontweight="bold")
    ax_d.set_ylabel("Death value", fontsize=9.2, fontweight="bold")
    ax_d.set_title("(a) Diagram split  $d_{\\mathrm{SW}_1}$", loc="left", fontsize=10.2, fontweight="bold", pad=6)
    diag_leg = [
        Line2D([], [], ls="", marker="x", ms=5.5, c=L.MCOL[1], mew=1.2, label="Foraging"),
        Line2D([], [], ls="", marker="o", ms=5.5, mfc="#8ebcf0", mec=L.MCOL[0], mew=0.8, label="D-Gate"),
        Line2D([], [], ls="", marker="o", ms=6.2, mfc=MF.CRIT_C, mec="white", mew=0.8, label=r"top $\mathrm{SW}_1$ share"),
    ]
    ax_d.legend(handles=diag_leg, loc="lower right", fontsize=7.6, frameon=True,
                facecolor="#faf9f6", edgecolor="#d4d3ca")
    ax_d.grid(True, color=L.GRID, lw=0.5); ax_d.set_axisbelow(True)

    # ---- Panel (b) static base: vorticity psi + 2,398 seeds + critical splits ----
    L.show_field(ax_s, F_grid, field)
    ax_s.scatter(S[:, 0], S[:, 1], s=1.0, c=L.MUTED, lw=0, zorder=3)
    crit_idx = np.where((share > 0) & np.isin(T_types, ("min", "max", "saddle")))[0]
    crit_order = crit_idx[np.argsort(share[crit_idx])]
    crit_s_full = 18.0 + 780.0 * (share[crit_order] / share.sum())
    crit_scats = []
    for v in crit_order:
        sc_k = ax_s.scatter([S[v, 0]], [S[v, 1]], s=0.0, marker=MF.TMARK[str(T_types[v])],
                            c=MF.CRIT_C, edgecolors="white", linewidths=0.85, zorder=6)
        crit_scats.append(sc_k)
    ax_s.set_title(r"(b) Back to critical seeds ($\psi_{\sigma_3}$)", loc="left", fontsize=10.2, fontweight="bold", pad=6)

    # Rotation center callouts (fade in during stage 2)
    ext_idx = np.where(np.isin(T_types, ("min", "max")))[0]
    ends = [v for v in ext_idx[np.argsort(share[ext_idx])[-2:]]]
    rot_anns = []
    for k, v in enumerate(ends):
        p_xy = S[v]
        other = S[ends[1 - k]]
        away = p_xy - other
        away = away / np.linalg.norm(away) + np.array([0.85, 0.0])
        off = 0.56 * away / np.linalg.norm(away)
        q_xy = np.array([np.clip(p_xy[0] + off[0], 0.35, L.ARENA - 0.38),
                         np.clip(p_xy[1] + off[1], 0.35, L.ARENA - 0.42)])
        an = ax_s.annotate(
            "rotation\ncenter", xy=p_xy, xytext=q_xy, fontsize=8.0, fontweight="bold", color=L.INK,
            ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.85, color=L.INK),
            path_effects=[PE.withStroke(linewidth=2.8, foreground="white")], alpha=0.0, zorder=8,
        )
        if an.arrow_patch is not None:
            an.arrow_patch.set_alpha(0.0)
        rot_anns.append(an)

    # ---- Panel (c) base: arena mask + cumulative gradient heatmap + robot trails & live markers ----
    ax_r.set_xlim(0, L.ARENA); ax_r.set_ylim(0, L.ARENA); ax_r.set_aspect("equal")
    ax_r.set_xticks([]); ax_r.set_yticks([])
    for sp in ax_r.spines.values():
        sp.set_visible(False)
    th = np.linspace(0, 2 * np.pi, 13)
    ax_r.plot(L.ARENA / 2 + 1.24 * np.cos(th), L.ARENA / 2 + 1.24 * np.sin(th), color="#9e9d97", lw=0.9, zorder=1)
    im_rob = ax_r.imshow(
        np.ma.masked_where(~inside, np.zeros((n_grid, n_grid))),
        origin="lower", extent=(0, L.ARENA, 0, L.ARENA), cmap=MF.ATTR_CMAP, vmin=0.0, vmax=1.0,
        interpolation="bilinear", zorder=2,
    )
    ax_r.set_title(r"(c) Back to robots ($\partial d / \partial \mathbf{x}_i(t)$)",
                   loc="left", fontsize=10.2, fontweight="bold", pad=6)

    trail_lcs = []
    past_lcs = []
    trail_cmap = MF.ATTR_CMAP
    trail_norm = Normalize(vmin=0.0, vmax=1.0)
    for _ in range(pts_mid.shape[1]):
        pl, = ax_r.plot([], [], color="#78808a", lw=0.40, alpha=0.28, zorder=3)
        lc = LineCollection([], cmap=trail_cmap, norm=trail_norm, linewidths=1.55, alpha=0.92, zorder=5)
        ax_r.add_collection(lc)
        past_lcs.append(pl)
        trail_lcs.append(lc)
    rob_scat = ax_r.scatter(pts_mid[0, :, 0], pts_mid[0, :, 1], s=26.0,
                            c=np.full(pts_mid.shape[1], 0.15), cmap=trail_cmap, vmin=0.0, vmax=1.0,
                            edgecolors="#78808a", linewidths=0.75, zorder=7)

    # Legend cleanly under Panel (b) & colorbar cleanly under Panel (c)
    seed_leg = [
        Line2D([], [], ls="", marker="^", ms=6.2, mfc=MF.CRIT_C, mec="white", label="max"),
        Line2D([], [], ls="", marker="v", ms=6.2, mfc=MF.CRIT_C, mec="white", label="min"),
        Line2D([], [], ls="", marker="X", ms=6.2, mfc=MF.CRIT_C, mec="white", label="saddle"),
    ]
    ax_s.legend(handles=seed_leg, loc="upper center", bbox_to_anchor=(0.5, -0.025), ncol=3, frameon=False,
                fontsize=8.2, handletextpad=0.25, columnspacing=0.9)

    fig.canvas.draw()
    pos_d, pos_s, pos_r = ax_d.get_position(), ax_s.get_position(), ax_r.get_position()
    cax = fig.add_axes([pos_r.x0 + 0.02, 0.145, pos_r.width - 0.04, 0.024])
    cb = fig.colorbar(im_rob, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    L.bold_colorbar(cb, r"gradient attribution (sqrt, max = 1)", size=8.0)

    mid_y = 0.5 * (pos_s.y0 + pos_s.y1)
    arr_ab = fig.text(0.5 * (pos_d.x1 + pos_s.x0), mid_y, r"$\Longrightarrow$", ha="center", va="center",
                      fontsize=17, fontweight="bold", color="#d2d0c8")
    arr_bc = fig.text(0.5 * (pos_s.x1 + pos_r.x0), mid_y, r"$\Longrightarrow$", ha="center", va="center",
                      fontsize=17, fontweight="bold", color="#d2d0c8")
    banner = fig.text(
        0.5, 0.038, "", ha="center", va="center", fontsize=9.8, fontweight="bold", color=L.INK,
        bbox=dict(facecolor="#f2f1ed", edgecolor="#c3c2b7", boxstyle="round,pad=0.36"),
    )

    # Precompute 24 trajectory frame indices covering 0..120s
    t_indices = np.linspace(35, len(pts_mid) - 1, 24, dtype=int)

    frames = []
    # Stage 1 (8 frames): Persistence diagram split activates
    for f in range(8):
        u = (f + 1) / 8.0
        s = u * u * (3.0 - 2.0 * u)
        sizes = s * (35.0 + 360.0 * top_sh_n)
        sc_diag_hi.set_sizes(sizes)
        ann_ext0.set_alpha(s)
        if ann_ext0.arrow_patch is not None:
            ann_ext0.arrow_patch.set_alpha(s)
        arr_ab.set_color("#c4b5fd" if f < 5 else MF.CRIT_C)
        banner.set_text(
            "[1/3] Persistence split:  SW1 distance splits onto top diagram pairs (Ext0 min→max dominates)"
        )
        frames.append(_fig_to_pil(fig, dpi=95))

    # Stage 2 (10 frames): 49 critical seeds bloom on psi_sigma3
    for f in range(10):
        u = (f + 1) / 10.0
        s = u * u * (3.0 - 2.0 * u)
        arr_ab.set_color(MF.CRIT_C)
        arr_bc.set_color("#c4b5fd" if f < 7 else MF.CRIT_C)
        for sc_k, target_s in zip(crit_scats, crit_s_full):
            sc_k.set_sizes([s * target_s])
        for an in rot_anns:
            an.set_alpha(s)
            if an.arrow_patch is not None:
                an.arrow_patch.set_alpha(s)
        banner.set_text(
            "[2/3] Seed back-propagation:  100% of attribution lands on 49 critical extrema & saddles (0% regular)"
        )
        frames.append(_fig_to_pil(fig, dpi=95))

    # Stage 3 (24 frames): Robots patrolling 0..120s with live gradient magnitude & cum map
    for t_end in t_indices:
        arr_bc.set_color(MF.CRIT_C)
        secs = int(round(120.0 * t_end / (len(pts_mid) - 1)))
        M_k = cum_map_scaled(t_end + 1)
        im_rob.set_data(np.ma.masked_where(~inside, M_k))

        t_start = max(0, t_end - 220)
        cur_pts = pts_mid[t_end]
        cur_g = np.clip(flow_smooth[t_end] / max(g_hi, 1e-12), 0.0, 1.0)
        cur_g_vis = np.sqrt(cur_g)
        for i in range(pts_mid.shape[1]):
            past_lcs[i].set_data(pts_mid[:t_end + 1:6, i, 0], pts_mid[:t_end + 1:6, i, 1])
            pts_i = pts_mid[t_start:t_end + 1:2, i, :]
            if len(pts_i) >= 2:
                segs = np.stack([pts_i[:-1], pts_i[1:]], axis=1)
                g_seg = np.clip(flow_smooth[t_start:t_end + 1:2, i][:-1] / max(g_hi, 1e-12), 0.0, 1.0)
                trail_lcs[i].set_segments(segs)
                trail_lcs[i].set_array(np.sqrt(g_seg))
        rob_scat.set_offsets(cur_pts)
        rob_scat.set_sizes(22.0 + 130.0 * cur_g_vis)
        rob_scat.set_array(cur_g_vis)

        banner.set_text(
            f"[3/3] Gradient back to robots (t = {secs:3d} / 120 s):  "
            + r"$\partial d / \partial \mathbf{x}_i(t)$ spikes at vortex turns (50% in 13.5% area)"
        )
        frames.append(_fig_to_pil(fig, dpi=95))

    # Hold final frame
    for _ in range(10):
        frames.append(frames[-1])

    plt.close(fig)
    _save_gif(frames, ASSETS / "attribution_to_robots.gif", fps=8, colors=256)


def main():
    targets = set(sys.argv[1:]) or {"footprint", "swarm", "persistence", "phase", "attribution"}
    if "footprint" in targets:
        make_footprint_gif()
    if "swarm" in targets:
        make_swarm_fields_gif()
    if "persistence" in targets:
        make_persistence_gif()
    if "phase" in targets:
        make_phase_diagrams_gif()
    if "attribution" in targets:
        make_attribution_gif()


if __name__ == "__main__":
    main()
