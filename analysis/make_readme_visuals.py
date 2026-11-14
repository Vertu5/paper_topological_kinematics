#!/usr/bin/env python3
"""
make_readme_visuals.py - animated educational GIFs for the repository README and notebooks
============================================================================================
Generates lightweight, publication-styled animated visualizations grounded directly in Paper 1's
closed-form equations and real Top-1 exemplar trajectories:
  1. assets/gaussian_splatting_demo.gif     - Two-robot Gaussian footprint & 2.35*sigma resolution limit
  2. assets/swarm_kinematic_fields.gif      - Live Top-1 exemplar trajectories unfolding into rho_sigma and psi_sigma
  3. assets/extended_persistence_demo.gif   - Sublevel/superlevel sweep -> Extended Persistence Diagram

Usage:
    python analysis/make_readme_visuals.py            # build all three GIFs
    python analysis/make_readme_visuals.py footprint  # build one by name: footprint | swarm | persistence
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


def main():
    targets = set(sys.argv[1:]) or {"footprint", "swarm", "persistence"}
    if "footprint" in targets:
        make_footprint_gif()
    if "swarm" in targets:
        make_swarm_fields_gif()
    if "persistence" in targets:
        make_persistence_gif()


if __name__ == "__main__":
    main()
