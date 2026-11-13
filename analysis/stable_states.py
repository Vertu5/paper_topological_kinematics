"""
stable_states.py - the resolution map over a wide window: its stable states and their transitions
==================================================================================================
The paper reads the count and the partition on the main grid (knn <= 25, t <= 25), at the resolution of one
controller. Here the same sweep (run_sweep.sweep, the code of the paper: rank normalization, one view per scale,
cross-scale SNF, affinity propagation with the method's settings) runs over a window that grows with the dataset
(--scaled --t-fixed, the window of the paper: on Top-n, knn = 1, n, 2n, ..., 50n and t = 1, ..., 50), so that the
stable states of the map (regions of settings where the partition does not change) and the transitions between them
can be read on every Top-n. Without --scaled, the window is fixed (knn = 3, ..., 50; t = 1, ..., 50).

Check built in: at the 99 settings of the main grid (grid.pkl), at the settings shared with the extended grid of Top-1
(extended_top1.pkl) and with the fixed window, every partition must be identical to the stored one.

    nohup nice -n 19 .venv_torch/bin/python analysis/stable_states.py [TOPS ...] --scaled --t-fixed [--cpu=N]
        -> results/epuck/paper/stable_states_scaled_top{n}.pkl  (one file per dataset, the partition of every setting)

The functions state_of, transition and power_law name the states and fit the law of the transitions; make_figures.py
and make_numbers.py use them, so that the figures and the text read the maps the same way.
"""
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import run_sweep as S  # noqa: E402   (sets the thread variables, then imports the paper's code)
from classification import fusion as F  # noqa: E402
from pipeline.checkpoint import load_pickle  # noqa: E402

KNNS = list(range(3, 51))
TS = list(range(1, 51))


def scaled_grid(top, t_grows):
    """The window that grows with the dataset (--scaled), the same linear grid in units of n on every Top-n: knn = 1
    (each recording alone: the views are only averaged), then n, 2n, ..., 50n, the window of Top-1 scaled with the
    number of runs; t = 1, then n, 2n, ..., 50n (t = 1, ..., 50 with --t-fixed)."""
    lin = lambda step: sorted({1} | {step * j for j in range(1, 51)})
    return lin(top), lin(top if t_grows else 1)


# --------------------------------------------------------------------------- reading the maps
ABBR = {"directional_gate": "G", "foraging": "F", "homing": "H", "shelter": "S", "xor_aggregation": "X"}
STATES = ("finer", "missions", "partial", "shelter_xor", "families", "two_families")


def named_clusters(pred, labels, behaviors):
    """The clusters of a partition named by the missions that hold at least a tenth of each, so that a few stray
    recordings do not make a merge while a block of recordings (a whole controller) does. A cluster of a single
    recording is set aside: it names no level. The missions name what the method finds; the method never sees them."""
    pred, labels = np.asarray(pred), np.asarray(labels)
    out = []
    for c in np.unique(pred):
        sel = pred == c
        if sel.sum() == 1:
            continue
        ys, cnt = np.unique(labels[sel], return_counts=True)
        out.append("".join(sorted(ABBR[behaviors[m]] for m, v in zip(ys, cnt) if v >= 0.1 * sel.sum())))
    return sorted(out)


def state_of(pred, labels, behaviors):
    """The state of the taxonomy that a partition reads: finer than the missions (each cluster inside one mission,
    more than five), the five missions, Shelter + XOR Aggregation merged, the three families (Directional Gate +
    Foraging, Homing, Shelter + XOR Aggregation), two families (Directional Gate + Foraging + Homing, Shelter + XOR
    Aggregation), or a mission partly merged with another."""
    cl = named_clusters(pred, labels, behaviors)
    if all(len(s) == 1 for s in cl) and len(set(cl)) == 5:
        return "missions" if len(cl) == 5 else "finer"
    return {("F", "G", "H", "SX"): "shelter_xor", ("FG", "H", "SX"): "families",
            ("FGH", "SX"): "two_families"}.get(tuple(cl), "partial")


def on_map(d):
    """The settings of a stored map that lie on its grid (the extra settings of the main grid are left out)."""
    grid = set(product(d["knns"], d["ts"]))
    return [r for r in d["results"] if (r["knn"], r["t"]) in grid]


def transition(d, behaviors, state="families", t_min=5):
    """For each t >= t_min, the smallest knn > 1 (in units of n: knn / n) at which the partition reads the state.
    Returns an array of (t, knn / n)."""
    n = int(round(max(d["knns"]) / 50))
    by = {(r["knn"], r["t"]): r["pred"] for r in d["results"]}
    pts = []
    for t in d["ts"]:
        if t < t_min:
            continue
        first = next((k for k in d["knns"] if k > 1 and state_of(by[(k, t)], d["labels"], behaviors) == state), None)
        if first is not None:
            pts.append((t, first / n))
    return np.array(pts, dtype=float)


def power_law(points):
    """Least-squares fit of knn / n = c t^(-alpha) on log-log axes; returns (c, alpha)."""
    P = np.asarray(points, dtype=float)
    slope, icpt = np.polyfit(np.log(P[:, 0]), np.log(P[:, 1]), 1)
    return float(np.exp(icpt)), float(-slope)


def stored(name):
    """A stored result of run_sweep, only if it was computed from the current distances."""
    src = S.OUT / (name + ".source")
    if not (S.OUT / name).exists() or not src.exists() or src.read_text().strip() != S.dist_source():
        return None
    return load_pickle(S.OUT / name)


def same_partitions(res, ref, label):
    """Compare the fine sweep with a stored sweep at the settings they share (C and the labels, run by run)."""
    fine = {(r["knn"], r["t"]): r for r in res}
    shared = [(r["knn"], r["t"]) for r in ref if (r["knn"], r["t"]) in fine]
    diff = [kt for kt in shared if fine[kt]["C"] != next(r for r in ref if (r["knn"], r["t"]) == kt)["C"]
            or not np.array_equal(np.asarray(fine[kt]["pred"]),
                                  np.asarray(next(r for r in ref if (r["knn"], r["t"]) == kt)["pred"]))]
    print(f"  check against {label}: {len(shared)} shared settings, {len(diff)} different"
          + (f" {diff[:5]}" if diff else ""), flush=True)
    return not diff


def main():
    # --cpu=N: the CPU path of the sweep with N workers (the same partitions as on the GPU, checked below), so that a
    # second dataset can run while the GPU is busy
    cpu = [int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--cpu=")]
    ap = dict(device="cpu", n_jobs=cpu[0]) if cpu else {}
    tops = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or [1, 2, 3, 4, 5, 6]
    scaled, t_grows = "--scaled" in sys.argv, "--t-fixed" not in sys.argv
    lab, rk, ctrl = S.load_labels()
    D, _ = S.load_distances()
    grid = stored("grid.pkl")
    ext = stored("extended_top1.pkl")
    for top in tops:
        name = f"stable_states_{'scaled_' if scaled else ''}top{top}.pkl"
        if S.done(name):
            print(f"Top-{top}: already computed from the current distances", flush=True)
            continue
        knns, ts = scaled_grid(top, t_grows) if scaled else (KNNS, TS)
        idx = S.subset(top, rk)
        t0 = time.time()
        views = F.scale_views(F.rank_normalise(D, F.ALL_FIELDS, idx), F.ALL_FIELDS)
        settings = list(product(knns, ts))
        if scaled:          # the 99 settings of the main grid as well, for the check (outside the map unless on it)
            settings += [kt for kt in product(S.KNNS, S.TS) if kt not in set(settings)]
        res = S.sweep(views, lab[idx], settings, **ap)
        ok = True
        if grid is not None and top in grid:
            ok &= same_partitions(res, grid[top]["results"], "the main grid (grid.pkl)")
        if top == 1 and ext is not None:
            ok &= same_partitions(res, ext["results"], "the extended grid (extended_top1.pkl)")
        if scaled and S.done(f"stable_states_top{top}.pkl"):
            ok &= same_partitions(res, load_pickle(S.OUT / f"stable_states_top{top}.pkl")["results"],
                                  "the fine grid with a fixed window")
        if not ok:
            sys.exit(f"Top-{top}: the fine sweep differs from the stored sweep; nothing saved")
        for r in res:
            r["pred"] = np.asarray(r["pred"], dtype=np.int16)
        S.save(dict(idx=idx, labels=lab[idx], controllers=ctrl[idx], ranks=rk[idx], results=res, knns=knns, ts=ts,
                    seconds=time.time() - t0, device=S.device_name()), name)
        C = np.array([r["C"] for r in res])
        vals, cnt = np.unique(C, return_counts=True)
        order = np.argsort(-cnt)[:6]
        print(f"Top-{top} (N={len(idx)}): {len(res)} settings in {time.time() - t0:.0f}s; most frequent C: "
              + ", ".join(f"{vals[i]} ({cnt[i]})" for i in order), flush=True)


if __name__ == "__main__":
    main()
