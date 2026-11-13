"""
io.py - reading the recordings
==============================
The ARGoS recordings (positions every 0.1 s) are read into one cached file, with the velocities by
forward differences; a recording that cannot be read stops the run.

Author: Olivier Vertu Ndinga Oba
"""
import os
import pickle
import time
import numpy as np
from pathlib import Path

from .config import ARGOS_ARENA, ARGOS_DT, DATA_DIR, BEHAVIORS, TOP_CONTROLLERS
from .checkpoint import save_pickle
from .progress import progress


def read_argos_data(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    pstart = lines.index('# Positions')
    spaces = [i for i, x in enumerate(lines) if x == '']
    positions = np.vstack([
        np.fromstring(line, sep=',') for line in lines[pstart + 1:spaces[0]]
    ])
    x_argos = np.transpose(positions[:, ::2])
    y_argos = np.transpose(positions[:, 1::2])
    x = -y_argos + ARGOS_ARENA / 2
    y = x_argos + ARGOS_ARENA / 2
    vx = (x[1:] - x[:-1]) / ARGOS_DT
    vy = (y[1:] - y[:-1]) / ARGOS_DT
    pos = np.stack([x[:-1], y[:-1]], axis=2).astype(np.float32)
    vel = np.stack([vx, vy], axis=2).astype(np.float32)
    return pos, vel


def load_trajectories(output_root: Path):
    meta_path = output_root / 'meta.pkl'
    if meta_path.exists():
        print(f"Loading cached metadata from {meta_path}")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        return meta['trajectories'], meta['labels'], meta['ranks']

    print("Loading trajectories from CSV files...")
    t0 = time.time()
    file_list = []
    for b_idx, beh in enumerate(BEHAVIORS):
        bdir = DATA_DIR / beh
        if not bdir.exists():
            print(f'  WARNING: {beh} missing at {bdir}')
            continue
        folders = sorted([f for f in os.listdir(bdir) if os.path.isdir(bdir / f)])
        for rank, c_idx in enumerate(TOP_CONTROLLERS[beh][:10]):
            if c_idx < len(folders):
                fpath = bdir / folders[c_idx]
                for fname in sorted(os.listdir(fpath)):
                    if fname.endswith('.csv') and not fname.startswith('robot_'):
                        file_list.append((fpath / fname, b_idx, rank))

    trajectories, labels_list, ranks_list, failed = [], [], [], []
    for idx, (fpath, b_idx, rank) in enumerate(file_list):
        try:
            pos, vel = read_argos_data(str(fpath))
        except Exception as err:                       # no run is dropped in silence
            failed.append(f'{fpath}: {err!r}')
            continue
        trajectories.append({'pos': pos, 'vel': vel})
        labels_list.append(b_idx)
        ranks_list.append(rank)
        if (idx + 1) % 500 == 0:
            progress('loaded', idx + 1, len(file_list))
    if failed:
        raise RuntimeError(f'{len(failed)} recordings could not be read:\n' + '\n'.join(failed))

    labels = np.array(labels_list, dtype=np.int32)
    ranks = np.array(ranks_list, dtype=np.int32)

    for b_idx, beh in enumerate(BEHAVIORS):
        count = int((labels == b_idx).sum())
        print(f'  [{beh:<20}] {count:3d} runs')

    print(f'Total: {len(trajectories)} trajectories loaded in {time.time() - t0:.1f}s')
    save_pickle({'trajectories': trajectories, 'labels': labels, 'ranks': ranks}, meta_path)
    print(f'Metadata cached to {meta_path}\n')
    return trajectories, labels, ranks