"""
stage3_distances.py - stage 3: the distances between runs
=========================================================
The sliced-Wasserstein distance of order one between the diagrams of every pair of runs, for every field and
scale, summed over the six sub-diagrams. The GPU computes all the pairs of one (field, scale) at once
(swd_field_condensed_gpu) and gives the same numbers, bit for bit, as the CPU routine
gaussian_splatting.distance.swd. Resumable: each (field, scale) is saved as soon as it is done,
and the final file is assembled when all twenty are there.

Author: Olivier Vertu Ndinga Oba
"""
import time
from pathlib import Path

import numpy as np

from gaussian_splatting.distance import sample_directions, swd
from gaussian_splatting.fields_cpu import AVAILABLE_FIELDS
from gaussian_splatting.persistence import N_SUB_DIAGRAMS

from .checkpoint import code_fingerprint, guard_settings, load_pickle, save_pickle, stage_fingerprint
from .config import N_DIRS, N_JOBS_SWD_CPU


def _rows_cpu(rows, diagrams, directions):
    """Condensed distances of rows i (pairs i < j) for one (field, scale), on the CPU."""
    out = []
    for i in rows:
        d_row = np.zeros(len(diagrams) - i - 1)
        for s in range(N_SUB_DIAGRAMS):
            d_row += swd(diagrams[i][s], [d[s] for d in diagrams[i + 1:]], directions)
        out.append(d_row)
    return np.concatenate(out) if out else np.zeros(0)


def _condensed_cpu(diagrams, n_jobs):
    from joblib import Parallel, delayed
    L = len(diagrams)
    directions = sample_directions(N_DIRS)
    chunks = np.array_split(np.arange(L - 1), max(1, min(L - 1, 8 * n_jobs)))
    parts = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_rows_cpu)(c, diagrams, directions) for c in chunks)
    return np.concatenate(parts)


def _condensed_gpu(diagrams):
    from gaussian_splatting.swd_gpu import swd_field_condensed_gpu
    return swd_field_condensed_gpu(diagrams, n_dirs=N_DIRS)


def settings(stage2_dir):
    here = Path(__file__).resolve()
    src = here.parents[1] / 'src' / 'gaussian_splatting'
    return dict(stage=3, n_dirs=N_DIRS, stage2=stage_fingerprint(stage2_dir),
                code=code_fingerprint(here, src / 'swd_gpu.py', src / 'distance.py'))


def run_stage3(L, ladder, backend, output_root: Path):
    stage2_dir = output_root / 'stage2_alpha_dgms'
    stage3_dir = output_root / 'stage3_distances'
    out_multiscale = stage3_dir / 'dist_multiscale.pkl'
    conf = settings(stage2_dir)
    guard_settings(stage3_dir, conf)
    if out_multiscale.exists():
        print('  [SWD]   Distance file exists, skip')
        return
    use_gpu = False
    if backend in ('gpu', 'auto'):
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            pass
    part_dir = stage3_dir / 'partial'
    iu = np.triu_indices(L, 1)
    t0 = time.time()
    for k in AVAILABLE_FIELDS:
        for si in range(len(ladder)):
            part = part_dir / f'{k}_s{si}.pkl'
            if part.exists():
                continue
            diagrams = [load_pickle(stage2_dir / f'traj{ti:04d}_s{si}.pkl')['dgms'][k] for ti in range(L)]
            dist = _condensed_gpu(diagrams) if use_gpu else _condensed_cpu(diagrams, N_JOBS_SWD_CPU)
            save_pickle(dist, part)
            print(f'  [SWD]   {k} sigma_{si + 1}: {len(dist):,} pairs on the {"GPU" if use_gpu else "CPU"} '
                  f'[{time.time() - t0:.0f}s]', flush=True)
    D = {k: {} for k in AVAILABLE_FIELDS}
    for k in AVAILABLE_FIELDS:
        for si in range(len(ladder)):
            M = np.zeros((L, L))
            M[iu] = load_pickle(part_dir / f'{k}_s{si}.pkl')
            D[k][si] = M + M.T
    save_pickle({'dist_matrices': D, 'field_keys': list(AVAILABLE_FIELDS), 'ladder': list(ladder), 'L': L,
                 'n_dirs': N_DIRS, 'settings': conf}, out_multiscale)
    for p in part_dir.glob('*.pkl'):
        p.unlink()
    part_dir.rmdir()
    print(f'  [SWD]   Saved to {out_multiscale.name} [{time.time() - t0:.0f}s]')
