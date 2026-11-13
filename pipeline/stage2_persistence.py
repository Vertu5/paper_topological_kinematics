"""
stage2_persistence.py - stage 2: extended persistence
=====================================================
The six extended-persistence diagrams of the five fields on the alpha complex of the seeds, for every
run and scale (GUDHI, on the CPU). Resumable: each (run, scale) is saved atomically and skipped when
present.

Author: Olivier Vertu Ndinga Oba
"""
import os
from pathlib import Path

import gudhi

from gaussian_splatting.alpha import AlphaSimplexTree
from gaussian_splatting.persistence import PersistenceComputer

from .checkpoint import code_fingerprint, guard_settings, load_pickle, save_pickle, stage_fingerprint, write_manifest
from .config import C_ALPHA


def settings(stage1_dir):
    src = Path(__file__).resolve().parents[1] / 'src' / 'gaussian_splatting'
    return dict(stage=2, c_alpha=C_ALPHA, gudhi=gudhi.__version__, stage1=stage_fingerprint(stage1_dir),
                code=code_fingerprint(src / 'alpha.py', src / 'persistence.py'))


def _stage2_worker(ti, si, stage1_dir, stage2_dir, sigma):
    out_path = stage2_dir / f'traj{ti:04d}_s{si}.pkl'
    if out_path.exists():
        return
    saved = load_pickle(stage1_dir / f'traj{ti:04d}_s{si}.pkl')
    pc = PersistenceComputer(AlphaSimplexTree(saved['seeds'], alpha_sq_max=(C_ALPHA * sigma) ** 2))
    save_pickle({'dgms': pc.compute_all(saved['fields']), 'sigma': sigma, 'sigma_idx': si, 'traj_idx': ti}, out_path)


def run_stage2(L, ladder, output_root: Path):
    from joblib import Parallel, delayed
    stage1_dir = output_root / 'stage1_alpha_fields'
    stage2_dir = output_root / 'stage2_alpha_dgms'
    guard_settings(stage2_dir, settings(stage1_dir))
    n_jobs = min(os.cpu_count(), 18)
    for si, sigma in enumerate(ladder):
        todo = [ti for ti in range(L) if not (stage2_dir / f'traj{ti:04d}_s{si}.pkl').exists()]
        if todo:
            Parallel(n_jobs=n_jobs, backend='loky', batch_size=16)(
                delayed(_stage2_worker)(ti, si, stage1_dir, stage2_dir, float(sigma)) for ti in todo)
        print(f'  [EPH]   sigma={sigma:.4f} : {L}/{L} ({len(todo)} new, {n_jobs} workers)', flush=True)
    write_manifest(stage2_dir, [stage2_dir / f'traj{ti:04d}_s{si}.pkl' for ti in range(L) for si in range(len(ladder))])
