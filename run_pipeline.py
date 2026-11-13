"""
run_pipeline.py - stages 1 to 3 of the method: fields, extended persistence, distances
=====================================================================================
Stage 1 builds the grid-free seeds of every run at every scale and evaluates the five fields there;
stage 2 computes the six extended-persistence diagrams of each field on the alpha complex of the
seeds; stage 3 computes the sliced-Wasserstein distance between every pair of runs for every field and
scale. The classification (stage 4) is run_sweep.py.

Stages 1 and 3 run on the GPU when one is available (same results as the CPU code: stage 3 bit for bit,
stage 1 to float32 rounding); stage 2 runs GUDHI on the CPU.

Every stage can be stopped and launched again: it continues where it stopped. A stage folder keeps
the settings and the code it was computed with, and refuses to be continued with other ones.

Usage:
    python run_pipeline.py                  # stages 1, 2 and 3
    python run_pipeline.py --stage splat    # stage 1 only (also: eph, swd, cluster)
    python run_pipeline.py --top 1          # only the runs of Top-1, in results/epuck_top1

Author: Olivier Vertu Ndinga Oba
"""
import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.chdir(ROOT)

from pipeline.checkpoint import file_fingerprint, load_pickle, save_pickle  # noqa: E402
from pipeline.config import (EPS_SIGMA, LADDER_INSTANTS, M_SCALES, OUTPUT_ROOT_DEFAULT,  # noqa: E402
                             SIGMA_MAX_FACTOR)
from pipeline.io import load_trajectories  # noqa: E402
from pipeline.stage1_seeds import run_stage1  # noqa: E402
from pipeline.stage2_persistence import run_stage2  # noqa: E402
from pipeline.stage3_distances import run_stage3  # noqa: E402


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_or_compute_ladder(trajectories, output_root: Path, n_scales: int):
    """The scale ladder, computed once from the data and kept with the settings it was computed with."""
    from gaussian_splatting.scale_ladder import compute_scale_ladder
    path = output_root / 'scale_ladder.pkl'
    conf = dict(M=n_scales, eps_sigma=EPS_SIGMA, max_factor=SIGMA_MAX_FACTOR, n_instants=LADDER_INSTANTS)
    if path.exists():
        saved = load_pickle(path)
        if all(saved.get(k) == v for k, v in conf.items()):
            print(f'Scale ladder loaded: {np.round(saved["ladder"], 5).tolist()}')
            return saved['ladder']
    ladder = compute_scale_ladder(trajectories, M=n_scales, epsilon_sigma=EPS_SIGMA,
                                  max_factor=SIGMA_MAX_FACTOR, n_instants=LADDER_INSTANTS)
    save_pickle(dict(ladder=ladder, **conf), path)
    print(f'Scale ladder computed: {np.round(ladder, 5).tolist()}')
    return ladder


STAGE_DIRS = {'splat': 'stage1_alpha_fields', 'eph': 'stage2_alpha_dgms', 'swd': 'stage3_distances'}


def timed(stage, output_root: Path, backend, work):
    """Run one stage and record its duration in timing.json; `from_scratch` tells whether the stage
    computed everything in this run (a resumed stage measures only the part it computed)."""
    folder = output_root / STAGE_DIRS[stage]
    fresh = not folder.exists() or not any(folder.iterdir())
    t0 = time.time()
    work()
    path = output_root / 'timing.json'
    timing = json.loads(path.read_text()) if path.exists() else {}
    if fresh or not timing.get(stage, {}).get('from_scratch'):    # keep a complete measurement
        timing[stage] = dict(seconds=round(time.time() - t0, 1), from_scratch=fresh, backend=backend)
        path.write_text(json.dumps(timing, indent=1, sort_keys=True) + '\n')
    return time.time() - t0


def main():
    parser = argparse.ArgumentParser(description='Stages 1-3: fields, extended persistence, distances')
    parser.add_argument('--stage', choices=['splat', 'eph', 'swd', 'cluster', 'all'], default='all')
    parser.add_argument('--scales', type=int, default=M_SCALES)
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()

    output_root = ROOT / (Path(f'results/epuck_top{args.top}') if args.top < 10 else OUTPUT_ROOT_DEFAULT)
    output_root.mkdir(parents=True, exist_ok=True)
    backend = 'gpu' if gpu_available() else 'cpu'

    trajectories, labels, ranks = load_trajectories(output_root)
    if args.top < 10:
        valid = np.where(ranks < args.top)[0]
        trajectories = [trajectories[i] for i in valid]
        print(f'Top-{args.top}: {len(trajectories)} runs.')
    L = len(trajectories)
    ladder = load_or_compute_ladder(trajectories, output_root, args.scales)
    stages = ['splat', 'eph', 'swd', 'cluster'] if args.stage == 'all' else [args.stage]
    print(f'{L} runs, {len(ladder)} scales, stages {stages}, backend {backend}, output {output_root}', flush=True)

    if 'splat' in stages:
        print('Stage 1: seeds and fields', flush=True)
        dt = timed('splat', output_root, backend, lambda: run_stage1(
            trajectories, ladder, backend, output_root, data_fingerprint=file_fingerprint(output_root / 'meta.pkl')))
        print(f'Stage 1 complete in {dt:.0f}s\n', flush=True)
    if 'eph' in stages:
        print('Stage 2: extended persistence (GUDHI, CPU)', flush=True)
        dt = timed('eph', output_root, 'cpu', lambda: run_stage2(L, ladder, output_root))
        print(f'Stage 2 complete in {dt:.0f}s\n', flush=True)
    if 'swd' in stages:
        print('Stage 3: sliced-Wasserstein distances', flush=True)
        dt = timed('swd', output_root, backend, lambda: run_stage3(L, ladder, backend, output_root))
        print(f'Stage 3 complete in {dt:.0f}s\n', flush=True)
    if 'cluster' in stages:
        if output_root != ROOT / OUTPUT_ROOT_DEFAULT:
            print('Stage 4: run_sweep.py reads results/epuck (all 2500 runs, one shared ladder); skipped.')
        else:
            t0 = time.time()
            subprocess.run([sys.executable, str(ROOT / 'run_sweep.py'), 'grid'], check=True)
            print(f'Stage 4 complete in {time.time() - t0:.0f}s\n', flush=True)
    print('Pipeline complete.')


if __name__ == '__main__':
    main()
