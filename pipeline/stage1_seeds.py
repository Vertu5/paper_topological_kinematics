"""
stage1_seeds.py - stage 1: the seeds and the five fields at the seeds
====================================================================
For every run and scale: the coreset, the seeds and the fields at the seeds (seeds_gridfree.py), on the
GPU when available. Resumable: each (run, scale) is saved atomically and skipped when present; the
folder keeps its settings and refuses to be continued with other ones.

Author: Olivier Vertu Ndinga Oba
"""
from collections import defaultdict
from pathlib import Path

from joblib import Parallel, delayed

from .checkpoint import code_fingerprint, guard_settings, save_pickle, write_manifest
from .config import (COVER, N_CORESET, N_CORESET_MAX, N_JACOBI, N_BIS, FRAC_LOW_GRAD, DEDUP_TOL, WINDOW,
                     N_THREADS_SEEDS, BATCH_SEEDS)
from .progress import progress

from gaussian_splatting.fields_cpu import AVAILABLE_FIELDS
from gaussian_splatting.seeds_gridfree import (CPUPointComputer, coreset, coreset_orders_gpu, coresets_gpu,
                                               delaunay_structure, prefix_for_radius, seeds_from_coresets)


def _save(stage_dir, ti, si, sigma, seeds, fields, n_core):
    """The seeds (the first n_core are the coreset) and the five fields at the seeds, atomically."""
    save_pickle({'seeds': seeds, 'fields': fields, 'sigma': sigma, 'sigma_idx': si, 'traj_idx': ti, 'n_core': int(n_core)},
                stage_dir / f'traj{ti:04d}_s{si}.pkl')


def _missing(stage_dir, ti, n_scales):
    return [si for si in range(n_scales) if not (stage_dir / f'traj{ti:04d}_s{si}.pkl').exists()]


def _run_batch(batch, trajectories, ladder, stage_dir, computer, computer_J):
    """One batch of runs. With a fixed coreset size, the coresets and their Delaunay triangulations are
    computed once (they do not depend on sigma); with COVER, the coreset at each scale is the shortest
    farthest-point prefix that leaves no agent position farther than COVER * sigma."""
    pos_list = [trajectories[ti]['pos'] for ti in batch]
    vel_list = [trajectories[ti]['vel'] for ti in batch]
    gpu = not isinstance(computer, CPUPointComputer)
    if COVER is None:
        if gpu:                                    # same points bit for bit, on the GPU
            fixed = coresets_gpu(pos_list, N_CORESET)
        else:
            fixed = Parallel(n_jobs=min(N_THREADS_SEEDS, len(batch)), backend='threading')(
                delayed(coreset)(p, N_CORESET) for p in pos_list)
        fixed_struct = [delaunay_structure(c) for c in fixed]
    else:
        orders = coreset_orders_gpu(pos_list, N_CORESET_MAX, stop_radius=COVER * float(min(ladder)),
                                    device='cuda' if gpu else 'cpu')
    for si, sigma in enumerate(ladder):
        need = [k for k, ti in enumerate(batch) if si in _missing(stage_dir, ti, len(ladder))]
        if not need:
            continue
        if COVER is None:
            cores, structs = [fixed[k] for k in need], [fixed_struct[k] for k in need]
        else:
            cores = [orders[k][0][:prefix_for_radius(orders[k][1], COVER * float(sigma))] for k in need]
            structs = [delaunay_structure(c) for c in cores]
        res = seeds_from_coresets([pos_list[k] for k in need], [vel_list[k] for k in need], cores, computer,
                                  float(sigma), computer_J=computer_J, frac_low_grad=FRAC_LOW_GRAD,
                                  n_jacobi=N_JACOBI, n_bis=N_BIS, dedup_tol=DEDUP_TOL, structures=structs)
        for k, core, (seeds, fields) in zip(need, cores, res):
            _save(stage_dir, batch[k], si, float(sigma), seeds, fields, len(core))


def settings(ladder, backend, data_fingerprint):
    """Everything the stage-1 files depend on: a folder computed with other values is not resumed."""
    src = Path(__file__).resolve().parents[1] / 'src' / 'gaussian_splatting'
    return dict(stage=1, n_coreset=N_CORESET, cover=COVER, n_coreset_max=N_CORESET_MAX,
                frac_low_grad=FRAC_LOW_GRAD, n_jacobi=N_JACOBI, n_bis=N_BIS,
                dedup_tol=DEDUP_TOL, window=WINDOW, fields=list(AVAILABLE_FIELDS), backend=backend,
                ladder=[float(s) for s in ladder], data=data_fingerprint,
                code=code_fingerprint(*(src / f for f in ('fields_cpu.py', 'fields_gpu.py', 'seeds_gridfree.py'))))


def run_stage1(trajectories, ladder, backend, output_root: Path, data_fingerprint=''):
    stage_dir = output_root / 'stage1_alpha_fields'
    computer = computer_J = None
    if backend == 'gpu':
        try:
            import torch
            if torch.cuda.is_available():
                from gaussian_splatting.fields_gpu import TorchFieldComputer
                computer = TorchFieldComputer(fields=AVAILABLE_FIELDS, kappa=WINDOW)
                computer_J = TorchFieldComputer(fields=("J",), kappa=1e6)
        except ImportError:
            pass
        if computer is None:
            print('  [SPLAT] GPU unavailable, falling back to CPU')
    if computer is None:
        computer = CPUPointComputer(fields=AVAILABLE_FIELDS, kappa=WINDOW)
        computer_J = CPUPointComputer(fields=("J",), kappa=1e6)
    guard_settings(stage_dir, settings(ladder, 'gpu' if computer_J.__class__.__name__ != 'CPUPointComputer' else 'cpu',
                                       data_fingerprint))

    L = len(trajectories)
    todo = [ti for ti in range(L) if _missing(stage_dir, ti, len(ladder))]
    files = [stage_dir / f'traj{ti:04d}_s{si}.pkl' for ti in range(L) for si in range(len(ladder))]
    if not todo:
        print(f'  [SPLAT] {L}/{L} runs, all scales cached')
        write_manifest(stage_dir, files)
        return
    groups = defaultdict(list)                     # runs of equal shape are evaluated together
    for ti in todo:
        groups[trajectories[ti]['pos'].shape[:2]].append(ti)
    done = 0
    for _, group in groups.items():
        for b0 in range(0, len(group), BATCH_SEEDS):
            batch = group[b0:b0 + BATCH_SEEDS]
            _run_batch(batch, trajectories, ladder, stage_dir, computer, computer_J)
            done += len(batch)
            progress('[SPLAT] runs', done, len(todo))
    print(f'  [SPLAT] {len(todo)} runs x {len(ladder)} scales computed ({type(computer).__name__})')
    write_manifest(stage_dir, files)
