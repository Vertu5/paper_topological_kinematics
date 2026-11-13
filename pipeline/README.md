# `pipeline/` - stages 1 to 3

Called by [`run_pipeline.py`](../run_pipeline.py). The classification (stage 4) is
[`run_sweep.py`](../run_sweep.py) with [`src/classification/fusion.py`](../src/classification/fusion.py).

| File | Role |
|---|---|
| `config.py` | The constants of the method (scales, coreset, Jacobi seeds, bisection, window, alpha cut, directions), listed in Appendix B of the paper and checked in `analysis/parameter_study.py`; execution settings |
| `io.py` | Reads the ARGoS recordings (positions, velocities by forward differences) into `meta.pkl`; a recording that cannot be read stops the run |
| `stage1_seeds.py` | Seeds and the five fields at the seeds, for every run and scale (GPU when available) |
| `stage2_persistence.py` | Extended persistence of every field on the alpha complex of the seeds (GUDHI) |
| `stage3_distances.py` | Sliced-Wasserstein distances between all pairs of runs, six diagrams summed per (field, scale) (GPU when available, same numbers as the CPU) |
| `checkpoint.py` | Resumable computations: atomic writes, settings kept per stage folder, manifests and input fingerprints |
| `progress.py` | Progress display |

Every stage can be interrupted and launched again: it continues where it stopped. A stage folder keeps the settings
and the code (its executable part) it was computed with, and refuses to be continued with other ones.

Outputs: `results/epuck/{meta.pkl, scale_ladder.pkl, stage1_alpha_fields/, stage2_alpha_dgms/, stage3_distances/dist_multiscale.pkl}`.
