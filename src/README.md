# `src/` - the method as a library

```
src/
├── gaussian_splatting/   fields in closed form, seeds, alpha complex, extended persistence, sliced Wasserstein
└── classification/       rank normalisation, similarity network fusion, affinity propagation, plateau consensus
```

Every computation that runs on the GPU has a CPU twin. Stage 3 (distances), the fusion (SNF) and affinity
propagation give the same numbers bit for bit on both; the fields agree to float32 rounding
(`analysis/verify_code.py`, `analysis/precision_check.py`).

## `gaussian_splatting/`

| Module | Role |
|---|---|
| `fields_cpu.py`, `fields_gpu.py` | The five run-averaged fields in closed form at any points: `rho`, `K` (Gaussian footprint, window 5 sigma), `phi`, `psi` (Helmholtz-Hodge potentials through the radial kernel `kappa_sigma`, with `kappa_sigma(0) = 1/2`), `J = sigma^2 det(grad phi, grad psi)` with analytic gradients |
| `scale_ladder.py` | `sigma_min = max(d_min / 2, 0.01 m)`, `sigma_max = D_s / (2.35 k)` with the cap k chosen without labels (k = 10), M geometric scales shared by all runs |
| `seeds_gridfree.py` | Seeds of a run at a scale: farthest-point coreset of the agent positions (no position farther than sigma / 2), barycentres of the flattest Delaunay triangles of each field, zeros of J by bisection; fields at the seeds |
| `alpha.py` | Alpha complex of the seeds (GUDHI), truncated at squared radius (3 sigma)^2 |
| `persistence.py` | Extended persistence of a field on that complex: six diagrams Ord0, Ord1, Rel1, Rel2, Ext0, Ext1 (no pair dropped) |
| `distance.py`, `swd_gpu.py` | Sliced-Wasserstein distance between diagrams (25 directions, diagonal projections), CPU reference and GPU twin |

## `classification/`

| Module | Role |
|---|---|
| `fusion.py` | The method of the paper: mutual then percentile ranks, equal weight per physical source (1/3 density, 1/3 kinetic intensity, 1/9 each for phi, psi, J), one view per scale, one SNF across the scales, affinity propagation at every setting (`sweep`), plateau consensus. `hierarchical_snf` (three stacked SNF stages) is kept for the comparison of fusion designs |
| `affinity.py` | Mutual ranks (Obayashi and Kinoshita 2009) and percentile ranks of a distance matrix, the affinity `1 - rank`, the affinity of standard SNF (Wang et al. 2014) for the baseline |
| `snf.py` | Similarity network fusion; one diffusion gives every number of iterations t |
| `ap.py` | Affinity propagation: scikit-learn on the CPU, the same arithmetic on the GPU for a batch of matrices |
| `kernels_gpu.py` | Triton kernels of `snf.py` and `ap.py` (fixed summation order, no fused multiply-add) |
