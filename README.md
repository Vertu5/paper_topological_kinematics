# Recovering a Taxonomy of Collective Behaviors from Motion Alone

From the tracked positions of a group of agents, the method recovers a taxonomy of
collective behaviors: how many distinct behaviors a set of recordings contains, which
recordings share each behavior, one real recording (the exemplar) that represents it,
and how the behaviors group across resolutions, from sub-behaviors to families.
Because the computation from the trajectories to the distances is differentiable,
the method also says what in the motion sets two behaviors apart: a distance is
traced back to critical points of the fields (sinks, sources, rotation centers,
saddles) and to the robots that shaped them.

Paper: [`manuscript/paper.pdf`](manuscript/paper.pdf).

## Method 

Each agent is spread over a Gaussian footprint of width sigma. The zeroth, first
and second moments of the velocity distribution give the mass density, the
momentum density and the kinetic intensity. Under rotation of the plane they
reduce to four invariant scalar fields: the density `rho`, the kinetic intensity
`K`, and the two Helmholtz-Hodge potentials `phi` (divergence) and `psi`
(vorticity), all in closed form. A fifth field, the Jacobi determinant
`J = sigma^2 det(grad phi, grad psi)`, records the coupling of the two potentials,
which their separate persistence diagrams cannot see. The five run-averaged fields
are evaluated exactly on grid-free seeds and summarized by extended persistence (six
diagrams per field, lower-star filtration on an alpha complex) at four data-driven
scales, and runs are compared by the sliced Wasserstein distance of order one.
The 20 distance matrices (5 fields x 4 scales) are fused by
**balanced rank fusion + cross-scale SNF**: rank-normalize every matrix (the mutual rank of
every pair, then its percentile rank among all pairs), average the
fields per scale with equal weight per moment (density from M0, kinetic intensity
from M2, flux M1 shared by phi, psi and J), and diffuse the four scale views with Similarity Network Fusion.
Affinity Propagation on the consensus returns the number of behaviors, the
partition and the exemplars. Swept over the neighborhood size and the diffusion length,
the partitions form a resolution map: its stable states give the levels of the
taxonomy, and its transitions scale with the size of the dataset.

## Repository layout

```
run_pipeline.py          stages 1-3: seeds + fields, extended persistence, SWD distances
run_sweep.py             clustering: grid, extended grid, standard SNF, ablation, taxonomy (plateau consensus)
pipeline/                stage implementations, data loading, configuration
src/gaussian_splatting/  closed-form fields, grid-free seeds, alpha complex, persistence, SWD (CPU and GPU)
src/classification/      ranks, SNF, affinity propagation (CPU and GPU), fusion.py (the paper method, plateau consensus)
analysis/                experiments, figures and numbers of the paper (see analysis/README.md)
notebooks/               00 every figure of the paper in order; 01-07 one notebook per step of the method, explained
figures/                 generated figures
manuscript/              paper.tex (the paper), sections/, numbers.tex and tables/ (generated), bibliography, SymPy checks
related_work/            notes on the reference papers of the related work (the PDFs stay local)
make_zenodo_bundle.sh    builds the Zenodo deposit: the code and the derived results
make_submission_bundle.sh builds the submission to Swarm Intelligence (submission/ and submission.zip, local only)
make_arxiv_bundle.sh     builds the arXiv deposit (arxiv/ and arxiv.tar.gz, local only)
data/, results/          local only (git-ignored): raw ARGoS tracks and computed checkpoints
```

## Reproducing the results

```bash
pip install -r requirements.txt
nohup nice -n 19 bash analysis/run_all.sh > results/run_all.log 2>&1 &
```

`analysis/run_all.sh` computes everything in order: stages 1-3 (`run_pipeline.py`: fields at the seeds,
extended persistence, sliced Wasserstein distances), the checks of the code against the formulas
(`analysis/verify_code.py`), the taxonomies (`run_sweep.py`), every analysis of the paper, the figures, the
numbers (`manuscript/numbers.tex`, `manuscript/tables/`), the notebooks (executed into `results/notebooks/`
when `nbconvert` and `ipykernel` are installed), the Lean proofs (when `elan` is installed) and the PDF. While
writing the numbers, `make_numbers.py` also checks on the results every statement of the text that the numbers
alone do not carry (which missions merge first, which datasets are hardest...) and prints `CLAIM FAILS` for any
that no longer holds. It can be stopped at any time and launched again: every step continues where it
stopped, and a result is reused only if it was computed from the current inputs.
`FROM=<step> bash analysis/run_all.sh` starts at a given step.

On a GPU (CUDA, with torch and triton) the fields, the distances, the fusion and affinity propagation run there;
the distances, the fusion and affinity propagation give the same numbers as the CPU code bit for bit, the fields
agree to float32 rounding. Affinity propagation uses the settings of its original implementation (damping 0.9,
convergence after 100 unchanged iterations, at most 1000). One scale ladder, derived from all 2500 runs, is
shared by every Top-n.

## Data

The runs analyzed are those of Legarda Herranz et al. (2026), publicly available at
https://github.com/glegarda/swarm-tda. In each run, 20 e-puck
robots are simulated in ARGoS, in a dodecagonal arena of 4.91 m^2, and their positions are recorded every 0.1 s.
All the robots execute the same controller, a probabilistic finite-state machine designed by AutoMoDe-Chocolate,
which Hasselmann et al. (Nat. Commun. 2021) ran ten times on each of five missions (Directional Gate, Foraging,
Homing, Shelter, XOR Aggregation); each controller is simulated in 50 runs. Top-n pools the runs of the n best
controllers of every mission (250 n runs).

## License

The code is released under the MIT License (`LICENSE`); the Springer Nature template files in `manuscript/`
(`sn-jnl.cls`, `sn-apacite.bst`) keep their own licenses. The derived results on Zenodo are released under the
Creative Commons Attribution 4.0 International license.

## Archive and publication

- **Zenodo deposit:** [`10.5281/zenodo.23440748`](https://doi.org/10.5281/zenodo.23440748), built with `make_zenodo_bundle.sh` (code and derived results).
- **Manuscript builds:** `manuscript/build.sh` compiles `manuscript/paper.pdf` (full arXiv version with appendices), `manuscript/journal.pdf` (Swarm Intelligence main manuscript), and `manuscript/supplementary.pdf` (Supplementary Information) from the same source.
- **Standalone bundles:** `make_submission_bundle.sh` and `make_arxiv_bundle.sh` assemble self-contained LaTeX bundles for Swarm Intelligence and arXiv (`submission/` and `arxiv/`, kept local).
- **Verification:** `analysis/verify_code.py` checks the implementation against closed-form formulas, `analysis/make_numbers.py` verifies all numerical macros and textual claims in `manuscript/`, and `manuscript/checks/` checks margins and float placement across all three PDFs.
