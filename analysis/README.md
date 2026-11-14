# `analysis/` - figures, tables, numbers and checks of the paper

| Script | Role | Output |
|---|---|---|
| `run_all.sh` | Every result of the paper in the order of their dependencies, from the recordings to the PDF (stages 1-3, checks, every analysis below, figures, numbers, notebooks, LaTeX). Resumable: launched again after an interruption, it continues where it stopped, and reuses a result only if it was computed from the current inputs | `results/`, `figures/`, `manuscript/` |
| `verify_code.py` | Checks that the code computes the formulas of the paper (kernel limit, Proposition 1, J, no persistence pair dropped) and that the GPU paths give the CPU numbers (fields to float32 rounding; distances, SNF and affinity propagation bit for bit) | exit status |
| `precision_check.py` | Where float64 is needed: float32 fields against float64, and their effect on the distances and on the zeros of J | printed |
| `parameter_study.py` | Each constant of the method varied alone on Top-1: the smallest value whose distances and taxonomy no longer change, label-free (three plans: main, extension, final) | `results/parameter_study*/summary.pkl` |
| `ap_check.py` | Affinity propagation on Top-1 and Top-10 with a longer convergence window, a higher ceiling, another damping and another tie-breaking noise, and with the shorter window of scikit-learn, compared setting by setting with the paper's settings | `ap_check.pkl` |
| `cache_top1.py` | Tracks of the 250 Top-1 runs and the reference consensus (knn = 7, t = 10) read by the figures and notebooks | `top1_tracks.pkl`, `reference_top1.pkl` |
| `fusion_designs.py` | Six fusion architectures on Top-1, same distances and clustering | `designs_top1*.pkl` |
| `baselines.py` | Distances of the baselines: order parameters, CROCKER curves of positions, field-value distributions on the same seeds | `baselines.pkl` |
| `baselines_eval.py` | Scores every baseline with the paper's clustering over the 99 settings | `baselines_eval.pkl` |
| `attribution.py` | Back-propagation of one distance (D-Gate vs Foraging exemplars, psi at sigma_3) to the field at the seeds and to every robot position, for persistence and for value distributions; checked step by step against the stored pipeline outputs; stability under noise and reproducibility over the 50 runs of the controller (GPU if free) | `attribution.pkl` |
| `summary_combined.py`, `summary_comparison.py` | Persistence diagrams and value distributions combined, and compared dataset by dataset under the same fusion | `summary_combined.pkl`, `summary_comparison.pkl` |
| `value_ablation.py` | The J test of the field ablation on the value distributions: (phi, psi), (phi, psi, J), (rho, K, phi, psi), same fusion and 99 settings | `value_ablation.pkl` |
| `value_variants.py` | Cruder summaries of the fields: values at the robot-position seeds only, at the added seeds only, mean and standard deviation (Top-1, Top-4, Top-10) | `value_variants.pkl` (distance cache outside `paper/`) |
| `merge_analysis.py` | The XOR Aggregation runs that percentile ranks alone place with Shelter on Top-3 and Top-4: controllers, field distances, when the robots stop, final positions | `merge_analysis.pkl` |
| `mirror_check.py` | Distance between every Top-1 run and its mirror image (vorticity potential of opposite sign), for persistence and for value distributions, relative to within-mission distances | `mirror_check.pkl` |
| `illustrations.py` | Didactic figures of Sections 3-4 on synthetic swarms, with the closed forms of the pipeline: footprint, moments, elementary flows, coupling, persistence and sliced Wasserstein | `figures/fig_footprint, fig_moments, fig_flows, fig_potentials, fig_persistence` |
| `invariance_check.py` | Rotates five e-puck runs, reprocesses them, compares with the originals | `results/invariance_check/report.pkl` |
| `cache_display_fields.py` | Display-grid fields of the exemplars (GPU, same closed forms as the pipeline) | `display_fields.pkl` |
| `figlib.py` | Shared style: figures drawn at their width on the page, every text checked at 8.5 pt or more on saving; colour-blind-safe mission colours with text labels, field colormaps | - |
| `make_figures.py` | One function per figure | `figures/*.pdf, *.png` |
| `make_numbers.py` | Every number quoted in the paper and every generated table; it also checks on the results every statement of the text that the numbers alone do not carry, and prints `CLAIM FAILS` for any that no longer holds | `manuscript/numbers.tex`, `manuscript/tables/*.tex` |

The scripts run by `run_all.sh` appear in its order. `precision_check.py` and `parameter_study.py` are studies run
separately; `make_numbers.py` reads the summaries of the parameter study. The checks step of `run_all.sh` also runs
`analysis/verify_propositions.py` (the propositions in SymPy). Superseded scripts are in the local `archive/` (not
published).
