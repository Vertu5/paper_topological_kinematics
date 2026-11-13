# `figures/` - generated figures

Every figure of the paper, in the order in which it appears. They are produced by
`analysis/illustrations.py` (synthetic examples) and `analysis/make_figures.py` (robot data), one
function per figure; `notebooks/00_paper_figures.ipynb` redraws them in the same order.
`manuscript/figures` is a symbolic link to this folder. Each figure is saved as PDF (in the paper,
without a creation date, so that the same figure always gives the same file) and as PNG. Each is drawn
at the width it has on the page, so that the font sizes in the code are those of the paper; on saving,
`figlib.save` reads that width in the LaTeX and warns if any text would be smaller than 8.5 pt.

| File | Label in the paper | Content | Script |
|---|---|---|---|
| `thumb_*` | `fig:overview` | thumbnails of the overview diagram (tracks, fields, diagram, distance, consensus, exemplars) | `make_figures.py` |
| `fig_footprint` | `fig:footprint` | the Gaussian footprint, its profile and the merging of two footprints | `illustrations.py` |
| `fig_moments` | `fig:moments` | from agents to fields on a synthetic configuration | `illustrations.py` |
| `fig_flows` | `fig:flows` | three elementary motions and their two potentials | `illustrations.py` |
| `fig_persistence` | `fig:persistence` | from a field to a distance: sweep, diagrams, sliced distance | `illustrations.py` |
| `fig_potentials` | `fig:potentials` | why the coupling field is needed | `illustrations.py` |
| `fig_pipeline_one` | `fig:one` | one recording followed through the method | `make_figures.py` |
| `fig_pipeline_all` | `fig:all` | the whole Top-1 collection, from distances to behaviours | `make_figures.py` |
| `fig_abstract` | (unnumbered, under the abstract) | the main results: the number of behaviours found, and their purity against the missions, as more controllers per mission are pooled | `make_figures.py` |
| `fig_fields` | `fig:fields` | the five fields of the exemplar of each behaviour of Top-1 | `make_figures.py` |
| `fig_resolution_compact` | `fig:resolution_two` | resolution maps of Top-1 and Top-10 | `make_figures.py` |
| `fig_resolution_map` | `fig:resolution` | resolution maps of the ten datasets (appendix) | `make_figures.py` |
| `fig_fusion_sequence` | `fig:fusion` | the hierarchy of missions on Top-1 and the merge order | `make_figures.py` |
| `fig_pair_separation` | `fig:pairs` | separation of every pair of missions by every field (Top-1) | `make_figures.py` |
| `fig_ablation` | `fig:ablation` | field ablation, purity and adjusted Rand index | `make_figures.py` |
| `fig_attribution` | `fig:attribution` | a distance carried back to the critical points and to the robots | `make_figures.py` |
| `fig_scales` | `fig:scales_all` | the five fields of one recording at the four scales (appendix) | `make_figures.py` |
| `figA_uncapped` | `fig:uncapped` | the fields near the merging width (appendix) | `make_figures.py` |
| `figA_cap` | `fig:cap` | the top of the ladder chosen without labels (appendix) | `make_figures.py` |
| `figA_merge` | `fig:merge` | XOR Aggregation runs placed in the Shelter cluster (appendix) | `make_figures.py` |

`figA_pooled` (mission-by-cluster tables of the pooled datasets) is drawn for
`notebooks/04_taxonomy.ipynb` only.
