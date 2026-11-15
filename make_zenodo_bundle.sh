#!/usr/bin/env bash
# make_zenodo_bundle.sh - the Zenodo deposit of the paper: its code and its derived results (built here, not uploaded).
# Build it from a clean commit: code.tar.gz is the committed code (git archive HEAD), and the results are those from
# which analysis/make_numbers.py wrote the committed manuscript/numbers.tex.
# Usage: bash make_zenodo_bundle.sh   ->  zenodo/  (archives + README + SHA256SUMS)
set -euo pipefail
cd "$(dirname "$0")"
OUT=zenodo; rm -rf "$OUT"; mkdir -p "$OUT"
R=results/epuck

tar -czf "$OUT/distances.tar.gz" "$R/stage3_distances/dist_multiscale.pkl" "$R/stage3_distances/SETTINGS.json" \
    "$R/scale_ladder.pkl" "$R/labels.npz"
tar -czf "$OUT/diagrams.tar.gz"  "$R/stage2_alpha_dgms"
tar -czf "$OUT/fields_at_seeds.tar.gz" "$R/stage1_alpha_fields"
# each result with its .source file, the fingerprint of the distances it was computed from, which make_numbers.py
# checks before reading it
tar -czf "$OUT/clustering_results.tar.gz" \
    $(ls $R/paper/*.pkl $R/paper/*.pkl.source | grep -v -e top1_tracks.pkl -e baselines.pkl) \
    results/invariance_check/report.pkl
[ -f "$R/paper/baselines.pkl" ] && tar -czf "$OUT/baseline_distances.tar.gz" "$R/paper/baselines.pkl" \
    $(ls $R/paper/baselines.pkl.source 2>/dev/null)
# the summaries of the study of the settings
tar -czf "$OUT/parameter_study.tar.gz" $(ls results/parameter_study{,_ext,_final}/summary.pkl{,.source} 2>/dev/null)
# the figures that the manuscript includes, in PDF and PNG
FIGS=$(grep -oh "figures/[A-Za-z0-9_]*\.pdf" manuscript/paper.tex manuscript/sections/*.tex | sort -u)
tar -czf "$OUT/figures.tar.gz" $FIGS $(for f in $FIGS; do [ -f "${f%.pdf}.png" ] && echo "${f%.pdf}.png"; done)
# the code as committed, with the checks of the code and propositions (analysis/verify_code.py,
# analysis/verify_propositions.py); without the text of the paper, the scripts that build its
# deposits, the notebooks, the working notes and the figures (in figures.tar.gz)
git archive --format=tar.gz --prefix=paper_topological_kinematics/ -o "$OUT/code.tar.gz" HEAD -- . \
  ":(exclude)notebooks" ":(exclude)figures" ":(exclude)assets" ":(exclude)CLAUDE.md" ":(exclude)PUBLICATION.md" \
  ":(exclude)make_submission_bundle.sh" ":(exclude)manuscript"

cat > "$OUT/README.md" <<'TXT'
# Recovering a Taxonomy of Collective Behaviors from Trajectories Alone: code and derived results

Code and derived results of the paper *Recovering a Taxonomy of Collective Behaviors from Trajectories Alone*
(O. V. Ndinga Oba, 2026). From the tracked positions of a robot swarm, the method recovers an unsupervised taxonomy of
collective behaviors: how many behaviors a set of recordings contains, which recordings share each behavior, one
exemplar per behavior, how the behaviors group across resolutions, and what in the motion distinguishes them. The code
matches https://github.com/Vertu5/paper_topological_kinematics at the commit of the paper.

## Data
The results are computed on the 2500 runs of Legarda Herranz, Garone and Birattari (Research Square, 2026,
doi:10.21203/rs.3.rs-8672197/v1), available at https://github.com/glegarda/swarm-tda. In each run, simulated in
ARGoS, 20 e-puck robots execute one of the controllers that Hasselmann, Ligot, Ruddick and Birattari (Nat. Commun.
12, 4345, 2021, doi:10.1038/s41467-021-24642-3) designed with AutoMoDe-Chocolate (ten controllers for each of five
missions, 50 runs per controller). Please cite both works when using the runs.

## Contents
| Archive | Content |
|---|---|
| `distances.tar.gz` | 20 pairwise distance matrices (5 fields x 4 scales, 2500 x 2500) using the sliced Wasserstein distance of order one (Carriere et al., 2017) summed over the six extended-persistence diagrams; scale ladder; mission labels and controller ranks defining Top-1 to Top-10; distance settings |
| `diagrams.tar.gz` | Six extended-persistence diagrams per run, field and scale |
| `fields_at_seeds.tar.gz` | Grid-free seeds of each run and scale, and the five field values at those seeds |
| `clustering_results.tar.gz` | Local resolution sweeps (99 settings on Top-1 to Top-10) and modal taxonomies; wide proportional resolution maps (`knn = 1, n, 2n, ..., 50n`, `t = 1..50`) and transition laws; fusion architecture comparisons and variants; field ablations; baseline evaluations; pairwise mission separation by field; attribution to critical points and robots; analysis of XOR Aggregation runs; Affinity Propagation and rotation-invariance checks |
| `baseline_distances.tar.gz` | Pairwise distances for the baselines: order parameters, CROCKER curves, and field-value distributions |
| `parameter_study.tar.gz` | Single-parameter sensitivity study on Top-1 |
| `figures.tar.gz` | All figures of the paper in PDF and PNG |
| `code.tar.gz` | Source code (`src/`, `pipeline/`, `analysis/`), numerical audit (`analysis/make_numbers.py`), and symbolic/numerical verification scripts (`analysis/verify_code.py`, `analysis/verify_propositions.py`) |

## Regenerating the numbers
Unpack `code.tar.gz`, then unpack the other archives at the root of the code so their paths match `results/` and
`figures/`. Running `python analysis/make_numbers.py` writes the numbers and tables of the paper (`manuscript/numbers.tex`
and `manuscript/tables/*.tex`) and verifies all 136 numerical claims of the text.

## License
The code is released under the MIT License (`LICENSE` in `code.tar.gz`), and the derived results under the Creative
Commons Attribution 4.0 International (CC BY 4.0) license.

## How to cite
O. V. Ndinga Oba, *Recovering a Taxonomy of Collective Behaviors from Trajectories Alone: code and derived results*,
Zenodo, 2026, doi:10.5281/zenodo.23440748.
TXT
( cd "$OUT" && sha256sum * > SHA256SUMS )
du -sh "$OUT"/*
