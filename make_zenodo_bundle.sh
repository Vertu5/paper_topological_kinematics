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
# the code as committed, with the numbers and tables that make_numbers.py writes and the checks of the propositions
# (manuscript/verify_propositions.py, manuscript/lean); without the text of the paper, the scripts that build its
# deposits, the notebooks, the working notes and the figures (in figures.tar.gz)
git archive --format=tar.gz --prefix=paper_topological_kinematics/ -o "$OUT/code.tar.gz" HEAD -- . \
  ":(exclude)notebooks" ":(exclude)figures" ":(exclude)CLAUDE.md" ":(exclude)PUBLICATION.md" \
  ":(exclude)make_submission_bundle.sh" ":(exclude)make_arxiv_bundle.sh" \
  ":(exclude)manuscript/*.pdf" ":(exclude)manuscript/paper.tex" ":(exclude)manuscript/journal.tex" \
  ":(exclude)manuscript/supplementary.tex" ":(exclude)manuscript/preamble.tex" ":(exclude)manuscript/fig_overview.tex" \
  ":(exclude)manuscript/cover_letter.tex" ":(exclude)manuscript/sections" ":(exclude)manuscript/references.bib" \
  ":(exclude)manuscript/sn-jnl.cls" ":(exclude)manuscript/sn-apacite.bst" ":(exclude)manuscript/build.sh" \
  ":(exclude)manuscript/checks" ":(exclude)manuscript/README.md" ":(exclude)manuscript/figures"

cat > "$OUT/README.md" <<'TXT'
# Recovering a Taxonomy of Collective Behaviors from Motion Alone: code and derived results

Code and derived results of the paper *Recovering a Taxonomy of Collective Behaviors from Motion Alone*
(O. V. Ndinga Oba, 2026). From the tracked positions of a robot swarm, the method recovers a taxonomy of collective
behaviors: how many behaviors a set of recordings contains, which recordings share each behavior, one exemplar per
behavior, how the behaviors group across resolutions, and what in the motion distinguishes them. The code is that of
https://github.com/Vertu5/paper_topological_kinematics at the commit of the paper.

## Data
The results are computed on the 2500 runs of Legarda Herranz, Garone and Birattari (Research Square, 2026,
doi:10.21203/rs.3.rs-8672197/v1), available at https://github.com/glegarda/swarm-tda. In each run, simulated in
ARGoS, 20 e-puck robots execute one of the controllers that Hasselmann, Ligot, Ruddick and Birattari (Nat. Commun.
12, 4345, 2021, doi:10.1038/s41467-021-24642-3) designed with AutoMoDe-Chocolate, ten for each of five missions;
each controller is simulated in 50 runs. Please cite both works when using the runs.

## Contents
| Archive | Content |
|---|---|
| `distances.tar.gz` | the 20 matrices of distances between the runs (5 fields x 4 scales, 2500 x 2500): the sliced Wasserstein distance of order one (Carriere et al., 2017), summed over the six extended-persistence diagrams; the scales; the mission of each run and the rank of each controller within its mission by mean score, which defines the Top-n datasets; the settings of the distance |
| `diagrams.tar.gz` | the six extended-persistence diagrams of each run, field and scale |
| `fields_at_seeds.tar.gz` | the seeds of each run and scale, and the values of the five fields at the seeds |
| `clustering_results.tar.gz` | the resolution sweeps of Top-1 to Top-10 (99 settings) and the taxonomy of each dataset; the resolution maps (on Top-n, knn = 1, n, 2n, ..., 50n and t = 1 to 50), with the partition of every setting, from which the stable states and their transitions are read; the extended grid of Top-1; the comparison with standard similarity network fusion; the variants of the fusion; the ablations of the fields; the evaluation of the baselines; the separation of every pair of missions by every field; the attribution of the distances to the critical points and to the robots; the analysis of the XOR Aggregation runs placed with Shelter; the checks of affinity propagation and of rotation invariance |
| `baseline_distances.tar.gz` | the distances of the baselines: order parameters, CROCKER curves, distributions of field values |
| `parameter_study.tar.gz` | the study of the settings of the method on Top-1 |
| `figures.tar.gz` | the figures of the paper, in PDF and PNG |
| `code.tar.gz` | the code, the numbers and tables of the paper as `analysis/make_numbers.py` writes them, and the checks of its propositions (SymPy and Lean) |

## Regenerating the numbers
Unpack `code.tar.gz`, then the other archives at the root of the code, where their paths match `results/` and
`figures/`. `python analysis/make_numbers.py` then regenerates the numbers of the paper and checks the statements of
its text; three statements about the final positions of the robots are checked with the trajectories as well,
which `analysis/cache_top1.py` reads.

## License
The code is released under the MIT License (`LICENSE` in `code.tar.gz`); the derived results under the Creative
Commons Attribution 4.0 International license.

## How to cite
O. V. Ndinga Oba, *Recovering a Taxonomy of Collective Behaviors from Motion Alone: code and derived results*,
Zenodo, 2026, doi:10.5281/zenodo.23440748.
TXT
( cd "$OUT" && sha256sum * > SHA256SUMS )
du -sh "$OUT"/*
