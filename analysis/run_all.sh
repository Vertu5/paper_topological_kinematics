#!/usr/bin/env bash
# run_all.sh - every result of the paper, from the recordings to the PDF, in the order of their
# dependencies. Every step continues where it stopped, and a result is reused only if it was computed
# from the current inputs; after an interruption, launch the same command again.
#
# Usage:  nohup nice -n 19 bash analysis/run_all.sh > results/run_all.log 2>&1 &
#         FROM=grid bash analysis/run_all.sh      (start at a given step, the earlier ones being done)
#
# Author: Olivier Vertu Ndinga Oba
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv_torch/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${JOBS:-16}
FROM=${FROM:-}

clean_shm() {    # shared-memory folders left by joblib workers whose process has ended
  for d in /dev/shm/joblib_memmapping_folder_*; do
    [ -e "$d" ] || continue
    pid=$(basename "$d" | sed -E 's/joblib_memmapping_folder_([0-9]+)_.*/\1/')
    kill -0 "$pid" 2>/dev/null || rm -rf "$d"
  done
}
trap clean_shm EXIT

step() {         # step <name> <command...>: run unless FROM names a later step
  local name=$1; shift
  if [ -n "$FROM" ] && [ "$FROM" != "$name" ]; then echo "-- skip $name"; return; fi
  FROM=""
  echo "== $(date '+%F %T') $name"
  "$@"
}

step pipeline     python run_pipeline.py                     # stages 1-3: fields, persistence, distances
step checks       bash -c "python analysis/verify_code.py && python analysis/verify_propositions.py"   # code = formulas; propositions
step grid         python run_sweep.py grid --n-jobs "$J"     # 99 settings on every Top-n
step taxonomy     python run_sweep.py taxonomy               # plateau consensus
step apcheck      python analysis/ap_check.py               # AP finishes: longer window, ceiling, damping, noise
step extended     python run_sweep.py extended --n-jobs "$J"
step standard     python run_sweep.py standard
step top1         python analysis/cache_top1.py              # Top-1 tracks and reference consensus
step designs      python analysis/fusion_designs.py
step baselines    python analysis/baselines.py               # order parameters, CROCKER, field values
step baselines2   python analysis/baselines_eval.py
step combined     python analysis/summary_combined.py
step summaries    python analysis/summary_comparison.py
step ablation     python run_sweep.py ablation --n-jobs "$J"
step values       python analysis/value_ablation.py "$J"
step variants     python analysis/value_variants.py
step merge        python analysis/merge_analysis.py
step mirror       python analysis/mirror_check.py
step invariance   python analysis/invariance_check.py
step display      python analysis/cache_display_fields.py
step attribution  python analysis/attribution.py
step illustrate   python analysis/illustrations.py
step fusion       python analysis/fusion_explore.py 1,2,3,4,5,6,7,8,9,10 "$J"   # each step of the fusion changed alone
run_study() {     # every constant varied alone on Top-1, in three plans, each resumable in its own folder
  python analysis/parameter_study.py
  STUDY_PLAN=extension STUDY_OUT="$PWD/results/parameter_study_ext" python analysis/parameter_study.py
  STUDY_PLAN=final STUDY_OUT="$PWD/results/parameter_study_final" python analysis/parameter_study.py
}
step study        run_study
step pairs        python analysis/make_figures.py pairs      # separation of every pair of missions by every field
step attrpairs    python analysis/attribution_pairs.py       # attribution on every pair of missions
step maps         python analysis/stable_states.py 1 2 3 4 5 6 7 8 9 10 --scaled --t-fixed   # resolution maps
step figures      python analysis/make_figures.py
step numbers      python analysis/make_numbers.py
run_notebooks() {     # every notebook runs; the copies with their outputs go to results/notebooks
  if python -c "import nbconvert, ipykernel" 2>/dev/null; then
    python -m nbconvert --to notebook --execute --output-dir results/notebooks notebooks/*.ipynb
  else
    echo "   not executed: install nbconvert and ipykernel in .venv_torch to run them here"
  fi
}
check_lean() {    # the propositions, proved in Lean
  if command -v lake >/dev/null 2>&1 || [ -x "$HOME/.elan/bin/lake" ]; then
    (cd manuscript/lean && PATH="$HOME/.elan/bin:$PATH" && lake build && lake env lean CheckAxioms.lean)
  else
    echo "   not checked: install elan (Lean 4) to build manuscript/lean"
  fi
}
step notebooks    run_notebooks
step lean         check_lean
step overview     bash -c "cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error fig_overview.tex > /dev/null && mv fig_overview.pdf ../figures/ && rm -f fig_overview.aux fig_overview.log"
step paper        bash -c "cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex"
echo "== $(date '+%F %T') all done"
