#!/usr/bin/env bash
# Build the three PDFs of the paper from one text:
#   paper.pdf          arXiv version, the paper with its appendices;
#   journal.pdf        Swarm Intelligence version, the paper alone;
#   supplementary.pdf  its Supplementary Information (the appendices, numbered S1, S2, ...).
# journal and supplementary read each other's labels (xr-hyper), so they are compiled in turn until both are stable.
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet paper.tex
run() { pdflatex -interaction=nonstopmode -halt-on-error "$1.tex" > /dev/null; }
run journal; run supplementary
bibtex -terse journal; bibtex -terse supplementary
for _ in 1 2 3; do run journal; run supplementary; done
for d in paper journal supplementary; do
  printf '%-14s %3s pages' "$d" "$(pdfinfo "$d.pdf" | awk '/^Pages/{print $2}')"
  if grep -q "There were undefined references\|Citation .* undefined\|multiply defined" "$d.log"; then
    echo "  WARNING: undefined or multiply defined references, see $d.log"
  else
    echo "  ok"
  fi
done
