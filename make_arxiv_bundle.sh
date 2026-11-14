#!/usr/bin/env bash
# Build the arXiv submission of the paper (nothing is uploaded): the LaTeX source of the version for publication
# (Springer Nature template, the same as the journal submission),
# the generated numbers and tables, the figures it uses (real files: arXiv refuses symbolic links) and the
# compiled bibliography (arXiv does not run BibTeX), then check that it compiles alone in an empty folder.
# Usage: bash make_arxiv_bundle.sh   ->  arxiv/ and arxiv.tar.gz (both git-ignored)
set -euo pipefail
cd "$(dirname "$0")"
OUT=arxiv; rm -rf "$OUT" arxiv.tar.gz; mkdir -p "$OUT/sections" "$OUT/tables" "$OUT/figures"
M=manuscript
( cd $M && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex > /dev/null )   # fresh .bbl
cp $M/paper.tex $M/paper.bbl $M/preamble.tex $M/numbers.tex $M/sn-jnl.cls $M/sn-apacite.bst "$OUT/"   # the Springer Nature class travels with the source
cp $M/sections/*.tex "$OUT/sections/"
cp $M/tables/*.tex "$OUT/tables/"
# the figures the source includes, through \figin, \includegraphics or the thumbnails of Figure 1
python3 - "$OUT" <<'EOF'
import re, shutil, sys
from pathlib import Path
out = Path(sys.argv[1]); src = ""
for f in [out / "paper.tex", out / "preamble.tex", *sorted((out / "sections").glob("*.tex"))]:
    src += re.sub(r"(?<!\\)%.*", "", f.read_text())
names = set(re.findall(r"figures/([A-Za-z0-9_]+)\.pdf", src)) | {f"{n}" for n in re.findall(r"\\thumb\{([A-Za-z0-9_]+)\}", src)}
for n in sorted(names):
    shutil.copy(Path("figures") / f"{n}.pdf", out / "figures" / f"{n}.pdf")
print(len(names), "figures copied")
EOF
tar -czf arxiv.tar.gz -C "$OUT" .
# compile the archive alone, twice, as arXiv does with pdflatex
TEST=$(mktemp -d); tar -xzf arxiv.tar.gz -C "$TEST"
( cd "$TEST" && pdflatex -interaction=nonstopmode paper.tex > /dev/null && pdflatex -interaction=nonstopmode paper.tex > /dev/null ) \
  || { echo "COMPILATION FAILED in $TEST"; rm -rf "$TEST"; exit 1; }
errors=$(python3 -c "print(sum(l.startswith('! ') for l in open('$TEST/paper.log', encoding='latin-1')))")
echo "compiles alone: $(pdfinfo "$TEST/paper.pdf" | awk '/^Pages/{print $2}') pages, $errors LaTeX errors;" \
     "undefined references: $(grep -cE 'undefined|multiply defined' "$TEST/paper.log" || true);" \
     "missing files: $(grep -c -i 'not found' "$TEST/paper.log" || true)"
if [ "$errors" != 0 ] || [ "$(pdftotext "$TEST/paper.pdf" - 2>/dev/null | md5sum)" != "$(pdftotext "$M/paper.pdf" - 2>/dev/null | md5sum)" ]; then
  echo "standalone arXiv compile differs from $M/paper.pdf"; rm -rf "$TEST"; exit 1
fi
rm -rf "$TEST"
du -sh arxiv.tar.gz
