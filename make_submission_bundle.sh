#!/usr/bin/env bash
# Build the submission to Swarm Intelligence (nothing is uploaded): every file at the root of one folder, as the
# Springer Nature submission system expects. The manuscript is one .tex file (sections, tables and generated
# numbers inserted), with the unmodified sn-jnl.cls, the APA style sn-apacite.bst, the bibliography (.bib and the
# compiled .bbl) and the figures as separate files named after their number (Fig1.pdf, ..., FigC1.pdf). The folder
# is compiled alone to check it, and zipped. The cover letter is built next to it when manuscript/cover_letter.tex
# exists. The code, as committed, goes next to them as supplementary material (ESM_1.zip), which the journal asks
# for; it stays out of submission.zip, and the derived data stay on Zenodo.
# Usage: bash make_submission_bundle.sh   ->  submission/ and submission.zip (both git-ignored)
set -euo pipefail
cd "$(dirname "$0")"
OUT=submission; rm -rf "$OUT" submission.zip; mkdir -p "$OUT"
M=manuscript
( cd $M && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex > /dev/null )   # fresh .aux and .bbl
python3 - "$OUT" <<'EOF'
import re, shutil, sys
from pathlib import Path
out, M = Path(sys.argv[1]), Path("manuscript")
tex = (M / "paper.tex").read_text()
# sections and tables inserted in place, so that the manuscript is a single file
tex = re.sub(r"\\input\{sections/([^}]+)\}", lambda m: (M / "sections" / (m.group(1) + ".tex")).read_text(), tex)
tex = re.sub(r"\\tabin\{([^}]+)\}", lambda m: (M / "tables" / (m.group(1) + ".tex")).read_text().rstrip("\n"), tex)
# numbers of the results: the generated file sits next to the manuscript
shutil.copy(M / "numbers.tex", out / "numbers.tex")
# figures renamed after their number in the article, read from the compiled .aux
aux = (M / "paper.aux").read_text()
num = {lab: n for lab, n in re.findall(r"\\newlabel\{(fig:[^}]+)\}\{\{([^}]*)\}", aux)}
def rename(m):
    width, path = m.group(1), m.group(2)
    block_start = tex.rfind("\\begin{figure}", 0, m.start())
    lab = re.search(r"\\label\{(fig:[^}]+)\}", tex[m.start():tex.find("\\end{figure}", m.start())])
    name = f"Fig{num[lab.group(1)]}" if lab else Path(path).stem
    shutil.copy(M / path, out / f"{name}.pdf")
    return "\\figin{" + width + "}{" + name + ".pdf}"
tex = re.sub(r"\\figin\{([^}]*)\}\{(figures/[^}]+\.pdf)\}", rename, tex)
(out / "manuscript.tex").write_text(tex)
for f in ("sn-jnl.cls", "sn-apacite.bst", "references.bib"):
    shutil.copy(M / f, out / f)
shutil.copy(M / "paper.bbl", out / "manuscript.bbl")
print(sorted(p.name for p in out.iterdir()))
EOF
# compile the folder alone, as the submission system will
( cd "$OUT" && for i in 1 2 3; do pdflatex -interaction=nonstopmode manuscript.tex > /dev/null 2>&1; done ) \
  || { echo "COMPILATION FAILED in $OUT"; exit 1; }
errors=$(python3 -c "print(sum(l.startswith('! ') for l in open('$OUT/manuscript.log', encoding='latin-1')))")
echo "manuscript: $(pdfinfo $OUT/manuscript.pdf | awk '/^Pages/{print $2}') pages, $errors LaTeX errors"
# once the folder is known to give the text of the paper, its reference PDF is the paper itself, byte for byte
if [ "$(pdftotext $OUT/manuscript.pdf - 2>/dev/null | md5sum)" = "$(pdftotext $M/paper.pdf - 2>/dev/null | md5sum)" ]; then
  cp $M/paper.pdf $OUT/manuscript.pdf
else
  echo "the folder compiles to another text than $M/paper.pdf"; exit 1
fi
if [ -f $M/cover_letter.tex ]; then
  ( cd $M && pdflatex -interaction=nonstopmode cover_letter.tex > /dev/null 2>&1 && mv cover_letter.pdf ../$OUT/ ) || echo "cover letter not built"
  if pdftotext "$OUT/cover_letter.pdf" - 2>/dev/null | grep -q 'identifier to add'; then
    echo "cover letter: the arXiv identifier is still to add (\\arxivid in $M/cover_letter.tex)"
  fi
  if [ "$(pdfinfo "$OUT/cover_letter.pdf" | awk '/^Pages/{print $2}')" != 1 ]; then
    echo "cover letter: more than one page"
  fi
fi
rm -f "$OUT"/*.aux "$OUT"/*.log "$OUT"/*.out "$OUT"/*.blg
( cd "$OUT" && zip -q -r ../submission.zip . )
git archive --format=zip --prefix=paper_topological_kinematics/ -o "$OUT/ESM_1.zip" HEAD -- . ":(exclude)notebooks" \
  ":(exclude)CLAUDE.md" ":(exclude)PUBLICATION.md" ":(exclude)manuscript/cover_letter.tex"   # the code, not the working notes
echo "supplementary material: $(du -h "$OUT/ESM_1.zip" | cut -f1) of code at commit $(git rev-parse --short HEAD)"
du -sh submission.zip
