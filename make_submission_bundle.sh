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
OUT=submission; SUP_TMP=$(mktemp -d); rm -rf "$OUT" submission.zip; mkdir -p "$OUT"
trap 'rm -rf "$SUP_TMP"' EXIT
M=manuscript
( cd $M && bash build.sh > /dev/null )   # fresh paper, journal and supplementary (.pdf, .aux, .bbl)

python3 - "$OUT" "$SUP_TMP" <<'EOF'
import re, shutil, sys
from pathlib import Path

out, sup_out, M = Path(sys.argv[1]), Path(sys.argv[2]), Path("manuscript")

def inline_inputs(tex: str) -> str:
    tex = re.sub(r"\\input\{preamble\}", lambda _: (M / "preamble.tex").read_text(), tex)
    # expand nested \input{sections/...} recursively (appendix_body includes the three appendices)
    while re.search(r"\\input\{sections/([^}]+)\}", tex):
        tex = re.sub(r"\\input\{sections/([^}]+)\}", lambda m: (M / "sections" / f"{m.group(1)}.tex").read_text(), tex)
    tex = re.sub(r"\\tabin\{([^}]+)\}", lambda m: (M / "tables" / f"{m.group(1)}.tex").read_text().rstrip("\n"), tex)
    return tex

def rewrite_figures(tex: str, aux_text: str, dest: Path) -> str:
    num = {lab: n for lab, n in re.findall(r"\\newlabel\{(fig:[^}]+)\}\{\{([^}]*)\}", aux_text)}
    def rename(m):
        width, path = m.group(1), m.group(2)
        lab = re.search(r"\\label\{(fig:[^}]+)\}", tex[m.start():tex.find("\\end{figure}", m.start())])
        name = f"Fig{num[lab.group(1)]}" if lab else Path(path).stem
        shutil.copy(M / path, dest / f"{name}.pdf")
        return "\\figin{" + width + "}{" + name + ".pdf}"
    return re.sub(r"\\figin\{([^}]*)\}\{(figures/[^}]+\.pdf)\}", rename, tex)

def _normalize_aux_label(line: str) -> str:
    m = re.match(r"^\\newlabel\{([^}]+@cref)\}\{(.*)\}$", line)
    if not m:
        return line
    key, body = m.group(1), m.group(2)
    if body.endswith("{}{}{}"):
        body = body[:-6]
    return f"\\@ptk@xcref{{{key}}}{{{body}}}"

def inject_xr_fallback(tex: str, other_name: str, other_aux: Path) -> str:
    raw_labels = re.findall(r"^\\newlabel\{.*$", other_aux.read_text(), re.M)
    labels = "\n".join(_normalize_aux_label(l) for l in raw_labels)
    probe = (
        "\\makeatletter\n"
        "\\newif\\if@ptk@five\n"
        "\\begingroup\n"
        "  \\@namedef{r@_ptk_probe@cref}{{X}{}{}{}{}}%\n"
        "  \\cref@getref{_ptk_probe}{\\@ptk@res}%\n"
        "  \\def\\@ptk@exp{X}%\n"
        "  \\ifx\\@ptk@res\\@ptk@exp\\global\\@ptk@fivetrue\\fi\n"
        "\\endgroup\n"
        "\\if@ptk@five\n"
        "  \\def\\@ptk@xcref#1#2{\\newlabel{#1}{#2{}{}{}}}%\n"
        "\\else\n"
        "  \\def\\@ptk@xcref#1#2{\\newlabel{#1}{#2}}%\n"
        "\\fi\n"
    )
    replacement = (
        f"{probe}"
        f"\\IfFileExists{{{other_name}.aux}}{{\\externaldocument[][nocite]{{{other_name}}}[]}}{{%\n"
        f"{labels}\n"
        "}\n"
        "\\makeatother"
    )
    return tex.replace(f"\\externaldocument[][nocite]{{{other_name}}}[]", replacement, 1)

def prepare_doc(tex_src: str, aux_src: Path, bbl_src: Path, dest: Path, doc_name: str,
                other_name: str, other_aux: Path) -> list[str]:
    tex = inline_inputs(tex_src)
    tex = rewrite_figures(tex, aux_src.read_text(), dest)
    tex = inject_xr_fallback(tex, other_name, other_aux)
    (dest / f"{doc_name}.tex").write_text(tex)
    shutil.copy(M / "numbers.tex", dest / "numbers.tex")
    for f in ("sn-jnl.cls", "sn-apacite.bst", "references.bib"):
        shutil.copy(M / f, dest / f)
    shutil.copy(bbl_src, dest / f"{doc_name}.bbl")
    return sorted(p.name for p in dest.iterdir())

# Main manuscript (journal.tex + paper.tex without the appendices block)
paper_body = (M / "paper.tex").read_text()
paper_body = re.sub(r"\\ifdefined\\journalversion\\else.*?\\fi\s*", "", paper_body, count=1, flags=re.S)
paper_body = paper_body.replace("\\ifdefined\\journalversion\\externaldocument[][nocite]{supplementary}[]\\fi", "\\externaldocument[][nocite]{supplementary}[]")
journal_src = "\\def\\journalversion{}\n\\RequirePackage{xr-hyper}\n" + paper_body
main_files = prepare_doc(journal_src, M / "journal.aux", M / "journal.bbl", out, "manuscript", "supplementary", M / "supplementary.aux")

# Supplementary Information source package
sup_files = prepare_doc((M / "supplementary.tex").read_text(), M / "supplementary.aux", M / "supplementary.bbl", sup_out, "supplementary", "journal", M / "journal.aux")

print("main files:", main_files)
print("supplementary files:", sup_files)
EOF

verify_compile() {
  local dir="$1" stem="$2" ref_pdf="$3"
  ( cd "$dir" && for _ in 1 2 3; do pdflatex -interaction=nonstopmode "${stem}.tex" > /dev/null 2>&1; done ) \
    || { echo "COMPILATION FAILED in $dir/${stem}.tex"; exit 1; }
  local errors undef
  errors=$(python3 -c "print(sum(l.startswith('! ') for l in open('$dir/${stem}.log', encoding='latin-1')))")
  undef=$(grep -cE 'undefined|multiply defined' "$dir/${stem}.log" || true)
  echo "${stem}: $(pdfinfo "$dir/${stem}.pdf" | awk '/^Pages/{print $2}') pages, $errors LaTeX errors, $undef unresolved refs"
  if [ "$errors" != 0 ] || [ "$undef" != 0 ]; then
    echo "COMPILATION WARNINGS/ERRORS in $dir/${stem}.log"; exit 1
  fi
  if [ "$(pdftotext "$dir/${stem}.pdf" - 2>/dev/null | md5sum)" = "$(pdftotext "$ref_pdf" - 2>/dev/null | md5sum)" ]; then
    cp "$ref_pdf" "$dir/${stem}.pdf"
  else
    echo "the standalone folder compiles to another text than $ref_pdf"; exit 1
  fi
  rm -f "$dir"/*.aux "$dir"/*.log "$dir"/*.out "$dir"/*.blg
}

# compile both independently in clean folders without .aux files, as a journal build server will
verify_compile "$OUT" manuscript "$M/journal.pdf"
verify_compile "$SUP_TMP" supplementary "$M/supplementary.pdf"

# zip ONLY the main manuscript package into submission.zip
( cd "$OUT" && zip -q -r ../submission.zip . )

# copy companion deliverables alongside submission.zip inside submission/
cp "$M/supplementary.pdf" "$OUT/supplementary.pdf"
( cd "$SUP_TMP" && zip -q -r - . ) > "$OUT/supplementary_source.zip"

if [ -f "$M/cover_letter.tex" ]; then
  ( cd "$M" && pdflatex -interaction=nonstopmode cover_letter.tex > /dev/null 2>&1 && mv cover_letter.pdf "../$OUT/" ) || { echo "cover letter not built"; exit 1; }
  rm -f "$M/cover_letter.aux" "$M/cover_letter.log" "$M/cover_letter.out"
  if pdftotext "$OUT/cover_letter.pdf" - 2>/dev/null | grep -qiE 'identi.*to add|to add|arXiv \('; then
    echo "cover letter: unresolved preprint placeholder in $M/cover_letter.tex"; exit 1
  fi
  if ! pdftotext "$OUT/cover_letter.pdf" - 2>/dev/null | tr -d ' \n\r\t' | grep -q '10\.5281/zenodo\.24254004'; then
    echo "cover letter: missing Zenodo preprint DOI (10.5281/zenodo.24254004) in $M/cover_letter.tex"; exit 1
  fi
  if [ "$(pdfinfo "$OUT/cover_letter.pdf" | awk '/^Pages/{print $2}')" != 1 ]; then
    echo "cover letter: more than one page"; exit 1
  fi
fi

git archive --format=zip --prefix=paper_topological_kinematics/ -o "$OUT/ESM_1.zip" HEAD -- . ":(exclude)notebooks" \
  ":(exclude)assets" ":(exclude)manuscript/*.pdf" ":(exclude)CLAUDE.md" ":(exclude)PUBLICATION.md" \
  ":(exclude)manuscript/cover_letter.tex"   # the code, not duplicate PDFs or working notes
echo "supplementary code archive: $(du -h "$OUT/ESM_1.zip" | cut -f1) at commit $(git rev-parse --short HEAD)"
du -sh submission.zip "$OUT/supplementary.pdf" "$OUT/supplementary_source.zip" "$OUT/cover_letter.pdf" "$OUT/ESM_1.zip"
