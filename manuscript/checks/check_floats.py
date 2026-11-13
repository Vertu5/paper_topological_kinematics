"""Placement of the floats in a compiled paper: each figure and table must sit on the page of its first citation or
on the next one, never before it. The page of a float comes from the .aux file, the pages that cite it from the text
of each page (pdftotext). A float of the appendices counts only the citations made in the appendices.
Usage: python3 check_floats.py <path of the paper without extension>, e.g. manuscript/paper"""
import re
import subprocess
import sys

base = sys.argv[1]
aux = open(base + ".aux").read()
info = subprocess.run(["pdfinfo", base + ".pdf"], capture_output=True, text=True).stdout
n = int(re.search(r"Pages:\s+(\d+)", info).group(1))
pages = [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), base + ".pdf", "-"], capture_output=True,
                        text=True).stdout.replace("\n", " ") for i in range(1, n + 1)]
app = next((i + 1 for i, t in enumerate(pages) if re.search(r"Appendix A\s", t[:400])), n + 1)  # first page of the appendices
bad = 0
for kind, word in (("fig", r"Figs?\."), ("tab", r"Tables?")):
    for lab, num, page in re.findall(r"\\newlabel\{(" + kind + r":[^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux):
        page = int(page)
        pat = re.compile(word + r"\s+(?:\d+[a-z]?(?:,|\s+and|\s*–)\s*)*" + re.escape(num) + r"(?![\d])")
        cites = [i + 1 for i, t in enumerate(pages) if pat.search(t)]
        if not num[:1].isdigit() and not num.startswith("S"):      # Supplementary Information: S1, S2, ...
            cites = [c for c in cites if c >= app]
        # the caption itself starts with "Fig. N" or "Table N": its page counts only with a second mention
        first = next((p for p in cites if p != page or len(pat.findall(pages[p - 1])) >= 2), None)
        ok = first is not None and page in (first, first + 1)
        bad += not ok
        print(f"{lab:28s} {kind} {num:>3s}  on p{page:<3d} first cited p{first}{'' if ok else '   <-- check'}")
print("pages", n, "| misplaced", bad)
