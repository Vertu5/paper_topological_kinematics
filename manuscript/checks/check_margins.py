"""Pages of a PDF where words stick out of the text block, given its width and its left edge on odd and even pages
(two-sided layout), all in PDF points. For the Springer Nature template (31 pc wide): 370.3 100 124.9.
Usage: python3 check_margins.py paper.pdf 370.3 100 124.9"""
import re
import subprocess
import sys

pdf, width, x_odd, x_even = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
html = subprocess.run(["pdftotext", "-bbox", pdf, "-"], capture_output=True, text=True).stdout
pages = re.findall(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', html, re.S)
for i, pg in enumerate(pages, 1):
    x0 = x_odd if i % 2 else x_even
    words = [(float(a), float(c), w) for a, b, c, d, w in
             re.findall(r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)<', pg[2])]
    out = [w for w in words if w[1] > x0 + width + 1.5 or w[0] < x0 - 1.5]
    if out:
        far = max(max(w[1] - x0 - width, x0 - w[0]) for w in out)
        print(f"p{i}: {len(out)} words outside, up to {far:.1f} pt; e.g. {[w[2] for w in out[:6]]}")
