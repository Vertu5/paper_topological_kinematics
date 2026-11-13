#!/usr/bin/env python3
"""
cache_display_fields.py - the fields drawn in the figures
=========================================================
The fields of the Top-1 exemplars of the plateau consensus at the four scales of the ladder, and of one
run on a scale series continued up to D_s / 2.35, on a display grid (GPU if available, the same closed
forms as the pipeline). Resumable; reused only for the same tracks and taxonomy.

Author: Olivier Vertu Ndinga Oba
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figlib as L  # noqa: E402
from pipeline.checkpoint import Store, file_fingerprint  # noqa: E402

out_path = L.PAPER / "display_fields.pkl"
d = L.load_pickle(L.PAPER / "top1_tracks.pkl")
ref = L.load_pickle(L.PAPER / "reference_top1.pkl")
ladder = np.asarray(ref["ladder"])
tax_path = L.PAPER / "taxonomy.pkl"
# resumable; reused only for the same tracks and the same taxonomy (whose exemplars are drawn)
cache = Store(out_path, "+".join(file_fingerprint(p) for p in (L.PAPER / "top1_tracks.pkl", tax_path) if p.exists()))
t0 = time.time()
exemplars = list(L.load_pickle(tax_path)[1]["exemplars"]) if tax_path.exists() else list(ref["exemplars"])
for e in exemplars:
    for si, s in enumerate(ladder):
        key = ("ladder", int(e), si)
        if key in cache:
            continue
        tr = d["tracks"][e]
        try:
            cache[key] = L.display_fields(tr["pos"], tr["vel"], float(s), n=160)
        except Exception:                                   # no usable GPU: CPU fallback
            cache[key] = L.display_fields(tr["pos"], tr["vel"], float(s), n=160, gpu=False, chunk=40)
        print(key, f"{time.time() - t0:.0f}s", flush=True)
lad = L.load_pickle(L.RES / "scale_ladder.pkl")
merging = float(lad["ladder"][-1]) * float(lad["max_factor"])      # D_s / 2.35, since sigma_max = D_s / (2.35 k)
uncapped = np.geomspace(ladder[0], merging, 6)
if tax_path.exists():                                # the Homing exemplar of the taxonomy
    e = int([x for x in exemplars if int(d["labels"][x]) == 2][0])
else:
    e = int(ref["exemplars"][2])
for si, s in enumerate(uncapped):
    key = ("uncapped", e, si)
    if key in cache:
        continue
    tr = d["tracks"][e]
    cache[key] = L.display_fields(tr["pos"], tr["vel"], float(s), n=160, fields=("rho", "psi"))
    cache[("uncapped_sigma", si)] = float(s)
    print(key, f"{time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
