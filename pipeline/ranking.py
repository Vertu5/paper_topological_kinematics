"""
ranking.py - the order of the controllers that defines the Top-n datasets
=========================================================================
Top-n pools, for every mission, the runs of the n controllers with the highest mean score over their runs: the score
that the mission's loop function prints at the end of a run, kept at the bottom of each recording CSV after
'# Objective function' (ties, which do not occur in our runs, would go to the higher median, then to the lower
controller id).

The runs themselves stay stored, and indexed, in the order of TOP_CONTROLLERS (pipeline.config). That order came from
AutomodeMission/Results/get_top_controllers.sh, which read only the first score of each controller (score.txt has no
final newline, so cat glued the fifty scores into one line and awk kept the first number); it is not a ranking by
performance, and it is used here only as the order in which the runs are stored.

Author: Olivier Vertu Ndinga Oba
"""
from pathlib import Path
import numpy as np

from .config import BEHAVIORS, DATA_DIR


def _read_score(csv_path: Path) -> float:
    with open(csv_path, 'rb') as f:
        f.seek(max(0, csv_path.stat().st_size - 256))
        tail = f.read().decode('utf-8', errors='replace').splitlines()
    for i, line in enumerate(tail):
        if line.strip() == '# Objective function' and i + 1 < len(tail):
            return float(tail[i + 1].strip())
    lines = csv_path.read_text().splitlines()
    idx = lines.index('# Objective function')
    return float(lines[idx + 1].strip())


def scores(data_root: Path = DATA_DIR):
    """{mission: {controller int: array of the 50 scores of its runs}} from data_root/{mission}/{ctrl}/*.csv."""
    out = {}
    for mission in BEHAVIORS:
        out[mission] = {}
        mdir = Path(data_root) / mission
        for cdir in sorted(d for d in mdir.iterdir() if d.is_dir() and d.name.isdigit()):
            vals = [_read_score(f) for f in sorted(cdir.glob('*.csv')) if not f.name.startswith('robot_')]
            if vals:
                out[mission][int(cdir.name)] = np.asarray(vals, dtype=np.float64)
    return out


def ranked_controllers(data_root: Path = DATA_DIR):
    """{mission: [controller ints, best first]}, by mean score, then median, then id."""
    S = scores(data_root)
    return {m: sorted(S[m], key=lambda c: (-S[m][c].mean(), -np.median(S[m][c]), c)) for m in BEHAVIORS}
