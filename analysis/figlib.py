"""
figlib.py - shared style and helpers of the figures
===================================================
Print-legible fonts, colour-blind-safe mission colours with text labels, field colormaps, and the data
helpers shared by make_figures.py, illustrations.py and the notebooks.

Author: Olivier Vertu Ndinga Oba
"""
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
RES = ROOT / "results" / "epuck"
PAPER = RES / "paper"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
# the figures are drawn at their printed width in Swarm Intelligence, whose text block is 122 mm wide (measured on
# a published article); the Springer Nature template of the manuscript, 31 pc wide, shows them 7 % larger
TEXT_WIDTH = 12.2 / 2.54                      # inches
MIN_PT = 8.5                                  # the smallest text on the page, in points
PAD = 0.02                                    # margin kept around the drawn content when saving, inches
TEX = [ROOT / "manuscript" / "paper.tex", *sorted((ROOT / "manuscript" / "sections").glob("*.tex"))]   # the paper and its sections

ARENA = 2 * 1.231
MISSIONS = ["Directional Gate", "Foraging", "Homing", "Shelter", "XOR Aggregation"]
MSHORT = ["D-Gate", "Foraging", "Homing", "Shelter", "XOR"]
# validated categorical order (adjacent CVD dE >= 9.1); always paired with text labels
MCOL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
FIELDS = ["rho", "K", "phi", "psi", "J"]
FLABEL = {"rho": r"$\rho_\sigma$", "K": r"$K_\sigma$", "phi": r"$\varphi_\sigma$",
          "psi": r"$\psi_\sigma$", "J": r"$J_\sigma$"}
FNAME = {"rho": "density", "K": "kinetic intensity", "phi": "divergence potential",
         "psi": "vorticity potential", "J": "Jacobi coupling"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

# zero is a very light blue, never white: white is reserved for "no data" (outside the observed area)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#e9f2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_bo", ["#104281", "#3987e5", "#9ec5f4", "#efeeea", "#f6b896", "#eb6834", "#a83a10"])
FCMAP = {"rho": SEQ, "K": SEQ, "phi": DIV, "psi": DIV, "J": DIV}
SIGNED = {"rho": False, "K": False, "phi": True, "psi": True, "J": True}


def style():
    """Print-legible defaults, in Arial as the journal asks (Helvetica or Arial). A figure is drawn at the width it
    has on the printed page (width()), so that these sizes are the sizes in print; save() checks it."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.fallback": "stixsans",
        "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "axes.labelweight": "bold", "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "figure.titlesize": 11, "figure.titleweight": "bold",
        "axes.edgecolor": INK2, "axes.linewidth": 0.7, "xtick.color": INK, "ytick.color": INK,
        "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42,
    })


def bold_colorbar(cb, label=None, size=9):
    """Colorbar with bold tick numbers and label, no outline."""
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=size, length=2.5)
    for t in cb.ax.get_yticklabels() + cb.ax.get_xticklabels():
        t.set_fontweight("bold")
    if label:
        cb.set_label(label, fontweight="bold", fontsize=size + 0.5)
    return cb


def bold_ticks(ax, axis="both"):
    """Bold tick numbers (call once the limits and ticks are final)."""
    labels = (ax.get_xticklabels() if axis in ("x", "both") else []) + (ax.get_yticklabels() if axis in ("y", "both") else [])
    for t in labels:
        t.set_fontweight("bold")


def panel_label(ax, letter, x=-0.02, y=1.02):
    """Bold panel letter (a), (b), ... placed above the top-left corner of an axis."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


PDF_META = {"CreationDate": None}      # no date in the file: the same figure gives the same bytes


def width(fraction=1.0):
    """Width in inches of a figure that the paper includes at ``fraction`` of the text width (the saved file
    adds the margin PAD on each side)."""
    return fraction * TEXT_WIDTH - 2 * PAD


def included_width(name):
    """The fraction of the text width at which the paper includes figures/<name>.pdf; when both versions include it,
    the narrower one, where its text is smallest (None if neither does)."""
    pattern = re.compile(r"\\figin\{([0-9.]*)\\linewidth\}\{figures/" + re.escape(name) + r"\.pdf\}")
    found = [float(m.group(1) or 1.0) for tex in TEX if tex.exists() for m in pattern.finditer(tex.read_text())]
    return min(found) if found else None


def _drawn_texts(fig, path, **kw):
    """Save the figure as a PNG and return (text, size in points) of every text drawn, halos included."""
    from matplotlib.text import Text
    drawn, draw = [], Text.draw

    def record(self, renderer):
        if self.get_visible() and self.get_text().strip():
            drawn.append((self.get_text(), self.get_fontsize()))
        return draw(self, renderer)
    Text.draw = record
    try:
        fig.savefig(path, **kw)
    finally:
        Text.draw = draw
    return drawn


SIZES = {}       # name -> (fraction of the text width, scale on the page, smallest text in points) of each figure saved


OVERFLOW = {}    # name -> (inches of content left of the canvas, inches right of it), when the content spills out


def save(fig, name):
    """Save PDF and PNG, then the size on the page of every text, from the width at which the paper includes
    the figure: a warning names each text smaller than MIN_PT (sub- and superscripts follow their base size), and
    how far the content spills out of the canvas, which makes the saved figure wider and shrinks it on the page."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    OVERFLOW[name] = (max(0.0, -bb.x0), max(0.0, bb.x1 - fig.get_figwidth()))
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=PAD, metadata=PDF_META)
    drawn = _drawn_texts(fig, FIG / f"{name}.png", bbox_inches="tight", pad_inches=PAD, dpi=200)
    plt.close(fig)
    print("saved", FIG / f"{name}.pdf")
    fraction = included_width(name)
    if fraction is None or not drawn:
        return
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)", (FIG / f"{name}.pdf").read_bytes())
    scale = fraction * TEXT_WIDTH * 72 / float(box.group(1))          # page width / drawn width (PDF points)
    final = sorted({(round(size * scale, 2), s) for s, size in drawn if s.strip()})
    SIZES[name] = (fraction, scale, final[0][0])
    small = [(p, s) for p, s in final if p < MIN_PT - 1e-6]
    if small:
        shown = ", ".join(f"{s!r} {p:.1f} pt" for p, s in small[:6])
        print(f"WARNING {name}: {len(small)} texts below {MIN_PT} pt on the page (scale {scale:.3f}): {shown}"
              f"{' ...' if len(small) > 6 else ''}")
        if OVERFLOW.get(name):
            print(f"        content outside the canvas, inches: left {OVERFLOW[name][0]:.3f}, right {OVERFLOW[name][1]:.3f}")


def report():
    """One line per figure saved: inclusion width, scale on the page and smallest text."""
    for name, (fraction, scale, smallest) in SIZES.items():
        flag = "" if smallest >= MIN_PT - 1e-6 else "   below the minimum"
        print(f"{name:22s} {fraction:4.2f} of the text width, scale {scale:.3f}, smallest text {smallest:4.1f} pt{flag}")


def median_interval(per_dataset, n_boot=2000, seed=0):
    """95 % interval of the median of the pooled values, from bootstrap resamplings of the datasets (the settings of
    one dataset are not independent of one another, so the datasets are the units that are resampled)."""
    rng = np.random.default_rng(seed)
    k = len(per_dataset)
    boot = [np.median(np.concatenate([per_dataset[i] for i in rng.integers(0, k, k)])) for _ in range(n_boot)]
    return tuple(float(q) for q in np.quantile(boot, [0.025, 0.975]))


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def display_fields(pos, vel, sigma, n=150, fields=FIELDS, chunk=20, gpu=True):
    """Evaluate the run-averaged fields on an n x n display grid (closed form, same code as the pipeline)."""
    xs = np.linspace(0, ARENA, n)
    X, Y = np.meshgrid(xs, xs)
    grid = SimpleNamespace(X=X, Y=Y, shape=X.shape, sigma=float(sigma), n_vertices=X.size)
    if gpu:
        from gaussian_splatting.fields_gpu import TorchFieldComputer
        return TorchFieldComputer(fields=fields, kappa=5.0).compute(pos, vel, grid)
    from gaussian_splatting.fields_cpu import FieldComputer
    return FieldComputer(fields=fields, kappa=5.0, chunk_size=chunk).compute(pos, vel, grid)


def field_limits(arr, signed):
    a = arr[np.isfinite(arr)]
    if signed:
        m = np.quantile(np.abs(a), 0.995) or 1.0
        return -m, m
    return 0.0, np.quantile(a, 0.995) or 1.0


def show_field(ax, F, key, vmin=None, vmax=None, arena=False):
    """Field on the display grid. White = no data (farther than 5 sigma from every agent at all times);
    zero is light blue (density, kinetic) or light grey (signed fields)."""
    lo, hi = field_limits(F, SIGNED[key])
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    cmap = FCMAP[key].copy(); cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(F), origin="lower", extent=(0, ARENA, 0, ARENA), cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear")
    if arena:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(ARENA / 2 + 1.236 * np.cos(th), ARENA / 2 + 1.236 * np.sin(th), color="#8a8984", lw=0.5)
    ax.set_xlim(0, ARENA); ax.set_ylim(0, ARENA)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im
