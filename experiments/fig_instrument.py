"""Figure 1, "the instrument": what the apparatus IS, before any result is shown.

Three panels at display size (5.5in text width, figstyle conventions -- monochrome, marker/dash
encoding, base-R box):

  (a) the ring CA: N token cells on a ring, the windowed conditional p_r(x_i | x_{i +/- r})
      resampling one centre cell, async random visit order. Drawn with matplotlib patches;
      schematic by declaration, so it renders no data and can invent none.
  (b) the CRN twin protocol: twin lattices sharing init, visit order and uniform stream. The
      null arm (no flip) must differ in exactly zero cells -- the exact-zero assertion the paper
      makes load-bearing -- while the probe arm flips one token and the damage field is their
      disagreement.
  (c) one measured damage-spreading spacetime field, regenerated from results/damage.npz
      (key T0.7_r4: the paper's operating temperature, mid radius; site-wise damage probability,
      time down the page). REAL DATA ONLY -- this panel is read from the results file that the
      existing damage-cone figures were built from, never synthesised.

Writes fig/instrument.png.
Usage:  .venv/bin/python experiments/fig_instrument.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figstyle import use_classic_r, finish, BAND, RULE  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAMAGE_KEY = "T0.7_r4"      # the paper's operating temperature; mid conditioning radius


def panel_ring(ax):
    """(a) The automaton: a ring of token cells, one being resampled through its window."""
    ax.set_aspect("equal")
    ax.axis("off")
    N, R0 = 16, 1.0
    th = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    # the centre cell being updated and its radius-2 window
    ci = 0
    win = {(ci - 2) % N, (ci - 1) % N, (ci + 1) % N, (ci + 2) % N}
    for i, t in enumerate(th):
        x, y = R0 * np.cos(t), R0 * np.sin(t)
        if i == ci:
            fc, lw = "black", 1.0
        elif i in win:
            fc, lw = BAND, 0.9
        else:
            fc, lw = "white", 0.7
        ax.add_patch(patches.Circle((x, y), 0.135, facecolor=fc, edgecolor="black", lw=lw))
    # window brace: arc spanning the window cells
    t0, t1 = th[ci] - 2.45 * (2 * np.pi / N), th[ci] + 2.45 * (2 * np.pi / N)
    arc = np.linspace(t0, t1, 60)
    ax.plot(1.32 * np.cos(arc), 1.32 * np.sin(arc), color=RULE, lw=0.8)
    ax.annotate(r"$p_r(x_i \mid x_{i\pm r})$", xy=(0, R0 + 0.32), ha="center", va="bottom",
                fontsize=7)
    # resampling arrow into the centre cell
    ax.annotate("", xy=(0, R0 + 0.02), xytext=(0, R0 + 0.30),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))
    # async random order: a few numbered visits
    order = {3: "1", 9: "2", 13: "3", 6: "4"}
    for i, lab in order.items():
        x, y = 0.72 * np.cos(th[i]), 0.72 * np.sin(th[i])
        ax.text(x, y, lab, ha="center", va="center", fontsize=6, color=RULE)
    ax.text(0, -1.52, "$N$ token cells,\nasync random visit order", ha="center", va="top",
            fontsize=6.5)
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-2.15, 1.72)
    ax.set_title("(a) the automaton", loc="left")


def panel_crn(ax):
    """(b) CRN twins: shared randomness, one flipped token, damage = disagreement."""
    ax.axis("off")
    n, w, h = 12, 0.072, 0.10
    rng = np.random.default_rng(3)          # decorative cell shading only -- no data claim
    shade = rng.choice([0.97, 0.90, 0.82], size=n)

    def lattice(y, flip=None, label=""):
        for j in range(n):
            fc = "black" if j == flip else str(shade[j])
            ax.add_patch(patches.Rectangle((0.08 + j * w, y), w * 0.92, h,
                                           facecolor=fc, edgecolor="black", lw=0.5))
        ax.text(0.055, y + h / 2, label, ha="right", va="center", fontsize=6.5)

    lattice(0.84, label="ref")
    lattice(0.56, flip=5, label="twin")
    ax.text(0.53, 0.735, "shared init, visit order, uniform stream", ha="center",
            va="center", fontsize=6, color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=RULE, lw=0.6))
    ax.annotate("one flipped token", xy=(0.08 + 5.5 * w, 0.56), xytext=(0.74, 0.47),
                fontsize=6, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))
    # the two readouts, one line each so nothing can collide
    ax.plot([0.08, 0.98], [0.40, 0.40], color=RULE, lw=0.6)
    ax.text(0.08, 0.31, r"null arm (no flip): $\mathbf{0}$ differing cells",
            fontsize=6.2)
    ax.text(0.08, 0.20, "probe arm: damage = disagreement", fontsize=6.2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.12, 0.99)
    ax.set_title("(b) CRN twin protocol", loc="left")


def panel_spacetime(ax):
    """(c) A measured damage field: probability of site disagreement, time downward."""
    z = np.load(ROOT / "results" / "damage.npz")
    fld = z[DAMAGE_KEY]                       # (time, N) damage probability, MEASURED
    im = ax.imshow(fld, aspect="auto", cmap="gray_r", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xlabel("lattice site")
    ax.set_ylabel(r"sweep $\rightarrow$")
    ax.set_title("(c) measured damage field", loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("P(site differs)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    return fld


def main():
    use_classic_r()
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05),
                             gridspec_kw=dict(width_ratios=[0.92, 1.08, 1.05]))
    panel_ring(axes[0])
    panel_crn(axes[1])
    fld = panel_spacetime(axes[2])
    # sanity: the data panel really is data -- a cone grows from a localised seed
    assert fld.shape[1] == 48 and 0.0 <= fld.min() and fld.max() <= 1.0
    finish(fig, ROOT / "fig" / "instrument.png")


if __name__ == "__main__":
    main()
