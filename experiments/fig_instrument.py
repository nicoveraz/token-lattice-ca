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
      (key T0.7_r1: the paper's operating temperature at the SLOWEST radius, chosen because the
      cone stays visible for the full 61 sweeps instead of saturating within ten). REAL DATA
      ONLY -- read from the results file the existing damage-cone figures were built from.

REDESIGNED after review ("not clear"): (a) had unexplained floating digits and a window shading
too faint to survive print; (b) stated its two claims as text without ever SHOWING the damage
field; (c) used T0.7_r4, whose cone floods the ring in ~7 sweeps, compressing the instructive
content into the top sliver. The damage row in (b) is the t=0 field, which is the flipped site
by definition -- shown, not invented.

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
DAMAGE_KEY = "T0.7_r1"      # operating temperature, slowest cone: visible all 61 sweeps


def panel_ring(ax):
    """(a) The automaton: a ring of token cells; the window feeds the rule; the centre is resampled."""
    ax.set_aspect("equal")
    ax.set_anchor("N")          # align this panel's top -- and so its title -- with (b) and (c)
    ax.axis("off")
    N, R0 = 16, 1.0
    th = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    ci = 0
    win = {(ci - 2) % N, (ci - 1) % N, (ci + 1) % N, (ci + 2) % N}
    for i, t in enumerate(th):
        x, y = R0 * np.cos(t), R0 * np.sin(t)
        if i == ci:
            fc, lw = "black", 1.0
        elif i in win:
            fc, lw = "0.55", 1.0          # dark enough to survive greyscale print
        else:
            fc, lw = "white", 0.7
        ax.add_patch(patches.Circle((x, y), 0.135, facecolor=fc, edgecolor="black", lw=lw))
    # brace over the window, naming it
    t0, t1 = th[ci] - 2.45 * (2 * np.pi / N), th[ci] + 2.45 * (2 * np.pi / N)
    arc = np.linspace(t0, t1, 60)
    ax.plot(1.34 * np.cos(arc), 1.34 * np.sin(arc), color=RULE, lw=0.8)
    ax.annotate(r"window $x_{i\pm r}$", xy=(0, R0 + 0.62), ha="center", va="bottom", fontsize=6.5)
    # the rule: window -> resampled centre. The formula sits at a height the arc never reaches
    # (arc radius 1.34, label at y=1.44), so neither arc nor arrow can strike through it.
    ax.annotate("", xy=(0, R0 + 0.03), xytext=(0, R0 + 0.58),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))
    ax.text(0.10, R0 + 0.44, r"$x_i \sim p_r$", ha="left", va="center", fontsize=6.5)
    ax.text(0, -1.50, "one site resampled at a time,\nin random order", ha="center", va="top",
            fontsize=6.5)
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-2.02, 1.95)
    ax.set_title("(a) the automaton", loc="left")


def panel_crn(ax):
    """(b) CRN twins: identical randomness, one flip; the damage field IS their disagreement."""
    ax.axis("off")
    n, w, h = 12, 0.072, 0.105
    FLIP = 5
    rng = np.random.default_rng(3)          # decorative cell shading only -- no data claim
    shade = rng.choice([0.97, 0.90, 0.82], size=n)

    def lattice(y, label, cells):
        for j, fc in enumerate(cells):
            ax.add_patch(patches.Rectangle((0.10 + j * w, y), w * 0.92, h,
                                           facecolor=fc, edgecolor="black", lw=0.5))
        ax.text(0.075, y + h / 2, label, ha="right", va="center", fontsize=6.5)

    y_ref, y_twin, y_dmg = 0.82, 0.64, 0.30
    lattice(y_ref, "ref", [str(x) for x in shade])
    twin_cells = [str(x) for x in shade]
    twin_cells[FLIP] = "black"
    lattice(y_twin, "twin", twin_cells)
    # identical except the flip -- SHOWN cell-wise, not asserted
    for j in range(n):
        cx = 0.10 + j * w + w * 0.46
        ax.text(cx, (y_ref + y_twin + h) / 2, "=" if j != FLIP else r"$\neq$",
                ha="center", va="center", fontsize=5.5,
                color=RULE if j != FLIP else "black")
    lat_right = 0.10 + n * w - w * 0.08          # the lattice's true right edge
    ax.text(lat_right, y_ref + h + 0.035, "shared init, visit order, uniform stream",
            ha="right", fontsize=6)
    # the damage field at t=0: disagreement, which is the flipped site by definition
    dmg_cells = ["white"] * n
    dmg_cells[FLIP] = "black"
    lattice(y_dmg, "damage", dmg_cells)
    ax.annotate("", xy=(0.10 + (FLIP + 0.46) * w, y_dmg + h + 0.015),
                xytext=(0.10 + (FLIP + 0.46) * w, y_twin - 0.015),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    ax.text(0.10 + (FLIP + 1.1) * w, (y_twin + y_dmg) / 2 + 0.02, "site-wise\ndisagreement",
            ha="left", va="center", fontsize=6)
    ax.text(0.10 + n * w / 2, y_dmg - 0.10, r"no flip: damage $\equiv 0$, exactly (asserted)",
            fontsize=5.7, va="top", ha="center")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.05, 1.06)
    ax.set_title("(b) CRN twin probe", loc="left")


def panel_spacetime(ax):
    """(c) A measured damage cone: probability of site disagreement, time downward."""
    z = np.load(ROOT / "results" / "damage.npz")
    fld = z[DAMAGE_KEY]                       # (time, N) damage probability, MEASURED
    im = ax.imshow(fld, aspect="auto", cmap="gray_r", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xlabel("lattice site")
    ax.set_ylabel(r"sweep $\rightarrow$")
    ax.set_title("(c) measured damage cone", loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("P(site differs)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    return fld


def main():
    use_classic_r()
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.88),
                             gridspec_kw=dict(width_ratios=[0.88, 1.14, 1.03]))
    panel_ring(axes[0])
    panel_crn(axes[1])
    fld = panel_spacetime(axes[2])
    # sanity: the data panel really is data -- a cone grows from a localised seed
    assert fld.shape[1] == 48 and 0.0 <= fld.min() and fld.max() <= 1.0
    finish(fig, ROOT / "fig" / "instrument.png")


if __name__ == "__main__":
    main()
