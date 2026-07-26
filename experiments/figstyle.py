"""Classic-R-style monochrome figure defaults, shared by every paper figure.

WHY MONOCHROME. The paper is read in print and photocopy as often as on screen, and a reviewer
with a greyscale printer is a reviewer who cannot read a hue-encoded series. Encoding by marker
shape and line dash instead of colour is safe under every kind of colour-vision deficiency, under
greyscale printing, and under forced-colors modes, without needing a palette check. Nothing here
is stylistic preference: a two-series figure distinguished by blue-vs-red alone fails all three.

WHY IT LOOKS LIKE R. Base R's default device draws a full box (bty="o"), ticks pointing OUT, no
grid, and plain black marks. That convention is legible at small sizes because nothing competes
with the data for ink -- which matters here, where a three-panel figure is reproduced at a 5.5in
text width.

SIZE. Figures are designed at their DISPLAY size and included at width=\\linewidth, so a 7pt label
in the script is 7pt on the page. The previous figures were authored at 13.2in wide and scaled to
4.4in, which shrank every label to a third of its nominal size and was the reason the headline
figure's text was unreadable.

Usage:
    from figstyle import use_classic_r, SERIES, finish
    use_classic_r()
    ...
    finish(fig, path)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Series are distinguished by MARKER and DASH, never by colour. Two entries is what the paper
# needs (N=48, N=96); more are provided so a third size does not tempt anyone into adding a hue.
SERIES = [
    dict(marker="o", mfc="white", ls="-",  label=None),   # open circle, solid
    dict(marker="s", mfc="black", ls="--", label=None),   # filled square, dashed
    dict(marker="^", mfc="white", ls=":",  label=None),   # open triangle, dotted
    dict(marker="D", mfc="0.55",  ls="-.", label=None),   # grey diamond, dash-dot
]

BAND = "0.88"        # light grey fill for the pre-transition band
RULE = "0.35"        # zero line / reference rules


def use_classic_r(base=7.0):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        # base-R device: full box, ticks out, no grid, thin black rules
        "axes.grid": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.4,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "axes.titlepad": 3.0,
    })


def series(i, label):
    """Style kwargs for series `i`, all black, distinguished by marker and dash."""
    s = dict(SERIES[i % len(SERIES)])
    s.pop("label")
    return dict(color="black", markeredgecolor="black", markeredgewidth=0.7,
                label=label, **s)


def finish(fig, path, pad=0.02):
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=pad, facecolor="white")
    print("wrote", path)
