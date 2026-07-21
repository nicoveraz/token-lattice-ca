import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SEC, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 150,
})

d = json.load(open("results/crystal.json"))
steps = [0, 1000, 2000, 3000, 4000, 5000, 6000]
g = lambda k: [d[str(s)].get(k, np.nan) for s in steps]

fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)

a = ax[0, 0]
a.plot(steps, g("bigram_T0.3"), "-o", color=BLUE, lw=2, ms=4, label="corpus-bigram order (T=0.3)")
a.plot(steps, g("census_overlap50"), "-o", color=AQUA, lw=2, ms=4, label="census top-50 recovery")
a.plot(steps, g("census_spearman"), "--o", color=AQUA, lw=1.2, ms=3, alpha=0.65, label="census rank corr ρ")
a.set_ylim(0, 1.05); a.legend(fontsize=7.5, loc="center right")
a.set_title("Order & prior recovery: crystallized by step 1000", loc="left")

b = ax[0, 1]
b.plot(steps, g("val_acc"), "-o", color=ORANGE, lw=2, ms=4, label="val masked accuracy")
b2 = [v / 8 for v in g("val_ce")]
b.plot(steps, b2, "--o", color=ORANGE, lw=1.2, ms=3, alpha=0.65, label="val cross-entropy (÷8)")
b.set_ylim(0, 1.05); b.legend(fontsize=7.5)
b.set_title("Prediction quality: keeps improving slowly", loc="left")

c = ax[1, 0]
c.plot(steps, g("bdmg_T0.3"), "-o", color=BLUE, lw=2, ms=4, label="T=0.3 (ordered phase)")
c.plot(steps, g("bdmg_T0.7"), "-o", color=MAGENTA, lw=2, ms=4, label="T=0.7 (near transition)")
c.set_ylim(0, 1.0); c.legend(fontsize=7.5)
c.set_ylabel("final damage (3-site flip, CRN twins)")
c.set_title("Self-healing is learned: fragility collapses at T=0.3", loc="left")
c.set_xlabel("training step")

e = ax[1, 1]
e.plot(steps, g("melt_retention"), "-o", color=YELLOW, lw=2, ms=4)
e.set_ylim(0, 0.3)
e.set_title("Pinning of real text (retention after 40 sweeps, T=0.3)", loc="left")
e.set_xlabel("training step")

fig.suptitle("Structure formation during training, read out by the CA instrument",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig/crystallization.png")
print("ok")
