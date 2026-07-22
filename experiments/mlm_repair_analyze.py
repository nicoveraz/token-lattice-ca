"""Phase B analysis (F23): the self-correction length, diversity-controlled.

The raw asymptotic damage D is diversity-confounded (a degenerate low-entropy model
snaps back trivially, scoring low D for the wrong reason -- the stability analog of
the A3 repetition confound). We control by the DIVERSITY FLOOR D0 (unperturbed
independent-noise drift) and use D_norm = D/D0. The spine test: does D_norm still
ORDER the models after this control (real xi_repair) or flatten onto the floor
(self-correction was diversity in disguise)?
  fig/repair_grid.png   D_norm(r,T) heatmaps per model + D=0.5 contour
  fig/repair_scale.png  D_norm(r) profiles across models; floor validation; verdict
Writes results/analysis_phaseB.json.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, SEC, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
M_COL = {"tiny": "#9ec5f4", "mini": "#2a78d6", "base": "#0d366b"}
DIV = LinearSegmentedColormap.from_list("hs", ["#1baf7a", "#f0efec", "#eb6834"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "legend.frameon": False, "figure.dpi": 150,
})
R = "results/mlm"
TAGS = [t for t in ["tiny", "mini", "base"] if os.path.exists(f"{R}/repair_{t}.json")]
analysis = {}


def load(tag):
    d = json.load(open(f"{R}/repair_{tag}.json"))
    RS, TS = d["RS"], d["TS"]
    def arr(k):
        return np.array([[d[k][str(r)][str(T)]["mean"] for T in TS] for r in RS])
    return d, RS, TS, arr


# ---------- grids ----------
if TAGS:
    fig, axes = plt.subplots(1, len(TAGS), figsize=(3.3 * len(TAGS), 3.3), squeeze=False)
    for j, tag in enumerate(TAGS):
        d, RS, TS, arr = load(tag)
        Dn = arr("D_norm")
        ax = axes[0, j]
        im = ax.imshow(Dn, aspect="auto", cmap=DIV, vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(TS))); ax.set_xticklabels(TS)
        ax.set_yticks(range(len(RS))); ax.set_yticklabels(RS)
        ax.set_xlabel("temperature T"); ax.set_title(f"{tag}: D_norm(r,T)", loc="left"); ax.grid(False)
        if j == 0:
            ax.set_ylabel("radius r")
    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, label="D / D0 (diversity-controlled damage)")
    fig.suptitle("Diversity-controlled self-correction map (green corrects, orange persists)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/repair_grid.png"); plt.close()

# ---------- profiles + verdict ----------
if TAGS:
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    summ = {}
    Dn_prof, Draw_prof = {}, {}
    for tag in TAGS:
        d, RS, TS, arr = load(tag)
        Dn, Draw, D0, dist = arr("D_norm"), arr("D"), arr("D0_floor"), arr("distinct")
        dn_r = Dn.mean(axis=1)      # D_norm averaged over T, vs r
        draw_r = Draw.mean(axis=1)
        Dn_prof[tag] = dn_r; Draw_prof[tag] = draw_r
        ax[0].plot(RS, draw_r, "-o", color=M_COL[tag], lw=2, ms=4, label=tag)
        ax[1].plot(RS, dn_r, "-o", color=M_COL[tag], lw=2, ms=4, label=tag)
        ax[2].scatter(dist.ravel(), Dn.ravel(), color=M_COL[tag], s=18, label=tag)
        summ[tag] = dict(D_raw_by_r=[round(float(x), 3) for x in draw_r],
                         D_norm_by_r=[round(float(x), 3) for x in dn_r],
                         D0_floor_mean=round(float(D0.mean()), 3),
                         distinct_mean=round(float(dist.mean()), 3),
                         r_star_norm=RS[int(np.argmax(dn_r))],
                         mean_D_norm=round(float(dn_r.mean()), 3))
    for a, t in [(ax[0], "raw D"), (ax[1], "D_norm = D/D0 (diversity-controlled)")]:
        a.set_xscale("log", base=2); a.set_xticks(RS); a.set_xticklabels(RS)
        a.set_xlabel("radius r"); a.legend(fontsize=8); a.set_title(t, loc="left")
    ax[2].set_xlabel("distinct-token fraction"); ax[2].set_ylabel("D_norm")
    ax[2].set_title("D_norm vs diversity (flat => controlled)", loc="left"); ax[2].legend(fontsize=8)
    # spine verdict: does the model ordering by mean D_norm survive, and is it separated?
    order_raw = sorted(TAGS, key=lambda t: Draw_prof[t].mean())
    order_norm = sorted(TAGS, key=lambda t: Dn_prof[t].mean())
    if len(TAGS) >= 2:
        means = {t: float(Dn_prof[t].mean()) for t in TAGS}
        spread = max(means.values()) - min(means.values())
        analysis["spine"] = dict(order_by_raw_D=order_raw, order_by_norm_D=order_norm,
                                 mean_D_norm=means, norm_spread=round(spread, 3),
                                 separates_after_control=bool(spread > 0.08))
    analysis["repair"] = summ
    v = analysis.get("spine", {})
    fig.suptitle(f"F23: does self-correction survive the diversity control? "
                 f"norm order={v.get('order_by_norm_D')} spread={v.get('norm_spread')}",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/repair_scale.png"); plt.close()

json.dump(analysis, open("results/analysis_phaseB.json", "w"), indent=1)
print(json.dumps(analysis, indent=1))
print("PHASE B ANALYSIS DONE (models:", TAGS, ")")
