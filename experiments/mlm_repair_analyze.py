"""Phase B analysis + figures (F23). The self-correction length xi_repair and the
kinematic (velocity, model-invariant) vs stability (D / T_c) decomposition.
  fig/repair_grid.png   D(r,T) heatmap per model with the D=0.5 heal/spread contour
  fig/repair_scale.png  xi_repair(T) and T_c(r) across models; the decomposition
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
DIV = LinearSegmentedColormap.from_list("hs", ["#1baf7a", "#f0efec", "#eb6834"])  # heal->spread
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
    D = np.array([[d["D"][str(r)][str(T)]["mean"] for T in TS] for r in RS])  # (r, T)
    return d, RS, TS, D


# ---------- 1. D(r,T) grids ----------
if TAGS:
    fig, axes = plt.subplots(1, len(TAGS), figsize=(3.3 * len(TAGS), 3.3), squeeze=False)
    for j, tag in enumerate(TAGS):
        d, RS, TS, D = load(tag)
        ax = axes[0, j]
        im = ax.imshow(D, aspect="auto", cmap=DIV, vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(TS))); ax.set_xticklabels(TS)
        ax.set_yticks(range(len(RS))); ax.set_yticklabels(RS)
        ax.set_xlabel("temperature T"); ax.set_title(f"{tag}: D(r,T)", loc="left")
        ax.grid(False)
        try:  # D=0.5 heal/spread contour
            ax.contour(D, levels=[0.5], colors=[INK], linewidths=1.4)
        except Exception:
            pass
        if j == 0:
            ax.set_ylabel("radius r")
    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, label="asymptotic damage D")
    fig.suptitle("Self-correction map: green heals, orange spreads, line = D=0.5 boundary",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/repair_grid.png"); plt.close()

# ---------- 2. xi_repair(T), T_c(r), and clean-ness ----------
if TAGS:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    summ = {}
    for tag in TAGS:
        d, RS, TS, D = load(tag)
        # T_c(r): interp T where D(T) crosses 0.5 at fixed r
        def cross(xs, ys, lv=0.5):
            xs, ys = np.asarray(xs, float), np.asarray(ys, float)
            for i in range(len(xs) - 1):
                a, b = ys[i], ys[i + 1]
                if (a - lv) * (b - lv) <= 0 and a != b:
                    return float(xs[i] + (a - lv) / (a - b) * (xs[i + 1] - xs[i]))
            return np.nan
        Tc = [cross(TS, D[i]) for i in range(len(RS))]
        ax[0].plot(RS, Tc, "-o", color=M_COL[tag], lw=2, ms=5, label=tag)
        # xi_repair(T): interp r (log2) where D(r) crosses 0.5 at fixed T
        xi = []
        for k in range(len(TS)):
            c = cross(np.log2(RS), D[:, k])
            xi.append(2 ** c if not np.isnan(c) else np.nan)
        ax[1].plot(TS, xi, "-o", color=M_COL[tag], lw=2, ms=5, label=tag)
        # monotonicity of T_c(r) (thesis: rises with r) and xi range
        tc_valid = [x for x in Tc if not np.isnan(x)]
        # r* = the maximally-UNSTABLE radius: argmin_r T_c (equivalently argmax_r D).
        # Averaged over the T where D straddles 0.5 for robustness.
        Tc_arr = np.array([np.nan if np.isnan(x) else x for x in Tc])
        r_star_Tc = RS[int(np.nanargmin(Tc_arr))] if not np.all(np.isnan(Tc_arr)) else None
        r_star_D = RS[int(np.argmax(D.mean(axis=1)))]     # radius of max mean damage
        # U-shaped (non-monotone) if T_c dips then rises
        u_shaped = (r_star_Tc not in (RS[0], RS[-1])) if r_star_Tc is not None else False
        summ[tag] = dict(T_c_by_r=[None if np.isnan(x) else round(x, 3) for x in Tc],
                         xi_repair_by_T=[None if np.isnan(x) else round(x, 2) for x in xi],
                         r_star_min_Tc=r_star_Tc, r_star_max_D=r_star_D,
                         Tc_min=round(float(np.nanmin(Tc_arr)), 3) if not np.all(np.isnan(Tc_arr)) else None,
                         Tc_max=round(float(np.nanmax(Tc_arr)), 3) if not np.all(np.isnan(Tc_arr)) else None,
                         U_shaped=bool(u_shaped))
    ax[0].set_xscale("log", base=2); ax[0].set_xticks([1, 2, 4, 8, 16]); ax[0].set_xticklabels([1, 2, 4, 8, 16])
    ax[0].set_xlabel("radius r"); ax[0].set_ylabel("T_c (heal/spread boundary)")
    ax[0].set_title("Stability boundary T_c(r)", loc="left"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("temperature T"); ax[1].set_ylabel("xi_repair (crossover radius)")
    ax[1].set_title("Repair length xi_repair(T)", loc="left"); ax[1].legend(fontsize=8)
    analysis["repair"] = summ
    # model separation: spread of T_c(r) across models at each r
    if len(TAGS) >= 2:
        seps = []
        for i in range(len(load(TAGS[0])[1])):
            vals = [summ[t]["T_c_by_r"][i] for t in TAGS if summ[t]["T_c_by_r"][i] is not None]
            if len(vals) >= 2:
                seps.append(max(vals) - min(vals))
        analysis["model_separation_Tc"] = round(float(np.mean(seps)), 3) if seps else None
    fig.suptitle("xi_repair as a measured scale, and the T_c(r) stability boundary",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/repair_scale.png"); plt.close()

json.dump(analysis, open("results/analysis_phaseB.json", "w"), indent=1)
print(json.dumps(analysis, indent=1))
print("PHASE B ANALYSIS DONE (models:", TAGS, ")")
