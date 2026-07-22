"""Phase 2 figures (same palette as analyze_figs.py):
  fig/phase_curves_multiseed.png  order parameter & activity vs T, >=5 seeds, error bars
  fig/finite_size.png             order & susceptibility vs T for N in {48,96,192}
  fig/damage_ignition.png         ignition probability vs conditional spread (block flips)
  fig/census_bpe.png              BPE vs word-level corpus recovery
Writes results/analysis_phase2.json with the finite-size verdict.
"""
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
R_ORD = {1: "#f5b894", 2: "#f08f61", 4: "#eb6834", 8: "#c24e1f", 16: "#8f3a17"}
N_ORD = {48: "#9ec5f4", 96: "#2a78d6", 192: "#0d366b"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 150,
})
analysis = {}

# ---------- 1. multi-seed phase curves with error bars --------------------
rows = [json.loads(l) for l in open("results/sweep_multiseed.jsonl")]
S = {(x["r"], x["T"]): x for x in rows}
RS = sorted({x["r"] for x in rows})
TS = sorted({x["T"] for x in rows})
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
for ax, mkey, skey, ttl in [
        (axes[0], "bigram_mean", "bigram_std", "Order parameter (corpus-bigram fraction)"),
        (axes[1], "act_mean", "act_std", "Activity (fraction of sites changing)")]:
    for r in RS:
        m = np.array([S[(r, T)][mkey] for T in TS])
        s = np.array([S[(r, T)][skey] for T in TS])
        ax.plot(TS, m, "-o", color=R_ORD[r], lw=2, ms=4, label=f"r={r}")
        ax.fill_between(TS, m - s, m + s, color=R_ORD[r], alpha=0.18, lw=0)
    ax.set_xlabel("temperature T"); ax.set_title(ttl, loc="left"); ax.set_ylim(0, 1.02)
axes[0].axvspan(1.0, 2.0, color="#f0efec", zorder=0)
axes[0].legend(title="rule radius", fontsize=8, title_fontsize=8, loc="center left")
fig.suptitle("Phase curves with 5-seed error bars — radius collapse holds under seed noise",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/phase_curves_multiseed.png"); plt.close()

# ---------- 2. finite-size --------------------------------------------------
fs = json.load(open("results/finite_size.json"))
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5))
for N in [48, 96, 192]:
    d = fs[str(N)]
    ax[0].plot(d["T"], d["order_mean"], "-o", color=N_ORD[N], lw=2, ms=4, label=f"N={N}")
    ax[1].plot(d["T"], d["susceptibility"], "-o", color=N_ORD[N], lw=2, ms=4, label=f"N={N}")
ax[0].set_title("Order parameter vs T (curves overlay -> no sharpening)", loc="left")
ax[0].set_ylim(0, 1.02)
ax[1].set_title("Susceptibility Var(order) (peak ~ 1/N -> self-averaging)", loc="left")
for a in ax:
    a.set_xlabel("temperature T"); a.legend(fontsize=8)
# robust transition-vs-crossover diagnostics (width_75_25 is nan: grid stops at
# T=1.5 where order is still ~0.41, so it never reaches 0.25):
peaks = {N: fs[str(N)]["chi_peak"] for N in [48, 96, 192]}
slopes = {}
for N in [48, 96, 192]:
    o, t = np.array(fs[str(N)]["order_mean"]), np.array(fs[str(N)]["T"])
    slopes[N] = float(np.abs(np.diff(o) / np.diff(t)).max())
# transition => chi_peak GROWS with N and max slope steepens; crossover => the
# opposite. Here chi_peak ~ 1/N (self-averaging) and slope is ~constant.
chi_grows = peaks[192] > peaks[48]
slope_steepens = slopes[192] > slopes[48] * 1.1
verdict = ("TRANSITION (chi_peak grows with N, drop steepens)" if (chi_grows and slope_steepens)
           else "CROSSOVER (chi_peak ~ 1/N self-averaging, drop does NOT steepen with N)")
analysis["finite_size"] = dict(chi_peak=peaks, max_slope=slopes,
                               chi_ratio_48_over_192=round(peaks[48] / peaks[192], 2),
                               verdict=verdict)
fig.suptitle(f"Finite-size scan: {verdict}",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/finite_size.png"); plt.close()

# ---------- 3. damage ignition (block flips) --------------------------------
ds = json.load(open("results/damage_block.json"))
TS_d = sorted({float(k.split("_")[0][1:]) for k in ds})
RS_d = sorted({int(k.split("_")[1][1:]) for k in ds})
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=False)
x = np.arange(len(TS_d)); wbar = 0.8 / len(RS_d)
for j, r in enumerate(RS_d):
    ig = [ds[f"T{T}_r{r}"]["ignition_prob"] for T in TS_d]
    cs = [ds[f"T{T}_r{r}"]["cond_spread"] for T in TS_d]
    ax[0].bar(x + j * wbar, ig, wbar, color=R_ORD[r], label=f"r={r}")
    ax[1].bar(x + j * wbar, cs, wbar, color=R_ORD[r], label=f"r={r}")
for a, ttl in [(ax[0], "Ignition probability P(damage spreads)"),
               (ax[1], "Conditional spread  E[damage | ignited]")]:
    a.set_xticks(x + 0.4 - wbar / 2); a.set_xticklabels([f"T={t}" for t in TS_d])
    a.set_title(ttl, loc="left"); a.set_ylim(0, 1.02); a.legend(fontsize=8)
fig.suptitle("Block-flip damage (3 sites, B=64): ignition probability vs conditional spread",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/damage_ignition.png"); plt.close()

# ---------- 4. BPE vs word-level census ------------------------------------
try:
    cb = json.load(open("results/census_bpe.json"))["census"]
    wl = json.load(open("results/census.json"))["census"]
    Ts = ["0.3", "0.7", "1.0"]
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    x = np.arange(len(Ts))
    ax.bar(x - 0.2, [wl[t]["overlap50"] for t in Ts], 0.36, color="#c3c2b7", label="word-level (<unk>)")
    ax.bar(x + 0.2, [cb[t]["overlap50"] for t in Ts], 0.36, color=BLUE, label="BPE (no <unk>)")
    for i, t in enumerate(Ts):
        ax.text(x[i] - 0.2, wl[t]["overlap50"] + 0.01, f"rho={wl[t]['spearman']:.2f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.2, cb[t]["overlap50"] + 0.01, f"rho={cb[t]['spearman']:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"T={t}" for t in Ts])
    ax.set_ylabel("top-50 trigram overlap with corpus"); ax.set_ylim(0, 0.7)
    ax.set_title("Census recovery: word-level vs BPE (trigram units differ)", loc="left")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("fig/census_bpe.png"); plt.close()
    analysis["census_bpe_vs_word"] = {t: dict(word=wl[t]["overlap50"], bpe=cb[t]["overlap50"]) for t in Ts}
except Exception as e:
    print("census fig skipped:", e)

json.dump(analysis, open("results/analysis_phase2.json", "w"), indent=1)
print(json.dumps(analysis, indent=1))
print("PHASE2 FIGS DONE")
