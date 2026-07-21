"""Figures + analysis for the token-CA pilot. Palette per dataviz reference."""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from collections import Counter

INK, SEC, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
T_ORD = {0.3: "#86b6ef", 0.7: "#5598e7", 1.0: "#2a78d6", 1.5: "#1c5cab", 2.5: "#0d366b"}
R_ORD = {1: "#f5b894", 2: "#f0906144", 4: "#eb6834", 8: "#c24e1f", 16: "#8f3a17"}
R_ORD = {1: "#f5b894", 2: "#f08f61", 4: "#eb6834", 8: "#c24e1f", 16: "#8f3a17"}
seq_cmap = LinearSegmentedColormap.from_list("seq", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 150,
})

TS = [0.3, 0.7, 1.0, 1.5, 2.5]
RS = [1, 2, 4, 8, 16]
rows = [json.loads(l) for l in open("results/summary.jsonl")]
S = {(x["mode"], x["r"], x["T"]): x for x in rows}
analysis = {}

# ---------- 1. phase curves ----------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
for ax, key, ttl in [(axes[0], "bigram_final", "Order parameter: corpus-bigram fraction"),
                     (axes[1], "act_final", "Activity (fraction of sites changing)")]:
    for r in RS:
        ys = [S[("async", r, T)][key] for T in TS]
        ax.plot(TS, ys, "-o", color=R_ORD[r], lw=2, ms=4.5, label=f"r={r}")
    ax.set_xlabel("temperature T"); ax.set_title(ttl, loc="left")
    ax.set_ylim(0, 1.02)
axes[0].axvspan(1.0, 2.0, color="#f0efec", zorder=0)
axes[0].text(1.5, 0.06, "reported MLM-Glauber\ncrossover zone", ha="center",
             fontsize=7.5, color=MUT)
axes[0].legend(title="rule radius", fontsize=8, title_fontsize=8, loc="center left")
fig.suptitle("Token-lattice CA phase behavior — curves collapse across radius",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/phase_curves.png"); plt.close()

# ---------- 2. space-time diagrams ----------
freq = Counter(np.load("data/train_ids.npy").tolist())
rank_of = {tok: i for i, (tok, _) in enumerate(freq.most_common())}
maxrank = len(rank_of)
def rankfield(snaps):
    f = np.vectorize(lambda t: rank_of.get(int(t), maxrank))(snaps)
    return np.log1p(f) / np.log1p(maxrank)

fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6), sharex=True, sharey=True)
for j, T in enumerate([0.3, 1.0, 2.5]):
    z = np.load(f"results/sweep_async_r2_T{T}.npz")
    snaps = z["snaps"][:, 0, :]                      # lattice 0: (sweeps+1, N)
    axes[0, j].imshow(rankfield(snaps), aspect="auto", cmap=seq_cmap, vmin=0, vmax=1)
    axes[0, j].set_title(f"T = {T}", loc="left")
    chg = (np.diff(snaps, axis=0) != 0)
    axes[1, j].imshow(chg, aspect="auto",
                      cmap=ListedColormap([SURF, "#2a78d6"]), vmin=0, vmax=1)
    for ax in (axes[0, j], axes[1, j]):
        ax.grid(False)
axes[0, 0].set_ylabel("sweep →\n(token corpus-rarity)", fontsize=8)
axes[1, 0].set_ylabel("sweep →\n(site changed)", fontsize=8)
for j in range(3): axes[1, j].set_xlabel("lattice site")
fig.suptitle("Space-time diagrams (async, r=2): random soup → ordered corpus-like phase vs churn",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig("fig/spacetime.png"); plt.close()

# ---------- 3. damage cones ----------
dz = np.load("results/damage.npz")
fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True)
widths = {}
for i, T in enumerate([0.3, 0.7, 1.5]):
    for j, r in enumerate([1, 4, 16]):
        cone = dz[f"T{T}_r{r}"]
        axes[i, j].imshow(cone, aspect="auto", cmap=seq_cmap, vmin=0, vmax=1)
        axes[i, j].grid(False)
        w = int((cone[-10:].mean(axis=0) > 0.05).sum())
        widths[f"T{T}_r{r}"] = dict(final_width_sites=w,
                                    total_damage=float(cone[-10:].mean()))
        axes[i, j].set_title(f"T={T}, r={r}   width≈{w} sites", loc="left", fontsize=9)
for j in range(3): axes[2, j].set_xlabel("site (flip at center)")
for i in range(3): axes[i, 0].set_ylabel("sweep →")
fig.suptitle("Damage spreading: probability twin runs differ (common random numbers)",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("fig/damage_cones.png"); plt.close()
analysis["damage"] = widths

# ---------- 4. melting curves ----------
mel = json.load(open("results/census.json"))["melts"]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for T in TS:
    k = mel[str(T)]
    ax.plot(range(len(k)), k, color=T_ORD[T], lw=2)
    ax.text(len(k) - 1 + 0.8, k[-1], f"T={T}", color=T_ORD[T], fontsize=8, va="center")
ax.set_xlabel("sweep"); ax.set_ylabel("fraction of original tokens intact")
ax.set_xlim(0, 72); ax.set_ylim(0, 1.02)
ax.set_title("Melting of real corpus text under the CA rule (r=2)", loc="left")
fig.tight_layout(); fig.savefig("fig/melting.png"); plt.close()

# ---------- 5. census validation ----------
cen = json.load(open("results/census.json"))["census"]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
xs = np.arange(3)
ov = [cen[t]["overlap50"] for t in ["0.3", "0.7", "1.0"]]
bs = [cen[t]["baseline_overlap50"] for t in ["0.3", "0.7", "1.0"]]
ax.bar(xs - 0.18, ov, 0.32, color="#2a78d6", label="CA census")
ax.bar(xs + 0.18, bs, 0.32, color="#c3c2b7", label="random-lattice baseline")
for x, v, t in zip(xs, ov, ["0.3", "0.7", "1.0"]):
    ax.text(x - 0.18, v + 0.02, f"{v:.2f}\nρ={cen[t]['spearman']:.2f}",
            ha="center", fontsize=8, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([f"T={t}" for t in ["0.3", "0.7", "1.0"]])
ax.set_ylabel("top-50 trigram overlap with corpus"); ax.set_ylim(0, 0.75)
ax.set_title("Ground-truth validation: census recovers corpus n-gram structure", loc="left")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("fig/census_validation.png"); plt.close()

# ---------- 6. sync oscillation metric ----------
p2 = {}
for mode in ["async", "sync"]:
    for T in [0.3, 0.7, 1.0]:
        z = np.load(f"results/sweep_{mode}_r2_T{T}.npz")["snaps"][-40:]
        s = z.astype(np.int32)
        per2 = float(((s[2:] == s[:-2]) & (s[2:] != s[1:-1])).mean())
        p2[f"{mode}_T{T}"] = round(per2, 4)
analysis["period2_fraction"] = p2
json.dump(analysis, open("results/analysis.json", "w"), indent=1)
print(json.dumps(analysis, indent=1))
print("FIGS DONE")
