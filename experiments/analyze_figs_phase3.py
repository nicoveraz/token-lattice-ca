"""Phase 3 figures + headline analysis for the real-MLM probes (same palette).
Robust to partially-complete runs: only plots models whose json exists.
  fig/mlm_radius.png     k-gram overlap vs r (radius test, headline a)
  fig/mlm_phase.png      order parameter vs T across tiny/mini/base (headline d)
  fig/mlm_damage.png     light-cone velocity vs r + self-healing vs T (headlines b,c)
  fig/mlm_differential.png  delta-order: null / apparatus / model arms (F9 certification)
Writes results/analysis_phase3.json with the headline answers.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SEC, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
M_COL = {"tiny": "#9ec5f4", "mini": "#2a78d6", "base": "#0d366b"}
K_COL = {2: "#eb6834", 3: "#2a78d6", 4: "#1baf7a"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 150,
})
R = "results/mlm"
TAGS = [t for t in ["tiny", "mini", "base"] if os.path.exists(f"{R}/{t}_sweep.json")]
analysis = {}


def sweep(tag):
    return {(x["r"], x["T"]): x for x in json.load(open(f"{R}/{tag}_sweep.json"))}


# ---------- 1. radius test (headline a) ----------
if TAGS:
    fig, axes = plt.subplots(1, len(TAGS), figsize=(3.4 * len(TAGS), 3.4), squeeze=False)
    rad = {}
    for j, tag in enumerate(TAGS):
        S = sweep(tag)
        RS = sorted({r for (r, T) in S})
        T0 = min({T for (r, T) in S})                      # most ordered T
        ax = axes[0, j]
        for k in (2, 3, 4):
            ys = [S[(r, T0)][f"k{k}"] for r in RS]
            ax.plot(RS, ys, "-o", color=K_COL[k], lw=2, ms=4, label=f"{k}-gram")
        ax.set_xscale("log", base=2); ax.set_xticks(RS); ax.set_xticklabels(RS)
        ax.set_xlabel("rule radius r"); ax.set_title(f"{tag} (T={T0})", loc="left")
        ax.set_ylim(0, max(0.3, max(S[(r, T0)]["k4"] for r in RS) * 1.2))
        if j == 0:
            ax.legend(fontsize=8); ax.set_ylabel("fraction of ring k-grams in corpus")
        k2 = [S[(r, T0)]["k2"] for r in RS]
        k4 = [S[(r, T0)]["k4"] for r in RS]
        rad[tag] = dict(T=T0, k2_spread=round(max(k2) - min(k2), 3),
                        k4_r1=round(k4[0], 3), k4_peak=round(max(k4), 3),
                        k4_peak_r=RS[int(np.argmax(k4))])
    analysis["radius_test"] = rad
    fig.suptitle("Radius test: local (bigram) order is ~radius-blind, but longer-range "
                 "structure grows with r (unlike the toy)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/mlm_radius.png"); plt.close()

# ---------- 2. phase curves across models (headline d) ----------
if TAGS:
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5))
    for tag in TAGS:
        S = sweep(tag)
        TS = sorted({T for (r, T) in S})
        o = [np.mean([S[(r, T)]["order"] for r in sorted({r for (r, T2) in S})]) for T in TS]
        a = [np.mean([S[(r, T)]["act_final"] for r in sorted({r for (r, T2) in S})]) for T in TS]
        ax[0].plot(TS, o, "-o", color=M_COL[tag], lw=2, ms=4, label=tag)
        ax[1].plot(TS, a, "-o", color=M_COL[tag], lw=2, ms=4, label=tag)
    ax[0].axvspan(1.5, 2.0, color="#f0efec", zorder=0)
    ax[0].set_title("Order parameter vs T (avg over r)", loc="left"); ax[0].set_ylim(0, 1.02)
    ax[1].set_title("Activity vs T (avg over r)", loc="left"); ax[1].set_ylim(0, 1.02)
    for a in ax:
        a.set_xlabel("temperature T"); a.legend(fontsize=8, title="model")
    fig.suptitle("Real-MLM phase behavior across scale (tiny -> mini -> base)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/mlm_phase.png"); plt.close()

# ---------- 3. damage: velocity vs r + healing vs T (headlines b,c) ----------
dtags = [t for t in TAGS if os.path.exists(f"{R}/{t}_damage.json")]
if dtags:
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5))
    heal = {}
    for tag in dtags:
        d = json.load(open(f"{R}/{tag}_damage.json"))
        vr = d["velocity_vs_r"]
        rs = sorted(int(k) for k in vr)
        ax[0].plot(rs, [vr[str(r)]["velocity"] for r in rs], "-o", color=M_COL[tag], lw=2, ms=4, label=tag)
        hv = d["healing_vs_T"]
        ts = sorted(float(k) for k in hv)
        ax[1].plot(ts, [hv[str(t)]["mean_damage"] for t in ts], "-o", color=M_COL[tag], lw=2, ms=4, label=tag)
        # healing boundary: lowest T where mean_damage>0.5
        boundary = next((t for t in ts if hv[str(t)]["mean_damage"] > 0.5), None)
        heal[tag] = dict(healing_boundary_T=boundary,
                         v_r1=round(vr[str(rs[0])]["velocity"], 2),
                         v_rmax=round(vr[str(rs[-1])]["velocity"], 2))
    ax[0].set_xscale("log", base=2)
    rs_all = sorted(int(k) for k in json.load(open(f"{R}/{dtags[0]}_damage.json"))["velocity_vs_r"])
    ax[0].set_xticks(rs_all); ax[0].set_xticklabels(rs_all)
    ax[0].set_xlabel("rule radius r"); ax[0].set_ylabel("front velocity (sites/sweep)")
    ax[0].set_title("Damage light-cone velocity vs r (T=1.0)", loc="left"); ax[0].legend(fontsize=8)
    ax[1].axvspan(1.5, 2.0, color="#f0efec", zorder=0)
    ax[1].text(1.75, 0.08, "tau~1.5-2\n(full-context\nMLM-Glauber)", ha="center", fontsize=7, color=MUT)
    ax[1].set_xlabel("temperature T"); ax[1].set_ylabel("mean damage (block flip)")
    ax[1].set_title("Self-healing vs T (r=4)", loc="left"); ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=8)
    analysis["damage"] = heal
    fig.suptitle("Damage transport: light cones (b) and the self-healing phase vs the tau crossover (c)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/mlm_damage.png"); plt.close()

# ---------- 4. differential certification (F9) ----------
diff_tags = [t for t in TAGS if os.path.exists(f"{R}/{t}_diff.json")]
if diff_tags:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels, nulls, apps, models_d = [], [], [], []
    cert = {}
    for tag in diff_tags:
        d = json.load(open(f"{R}/{tag}_diff.json"))
        nul = np.mean([v["delta_order"] for k, v in d.items() if k.startswith("null")])
        app = np.mean([v["delta_order"] for k, v in d.items() if k.startswith("apparatus:order")
                       or k.startswith("apparatus:cdfperm")])
        labels.append(tag); nulls.append(nul); apps.append(app)
        cert[tag] = dict(null_delta=round(float(nul), 4), apparatus_delta=round(float(app), 4))
    # model-arm deltas
    marm = {}
    for f in os.listdir(R):
        if f.startswith("model_arm_"):
            d = json.load(open(f"{R}/{f}"))
            marm[f[:-5]] = round(float(np.mean([v["delta_order"] for v in d.values()])), 4)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, nulls, 0.38, color="#c3c2b7", label="null (nothing differs)")
    ax.bar(x + 0.2, apps, 0.38, color=BLUE, label="apparatus (order/cdfperm swap)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Δ order parameter"); ax.set_title(
        "Differential certification: apparatus swaps null the order parameter; "
        "model swaps move it", loc="left")
    ax.legend(fontsize=8)
    if marm:
        txt = "model-arm Δorder:  " + "   ".join(f"{k.replace('model_arm_','')}={v}" for k, v in marm.items())
        ax.text(0.01, 0.95, txt, transform=ax.transAxes, fontsize=8, color=MAGENTA, va="top")
    cert["model_arm"] = marm
    analysis["differential"] = cert
    fig.tight_layout(); fig.savefig("fig/mlm_differential.png"); plt.close()

# census summary into analysis
cen = {}
for tag in TAGS:
    p = f"{R}/{tag}_census.json"
    if os.path.exists(p):
        c = json.load(open(p))
        cen[tag] = {T: dict(overlap50=c[T]["overlap50"], spearman=c[T]["spearman"],
                            baseline=c[T]["baseline_overlap50"]) for T in c}
analysis["census_proxy"] = cen

# ---------- 5. space-time diagram (soup -> ordered English) ----------
st_tag = "base" if "base" in TAGS else (TAGS[-1] if TAGS else None)
if st_tag and os.path.exists(f"{R}/{st_tag}_spacetime.npz"):
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    seq_cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    import mlm_ca
    from collections import Counter
    ref = np.load("data_mlm/ref_ids.npy")
    freq = Counter(ref.tolist())
    rank = {t: i for i, (t, _) in enumerate(freq.most_common())}
    maxr = len(rank)
    def rankfield(snaps):
        f = np.vectorize(lambda t: rank.get(int(t), maxr))(snaps)
        return np.log1p(f) / np.log1p(maxr)
    z = np.load(f"{R}/{st_tag}_spacetime.npz")
    Ts = [t for t in [0.5, 1.0, 2.0] if f"snaps_T{t}" in z]
    fig, axes = plt.subplots(2, len(Ts), figsize=(3.1 * len(Ts), 5.2), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for j, T in enumerate(Ts):
        s = z[f"snaps_T{T}"]
        axes[0, j].imshow(rankfield(s), aspect="auto", cmap=seq_cmap, vmin=0, vmax=1)
        axes[0, j].set_title(f"T = {T}", loc="left"); axes[0, j].grid(False)
        chg = (np.diff(s, axis=0) != 0)
        axes[1, j].imshow(chg, aspect="auto", cmap=ListedColormap([SURF, "#2a78d6"]), vmin=0, vmax=1)
        axes[1, j].grid(False); axes[1, j].set_xlabel("lattice site")
    axes[0, 0].set_ylabel("sweep →\n(token corpus-rarity)", fontsize=8)
    axes[1, 0].set_ylabel("sweep →\n(site changed)", fontsize=8)
    fig.suptitle(f"{st_tag}: space-time (r=2) — random soup -> ordered English vs churn",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("fig/mlm_spacetime.png"); plt.close()

json.dump(analysis, open("results/analysis_phase3.json", "w"), indent=1)
print(json.dumps(analysis, indent=1))
print("PHASE3 FIGS DONE  (models:", TAGS, ")")
