"""Phase A analysis + figures + acceptance verdicts (F20).
  A1: certify F15's radius profile as a MODEL claim, not apparatus — cross-model
      profile shift (fixed scheme) vs the distribution-preserving apparatus floor
      and the scheme-swap magnitude, with 5-seed error bars.
  A2: velocity ceiling — v(N) per r; does the 11.5 plateau lift with N?
  A3: repetition control — does the intermediate-radius peak survive the distinct-
      corpus-kgram metric and show up as a longer MI-decay length, with distinct-
      token fraction flat (real structure) rather than falling (repetition)?
Writes results/analysis_phaseA.json; figs fig/phaseA_{radius,velocity,repetition}.png.
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
N_COL = {48: "#f5b894", 96: "#eb6834", 192: "#c24e1f", 384: "#8f3a17"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5, "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
})
R = "results/mlm"
RS = [1, 2, 4, 8, 16]
TAGS = [t for t in ["tiny", "mini", "base"] if os.path.exists(f"{R}/phaseA_radius_{t}.json")]
analysis = {}


def prof(tag, scheme, metric):
    d = json.load(open(f"{R}/phaseA_radius_{tag}.json"))[scheme]
    return np.array([d[str(r)][metric] for r in RS])


def prof_std(tag, scheme, metric):
    d = json.load(open(f"{R}/phaseA_radius_{tag}.json"))[scheme]
    return np.array([d[str(r)].get(metric + "_std", 0.0) for r in RS])


# ---------------- A1: scheme certification ----------------
if len(TAGS) >= 2:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for tag in TAGS:                                     # order profile, fixed cls_sep, error bars
        m, s = prof(tag, "cls_sep", "order"), prof_std(tag, "cls_sep", "order")
        ax[0].errorbar(RS, m, yerr=s, fmt="-o", color=M_COL[tag], lw=2, ms=4, capsize=2, label=tag)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(RS); ax[0].set_xticklabels(RS)
    ax[0].set_xlabel("radius r"); ax[0].set_ylabel("order parameter"); ax[0].legend(fontsize=8)
    ax[0].set_title("Radius profiles, fixed scheme (cls_sep), 5 seeds", loc="left")

    # profile distances (mean_r |Δ|) + SE
    def dist(a, b, scheme="cls_sep", metric="order"):
        pa, pb = prof(a, scheme, metric), prof(b, scheme, metric)
        sa, sb = prof_std(a, scheme, metric), prof_std(b, scheme, metric)
        d = float(np.mean(np.abs(pa - pb)))
        se = float(np.mean(np.sqrt(sa**2 + sb**2)) / np.sqrt(5))
        return d, se
    a1 = {"cross_model_fixed_scheme": {}, "scheme_apparatus": {}}
    pairs = [(TAGS[i], TAGS[j]) for i in range(len(TAGS)) for j in range(i+1, len(TAGS))]
    for a, b in pairs:
        d, se = dist(a, b)
        a1["cross_model_fixed_scheme"][f"{a}_vs_{b}"] = dict(dist=round(d, 4), se=round(se, 4))
    for tag in TAGS:                                     # cls_sep vs none, same model
        pa, pb = prof(tag, "cls_sep", "order"), prof(tag, "none", "order")
        a1["scheme_apparatus"][tag] = round(float(np.mean(np.abs(pa - pb))), 4)
    # distribution-preserving apparatus floor from the Phase-3 diff jsons
    floor = []
    for tag in TAGS:
        p = f"{R}/{tag}_diff.json"
        if os.path.exists(p):
            dd = json.load(open(p))
            floor += [v["delta_order"] for k, v in dd.items()
                      if k.startswith("apparatus:order") or k.startswith("apparatus:cdfperm")]
    a1["apparatus_floor_order_cdf"] = round(float(np.mean(floor)), 4) if floor else None
    tm = a1["cross_model_fixed_scheme"].get("tiny_vs_mini", {})
    fl = a1["apparatus_floor_order_cdf"] or 0
    a1["verdict"] = ("F15 CERTIFIED as a model claim: tiny->mini profile shift %.3f±%.3f "
                     "exceeds the fixed-scheme apparatus floor %.3f"
                     % (tm.get("dist", 0), tm.get("se", 0), fl)) if tm.get("dist", 0) - 2*tm.get("se", 0) > fl \
                    else "F15 NOT certified: model shift within apparatus floor"
    analysis["A1_scheme_certification"] = a1
    # bar: model shift vs scheme apparatus vs floor
    labels = [f"{a[:4]}→{b[:4]}" for a, b in pairs]
    ax[1].bar(range(len(pairs)), [a1["cross_model_fixed_scheme"][f"{a}_vs_{b}"]["dist"] for a, b in pairs],
              yerr=[a1["cross_model_fixed_scheme"][f"{a}_vs_{b}"]["se"] for a, b in pairs],
              color=BLUE, capsize=3, label="model shift (fixed scheme)")
    ax[1].axhline(fl, color=ORANGE, lw=1.5, ls="--", label="apparatus floor (order/cdf)")
    sm = np.mean(list(a1["scheme_apparatus"].values()))
    ax[1].axhline(sm, color=MAGENTA, lw=1.5, ls=":", label="scheme swap (mean)")
    ax[1].set_xticks(range(len(pairs))); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("mean_r |Δ order|"); ax[1].legend(fontsize=7.5)
    ax[1].set_title("Model shift vs apparatus (A1 certification)", loc="left")
    fig.suptitle("A1: is the radius-profile shift a MODEL effect once the scheme is held fixed?",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/phaseA_radius.png"); plt.close()

# ---------------- A2: velocity ceiling ----------------
if os.path.exists(f"{R}/phaseA_velocity_tiny.json"):
    v = json.load(open(f"{R}/phaseA_velocity_tiny.json"))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    a2 = {}
    rs = sorted({int(k.split("_r")[1]) for k in v if k.startswith("N")})
    ns = sorted({int(k.split("_r")[0][1:]) for k in v if k.startswith("N")})
    for r in rs:
        vs = [v.get(f"N{N}_r{r}", {}).get("velocity", np.nan) for N in ns]
        ax.plot(ns, vs, "-o", lw=2, ms=5, label=f"r={r}", color=N_COL.get(r*3, BLUE) if r in N_COL else None)
        a2[f"r{r}"] = dict(zip([f"N{N}" for N in ns], vs))
        lifts = (not np.isnan(vs[-1])) and vs[-1] > vs[0] * 1.15
        a2[f"r{r}"]["ceiling_lifts_with_N"] = bool(lifts)
    ax.plot(ns, [n/2 for n in ns], "--", color=MUT, lw=1, label="N/2 (wraparound bound)")
    ax.set_xlabel("ring size N"); ax.set_ylabel("front velocity (sites/sweep)")
    ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.legend(fontsize=8, title="radius")
    lifted = all(a2[f"r{r}"].get("ceiling_lifts_with_N") for r in rs if r >= 8)
    a2["verdict"] = ("ceiling was finite-size wraparound: velocity keeps rising with N (v~r holds)"
                     if lifted else "velocity saturates across N: a real bound")
    analysis["A2_velocity_ceiling"] = a2
    ax.set_title(f"A2: velocity vs N — {a2['verdict'][:48]}", loc="left", fontsize=9)
    fig.suptitle("A2: is the r=8,16 velocity plateau a finite-size artifact?",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig("fig/phaseA_velocity.png"); plt.close()

# ---------------- A3: repetition control ----------------
if TAGS:
    fig, axes = plt.subplots(1, len(TAGS), figsize=(3.4*len(TAGS), 3.6), squeeze=False)
    a3 = {}
    for j, tag in enumerate(TAGS):
        ax = axes[0, j]
        k4 = prof(tag, "cls_sep", "k4"); k4d = prof(tag, "cls_sep", "k4d")
        milen = prof(tag, "cls_sep", "mi_len"); dist_t = prof(tag, "cls_sep", "distinct")
        ax.plot(RS, k4, "-o", color=ORANGE, lw=2, ms=4, label="k4 raw (F15)")
        ax.plot(RS, k4d, "-o", color=BLUE, lw=2, ms=4, label="k4 distinct (repetition-robust)")
        ax.plot(RS, dist_t, "--", color=MUT, lw=1.4, label="distinct-token frac")
        ax2 = ax.twinx(); ax2.plot(RS, milen, ":s", color=AQUA, lw=1.6, ms=3, label="MI decay len")
        ax2.set_ylim(0, 25); ax2.grid(False)
        ax.set_xscale("log", base=2); ax.set_xticks(RS); ax.set_xticklabels(RS)
        ax.set_xlabel("radius r"); ax.set_title(tag, loc="left")
        if j == 0:
            ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
        peak_raw = RS[int(np.argmax(k4))]; peak_dist = RS[int(np.argmax(k4d))]
        a3[tag] = dict(k4_peak_r=peak_raw, k4distinct_peak_r=peak_dist,
                       k4distinct=[round(x, 3) for x in k4d],
                       mi_len=[round(float(x), 1) for x in milen],
                       distinct=[round(float(x), 2) for x in dist_t],
                       survives=bool(max(k4d) > k4d[0] * 1.3 and peak_dist in (2, 4, 8)))
    analysis["A3_repetition"] = a3
    surv = [t for t in TAGS if a3[t]["survives"]]
    fig.suptitle(f"A3: intermediate-r peak survives repetition control in {surv or 'NONE'}",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("fig/phaseA_repetition.png"); plt.close()

json.dump(analysis, open("results/analysis_phaseA.json", "w"), indent=1)
print(json.dumps(analysis, indent=1))
print("PHASE A ANALYSIS DONE (models:", TAGS, ")")
