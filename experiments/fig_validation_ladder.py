"""The validation ladder: five rungs, ordered by how much of the instrument's regime each shares.

REGENERATED in Phase 4.2. The previous version shipped two claims the project has since
retracted, and a figure that contradicts its own caption is the last way a retraction can
still reach a reviewer:

  * panel (3) was titled "ECA classes: ord -0.32 < edge +0.19 < chaos +0.26" -- the
    three-class ordering demoted by F33/F34 and re-tested to p=0.470 by F36. Only the
    coarse ordered-vs-rest split survives, and it must be measured on IGNITION PROBABILITY,
    the DP-class order parameter, not on lambda.
  * that same panel drew Rule 128 at lambda = -0.92, which is not a measurement at all: it
    is the estimator's DEAD_DAMAGE_FLOOR (= -0.4 ln 10), the constant a log-linear fit
    returns when damage dies immediately (F40). Plotting a sentinel as a bar is how the
    -0.32 group mean looked like an exponent in the first place.

Rungs now shown, in the paper's order:
  (1) logistic map      -- smooth-limit arithmetic UNIT TEST, labelled as such (F30/F31)
  (2) coupled-map lattice against an exact Benettin/Jacobian reference (F37)
  (3) elementary CA, ignition probability, ordered vs rest (F36)
  (4) Domany-Kinzel: the bit-exact damage identity, and its critical points (F38)
  (5) synthetic-Markov census against known transition matrices

Reads only results/*.json; writes fig/validation_ladder.png.
"""
import pathlib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
rep = json.load(open(R / "reproduce_lyapunov.json"))
ben = json.load(open(R / "cml_benettin.json"))
ign = json.load(open(R / "eca_ordered_vs_rest.json"))
dk = json.load(open(R / "dk_calib.json"))
cen = json.load(open(R / "calib_census.json"))

C = dict(known="#2c6fbb", ours="#b0413e", ord="#3a7d44", edge="#c78a1e",
         chaos="#b0413e", cml="#6a4fb0", grey="#999999")
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.5, "figure.dpi": 200})
fig, ax = plt.subplots(2, 3, figsize=(14.5, 7.2))

# ---------------------------------------------------------------- (1) logistic: a unit test
r = np.array(rep["logistic"]["r"]); ex = np.array(rep["logistic"]["exact"])
cr = np.array(rep["logistic"]["crn"])
a = ax[0, 0]
a.plot(r, ex, "-", color=C["known"], lw=2, label="known (analytic) ⟨ln|f'|⟩")
a.plot(r, cr, "o", color=C["ours"], ms=2.5, label="renormalized tangent estimator")
a.axhline(0, color="gray", lw=0.7, ls=":"); a.axhline(rep["ln2"], color="green", lw=0.8, ls="--")
a.set_xlabel("logistic r"); a.set_ylabel("Lyapunov λ"); a.legend(fontsize=7, loc="lower right")
a.set_title("(1) logistic map — SMOOTH-LIMIT UNIT TEST\nnot validation: the estimator is a "
            "finite-difference\nderivative of what it is compared to (F30)", fontsize=8.5)

# ---------------------------------------------------------------- (2) CML vs exact Benettin
a = ax[0, 1]
eps = sorted(float(k) for k in ben["by_eps"])
bmean = [ben["by_eps"][f"{e:g}"]["benettin_mean"] for e in eps]
cmean = [ben["by_eps"][f"{e:g}"]["cml_lyap_mean"] for e in eps]
a.plot(eps, bmean, "s-", color=C["known"], lw=1.6, ms=6, label="Benettin (exact Jacobian)")
a.plot(eps, cmean, "o--", color=C["cml"], lw=1.4, ms=5, label="instrument's estimator")
a.axhline(ben["ln2"], color="green", lw=0.8, ls="--", label=f"single-site ln2={ben['ln2']}")
a.set_xlabel("diffusive coupling ε"); a.set_ylabel("lattice λ"); a.legend(fontsize=7)
a.set_title(f"(2) coupled-map lattice vs exact reference\nmax |Benettin − estimator| = "
            f"{ben['max_abs_diff']} (F37)\nnon-monotone in ε", fontsize=8.5)

# ---------------------------------------------------------------- (3) ECA: ignition probability
a = ax[0, 2]
order = [("ordered", C["ord"]), ("edge", C["edge"]), ("chaotic", C["chaos"])]
xs, hs, cols, ticks = [], [], [], []
x = 0
for g, col in order:
    for rn, p in sorted(ign["groups"][g]["rules"].items(), key=lambda kv: int(kv[0])):
        xs.append(x); hs.append(p); cols.append(col); ticks.append(rn); x += 1
    x += 0.8
a.bar(xs, hs, color=cols, width=0.82)
for g, col in order:                                   # group means with bootstrap CI
    idx = [i for i, t in enumerate(ticks) if t in ign["groups"][g]["rules"]]
    seg = [xs[i] for i in idx]
    m = ign["groups"][g]["mean"]; lo, hi = ign["groups"][g]["ci95"]
    a.hlines(m, min(seg) - 0.45, max(seg) + 0.45, color="k", lw=1.6)
    a.fill_between([min(seg) - 0.45, max(seg) + 0.45], lo, hi, color="k", alpha=0.10, lw=0)
a.set_xticks(xs); a.set_xticklabels(ticks, fontsize=6, rotation=90)
a.set_ylim(0, 1.05); a.set_xlabel("ECA rule (green ordered / amber edge / red chaotic)")
a.set_ylabel("ignition probability")
t = ign["tests"]
a.set_title(f"(3) ECA: ordered vs rest on IGNITION PROB.\n"
            f"p={t['ordered_lt_rest_p']:.3f}, Cohen d={t['cohens_d_ordered_vs_rest']}  |  "
            f"edge vs chaotic p={t['edge_lt_chaotic_p']:.2f}\n"
            f"(the 3-class ordering does NOT survive, F36)", fontsize=8.5)

# ---------------------------------------------------------------- (4) DK: the exact identity
a = ax[1, 0]
pa = dk["part_a_exact_identity"]
labels = [f"$p_1$={k}" for k in pa["by_p1"]] + ["control\n(0.6, 0.5)"]
vals = [v["mismatching_cells"] for v in pa["by_p1"].values()] + [pa["control_offline_mismatch"]]
cols = [C["ord"]] * len(pa["by_p1"]) + [C["ours"]]
a.bar(range(len(vals)), vals, color=cols, width=0.7)
for i, v in enumerate(vals):
    a.text(i, v + 0.4, str(v), ha="center", fontsize=8,
           fontweight="bold" if v == 0 else "normal")
a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
a.set_ylim(0, max(vals) * 1.35); a.set_ylabel("mismatching cells vs prediction")
a.set_title("(4) Domany–Kinzel: the damage field IS the\nautomaton on $p_2$=0 — predicted "
            f"independently,\n{pa['N']}×{pa['steps']}, ZERO mismatches (F38)", fontsize=8.5)

# ---------------------------------------------------------------- (5) DK critical points
a = ax[1, 1]
cal = dk["calibration"]
rows = [("site DP\n$p_1$=$p_2$", cal["site DP"]["estimate"], 0.705489, None),
        ("W18 $p_2$=0\nactivity", cal["W18 activity"]["estimate"], 0.8087, 0.801),
        ("W18 $p_2$=0\ndamage", cal["W18 damage"]["estimate"], 0.8087, 0.801)]
xx = np.arange(len(rows))
# points, not bars: the axis cannot start at 0 here, and truncated bars misencode ratios
for i, (_, est, pub, pub2) in enumerate(rows):
    a.hlines(pub, i - 0.30, i + 0.30, color=C["known"], lw=2.4,
             label="published" if i == 0 else None)
    if pub2 is not None:
        a.hlines(pub2, i - 0.30, i + 0.30, color=C["grey"], lw=2.0, ls="--",
                 label="published (disputed 2nd value)" if i == 1 else None)
    # our own ~1% method error, shown so the reader sees why we cannot pick a side
    a.errorbar(i, est, yerr=0.01 * est, fmt="o", color=C["ours"], ms=7, capsize=4,
               lw=1.4, label="measured (±1%, the method's accuracy)" if i == 0 else None)
    a.text(i, est - 0.006, f"{est:.4f}", ha="center", va="top", fontsize=7.5)
a.set_xticks(xx); a.set_xticklabels([r[0] for r in rows], fontsize=7.5)
a.set_xlim(-0.5, len(rows) - 0.3)
a.set_ylim(0.685, 0.828); a.set_ylabel("$p_c$")
a.legend(fontsize=6.2, loc="lower right", framealpha=0.95)
a.set_title("(5) DK critical points — the WEAKER check\n~1% method, so it cannot separate the\n"
            "two disputed $p_2$=0 values; both shown", fontsize=8.5)

# ---------------------------------------------------------------- (6) census
a = ax[1, 2]
srcs = ["a", "b", "c"]
selfv = [cen["self_recovery"][s] for s in srcs]
crossv = [np.mean([cen["TV_matrix"][s][o] for o in srcs if o != s]) for s in srcs]
base = [cen["baseline_TV"][s] for s in srcs]
xx = np.arange(3); w = 0.26
a.bar(xx - w, selfv, w, color=C["known"], label=f"self-recovery (μ={cen['mean_self']:.2f})")
a.bar(xx, crossv, w, color=C["ours"], label=f"cross (μ={cen['mean_cross']:.2f})")
a.bar(xx + w, base, w, color=C["grey"], label="random baseline")
a.set_xticks(xx); a.set_xticklabels([f"source {s.upper()}" for s in srcs])
a.set_ylabel("row total-variation to true P"); a.legend(fontsize=7)
a.set_title("(6) census recovers known transition matrices", fontsize=8.5)

fig.suptitle("Validation ladder — rungs ordered by how much of the instrument's regime each shares. "
             "Only (3)–(5) are weight-bearing.", fontsize=11, y=1.005)
fig.tight_layout()
fp = str(ROOT / "fig" / "validation_ladder.png")
fig.savefig(fp, bbox_inches="tight"); print("wrote", fp)
