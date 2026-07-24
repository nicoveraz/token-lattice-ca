"""The validation ladder: the instrument reproduces KNOWN metrics at four rungs of rigor.
(1) logistic-map Lyapunov (exact analytic), (2) coupled-map-lattice finite-size Lyapunov,
(3) elementary-CA criticality classes, (4) synthetic-Markov census (known transition matrices).
This is the credibility spine of the reframed paper: the CA/damage-spreading instrument is
validated by recovering established quantities before it is applied to LMs.
"""
import pathlib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
rep = json.load(open(ROOT / "results" / "reproduce_lyapunov.json"))
eca = json.load(open(ROOT / "results" / "eca_calib.json"))
cen = json.load(open(ROOT / "results" / "calib_census.json"))

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "figure.dpi": 200})
fig, ax = plt.subplots(2, 2, figsize=(12.5, 8))

# (1) logistic map — exact analytic vs CA damage-spreading
r = np.array(rep["logistic"]["r"]); ex = np.array(rep["logistic"]["exact"]); cr = np.array(rep["logistic"]["crn"])
ax[0, 0].plot(r, ex, "-", color="#2c6fbb", lw=2, label="known (analytic) ⟨ln|f'|⟩")
ax[0, 0].plot(r, cr, "o", color="#b0413e", ms=2.5, label="CA damage-spreading (CRN)")
ax[0, 0].axhline(0, color="gray", lw=0.7, ls=":"); ax[0, 0].axhline(rep["ln2"], color="green", lw=0.8, ls="--")
ax[0, 0].set_xlabel("logistic r"); ax[0, 0].set_ylabel("Lyapunov λ")
ax[0, 0].legend(fontsize=8, loc="lower right")
ax[0, 0].set_title(f"(1) logistic map — EXACT (mean err {rep['logistic']['mean_abs_err']:.4f})")

# (2) coupled map lattice
ec = [float(k) for k in rep["cml"]]; lam = [rep["cml"][k] for k in rep["cml"]]
ax[0, 1].plot(ec, lam, "o-", color="#6a4fb0", lw=1.6, ms=6)
ax[0, 1].axhline(rep["ln2"], color="green", lw=0.8, ls="--", label=f"single-site ln2={rep['ln2']}")
ax[0, 1].set_xlabel("diffusive coupling ε"); ax[0, 1].set_ylabel("lattice λ")
ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("(2) coupled-map lattice (spatial CA)")

# (3) elementary CA criticality classes
grp = [("ordered", [128, 232, 4], "#3a7d44"), ("edge", [110, 54], "#c78a1e"), ("chaotic", [30, 150, 22, 90], "#b0413e")]
x = 0; ticks = []; tlab = []
for gname, rules, c in grp:
    for rn in rules:
        ax[1, 0].bar(x, eca[str(rn)]["lambda_ca"], color=c, width=0.82); ticks.append(x); tlab.append(str(rn)); x += 1
    x += 0.6
ax[1, 0].axhline(0, color="gray", lw=0.8); ax[1, 0].set_xticks(ticks); ax[1, 0].set_xticklabels(tlab, fontsize=8)
gm = eca["_group_means"]
ax[1, 0].set_xlabel("ECA rule  (green ord / amber edge / red chaos)"); ax[1, 0].set_ylabel("λ_ca")
ax[1, 0].set_title(f"(3) ECA classes: ord {gm['ordered']:+.2f} < edge {gm['edge']:+.2f} < chaos {gm['chaotic']:+.2f}")

# (4) census recovery of known transition matrices
srcs = ["a", "b", "c"]
selfv = [cen["self_recovery"][s] for s in srcs]
crossv = [np.mean([cen["TV_matrix"][s][o] for o in srcs if o != s]) for s in srcs]
base = [cen["baseline_TV"][s] for s in srcs]
xx = np.arange(3); w = 0.26
ax[1, 1].bar(xx - w, selfv, w, color="#2c6fbb", label=f"self-recovery (μ={cen['mean_self']:.2f})")
ax[1, 1].bar(xx, crossv, w, color="#b0413e", label=f"cross (μ={cen['mean_cross']:.2f})")
ax[1, 1].bar(xx + w, base, w, color="#999999", label="random baseline")
ax[1, 1].set_xticks(xx); ax[1, 1].set_xticklabels([f"source {s.upper()}" for s in srcs])
ax[1, 1].set_ylabel("row total-variation to true P"); ax[1, 1].legend(fontsize=8)
ax[1, 1].set_title("(4) census recovers known transition matrices")

fig.suptitle("Validation ladder — the CA instrument reproduces known metrics before it measures LMs",
             fontsize=12.5, y=1.005)
fig.tight_layout()
fp = str(ROOT / "fig" / "validation_ladder.png")
fig.savefig(fp, bbox_inches="tight"); print("wrote", fp)
