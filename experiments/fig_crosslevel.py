"""Figure for the cross-level story (honest negative + calibration). Three panels:
(1) the instrument is criticality-calibrated on classical CA rules (ordered vs rest);
(2) the cross-model white-vs-black pairing is family-dependent (Pythia + vs GPT-2 -);
(3) black-box lambda_ca(r) is model-invariant (all models/families overlap) -> no model
signal to proxy the white-box rho(F_r).
"""
import sys, pathlib, json
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / "src")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlm_lib import RESDIR

ROOT = pathlib.Path(__file__).resolve().parents[1]
cl = json.load(open(f"{RESDIR}/crosslevel.json"))
rad = json.load(open(f"{RESDIR}/crosslevel_radius.json"))
ign = json.load(open(f"{ROOT}/results/eca_ordered_vs_rest.json"))

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "figure.dpi": 200})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

# panel 1: ECA calibration on IGNITION PROBABILITY (F36).
# Was: lambda_ca with an "ordered < edge < chaotic" title. Both are retracted -- the 3-class
# ordering fails (p=0.470, F36) and the ordered-group lambda is the estimator's dead-damage
# floor, not a measurement (F40). A figure that still showed them would contradict the text.
x = 0; ticks = []; tlab = []
for gname, c in [("ordered", "#3a7d44"), ("edge", "#c78a1e"), ("chaotic", "#b0413e")]:
    rules = sorted(ign["groups"][gname]["rules"].items(), key=lambda kv: int(kv[0]))
    seg0 = x
    for rn, p_ig in rules:
        ax[0].bar(x, p_ig, color=c, width=0.8)
        ticks.append(x); tlab.append(rn); x += 1
    m = ign["groups"][gname]["mean"]; lo, hi = ign["groups"][gname]["ci95"]
    ax[0].hlines(m, seg0 - 0.45, x - 0.55, color="k", lw=1.6)
    ax[0].fill_between([seg0 - 0.45, x - 0.55], lo, hi, color="k", alpha=0.10, lw=0)
    x += 0.6
ax[0].set_xticks(ticks); ax[0].set_xticklabels(tlab, fontsize=6.5, rotation=90)
ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel("ignition probability (DP order parameter)")
ax[0].set_xlabel("ECA rule (green=ordered, amber=edge, red=chaotic)")
t = ign["tests"]
ax[0].set_title(f"Instrument calibrated on known rules\nordered vs rest: p={t['ordered_lt_rest_p']:.3f}, "
                f"d={t['cohens_d_ordered_vs_rest']}\n(edge vs chaotic p={t['edge_lt_chaotic_p']:.2f} — "
                f"no 3-class ordering)")

# panel 2: cross-model, family-dependent
def pts(fam):
    xs, ys, lab = [], [], []
    for t in cl:
        if not isinstance(cl[t], dict) or "black_lyap" not in cl[t]:
            continue
        if (fam == "pythia") != t.startswith("pythia"):
            continue
        xs.append(cl[t]["black_lyap"]["lambda_ca_r2"]); ys.append(cl[t]["white"]["lambda_top"])
        lab.append(t.replace("pythia-", "").replace("gpt2", "g2"))
    return np.array(xs), np.array(ys), lab
for fam, c, mk in [("pythia", "#2c6fbb", "o"), ("gpt2", "#d1701f", "s")]:
    xs, ys, lab = pts(fam)
    ax[1].scatter(xs, ys, c=c, marker=mk, label=f"{fam} (r={'+0.71' if fam=='pythia' else '-0.43'})")
    if len(xs) > 2:
        z = np.polyfit(xs, ys, 1); xx = np.linspace(xs.min(), xs.max(), 10)
        ax[1].plot(xx, np.polyval(z, xx), "--", color=c, lw=1, alpha=0.6)
ax[1].set_xlabel(r"black-box $\lambda_{ca}$ (token-space, r=2)")
ax[1].set_ylabel(r"white-box $\lambda_{top}$ (activation-space)")
ax[1].set_title("Cross-model pairing is family-dependent\n(pooled p=0.025 is a Simpson artifact)")
ax[1].legend(fontsize=8, loc="upper left")

# panel 3: lambda_ca(r) model-invariance
Rs = rad[[k for k in rad if isinstance(rad[k], dict) and "R_grid" in rad[k]][0]]["R_grid"]
for t in rad:
    if not isinstance(rad[t], dict) or "black_lambda_ca" not in rad[t]:
        continue
    c = "#2c6fbb" if t.startswith("pythia") else "#d1701f"
    y = [rad[t]["black_lambda_ca"][str(r)] for r in Rs]
    ax[2].plot(Rs, y, "o-", color=c, alpha=0.7, lw=1.2, ms=4)
ax[2].axhline(0, color="gray", lw=0.8, ls=":")
ax[2].set_xscale("log", base=2); ax[2].set_xticks(Rs); ax[2].set_xticklabels(Rs)
ax[2].set_xlabel("radius r (fixed T=0.7)")
ax[2].set_ylabel(r"black-box $\lambda_{ca}(r)$")
ax[2].plot([], [], color="#2c6fbb", label="Pythia"); ax[2].plot([], [], color="#d1701f", label="GPT-2")
ax[2].legend(fontsize=8, loc="lower right")
ax[2].set_title("$\\lambda_{ca}(r)$ is model-invariant (all 10 curves overlap)\n"
                "-> pure kinematics, no model signal")

fig.tight_layout()
out = str(ROOT / "fig" / "crosslevel.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
