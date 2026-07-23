"""Figure for the cross-level story (honest negative + calibration). Three panels:
(1) the instrument is criticality-calibrated on classical CA rules (ordered<edge<chaotic);
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
eca = json.load(open(f"{ROOT}/results/eca_calib.json"))

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "figure.dpi": 200})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

# panel 1: ECA calibration
grp = [("ordered", [128, 232, 4], "#3a7d44"), ("edge", [110, 54], "#c78a1e"),
       ("chaotic", [30, 150, 22, 90], "#b0413e")]
x = 0; ticks = []; tlab = []
for gname, rules, c in grp:
    for rn in rules:
        v = eca[str(rn)]["lambda_ca"]
        ax[0].bar(x, v, color=c, width=0.8)
        ticks.append(x); tlab.append(str(rn)); x += 1
    x += 0.6
ax[0].axhline(0, color="gray", lw=0.8)
ax[0].set_xticks(ticks); ax[0].set_xticklabels(tlab, fontsize=8)
ax[0].set_ylabel(r"$\lambda_{ca}$ (token-space Lyapunov)")
ax[0].set_xlabel("ECA rule (green=ordered, amber=edge, red=chaotic)")
gm = eca["_group_means"]
ax[0].set_title(f"Instrument calibrated on known rules\nordered {gm['ordered']:+.2f} < "
                f"edge {gm['edge']:+.2f} < chaotic {gm['chaotic']:+.2f}")

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
