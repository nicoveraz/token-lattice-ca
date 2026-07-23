"""Figure for issue #4: cross-level (black-box token-space vs white-box activation-space).
Three honest panels: (1) type-MISMATCHED pairing (white lambda_top vs black D_norm) = null;
(2) type-MATCHED pairing (white lambda_top vs black lambda_ca) = suggestive positive but
underpowered; (3) the clean white-box scaling law (lambda_top vs size).
"""
import sys, pathlib, json
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / "src")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlm_lib import RESDIR

d = json.load(open(f"{RESDIR}/crosslevel.json"))
order = ["pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"]
size = np.array([d[t]["size_M"] for t in order], float)
wtop = np.array([d[t]["white"]["lambda_top"] for t in order])
wtop_se = np.array([d[t]["white"]["lambda_top_se"] for t in order])
dnorm = np.array([d[t]["black"]["D_norm"] for t in order])
dnorm_se = np.array([d[t]["black"]["D_norm_se"] for t in order])
lca = np.array([d[t]["black_lyap"]["lambda_ca_r2"] for t in order])
lab = [t.replace("pythia-", "") for t in order]

cm = json.load(open(f"{RESDIR}/crosslevel.json"))["_correlation"]
cl = json.load(open(f"{RESDIR}/crosslevel.json"))["_correlation_lyap"]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "figure.dpi": 200})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))

def annotate(a, x, y):
    for xi, yi, li in zip(x, y, lab):
        a.annotate(li, (xi, yi), fontsize=8, xytext=(4, 3), textcoords="offset points")

# panel 1: MISMATCHED -- null
c = cm["pearson_white_vs_black"]
ax[0].errorbar(dnorm, wtop, xerr=dnorm_se, yerr=wtop_se, fmt="o", color="#b0413e", capsize=2)
annotate(ax[0], dnorm, wtop)
ax[0].set_xlabel("black-box  $D_{norm}$  (asymptotic persistence)")
ax[0].set_ylabel(r"white-box  $\lambda_{top}$  (activation-space)")
ax[0].set_title(f"Mismatched types: NULL\nPearson r={c['r']}, p={c['p']} (n=6)")

# panel 2: MATCHED -- suggestive
c = cl["white_top_vs_lambda_ca_r2"]
ax[1].errorbar(lca, wtop, yerr=wtop_se, fmt="o", color="#2c6fbb", capsize=2)
annotate(ax[1], lca, wtop)
z = np.polyfit(lca, wtop, 1); xs = np.linspace(lca.min(), lca.max(), 20)
ax[1].plot(xs, np.polyval(z, xs), "--", color="#2c6fbb", lw=1, alpha=0.6)
ax[1].set_xlabel(r"black-box  $\lambda_{ca}$ (token-space Lyapunov, r=2)")
ax[1].set_ylabel(r"white-box  $\lambda_{top}$  (activation-space)")
ax[1].set_title(f"Matched types (Lyapunov vs Lyapunov)\nPearson r={c['pearson_r']}, p={c['pearson_p']} "
                f"(n=6, suggestive)")

# panel 3: white-box scaling law
c = cm["white_lambda_top_vs_size"]
ax[2].errorbar(size, wtop, yerr=wtop_se, fmt="o-", color="#3a7d44", capsize=2)
annotate(ax[2], size, wtop)
ax[2].set_xscale("log")
ax[2].axhline(0, color="gray", lw=0.8, ls=":")
ax[2].set_xlabel("model size (M params, log)")
ax[2].set_ylabel(r"white-box  $\lambda_{top}$")
ax[2].set_title(f"White-box scaling law\nSpearman $\\rho$={c['rho']}, p={c['p']}")

fig.tight_layout()
out = str(pathlib.Path(__file__).resolve().parents[1] / "fig" / "crosslevel.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
