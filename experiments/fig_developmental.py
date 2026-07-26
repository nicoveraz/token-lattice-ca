"""The developmental transition (F39) -- the paper's headline figure.

Three panels, chosen so the figure states the claim AND its two honest qualifications:

  (A) lambda_ca vs checkpoint, both lattice sizes. The claim. lambda_ca is the quantity that
      carries it, because W9 turned out favourably for it: 104% retention at N=96, and the
      plateau levels agree to within +-14% (95% CI on the difference, not a null p-value).
  (B) D_norm vs checkpoint, both sizes. The corroboration -- same shape, same direction,
      but the two curves are visibly offset, which is exactly the point: D_norm's absolute
      scale moves with N (plateau 0.569 vs 0.306, p=1.3e-08), so it is reported at a stated
      lattice size and never as a lattice-free property.
  (C) Seed spread across the transition. Before it, seeds disagree about the SIGN of lambda;
      after, none of 48 plateau runs is negative (sd collapses 3.7x / 3.1x). Not pre-registered, so it
      is drawn as an observation.

The plateau band is drawn from steps 2000/8000/143000 -- NOT from the step-1000 peak, whose
overshoot is +1.4% to +22.4% and survives BH-FDR in only 1 of 4 cells (a D_norm cell). Quoting the peak as the
level would inflate the N=48 D_norm effect; using step256 ALONE as the pre value inflates
lambda_ca by 1.7x. Both ends use the pre-registered sets.

Reads results/dev_transition_phase3.json and dev_transition_shape.json (writes neither);
writes fig/developmental.png.
"""
import pathlib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
raw = json.load(open(ROOT / "results" / "dev_transition_phase3.json"))
shape = json.load(open(ROOT / "results" / "dev_transition_shape.json"))

rows = raw if isinstance(raw, list) else raw.get("runs", raw)
if isinstance(rows, dict):
    rows = list(rows.values())
rows = [r for r in rows if isinstance(r, dict) and "lambda_ca" in r]

STEPS = [256, 512, 1000, 2000, 8000, 143000]
PLATEAU = {2000, 8000, 143000}
SIZES = [(48, "#2c6fbb", "o", "N=48"), (96, "#b0413e", "s", "N=96")]
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.5, "figure.dpi": 200})
fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.5))


def cell(N, step, m):
    return np.array([r[m] for r in rows if r["N"] == N and r["step"] == step])


def panel(a, metric, ylabel, title):
    for N, col, mk, lab in SIZES:
        mu = np.array([cell(N, s, metric).mean() for s in STEPS])
        sd = np.array([cell(N, s, metric).std(ddof=1) for s in STEPS])
        a.errorbar(STEPS, mu, yerr=sd, fmt=mk + "-", color=col, ms=5, lw=1.6,
                   capsize=3, label=lab, zorder=3)
        pl = np.mean([cell(N, s, metric).mean() for s in sorted(PLATEAU)])
        a.axhline(pl, color=col, lw=0.9, ls=":", alpha=0.8, zorder=1)
    a.set_xscale("log")
    a.axvspan(200, 700, color="0.85", alpha=0.55, lw=0, zorder=0)   # the pre-transition set
    a.set_xlabel("training step (Pythia-410m checkpoint)")
    a.set_ylabel(ylabel); a.set_title(title, fontsize=9)
    a.legend(fontsize=7.5, loc="lower right", frameon=False)


h = shape["headline"]
panel(ax[0], "lambda_ca", r"$\lambda_{\mathrm{ca}}$",
      f"(A) the claim: 0/48 plateau runs negative (min +0.107)\\n"
      f"$d$={h['N48_lambda_ca']['cohens_d']} / {h['N96_lambda_ca']['cohens_d']} vs the pre-registered pre set\\n"
      f"plateau levels agree within $\\pm$13.6\\% (95% CI [-0.0229, +0.0223])")
ax[0].axhline(0, color="k", lw=0.8, ls="-", alpha=0.5, zorder=2)

panel(ax[1], "D_norm", r"$D_{\mathrm{norm}}$",
      f"(B) corroboration, at a stated lattice size:\n"
      f"same shape, but the LEVEL moves with $N$\n"
      f"(plateau 0.569 vs 0.306, $p$=1.3e-08)")

# (C) seed spread -- the collapse, drawn as raw per-seed points plus the sd
a = ax[2]
for N, col, mk, lab in SIZES:
    for i, s in enumerate(STEPS):
        v = cell(N, s, "lambda_ca")
        jitter = (np.random.default_rng(s + N).random(len(v)) - 0.5) * 0.13
        a.scatter(np.full(len(v), i) + (0.16 if N == 96 else -0.16) + jitter, v,
                  s=9, color=col, alpha=0.65, lw=0, zorder=3,
                  label=lab if i == 0 else None)
a.axhline(0, color="k", lw=0.8, alpha=0.5)
a.axvspan(-0.5, 1.5, color="0.85", alpha=0.55, lw=0, zorder=0)
a.set_xticks(range(len(STEPS))); a.set_xticklabels([str(s) for s in STEPS], fontsize=7.5)
a.set_xlabel("training step"); a.set_ylabel(r"$\lambda_{\mathrm{ca}}$ (per seed)")
v48, v96 = shape["variance"]["N48"], shape["variance"]["N96"]
a.set_title(f"(C) seeds stop disagreeing about the SIGN\n"
            f"sd collapses {v48['ratio']}× (N=48) / {v96['ratio']}× (N=96)\n"
            f"observation, not pre-registered", fontsize=9)
a.legend(fontsize=7.5, loc="lower right", frameon=False)

fig.tight_layout()
fp = str(ROOT / "fig" / "developmental.png")
fig.savefig(fp, bbox_inches="tight"); print("wrote", fp)
