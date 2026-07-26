"""The developmental transition (F39) -- the paper's headline figure.

Three panels, chosen so the figure states the claim AND its two honest qualifications:

  (A) lambda_ca vs checkpoint, both lattice sizes. The claim, and the quantity that carries it:
      lambda_ca is intensive across a 4x range of N (F45), so the two curves lie on top of each
      other.
  (B) D_norm vs checkpoint, both sizes. The corroboration -- same shape, same direction, but the
      two curves are visibly OFFSET, which is exactly the point: D_norm's absolute scale moves
      with N, so it is reported at a stated lattice size and never as a lattice-free property.
  (C) Seed spread across the transition. Before it, seeds disagree about the SIGN of lambda;
      after, none of 48 plateau runs is negative. Not pre-registered, so it is drawn as an
      observation.

The plateau band is drawn from steps 2000/8000/143000 -- NOT from the step-1000 peak. Quoting the
peak as the level would inflate the N=48 D_norm effect; using step256 ALONE as the pre value
inflates lambda_ca by 1.7x. Both ends use the pre-registered sets.

TWO DEFECTS THIS REWRITE FIXES, both of which shipped in a built PDF:

  * Panel A's title used a doubled backslash-n inside a non-raw f-string, so the escape reached
    matplotlib as a LITERAL backslash-n rather than a line break. The title rendered as one long
    line that overran its axes and printed on top of panel B's title. The headline figure was
    unreadable across the middle, and nobody caught it because the figure was only ever checked
    as a thumbnail.
  * That same title hardcoded "+-13.6%" and "95% CI [-0.0229, +0.0223]" -- the two-size
    equivalence bound, which the third lattice size superseded and which has since been cut from
    paper.tex. The figure would have been the last surviving site of a retired number, which is
    exactly the drift the paper-number manifest exists to prevent.

All titles are now short. Statistics belong in the caption, where they are set at full size and
where the manifest already checks them.

Monochrome by construction -- series are separated by marker and dash, never by hue -- so the
figure survives greyscale printing and every colour-vision deficiency. See figstyle.py.

Reads results/dev_transition_phase3.json and dev_transition_shape.json (writes neither);
writes fig/developmental.png.
"""
import pathlib, json, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figstyle import use_classic_r, series, finish, BAND, RULE
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
SIZES = [48, 96]

use_classic_r(base=7.5)
# Authored at the size it is DISPLAYED: width=\linewidth is 5.5in in the NeurIPS geometry, so a
# 7pt label here is 7pt on the page. The previous version was authored 13.2in wide and scaled to
# 4.4in, which shrank every label to a third of nominal and is why its text was illegible.
fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.44))


def cell(N, step, m):
    return np.array([r[m] for r in rows if r["N"] == N and r["step"] == step])


def panel(a, metric, ylabel, title, legend_loc="lower right"):
    for i, N in enumerate(SIZES):
        mu = np.array([cell(N, s, metric).mean() for s in STEPS])
        sd = np.array([cell(N, s, metric).std(ddof=1) for s in STEPS])
        a.errorbar(STEPS, mu, yerr=sd, capsize=1.5, elinewidth=0.6, zorder=3,
                   **series(i, f"$N$={N}"))
        pl = np.mean([cell(N, s, metric).mean() for s in sorted(PLATEAU)])
        a.axhline(pl, color=RULE, lw=0.5, ls=(0, (1, 2)), zorder=1)
    a.set_xscale("log")
    a.axvspan(200, 700, color=BAND, lw=0, zorder=0)          # the pre-registered pre set
    a.set_xlabel("training step")
    a.set_ylabel(ylabel)
    a.set_title(title)
    a.legend(loc=legend_loc)


panel(ax[0], "lambda_ca", r"$\lambda_{\mathrm{ca}}$", r"(A) $\lambda_{\mathrm{ca}}$: sizes agree")
ax[0].axhline(0, color=RULE, lw=0.7, zorder=2)

# (B) seed spread -- raw per-seed points, so the collapse is drawn rather than summarised
a = ax[1]
for i, N in enumerate(SIZES):
    st = series(i, f"$N$={N}")
    for j, s in enumerate(STEPS):
        v = cell(N, s, "lambda_ca")
        jit = (np.random.default_rng(s + N).random(len(v)) - 0.5) * 0.16
        a.plot(np.full(len(v), j) + (0.17 if N == 96 else -0.17) + jit, v,
               ls="none", marker=st["marker"], mfc=st["mfc"], color="black",
               markeredgecolor="black", markeredgewidth=0.5, ms=2.6, zorder=3,
               label=st["label"] if j == 0 else None)
a.axhline(0, color=RULE, lw=0.7)
a.axvspan(-0.5, 1.5, color=BAND, lw=0, zorder=0)
a.set_xticks(range(len(STEPS)))
a.set_xticklabels([str(s) for s in STEPS], rotation=45, ha="right")
a.set_xlabel("training step")
a.set_ylabel(r"$\lambda_{\mathrm{ca}}$ per seed")
a.set_title("(B) seeds agree on the sign")
a.legend(loc="lower right")

finish(fig, str(ROOT / "fig" / "developmental.png"))
