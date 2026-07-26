"""The validation ladder: the three WEIGHT-BEARING rungs, numbered as the text numbers them.

REGENERATED TWICE. Phase 4.2 removed two retracted claims (see the history note at the bottom).
This pass fixes three defects that survived that one and shipped in a built PDF:

  * ILLEGIBLE ON THE PAGE. The figure was authored 14.5in wide and included at 0.80\\linewidth
    = 4.4in, so every label was reproduced at 30% of nominal -- 9.5pt text arriving as ~3pt.
    It looked fine as a PNG and was unreadable in the paper. Figures are now authored at their
    DISPLAY size, so a 6.5pt label is 6.5pt on the page.

  * COLOUR-ONLY ENCODING. The ECA panel's x-label read "green ordered / amber edge / red
    chaotic" and the census legend separated three series by hue alone. Both are unreadable in
    greyscale, in photocopy, and under red-green colour-vision deficiency -- which is a large
    fraction of any reviewer pool. Groups are now separated by fill (white / hatched / solid)
    and by position, with the names written on the axis rather than in a colour key.

  * PANEL NUMBERS CONTRADICTED THE TEXT. The figure had six panels (1)-(6) while the caption
    and Section 3 speak of five RUNGS, so "only (3)-(5) are weight-bearing" pointed at panels
    3,4,5 -- excluding the census, which the text calls weight-bearing. Panels are now numbered
    by rung and only the weight-bearing rungs are drawn. Not plotting the smooth-limit unit
    tests is also the honest choice: the text says they are "not evidence about damage
    spreading", and giving them equal visual weight said the opposite.

Rungs drawn (paper numbering):
  (3) elementary CA, ignition probability, ordered vs rest (F36)
  (4) Domany-Kinzel: the bit-exact damage identity, with its off-line control (F38)

Two panels rather than three, so each is large enough to read. Rung (5), the synthetic-Markov
census, is reported in the text: its three numbers say everything the bar chart did, and a third
panel bought nothing at this width.

Rungs (1) logistic map and (2) coupled-map lattice are smooth-limit arithmetic unit tests,
described in the text and deliberately not plotted. The DK critical points (~1% accuracy, and
they cannot separate the two disputed published values) are likewise reported in the text only.

History, kept because a figure that contradicts its own caption is the last way a retraction
can still reach a reviewer: the pre-Phase-4.2 version titled a panel "ECA classes: ord -0.32 <
edge +0.19 < chaos +0.26" -- the three-class ordering demoted by F33/F34 and re-tested to
p=0.470 by F36 -- and drew Rule 128 at lambda = -0.92, which is not a measurement but the
estimator's DEAD_DAMAGE_FLOOR (= -0.4 ln 10), the constant a log-linear fit returns when damage
dies immediately (F40).

Reads only results/*.json; writes fig/validation_ladder.png.
"""
import pathlib, json, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figstyle import use_classic_r, finish, RULE
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
ign = json.load(open(R / "eca_ordered_vs_rest.json"))
dk = json.load(open(R / "dk_calib.json"))
cen = json.load(open(R / "calib_census.json"))

use_classic_r(base=7.0)
fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.40))

# ---------------------------------------------------- (3) ECA: ignition probability, by class
# Fill encodes the class: white = ordered, hatched = edge, solid grey = chaotic. The surviving
# claim is ordered-vs-REST (the three-class ordering does not survive, F36), so the two "rest"
# groups are deliberately drawn closer in weight to each other than to ordered.
a = ax[0]
STYLE = {"ordered": dict(fc="white", hatch=None),
         "edge":    dict(fc="0.75",  hatch="///"),
         "chaotic": dict(fc="0.45",  hatch=None)}
xs, x, spans = [], 0, {}
for g in ("ordered", "edge", "chaotic"):
    rules = sorted(ign["groups"][g]["rules"].items(), key=lambda kv: int(kv[0]))
    seg0 = x
    for rn, p in rules:
        a.bar(x, p, width=0.82, edgecolor="black", linewidth=0.5, zorder=3, **STYLE[g])
        xs.append((x, rn)); x += 1
    spans[g] = (seg0, x - 1); x += 1.1
for g, (lo, hi) in spans.items():                       # group mean + bootstrap CI
    m = ign["groups"][g]["mean"]; c0, c1 = ign["groups"][g]["ci95"]
    a.hlines(m, lo - 0.5, hi + 0.5, color="black", lw=1.1, zorder=4)
    a.fill_between([lo - 0.5, hi + 0.5], c0, c1, color="black", alpha=0.12, lw=0, zorder=1)
# Tick per GROUP, not per rule. Nineteen rotated rule numbers collided with the group names
# once the figure was shortened, and no claim here is about an individual rule -- the unit of
# analysis is the rule only for the statistics, and the reader needs the three classes.
a.set_xticks([(lo + hi) / 2 for lo, hi in spans.values()])
a.set_xticklabels(list(spans), fontsize=6.5)
a.tick_params(axis="x", length=0)
a.set_ylim(0, 1.05)
a.set_ylabel("ignition probability")
a.set_xlabel("ECA class (19 rules)")
t = ign["tests"]
a.set_title(f"(3) ECA: ordered vs rest\n$p$={t['ordered_lt_rest_p']:.3f}, "
            f"$d$={t['cohens_d_ordered_vs_rest']}; 3-class $p$={t['edge_lt_chaotic_p']:.2f}")

# ------------------------------------------------ (4) DK: the bit-exact identity + its control
a = ax[1]
pa = dk["part_a_exact_identity"]
labels = [f"{k}" for k in pa["by_p1"]] + ["ctrl"]
vals = [v["mismatching_cells"] for v in pa["by_p1"].values()] + [pa["control_offline_mismatch"]]
for i, v in enumerate(vals):
    ctrl = i == len(vals) - 1
    a.bar(i, v, width=0.7, fc="0.45" if ctrl else "white", edgecolor="black",
          linewidth=0.6, zorder=3)
    a.text(i, v + max(vals) * 0.04, str(v), ha="center", fontsize=6.4,
           fontweight="bold" if v == 0 else "normal")
a.set_xticks(range(len(labels)))
a.set_xticklabels(labels, fontsize=6.0, rotation=45, ha="right")
a.set_ylim(0, max(vals) * 1.30)
a.set_xlabel("$p_1$  (ctrl = off-line $(0.6,0.5)$)")
a.set_ylabel("mismatching cells")
a.set_title(f"(4) Domany\u2013Kinzel: exact\n{pa['N']}$\\times${pa['steps']}, zero mismatches")

finish(fig, str(ROOT / "fig" / "validation_ladder.png"))
