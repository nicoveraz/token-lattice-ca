"""Does F104's revival replicate at a second checkpoint? (paper 2, row 4 of the discriminator table)

WHAT F104 FOUND. With the early attention block ablated, the lattice is nearly frozen -- ignition
0.181 +/- 0.032 at step143000. Adding ONE further attention ablation revives it: L8 0.516,
L21 0.438, L20 0.400, L18 0.391, L22 0.369, all clearing Bonferroni at alpha/16, four of the five
inside `attn_late`. The direction agrees with F79, where ablating attn_late alone RAISES lambda_ca
(0.3566 -> 0.3960, the only group arm with that sign). Removing more of a network making its
dynamics livelier is not what any monotone account of ablation predicts.

WHY IT NEEDS REPLICATING RATHER THAN EXTENDING. plan_paper2 section 7 makes this measurement row 4
of the discriminator table -- the row showing the instrument responds when the MODEL is varied and
the construction is held fixed. Rows 3 and 4 are what stop the paper reading as "the instrument
measures nothing". Row 4 is currently one model at one checkpoint, which is the weakest evidence in
the table it is load-bearing for.

THE DESIGN IS LAYER-SPECIFICITY, NOT JUST REVIVAL. A replication that only re-measured the five
revivers could not distinguish "these layers revive" from "any further ablation revives", and the
second reading would make F104 a statement about ablation count rather than about late attention.
So two arms that did NOT revive at step143000 are carried as controls: L11 (0.169, t = -0.30) and
L23 (0.203, t = +0.55), both indistinguishable from the reference there. The registered prediction
is a PATTERN -- revivers up, controls flat -- and a replication in which everything rises refutes
the layer-specific reading as surely as one in which nothing does.

THE ANTI-VACUITY GATE COMES FIRST. Revival is only defined against a frozen reference. If
`attn_early` does not freeze the lattice at this checkpoint -- if its ignition is not far below the
unablated model's -- then there is nothing to revive from and the comparison is undefined, whatever
the arms do. That is checked before any arm is read, and it is the one outcome that makes the run
uninformative rather than negative.

SEEDS ARE PAIRED, NOT FRESH. The same 20 seeds as F104. A seed indexes the lattice initialisation
and visit order, not the model, so reusing them at a different checkpoint gives matched starting
conditions with the model varied -- which is exactly the manipulation row 4 claims to test. Fresh
seeds would add noise without adding independence, since the checkpoint already supplies it.

Usage:
    .venv/bin/python experiments/revival_replication.py --smoke
    .venv/bin/python experiments/revival_replication.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src"), str(_ROOT / "gatecheck" / "src")]

import argparse
import json
import os
import time

import numpy as np
from scipy import stats

from provenance import stamp, rel
from dev_transition_phase3 import measure, BASE, T, bh_fdr
from gatecheck import NOT_DECIDABLE, carries_verdict, distinct_units, dynamic_range
from ablate_compensators import ablating_many, all_seeds, R, N, B, EARLY

STEP = "step8000"                    # post-crossing plateau, ~18x earlier than F104's step143000
REVIVERS = [8, 18, 20, 21, 22]       # cleared Bonferroni at step143000
CONTROLS = [11, 23]                  # indistinguishable from the reference at step143000
SEEDS = all_seeds()                  # paired with F104: same lattice inits, different checkpoint
ARMS = ["none", EARLY] + [f"{EARLY}+attn_L{L:02d}" for L in REVIVERS + CONTROLS]
OUT = str(_ROOT / "results" / "revival_replication.json")

# step143000 ignition, for the record and for the pattern comparison. Not thresholds.
F104 = {"none": 0.977, EARLY: 0.181,
        8: 0.516, 18: 0.391, 20: 0.400, 21: 0.438, 22: 0.369, 11: 0.169, 23: 0.203}
FREEZE_RATIO = 0.5    # the reference must ignite at most half as often as the unablated model


def ign(res, arm):
    """Per-run ignition fractions for an arm. The RUN is the unit -- replicas within one run share
    a lattice and a visit order, so pooling them would shrink the error bar by a factor it has not
    earned (F57, which cost a retracted verdict)."""
    return np.array([r["ignition_prob"] for r in res["runs"].values() if r["arm"] == arm])


def analyse(res):
    parts, gates = [], []
    unab, ref = ign(res, "none"), ign(res, EARLY)
    if len(unab) < 2 or len(ref) < 2:
        res["analysis"] = dict(decided=False)
        res["verdict"] = "NOT DECIDABLE: the reference arms have too few runs to compare."
        return res["verdict"]

    # --- anti-vacuity: is there anything to revive FROM? -------------------------------------
    frozen = ref.mean() <= FREEZE_RATIO * unab.mean()
    parts.append(
        f"FREEZE CHECK at {STEP}: unablated ignites at {unab.mean():.3f}, `{EARLY}` at "
        f"{ref.mean():.3f} "
        f"({ref.mean()/unab.mean() if unab.mean() else float('nan'):.2f}x) -> "
        f"{'frozen, revival is defined' if frozen else 'NOT FROZEN'}.")
    if not frozen:
        res["analysis"] = dict(decided=False, unablated=round(float(unab.mean()), 4),
                               reference=round(float(ref.mean()), 4))
        res["verdict"] = (" ".join(parts) + f" NOT DECIDABLE: the early block does not freeze the "
                          f"lattice at {STEP}, so there is nothing to revive from and the arms "
                          f"cannot be read either way. This is an uninformative run, not a "
                          f"negative one -- F104's effect is undefined outside a frozen reference.")
        return res["verdict"]

    # --- the pattern: revivers up, controls flat ---------------------------------------------
    rows = []
    for L in REVIVERS + CONTROLS:
        v = ign(res, f"{EARLY}+attn_L{L:02d}")
        if len(v) < 2:
            continue
        t, p = stats.ttest_ind(v, ref, equal_var=False)
        rows.append(dict(layer=L, kind="reviver" if L in REVIVERS else "control",
                         ignition=round(float(v.mean()), 4),
                         se=round(float(v.std(ddof=1) / np.sqrt(len(v))), 4),
                         t=round(float(t), 3), p=float(p), step143000=F104[L]))
    if len(rows) < len(REVIVERS + CONTROLS):
        parts.append(f"NOTE: {len(REVIVERS + CONTROLS) - len(rows)} arm(s) produced too few runs "
                     f"to test and are absent from the pattern below.")

    q = bh_fdr([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q_bh"] = round(float(qq), 5)
        r["up"] = bool(r["t"] > 0 and qq < 0.05)

    gates.append(distinct_units([r["layer"] for r in rows], minimum=5, name="ablation arm"))
    gates.append(dynamic_range([r["ignition"] for r in rows] + [float(ref.mean())],
                               floor=float(ref.std(ddof=1) / np.sqrt(len(ref))),
                               name="ignition across the compared arms"))
    verdict = carries_verdict(gates, value=len(rows))

    up_rev = [r["layer"] for r in rows if r["kind"] == "reviver" and r["up"]]
    up_ctl = [r["layer"] for r in rows if r["kind"] == "control" and r["up"]]
    parts.append(
        "PATTERN: " + "; ".join(
            f"L{r['layer']}({r['kind'][:3]}) {r['ignition']:.3f} vs {r['step143000']:.3f} at "
            f"step143000, t={r['t']:+.2f} q={r['q_bh']:.4f}" for r in rows) + ".")

    if verdict.status == NOT_DECIDABLE:
        parts.append(f"NOT DECIDABLE: {verdict.reason}.")
        decided = False
    elif up_ctl:
        parts.append(
            f"REFUTES THE LAYER-SPECIFIC READING: control layer(s) {up_ctl} revive too, so the "
            f"effect is not specific to the layers F104 named. F104's revival would then be a "
            f"statement about adding an ablation at all, not about late attention, and row 4 of "
            f"the discriminator table cannot rest on it as written.")
        decided = True
    elif len(up_rev) >= 3:
        parts.append(
            f"REPLICATES: {len(up_rev)} of {len(REVIVERS)} revivers rise at q<0.05 ({up_rev}) "
            f"while both controls stay flat. The effect is layer-specific and survives a change of "
            f"checkpoint, so row 4 of plan_paper2's discriminator table is replicated rather than "
            f"measured once. Gates: {verdict.reason}")
        decided = True
    elif up_rev:
        parts.append(
            f"PARTIAL: only {up_rev} rise where five did at step143000, with controls flat. The "
            f"direction survives but the specific layer set does not, which is weaker than F104 "
            f"claims and should be reported as such.")
        decided = True
    else:
        parts.append(
            f"DOES NOT REPLICATE: no reviver rises at q<0.05 against a reference that IS frozen "
            f"({ref.mean():.3f} vs {unab.mean():.3f} unablated), so the freeze gate cannot excuse "
            f"it. F104 is checkpoint-specific and row 4 must be rewritten around a single "
            f"measurement or dropped.")
        decided = True

    parts.append(
        f"BOUNDARY: two checkpoints of one model, one radius, greedy. Seeds are PAIRED with F104 "
        f"(same lattice inits, different checkpoint), so this varies the model and holds the "
        f"construction fixed -- which is the manipulation row 4 claims, and nothing more.")

    res["analysis"] = dict(step=STEP, unablated=round(float(unab.mean()), 4),
                           reference=round(float(ref.mean()), 4), frozen=bool(frozen),
                           rows=rows, revivers_up=up_rev, controls_up=up_ctl,
                           gates=[g.block() for g in gates], decided=decided)
    res["verdict"] = " ".join(parts)
    return res["verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seeds = SEEDS[:1] if a.smoke else SEEDS
    arms = ARMS[:3] if a.smoke else ARMS

    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        replicates="F104", step=STEP, compared_against="step143000", base=BASE,
        r=R, N=N, B=B, T=T, seeds=seeds, revivers=REVIVERS, controls=CONTROLS,
        primary="the PATTERN: revivers rise above the frozen reference at q<0.05 (BH) while "
                "controls stay flat. Run as the unit of analysis (F57)",
        anti_vacuity=f"`{EARLY}` must ignite at most {FREEZE_RATIO}x the unablated model, or there "
                     f"is nothing to revive from and the run is UNINFORMATIVE rather than negative",
        refuted="a control layer rises too -> the effect is about adding an ablation, not about "
                "late attention, and row 4 of plan_paper2 cannot rest on it as written",
        does_not_replicate="no reviver rises against a reference that IS frozen -> F104 is "
                           "checkpoint-specific",
        seeds_paired="the same seeds as F104: a seed indexes the lattice init and visit order, not "
                     "the model, so pairing holds the construction fixed while the model varies",
        boundary="two checkpoints of one model; this is row 4 of the discriminator table, not a "
                 "general claim about ablation",
        resumable="keyed by (arm, seed)")

    todo = [(ar, sd) for ar in arms for sd in seeds if f"{ar}|s{sd}" not in res["runs"]]
    print(f"{len(res['runs'])} cached, {len(todo)} cells (~{len(todo)*135/3600:.1f} h)\n", flush=True)
    for arm, seed in todo:
        t0 = time.time()
        with ablating_many(arm):
            lam, dn, md, ig = measure(STEP, N, B, seed, r=R)
        res["runs"][f"{arm}|s{seed}"] = dict(
            arm=arm, seed=seed, step=STEP, r=R, N=N, B=B, T=T,
            lambda_ca=lam, D_norm=dn, mean_damage=md, ignition_prob=ig,
            secs=round(time.time() - t0, 1))
        print(f"  {arm:22s} s={seed}  ign={ig:.3f}  lambda={lam:+.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    if a.smoke:
        print("\nSMOKE: plumbing only, no verdict.")
        json.dump(res, open(OUT, "w"), indent=1)
        return 0

    print("\n  -> " + analyse(res))
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
