"""Confirmatory test of F103's single positive: does L23 still compensate? (#103)

WHY THIS EXISTS. F103 reported COMPENSATION at layer 23 -- delta = +0.07722, family-wise p = 0.0080
over 14 layers -- and recorded three reasons to hold it loosely: the verdict turned on a seed-floor
bug found AFTER the run returned a null, the registered criterion was one-sided while the bulk of
the deltas ran negative (L22 z = -5.30, L18 -5.13, L20 -4.13 against a best positive of +3.25), and
L23's own estimate shrank 16% between n=8 and n=20. A provisional positive reached that way is
confirmed by an independent test or it is not a result.

WHAT MAKES THIS INDEPENDENT RATHER THAN A RE-READ.

  FRESH SEEDS. 41-60, disjoint from the 21-40 that produced F103. No cell is reused.

  NOTHING IS BORROWED. F103 computed contrib_intact from F80's recorded lambda(attn_L23) -- a
  different run, gated on a calibration rung. Here all FOUR arms (none, attn_early, attn_L23,
  attn_early+attn_L23) are measured in this run, so the difference of differences is internal and
  needs no rung to license it.

  ONE PRE-SPECIFIED COMPARISON, so no multiple-comparisons penalty. F103 paid family-wise p = 0.0080
  because L23 was the maximum of fourteen. It is now NAMED IN ADVANCE, which is the entire
  statistical advantage of confirming a specific layer rather than re-running the sweep.

  DIRECTION FIXED IN ADVANCE. F103 specifies delta > 0. The test is one-sided, and that is
  legitimate here precisely because the direction was published before this run existed.

THE SECOND CHECKPOINT IS A GENERALITY PROBE, NOT PART OF THE CONFIRMATION. step143000 is the direct
replication and carries the verdict alone. step8000 -- also post-crossing, also on the plateau, an
order of magnitude earlier in training -- asks whether the effect is a property of one model state.
Folding it into the primary would reintroduce the multiple-comparisons problem this design exists
to escape, so it is reported separately and cannot rescue a failed replication.

WHAT A CONFIRMATION WOULD AND WOULD NOT BUY. It would make L23 a real measurement of a named
mechanism -- the first positive the explanandum programme has produced. It would NOT resolve F103's
deeper problem: that de-recruitment dominates the layer population while L23 runs against it. A
confirmed L23 is one layer behaving unlike its neighbours, which is a finding about L23, not a
vindication of self-repair as the account of F80's non-additivity.

Usage:
    .venv/bin/python experiments/confirm_L23.py --smoke   # 4 arms x 1 seed, one checkpoint
    .venv/bin/python experiments/confirm_L23.py           # the run (resumable)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src"), str(_ROOT / "gatecheck" / "src")]

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import stats

from provenance import stamp, rel
from dev_transition_phase3 import measure, BASE, T
from lyapunov import lambda_of, run_ignited
from gatecheck import NOT_DECIDABLE, carries_verdict, directional, noise_gate
from ablate_compensators import (ablating_many, held_out_loss_many, R, N, B, MIN_DETECTABLE,
                                 IGN_TOL)

LAYER = 23
PRIMARY_STEP = "step143000"          # F103's checkpoint: the direct replication
PROBE_STEP = "step8000"              # post-crossing plateau, ~18x earlier; generality only
STEPS = [PRIMARY_STEP, PROBE_STEP]
SEEDS = list(range(41, 61))          # FRESH: disjoint from F103's 21-40
ARMS = ["none", "attn_early", f"attn_L{LAYER:02d}", f"attn_early+attn_L{LAYER:02d}"]
OUT = str(_ROOT / "results" / "confirm_L23.json")

F103_DELTA = 0.07722                 # the estimate being confirmed


def cells(res, step, arm):
    return [v for v in res["runs"].values()
            if v["step"] == step and v["arm"] == arm and run_ignited(v)]


def centre(res, step, arm):
    c = cells(res, step, arm)
    return (float(np.median(lambda_of(c))) if c else None), len(c)


def se(res, step, arm):
    c = cells(res, step, arm)
    if len(c) < 2:
        return None
    sd = float(np.std(lambda_of(c)))
    return sd / np.sqrt(len(c)) if np.isfinite(sd) and sd > 0 else None


def ignition(res, step, arm):
    v = [r["ignition_prob"] for r in res["runs"].values()
         if r["step"] == step and r["arm"] == arm]
    return float(np.mean(v)) if v else 0.0


def one_step(res, step):
    """delta(L23) and its gates at one checkpoint. Returns a dict, decides nothing globally."""
    lam = {a: centre(res, step, a)[0] for a in ARMS}
    n = {a: centre(res, step, a)[1] for a in ARMS}
    single, compound = f"attn_L{LAYER:02d}", f"attn_early+attn_L{LAYER:02d}"
    if any(lam[a] is None for a in ARMS):
        return dict(step=step, decided=False, n=n,
                    reason="an arm produced no ignited estimate, so lambda is undefined (F42)")

    contrib_intact = lam["none"] - lam[single]
    contrib_early = lam["attn_early"] - lam[compound]
    delta = contrib_early - contrib_intact

    ses = [se(res, step, a) for a in ARMS]
    ses = [s for s in ses if s]
    # Difference of two differences over four independent arms: the standard errors add in
    # quadrature. Averaging them, as the sweep did, understates the uncertainty of THIS statistic.
    floor = float(np.sqrt(sum(s ** 2 for s in ses))) if len(ses) == 4 else None
    if floor is None:
        return dict(step=step, decided=False, n=n,
                    reason="an arm has too few ignited runs to carry a standard error")

    ign_ref, ign_comp = ignition(res, step, "attn_early"), ignition(res, step, compound)
    comparable = abs(ign_comp - ign_ref) <= IGN_TOL
    z = delta / floor
    p = float(stats.norm.sf(z))                      # one-sided: direction fixed by F103
    return dict(step=step, lam={a: round(v, 5) for a, v in lam.items()}, n=n,
                contrib_intact=round(contrib_intact, 5), contrib_early=round(contrib_early, 5),
                delta=round(delta, 5), floor=round(floor, 5), z=round(z, 4), p_one_sided=p,
                ignition_reference=round(ign_ref, 4), ignition_compound=round(ign_comp, 4),
                comparable=bool(comparable), decided=True)


def analyse(res):
    parts, gates = [], []
    per = {s: one_step(res, s) for s in STEPS}
    res["per_step"] = per
    pri = per[PRIMARY_STEP]

    if not pri.get("decided"):
        res["analysis"] = dict(decided=False)
        res["verdict"] = f"NOT DECIDABLE at {PRIMARY_STEP}: {pri['reason']}."
        return res["verdict"]

    parts.append(
        f"REPLICATION at {PRIMARY_STEP} on fresh seeds {SEEDS[0]}-{SEEDS[-1]}: delta(L{LAYER}) = "
        f"{pri['delta']:+.5f} against F103's {F103_DELTA:+.5f}, on a floor of {pri['floor']:.5f} "
        f"(quadrature over four arms), z = {pri['z']:+.3f}, one-sided p = {pri['p_one_sided']:.5f}. "
        f"Single pre-specified comparison -- no family-wise correction applies.")

    gates.append(noise_gate(MIN_DETECTABLE, pri["floor"]))
    verdict = carries_verdict(gates, value=pri["delta"])
    dirn = directional(pri["delta"], expect="increase", floor=0.0)

    if not pri["comparable"]:
        parts.append(
            f"COMPARABILITY: the compound arm ignites at {pri['ignition_compound']:.3f} against "
            f"the reference's {pri['ignition_reference']:.3f}, past the {IGN_TOL} tolerance, so "
            f"the two centres are taken over differently-selected replica subsets. NOT DECIDABLE.")
        decided = False
    elif verdict.status == NOT_DECIDABLE:
        parts.append(f"NOT DECIDABLE: underpowered -- {verdict.reason}.")
        decided = False
    elif not dirn.usable:
        parts.append(
            f"REFUTED: {dirn.reason} F103's positive does not replicate on fresh seeds, and the "
            f"provisional reading should be withdrawn.")
        decided = True
    elif pri["p_one_sided"] < 0.05:
        parts.append(
            f"CONFIRMED: L{LAYER} compensates on an independent sample, with every arm measured in "
            f"this run and nothing borrowed. Gates: {verdict.reason}")
        decided = True
    else:
        parts.append(
            f"NOT REPLICATED: the estimate is positive but does not clear its own floor "
            f"(p = {pri['p_one_sided']:.4f}). With power adequate for {MIN_DETECTABLE}, this is a "
            f"failure to replicate rather than an absence of power, and F103's positive should be "
            f"treated as unconfirmed.")
        decided = True

    probe = per[PROBE_STEP]
    parts.append(
        f"GENERALITY PROBE at {PROBE_STEP}, reported separately and unable to rescue the above: "
        + (f"delta = {probe['delta']:+.5f}, z = {probe['z']:+.3f}, p = {probe['p_one_sided']:.4f}, "
           f"comparable={probe['comparable']}." if probe.get("decided")
           else f"{probe['reason']}."))

    parts.append(
        f"BOUNDARY: a confirmation makes L{LAYER} a real measurement of a named mechanism in one "
        f"model. It does NOT resolve F103's population problem -- de-recruitment dominated the "
        f"other layers (L22 z = -5.30, L18 -5.13, L20 -4.13) while L{LAYER} ran against them -- so "
        f"a confirmed L{LAYER} is a finding about L{LAYER}, not a vindication of self-repair as "
        f"the account of F80's non-additivity.")

    res["analysis"] = dict(primary=pri, probe=probe, gates=[g.block() for g in gates],
                           directional=dirn.block(), decided=decided)
    res["verdict"] = " ".join(parts)
    return res["verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    steps = STEPS[:1] if a.smoke else STEPS
    seeds = SEEDS[:1] if a.smoke else SEEDS

    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}, "loss": {}}
    res["_preregistration"] = dict(
        issue=103, confirms="F103", layer=LAYER, base=BASE, steps=steps, r=R, N=N, B=B, T=T,
        seeds=seeds, arms=ARMS,
        primary=f"delta(L{LAYER}) > 0 at {PRIMARY_STEP} on FRESH seeds, as a SINGLE pre-specified "
                f"comparison. Direction fixed in advance by F103, so the test is one-sided",
        independence="seeds 41-60 are disjoint from F103's 21-40, and all four arms are measured "
                     "in this run -- nothing is borrowed from F80, so no calibration rung is "
                     "needed to license the comparison",
        floor="standard errors of the four arm centres added in QUADRATURE; averaging them, as the "
              "sweep did, understates the uncertainty of a difference of differences",
        refuted=f"delta <= 0 -- F103's positive does not replicate and should be withdrawn",
        not_replicated=f"delta > 0 but p >= 0.05 with power adequate for {MIN_DETECTABLE}: a "
                       f"failure to replicate rather than an absence of power",
        not_decidable=f"underpowered for {MIN_DETECTABLE}, or the compound arm ignites more than "
                      f"{IGN_TOL} from the reference (differently-selected subsets), or an arm "
                      f"fails to ignite (F42)",
        probe=f"{PROBE_STEP} is a generality probe reported SEPARATELY; it cannot rescue a failed "
              f"replication, and folding it into the primary would reintroduce the "
              f"multiple-comparisons problem this design exists to escape",
        boundary="a confirmation is a finding about one layer, not a vindication of self-repair as "
                 "the account of F80's non-additivity",
        resumable="keyed by (step, arm, seed)")

    for step in steps:
        for arm in ARMS:
            k = f"{step}|{arm}"
            if k not in res["loss"]:
                res["loss"][k] = round(held_out_loss_many(arm), 4)
                json.dump(res, open(OUT, "w"), indent=1)

    todo = [(st, ar, sd) for st in steps for ar in ARMS for sd in seeds
            if f"{st}|{ar}|s{sd}" not in res["runs"]]
    print(f"{len(res['runs'])} cached, {len(todo)} cells to run "
          f"(~{len(todo)*135/3600:.1f} h)\n", flush=True)

    for step, arm, seed in todo:
        t0 = time.time()
        with ablating_many(arm):
            lam, dn, md, ig = measure(step, N, B, seed, r=R)
        res["runs"][f"{step}|{arm}|s{seed}"] = dict(
            step=step, arm=arm, seed=seed, r=R, N=N, B=B, T=T,
            lambda_ca=lam, D_norm=dn, mean_damage=md, ignition_prob=ig,
            secs=round(time.time() - t0, 1))
        print(f"  {step:12s} {arm:22s} s={seed}  lambda={lam:+.4f}  ign={ig:.3f}  "
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
