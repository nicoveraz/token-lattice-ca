"""Is F104 revival, or regression to a common level? The slope test across checkpoints.

WHAT F104 CLAIMED. With the early attention block ablated, adding ONE further attention ablation
raised ignition from 0.181 to 0.516 (L8) and 0.369 (L22) at step143000 -- five layers clearing
Bonferroni. It was written up as "removing MORE of the network makes damage spread FURTHER", which
no monotone account of ablation predicts.

WHAT THREATENS IT. The replication at step8000 found the reference is not frozen there
(`attn_early` ignites at 0.581 against the unablated 0.981), and the interim L08 arm SUPPRESSED
rather than revived: 0.397 against 0.581, t = -3.58. Two checkpoints, opposite signs, same
manipulation. But look at where the COMPOUND lands: 0.516 and 0.397, while its reference moved
0.181 -> 0.581. That is what regression toward a characteristic level looks like, and it reads as
revival only when the reference happens to sit below that level.

THE TWO HYPOTHESES MAKE OPPOSITE QUANTITATIVE PREDICTIONS, which is why this is decidable rather
than a matter of framing. Regress compound ignition on reference ignition across checkpoints:

    H_revival (F104 as written): the compound tracks its reference with a positive offset.
                                 SLOPE ~ 1, and delta > 0 at every checkpoint.
    H_level  (the threat):       the compound sits at a fixed level whatever the reference does.
                                 SLOPE ~ 0, and sd(compound) << sd(reference) across checkpoints.

The slope is the discriminator. A confidence interval that excludes 0 and covers 1 is revival; one
that covers 0 and excludes 1 is a common level; anything else is NOT DECIDABLE, and with only a
handful of checkpoints that is a real possibility rather than a formality.

THE RANGE GATE COMES FIRST AND IS NOT A FORMALITY HERE. A slope is meaningless if the predictor
does not move: if reference ignition is similar at every checkpoint, both hypotheses predict the
same scatter and nothing is being tested. The reference must span several times its own standard
error before any slope is read -- the defect class this project has caught by hand six times, in
the one place where it would be easiest to miss because the x-axis is a measurement rather than a
knob.

MOST OF THE DATA ALREADY EXISTS. step143000 comes from #103's 360-cell run (all four arms, 20
seeds) and step8000 from the killed replication (three arms). Only the intermediate checkpoints are
new. Ignition is a per-run proportion measured under an identical protocol, so pooling across runs
is reading the same quantity, not merging two statistics -- and the RUN stays the unit of analysis
throughout (F57).

Usage:
    .venv/bin/python experiments/ignition_level.py --smoke
    .venv/bin/python experiments/ignition_level.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src"), str(_ROOT / "gatecheck" / "src")]

import argparse
import json
import os
import time

import numpy as np

from provenance import stamp, rel
from dev_transition_phase3 import measure, BASE, T
from gatecheck import NOT_DECIDABLE, carries_verdict, distinct_units, dynamic_range
from ablate_compensators import ablating_many, R, N, B, EARLY

LAYERS = [8, 22]                     # the strongest reviver and a second Bonferroni one (F104)
STEPS = ["step1000", "step2000", "step4000", "step8000", "step143000"]
NEW_SEEDS = list(range(21, 33))      # 12 seeds for the newly measured checkpoints
ARMS = ["none", EARLY] + [f"{EARLY}+attn_L{L:02d}" for L in LAYERS]
OUT = str(_ROOT / "results" / "ignition_level.json")

# Cells already measured under the identical protocol, harvested rather than re-run.
CACHES = {"step143000": "results/ablate_compensators.json",
          "step8000": "results/revival_replication.json"}
MIN_POINTS = 4
RANGE_K = 3.0                        # the reference must span RANGE_K x its own mean standard error


def harvest(res):
    """Pull matching cells out of prior runs. Same protocol, same geometry, run as the unit."""
    got = 0
    for step, path in CACHES.items():
        p = _ROOT / path
        if not p.exists():
            continue
        for rec in (json.load(open(p)).get("runs") or {}).values():
            if rec.get("arm") not in ARMS:
                continue
            if rec.get("step", step) != step:
                continue
            k = f"{step}|{rec['arm']}|s{rec['seed']}"
            if k not in res["runs"]:
                res["runs"][k] = dict(step=step, arm=rec["arm"], seed=rec["seed"],
                                      ignition_prob=rec["ignition_prob"],
                                      lambda_ca=rec.get("lambda_ca"), source=path)
                got += 1
    return got


def cell(res, step, arm):
    v = np.array([r["ignition_prob"] for r in res["runs"].values()
                  if r["step"] == step and r["arm"] == arm])
    if len(v) < 2:
        return None
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))), n=int(len(v)))


def wls_slope(x, y, sy):
    """Weighted least squares slope with its standard error. Weights are 1/se^2 on the response."""
    w = 1.0 / np.maximum(np.asarray(sy), 1e-9) ** 2
    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ (X.T @ W @ y)
    return float(beta[1]), float(np.sqrt(cov[1, 1])), float(beta[0])


def analyse(res):
    parts, gates = [], []
    pts = []
    for st in STEPS:
        ref, unab = cell(res, st, EARLY), cell(res, st, "none")
        for L in LAYERS:
            comp = cell(res, st, f"{EARLY}+attn_L{L:02d}")
            if ref and comp:
                pts.append(dict(step=st, layer=L, ref=round(ref["mean"], 4),
                                ref_se=round(ref["se"], 4), comp=round(comp["mean"], 4),
                                comp_se=round(comp["se"], 4), n_ref=ref["n"], n_comp=comp["n"],
                                unablated=None if not unab else round(unab["mean"], 4),
                                delta=round(comp["mean"] - ref["mean"], 4)))
    res["points"] = pts
    by_layer = {L: [p for p in pts if p["layer"] == L] for L in LAYERS}

    usable = {L: v for L, v in by_layer.items() if len(v) >= MIN_POINTS}
    if not usable:
        res["analysis"] = dict(decided=False, n_points={L: len(v) for L, v in by_layer.items()})
        res["verdict"] = (f"NOT DECIDABLE: no layer has {MIN_POINTS} checkpoints with both a "
                          f"reference and a compound estimate.")
        return res["verdict"]

    # --- the range gate, on the PREDICTOR, before any slope is read -------------------------
    allref = [p["ref"] for v in usable.values() for p in v]
    ref_se = float(np.mean([p["ref_se"] for v in usable.values() for p in v]))
    gates.append(dynamic_range(allref, floor=ref_se, k=RANGE_K,
                               name="reference ignition across checkpoints"))
    gates.append(distinct_units([p["step"] for v in usable.values() for p in v],
                                minimum=MIN_POINTS, name="checkpoint"))
    verdict = carries_verdict(gates, value=len(pts))

    parts.append(
        "GRID: " + "; ".join(
            f"{p['step']} L{p['layer']}: ref {p['ref']:.3f} -> comp {p['comp']:.3f} "
            f"(delta {p['delta']:+.3f})" for p in pts) + ".")

    if verdict.status == NOT_DECIDABLE:
        res["analysis"] = dict(decided=False, gates=[g.block() for g in gates], points=pts)
        res["verdict"] = (" ".join(parts) + f" NOT DECIDABLE: {verdict.reason}. Without spread in "
                          f"the reference both hypotheses predict the same scatter, so the slope "
                          f"tests nothing.")
        return res["verdict"]

    fits = {}
    for L, v in usable.items():
        x = np.array([p["ref"] for p in v]); y = np.array([p["comp"] for p in v])
        sy = np.array([max(p["comp_se"], 1e-6) for p in v])
        m, sm, b = wls_slope(x, y, sy)
        lo, hi = m - 1.96 * sm, m + 1.96 * sm
        fits[L] = dict(slope=round(m, 4), se=round(sm, 4), intercept=round(b, 4),
                       ci=[round(lo, 4), round(hi, 4)],
                       excludes_0=bool(lo > 0 or hi < 0), covers_1=bool(lo <= 1 <= hi),
                       sd_comp=round(float(np.std(y, ddof=1)), 4),
                       sd_ref=round(float(np.std(x, ddof=1)), 4),
                       deltas_all_positive=bool(all(p["delta"] > 0 for p in v)))
        parts.append(
            f"L{L}: slope {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(v)} checkpoints; "
            f"sd(compound) {np.std(y, ddof=1):.3f} against sd(reference) {np.std(x, ddof=1):.3f}; "
            f"delta positive at {sum(1 for p in v if p['delta'] > 0)}/{len(v)}.")

    level = [L for L, f in fits.items() if not f["excludes_0"] and not f["covers_1"]]
    reviv = [L for L, f in fits.items() if f["excludes_0"] and f["covers_1"]
             and f["deltas_all_positive"]]

    if level and not reviv:
        parts.append(
            f"COMMON LEVEL: for {level} the slope covers 0 and excludes 1, and the compound varies "
            f"less across checkpoints than its reference does. Adding an ablation drives ignition "
            f"toward a characteristic value rather than raising it, so F104's reading -- removing "
            f"more of the network makes damage spread further -- is an artifact of measuring at a "
            f"checkpoint where the reference happened to sit below that value. F104 must be "
            f"amended and plan_paper2's row 4 rewritten around the level, not the revival.")
        decided = True
    elif reviv and not level:
        parts.append(
            f"REVIVAL SURVIVES: for {reviv} the slope excludes 0, covers 1, and delta is positive "
            f"at every checkpoint. The compound tracks its reference with a positive offset, which "
            f"is F104 as written and not regression to a level. Gates: {verdict.reason}")
        decided = True
    else:
        parts.append(
            f"NOT DECIDABLE: the slopes separate neither hypothesis cleanly (level={level}, "
            f"revival={reviv}). With {min(len(v) for v in usable.values())} checkpoints the "
            f"interval is wide; the fix is more checkpoints spanning more reference ignition, not "
            f"more seeds at the ones already measured.")
        decided = False

    parts.append(
        "BOUNDARY: one model family, one radius, greedy, and the checkpoints are not independent "
        "of each other -- they are a training trajectory, so a slope across them is not a random "
        "sample of models. This distinguishes two readings of an existing effect; it does not "
        "establish what sets the level.")

    res["analysis"] = dict(fits=fits, points=pts, gates=[g.block() for g in gates],
                           reference_span=round(max(allref) - min(allref), 4),
                           reference_se=round(ref_se, 4), decided=decided)
    res["verdict"] = " ".join(parts)
    return res["verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    steps = STEPS[:2] if a.smoke else STEPS
    seeds = NEW_SEEDS[:1] if a.smoke else NEW_SEEDS

    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        question="is F104 revival, or regression to a common ignition level?",
        base=BASE, steps=steps, layers=LAYERS, arms=ARMS, r=R, N=N, B=B, T=T, new_seeds=seeds,
        h_revival="compound tracks reference with a positive offset: slope ~ 1, delta > 0 at every "
                  "checkpoint (F104 as written)",
        h_level="compound sits at a fixed value whatever the reference does: slope ~ 0, "
                "sd(compound) << sd(reference)",
        primary="weighted least-squares slope of compound ignition on reference ignition across "
                "checkpoints, with a 95% interval. Excludes 0 and covers 1 -> revival; covers 0 "
                "and excludes 1 -> common level; otherwise NOT DECIDABLE",
        range_gate=f"reference ignition must span {RANGE_K}x its own mean standard error before "
                   f"any slope is read -- without spread in the predictor both hypotheses predict "
                   f"the same scatter",
        harvested="step143000 from #103's run and step8000 from the killed replication; identical "
                  "protocol and geometry, ignition is a per-run proportion, RUN is the unit (F57)",
        boundary="checkpoints are a training trajectory, not a random sample of models; this "
                 "separates two readings of an existing effect and does not explain the level",
        resumable="keyed by (step, arm, seed)")

    n = harvest(res)
    print(f"harvested {n} cells from prior runs", flush=True)

    todo = [(st, ar, sd) for st in steps for ar in ARMS for sd in seeds
            if f"{st}|{ar}|s{sd}" not in res["runs"]]
    print(f"{len(res['runs'])} cached, {len(todo)} new cells (~{len(todo)*135/3600:.1f} h)\n",
          flush=True)

    for step, arm, seed in todo:
        t0 = time.time()
        with ablating_many(arm):
            lam, dn, md, ig = measure(step, N, B, seed, r=R)
        res["runs"][f"{step}|{arm}|s{seed}"] = dict(
            step=step, arm=arm, seed=seed, r=R, N=N, B=B, T=T,
            lambda_ca=lam, D_norm=dn, mean_damage=md, ignition_prob=ig,
            secs=round(time.time() - t0, 1), source="measured here")
        print(f"  {step:12s} {arm:22s} s={seed}  ign={ig:.3f}  ({time.time()-t0:.0f}s)", flush=True)
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
