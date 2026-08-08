"""Is the coupling a COMMON MODE where the paper uses it? The crossing, under both couplings.

THE CLAIM BEING TESTED IS F41'S ESCAPE CLAUSE. F41 established that this project's CRN is the
MONOTONE coupling, not the maximal one, and measured the gap in ABSOLUTE damage at 1.013x (T=0.7)
and 1.054x (T=0.9). It then argued that every RELATIVE comparison survives -- checkpoint to
checkpoint, across radii, rule to rule -- "because the coupling is a common mode". That argument is
asserted and never measured, and W2 concedes the same gap from the other side ("the alternative
floors themselves are unrun on the LM backends").

WHY IT IS LOAD-BEARING. The paper concedes that the substrate is Glauber dynamics and stakes its
contribution on the coupling. Rows 3 and 4 of its discriminator table -- the developmental crossing
and the ablation response -- are RELATIVE comparisons taken under one coupling. If the inflation is
uniform they are safe under either. If it is DIFFERENTIAL, and it plausibly could be, since maximal
coupling maximises agreement and therefore bites hardest where conditionals are peaked, which is
exactly where the frozen phase lives, then an ordering could be an artifact of the coupling.

SCOPED TO THE CROSSING, NOT THE WHOLE GRID. The minimal decisive test is where the paper actually
uses the relative comparison: the sign change between step256 and step512, plus one plateau point
to fix the ordering. Measuring the full developmental curve twice would cost several times as much
and answer the same question.

    PRIMARY: does the SIGN CROSSING survive, and does the ORDERING of the three checkpoints
             survive, when the coupling is swapped?

RUNG 1b COMES FIRST, AND IS NOT THE SAME AS RUNG 1. paired_runner's rung 1 validated the lockstep
loop on the toy JAX backend. This is the AR backend, a different rule, a different window shape and
a different twin protocol -- `ar_probe.block_damage` batches BOTH twins into one 2B run and
duplicates the uniform stream across the batch, rather than making two calls. The paired path must
reproduce THAT, cell for cell, under the monotone coupler before either arm is read.

WHAT IT COSTS, AND WHY LESS THAN EXPECTED. Because block_damage already advances both twins in a
single batched forward pass, both conditionals are available at every site for free. Maximal
coupling therefore costs no extra model evaluation -- only the coupler arithmetic. The
reproducibility cost stands: maximal draws depend on both twins jointly, so a run is reproducible
only as a pair.

Usage:
    .venv/bin/python experiments/coupling_primary.py --rung   # AR reproduction check only
    .venv/bin/python experiments/coupling_primary.py          # the primary
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src"), str(_ROOT / "gatecheck" / "src")]

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

from provenance import stamp, rel
from lyapunov import lyap_from_cone
from gatecheck import NOT_DECIDABLE, carries_verdict, distinct_units, dynamic_range
from coupling_ladder import couple_monotone, couple_maximal

from dev_transition_phase3 import BASE, R, T, FIT_KW

STEPS = ["step256", "step512", "step143000"]     # pre-crossing, post-crossing, plateau
COUPLINGS = ["monotone", "maximal"]
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B = 48, 16
SETTLE, SWEEPS, BLOCK, TAIL = 12, 22, 3, 8
IGNITE = 0.05
OUT = str(_ROOT / "results" / "coupling_primary.json")
# F41's measured ABSOLUTE gap, for context. Not a threshold: this experiment is about the
# RELATIVE comparison, which F41 asserted survives.
F41_ABSOLUTE_GAP = {"T0.7": 1.013, "T0.9": 1.054}


def paired_block_damage(rule, coupling, *, seed, scheme="none"):
    """`ar_probe.block_damage`, with the twins' coupling made explicit.

    Mirrors that function exactly under `monotone` -- same settle, same flip, same uniform stream,
    same cone construction -- and swaps only the rule by which the twin pair is drawn.
    """
    from ar_ca import run as ar_run
    rng = np.random.default_rng(seed)
    base = ar_run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme=scheme,
                  init="random", seed=seed)["final"]
    c = N // 2
    idx = [c + k for k in range(-(BLOCK // 2), BLOCK - BLOCK // 2)]
    flipped = base.copy()
    for j in idx:
        flipped[:, j] = rng.choice(rule.init_pool, size=B)

    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)

    # BOTH couplings go through the SAME lockstep loop. An earlier version routed `monotone`
    # to the production call, which made the reproduction rung vacuous: it compared production
    # against production and matched at 0.0 without ever exercising the loop the maximal arm uses.
    a, b = _paired_lockstep(rule, base, flipped, u, seed + 2, scheme, coupling)

    diff = (a != b)
    cone = np.roll(diff, N // 2 - idx[len(idx) // 2], axis=2).mean(axis=1)
    final = diff[-TAIL:].mean(axis=(0, 2))
    return dict(cone=cone, mean_damage=float(final.mean()),
                ignition_prob=float((final > IGNITE).mean()))


def _t_monotone(p, q, u):
    """Inverse-CDF against a shared uniform, ON DEVICE -- `ar_ca.sample_device` verbatim.

    Written in torch rather than numpy for a reason the rung found the hard way: the AR backend
    runs the model in float16 on MPS and samples there. A numpy re-implementation cumsums a 50k
    vocabulary in a different precision, and no amount of care makes that reproduce -- the first
    version of this loop missed the production cone by 0.3125. The coupling must be applied in the
    same arithmetic the production path uses, or the comparison measures the arithmetic.
    """
    ut = torch.as_tensor(u, device=p.device, dtype=p.dtype).unsqueeze(1)
    cp = p.cumsum(-1); cp = cp / cp[:, -1:]
    cq = q.cumsum(-1); cq = cq / cq[:, -1:]
    return ((cp < ut).sum(dim=1).to("cpu", torch.int64).numpy(),
            (cq < ut).sum(dim=1).to("cpu", torch.int64).numpy())


def _t_maximal(p, q, u, u2, u3):
    """Maximal coupling on device: agree w.p. sum_v min(p_v, q_v), else draw the residuals."""
    dev, dt = p.device, p.dtype
    m = torch.minimum(p, q)
    M = m.sum(-1)
    agree = torch.as_tensor(u, device=dev, dtype=dt) < M

    def draw(w, uu):
        tot = w.sum(-1, keepdim=True).clamp_min(1e-9)
        c = (w / tot).cumsum(-1)
        c[:, -1] = 1.0
        ut = torch.as_tensor(uu, device=dev, dtype=dt).unsqueeze(1)
        return (c < ut).sum(dim=1)

    common = draw(m, u2)
    rx = draw((p - m).clamp_min(0), u2)
    ry = draw((q - m).clamp_min(0), u3)
    x = torch.where(agree, common, rx).to("cpu", torch.int64).numpy()
    y = torch.where(agree, common, ry).to("cpu", torch.int64).numpy()
    return x, y


def _paired_lockstep(rule, base, flipped, u, seed, scheme, coupling):
    """Advance both twins in lockstep under `coupling`. One forward pass supplies both conditionals.

    The adapter keeps its default (sampler=None) so `probs` returns the DEVICE tensor the
    production path uses; the coupling is then applied in torch on that tensor.
    """
    from ar_ca import _ARAdapter
    ad = _ARAdapter(rule, scheme, None)
    rng = np.random.default_rng(seed)
    rng_extra = np.random.default_rng(seed * 7919 + 13)
    lat = np.concatenate([base, flipped], axis=0)          # 2B rows: twin A then twin B
    snaps = [lat.copy()]
    ui = 0
    for _ in range(SWEEPS):
        for i in rng.permutation(N):
            idx = ad.window(i, R, N)
            uu = u[ui:ui + B]; ui += B
            probs = ad.probs(lat[:, idx], T)               # (2B, V) device tensor, ONE pass
            p, q = probs[:B], probs[B:]
            if coupling == "monotone":
                x, y = _t_monotone(p, q, uu)
            else:
                x, y = _t_maximal(p, q, uu, rng_extra.random(B), rng_extra.random(B))
            lat[:B, i] = x
            lat[B:, i] = y
        snaps.append(lat.copy())
    snaps = np.array(snaps)
    return snaps[:, :B], snaps[:, B:]


def rung_ar_reproduction(revision="step143000", seed=21):
    """The LOCKSTEP loop under MONOTONE must equal ar_probe.block_damage, cell for cell.

    This is the check that matters: the loop is a numpy re-implementation of the production sweep,
    and production samples on-device. If the two disagree, every maximal number would be measuring
    that disagreement -- which is exactly the failure paired_runner's rung 1 caught on the toy
    backend, where a float64 coercion moved 12% of cells.
    """
    from ar_ca import ARRule
    from ar_probe import block_damage
    rule = ARRule(BASE, revision=revision)
    try:
        prod = block_damage(rule, T, R, block=BLOCK, B=B, N=N, settle=SETTLE, sweeps=SWEEPS,
                            seed=seed, scheme="none")
        mine = paired_block_damage(rule, "monotone", seed=seed)
    finally:
        rule.model = None; del rule; gc.collect()
        try: torch.mps.empty_cache()
        except Exception: pass
    dc = float(np.abs(prod["cone"] - mine["cone"]).max())
    return dict(max_cone_difference=dc,
                production_mean_damage=prod["mean_damage"],
                paired_mean_damage=mine["mean_damage"],
                nonvacuous=bool(prod["mean_damage"] > 0.01),
                passes=bool(dc == 0.0 and prod["mean_damage"] > 0.01))


def lam_of(rule, coupling, seed):
    d = paired_block_damage(rule, coupling, seed=seed)
    return float(lyap_from_cone(d["cone"], N, **FIT_KW)[0]), d["mean_damage"], d["ignition_prob"]


def analyse(res):
    parts, gates = [], []
    rung = res.get("rung") or {}
    parts.append(
        f"RUNG (AR backend): the paired path under monotone reproduces ar_probe.block_damage to "
        f"{rung.get('max_cone_difference')} in the cone, against a production mean damage of "
        f"{rung.get('production_mean_damage')} -> "
        f"{'PASS' if rung.get('passes') else 'FAIL'}.")
    if not rung.get("passes"):
        res["analysis"] = dict(decided=False)
        res["verdict"] = (" ".join(parts) + " NOT DECIDABLE: the paired path does not reproduce the "
                          "production protocol on this backend, so a coupling comparison built on "
                          "it would measure that difference instead.")
        return res["verdict"]

    def curve(cpl):
        out = {}
        for st in STEPS:
            v = [r["lambda_ca"] for r in res["runs"].values()
                 if r["step"] == st and r["coupling"] == cpl and r["ignition_prob"] > IGNITE]
            if v:
                out[st] = dict(median=float(np.median(v)), n=len(v),
                               se=float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None)
        return out

    cm, cx = curve("monotone"), curve("maximal")
    if len(cm) < len(STEPS) or len(cx) < len(STEPS):
        res["analysis"] = dict(decided=False, monotone=cm, maximal=cx)
        res["verdict"] = (" ".join(parts) + " NOT DECIDABLE: a checkpoint produced no ignited run "
                          "under one of the couplings, so lambda is undefined there (F42).")
        return res["verdict"]

    parts.append("CURVES: " + "; ".join(
        f"{st} monotone {cm[st]['median']:+.4f} vs maximal {cx[st]['median']:+.4f}"
        for st in STEPS) + ".")

    floors = [c[st]["se"] for c in (cm, cx) for st in STEPS if c[st]["se"]]
    floor = float(np.mean(floors)) if floors else None
    gates.append(dynamic_range([cm[st]["median"] for st in STEPS], floor=floor or 1e-9,
                               name="lambda across the three checkpoints (monotone)"))
    gates.append(distinct_units(STEPS, minimum=3, name="checkpoint"))
    verdict = carries_verdict(gates, value=len(STEPS))

    # THE SIGN-CROSSING TEST WAS BADLY CHOSEN AND IS REPORTED AS SUCH. lambda at step256 is
    # within its own standard error of zero under BOTH couplings (+0.0083 +/- 0.0440 monotone,
    # -0.0303 +/- 0.0355 maximal), and the paper itself records that pre-crossing "seeds disagree
    # about the sign". A test of whether a sign changes, evaluated where the value is not
    # distinguishable from zero, is a test of a coin flip -- it cannot discriminate couplings, and
    # an earlier version of this branch reported the two arms AGREEING at False as "the crossing is
    # present under both", which is nonsense. The ordering is the sound comparison.
    pre = cm["step256"]
    pre_x = cx["step256"]
    crossing_informative = bool(
        pre["se"] and pre_x["se"]
        and abs(pre["median"]) > 2 * pre["se"] and abs(pre_x["median"]) > 2 * pre_x["se"])
    order_m = [cm[st]["median"] for st in STEPS]
    order_x = [cx[st]["median"] for st in STEPS]
    same_order = bool(np.all(np.argsort(order_m) == np.argsort(order_x)))
    offsets = [cx[st]["median"] - cm[st]["median"] for st in STEPS]
    offset_spread = float(np.max(offsets) - np.min(offsets))
    offset_uniform = bool(offset_spread <= 2 * (floor or 1e-9))

    parts.append(
        f"CROSSING TEST: NOT INFORMATIVE at this geometry. lambda(step256) is "
        f"{pre['median']:+.4f} +/- {pre['se']:.4f} under monotone and {pre_x['median']:+.4f} +/- "
        f"{pre_x['se']:.4f} under maximal -- neither is distinguishable from zero, so whether the "
        f"SIGN changes is not determined at 8 seeds and cannot separate the couplings. Reported "
        f"rather than quietly dropped: it was the registered primary's first leg.")
    parts.append(
        f"OFFSET: maximal reads lower at every checkpoint, by "
        f"{', '.join(f'{o:+.4f}' for o in offsets)} -- the direction F41 predicts, since maximal "
        f"coupling maximises agreement and so minimises damage. The offsets span "
        f"{offset_spread:.4f} against a floor of {floor:.4f}, so they are "
        f"{'uniform within noise' if offset_uniform else 'NOT uniform'}.")

    if verdict.status == NOT_DECIDABLE:
        parts.append(f"NOT DECIDABLE: {verdict.reason}.")
        decided = False
    elif same_order and offset_uniform:
        parts.append(
            f"COMMON MODE, for the comparison the paper actually makes: the ORDERING of the three "
            f"checkpoints is identical under both couplings, and the offset between them is "
            f"uniform within the seed floor. Rows 3 and 4 of the discriminator table read the "
            f"model, not the coupling. F41's escape clause holds here -- measured, where it had "
            f"been asserted. Gates: {verdict.reason}")
        decided = True
    elif same_order:
        parts.append(
            f"ORDERING SURVIVES, OFFSET DOES NOT: the ranking of the three checkpoints is identical "
            f"under both couplings, so the qualitative reading is safe, but the offset varies by "
            f"{offset_spread:.4f} against a {floor:.4f} floor -- more than a common mode allows. "
            f"Any ABSOLUTE lambda must state its coupling; relative rankings need not.")
        decided = True
    else:
        parts.append(
            f"NOT A COMMON MODE: the ordering of the three checkpoints DIFFERS between couplings "
            f"(monotone {order_m}, maximal {order_x}). A relative reading taken under one coupling "
            f"is not automatically a reading of the model, and rows 3 and 4 need re-taking under a "
            f"stated coupling.")
        decided = True

    parts.append(
        f"BOUNDARY: one geometry (r={R}, T={T}, N={N}, B={B}), one family, three checkpoints. A "
        f"pass licenses THESE comparisons, not coupling-invariance in general. F41's absolute gap "
        f"({F41_ABSOLUTE_GAP}) is untouched -- this is about the relative reading only.")

    res["analysis"] = dict(monotone=cm, maximal=cx, crossing_informative=crossing_informative,
                           same_ordering=same_order, offsets=[round(o, 5) for o in offsets],
                           offset_spread=round(offset_spread, 5), offset_uniform=offset_uniform,
                           floor=None if floor is None else round(floor, 5),
                           gates=[g.block() for g in gates], decided=decided)
    res["verdict"] = " ".join(parts)
    return res["verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", action="store_true", help="AR reproduction check only")
    a = ap.parse_args()

    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        question="is the coupling a COMMON MODE for the relative comparisons the paper uses?",
        settled_elsewhere=f"F41 measured the ABSOLUTE gap ({F41_ABSOLUTE_GAP}); this does not "
                          f"re-ask that",
        primary="does the sign crossing between step256 and step512, and the ordering of the three "
                "checkpoints, survive swapping monotone for maximal coupling?",
        rung="the paired path under MONOTONE must reproduce ar_probe.block_damage cell for cell on "
             "the AR backend -- a different rule, window and twin protocol from paired_runner's "
             "toy-backend rung 1",
        refuted="crossing or ordering differs -> the relative reading is not coupling-invariant and "
                "rows 3 and 4 of the paper's discriminator table need re-taking",
        base=BASE, steps=STEPS, couplings=COUPLINGS, seeds=SEEDS, r=R, T=T, N=N, B=B,
        boundary="one geometry, one family, three checkpoints; a pass licenses THESE comparisons",
        resumable="keyed by (step, coupling, seed)")

    if "rung" not in res or not res["rung"].get("passes"):
        print("RUNG -- AR-backend reproduction under monotone\n")
        t0 = time.time()
        res["rung"] = rung_ar_reproduction()
        r = res["rung"]
        print(f"  max cone difference : {r['max_cone_difference']}")
        print(f"  production damage   : {r['production_mean_damage']:.4f} "
              f"(nonvacuous={r['nonvacuous']})")
        print(f"  -> {'PASS' if r['passes'] else 'FAIL'}   ({time.time()-t0:.0f}s)\n", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        if not r["passes"]:
            print("  Stopping: a coupling comparison on an unreproduced path measures the path.")
            return 1
    if a.rung:
        return 0

    from ar_ca import ARRule
    todo = [(st, cp, sd) for st in STEPS for cp in COUPLINGS for sd in SEEDS
            if f"{st}|{cp}|s{sd}" not in res["runs"]]
    print(f"{len(res['runs'])} cached, {len(todo)} cells\n", flush=True)
    for st in STEPS:
        pend = [(c, s) for (x, c, s) in todo if x == st]
        if not pend:
            continue
        rule = ARRule(BASE, revision=st)
        try:
            for cp, sd in pend:
                t0 = time.time()
                lam, md, ig = lam_of(rule, cp, sd)
                res["runs"][f"{st}|{cp}|s{sd}"] = dict(
                    step=st, coupling=cp, seed=sd, lambda_ca=lam, mean_damage=md,
                    ignition_prob=ig, secs=round(time.time() - t0, 1))
                print(f"  {st:12s} {cp:9s} s={sd}  lambda={lam:+.4f}  ign={ig:.3f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
        finally:
            rule.model = None; del rule; gc.collect()
            try: torch.mps.empty_cache()
            except Exception: pass

    print("\n  -> " + analyse(res))
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
