"""Annealed mean field, tested by INTERVENTION: does lambda follow s when the model is damaged?

WHY THIS IS ITS OWN EXPERIMENT AND NOT A DIAGNOSTIC INSIDE #103. Three arms measured while checking
#103's plumbing showed single-token sensitivity RISING with ablation depth (0.8174 -> 0.8597 ->
0.8758) while measured lambda_ca COLLAPSED (0.3566 -> 0.0115). Mean field says lambda = log(r*s),
so s up must mean lambda up. Burying that in a compensator experiment would be filing a finding as
a footnote.

THE FIRST DRAFT OF THIS FILE REGISTERED THE WRONG STATISTIC, and the record is kept because the
correction is the point. The primary was Spearman(s, lambda), read as a directional falsification.
Exercised on realistic values before running, `correlation_leverage` refused it: lambda_MF(s) spans
about 0.046 against the target's 0.407, a ratio of 0.11 against a 0.5 gate, so even a perfect
rho = -1 would have been evidence in NEITHER direction. That is F94's own defect, reproduced by an
experiment written to test F94.

THE REFUSAL IS THE RESULT, once the question is asked properly. A theory earns its keep by telling
interventions apart. Handed the model's own exactly-measured s, annealed mean field predicts
essentially ONE lambda for every arm -- for an intact lattice and for one frozen at ignition 0.156
alike -- while the measured exponent spans nearly half a nat. So the registered primary is the
RANGE RATIO, range(lambda_MF) / range(lambda_measured), and the correlation is reported underneath
it as description rather than as evidence. Non-discrimination is a decidable outcome; an
uninterpretable correlation is not.

WHAT IS NEW AND WHAT IS BORROWED. lambda_ca for all 33 arms is already measured -- F79's groups
(ablate_lambda.json) and F80's 24 singles (ablate_layers.json) -- and is READ, not re-run. Only s
is new, and s is exact: `s_crn` is a deterministic functional of a conditional pair under a shared
uniform, so this costs forward passes and carries no seed noise and no ignited-vs-unignited
selection.

THE ENSEMBLE IS HELD FIXED, FOLLOWING F99. s is a property of the ensemble as much as of the model
(F96: flat 0.833-0.876 on random windows, spanning 0.331 on the states the ring occupies). Every
arm is measured on the SAME pool, settled from the UNABLATED ring, so the model varies and the
ensemble does not. Each arm's own settled state would reintroduce F96's circularity, and a heavily
ablated ring barely settles at all.

THE PREDICTOR IS MAPPED INTO THE TARGET'S UNITS BEFORE ANY RANGE IS COMPARED. `correlation_leverage`
requires that, and F94 is the reason it exists: a predictor that cannot move as much as the target
gives no evidence in either direction, and quoting a weak correlation from one is the defect that
appeared three times before it was named. So the predictor is lambda_MF = log(r*s), not s.

WHAT A FAILURE WOULD AND WOULD NOT MEAN. It would show annealed mean field does not discriminate
between interventions on this model: given the model's own s, it calls a frozen lattice and a live
one equally supercritical. It would NOT show s is irrelevant -- F94's rung 2 got 17 of 19 ECA rules
right, and the one it missed was rule 232, MAJORITY, the canonical canalizing function. The natural
reading is the one `canalization.py` already proposes: the ANNEALED MEAN discards the spread of
sensitivity and its sub-additivity, and the residual lives there. This measures that premise under
intervention rather than across training, which is a controlled test the developmental grid cannot
give.

Usage:
    .venv/bin/python experiments/meanfield_under_ablation.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src"), str(_ROOT / "gatecheck" / "src")]

import json
import os
import time

import numpy as np
from scipy import stats

from provenance import stamp, rel
from lyapunov import lambda_of, run_ignited
from gatecheck import (NOT_DECIDABLE, carries_verdict, correlation_leverage, directional,
                       distinct_units, dynamic_range)
from meanfield_lambda import lambda_mf
from ablate_compensators import settled_pool, s_for_arm, R, T, STEP, BASE, N_CTX_S

OUT = str(_ROOT / "results" / "meanfield_under_ablation.json")
SOURCES = ["results/ablate_layers.json", "results/ablate_lambda.json"]
MIN_ARMS = 12
# A predictor must span at least half its target's range to be informative in the target's units
# -- gatecheck.correlation_leverage's own default, adopted here as the discrimination threshold.
DISCRIMINATION_GATE = 0.5


def measured_lambdas():
    """{arm: median lambda over its IGNITED runs}, read from F79 and F80. Never re-measured."""
    by = {}
    for src in SOURCES:
        p = _ROOT / src
        if not p.exists():
            continue
        for rec in (json.load(open(p)).get("runs") or {}).values():
            by.setdefault(rec.get("ablation"), []).append(rec)
    out = {}
    for arm, rs in by.items():
        ign = [r for r in rs if run_ignited(r)]
        if arm and ign:
            out[arm] = dict(lambda_ca=float(np.median(lambda_of(ign))),
                            n_ignited=len(ign), n_runs=len(rs),
                            ignition_rate=float(np.mean([r.get("ignition_prob", 0.0)
                                                         for r in rs])))
    return out


def floor_of(res):
    """Seed floor for measured lambda: the mean within-arm spread, read from the source runs."""
    sds = []
    for src in SOURCES:
        p = _ROOT / src
        if not p.exists():
            continue
        by = {}
        for rec in (json.load(open(p)).get("runs") or {}).values():
            if run_ignited(rec):
                by.setdefault(rec.get("ablation"), []).append(rec)
        sds += [float(np.std(lambda_of(rs))) for rs in by.values() if len(rs) > 1]
    return float(np.mean([v for v in sds if np.isfinite(v) and v > 0])) if sds else 0.05


def analyse(res):
    lam = res["lambda_measured"]
    sblk = res["s"]
    arms = sorted(a for a in sblk if a in lam)
    parts, gates = [], []

    if len(arms) < MIN_ARMS:
        res["analysis"] = dict(n_arms=len(arms), decided=False)
        res["verdict"] = (f"NOT DECIDABLE: only {len(arms)} arms carry both an s and an ignited "
                          f"lambda, under the {MIN_ARMS} required.")
        return res["verdict"]

    s_vals = np.array([sblk[a]["s"] for a in arms])
    y = np.array([lam[a]["lambda_ca"] for a in arms])
    pred = np.array([lambda_mf(R, v) for v in s_vals])          # the predictor, in lambda's units

    parts.append(
        f"GRID: {len(arms)} ablation arms, s in [{s_vals.min():.4f}, {s_vals.max():.4f}] against "
        f"the mean-field critical 1/r = {1/R:.4f}; measured lambda in [{y.min():+.4f}, "
        f"{y.max():+.4f}]; predicted lambda_MF in [{pred.min():+.4f}, {pred.max():+.4f}].")

    # THE PRIMARY IS DISCRIMINATION, NOT CORRELATION, and the first draft of this file had it
    # wrong in a way worth recording. Spearman(s, lambda) was registered as the primary, and on
    # realistic values `correlation_leverage` refused it: lambda_MF(s) spans ~0.046 against the
    # target's ~0.407, ratio 0.11 against a 0.5 gate, so even a perfect rho = -1 would have been
    # uninterpretable -- exactly F94's defect, reproduced by an experiment written to test F94.
    #
    # But the refusal IS the result once the question is asked properly. A theory earns its keep by
    # telling interventions apart. Mean field, handed the model's own exactly-measured s, predicts
    # essentially ONE lambda for every arm -- for a healthy lattice and for one frozen at
    # ignition 0.156 alike. So the registered quantity is the RANGE RATIO, and the correlation is
    # reported underneath it as a description rather than as evidence.
    lev = correlation_leverage(pred, y, name="lambda_MF(s) across ablation arms")
    # The blocking gate is on the TARGET: if measured lambda does not vary across arms there is
    # nothing for any theory to discriminate, and non-discrimination would be vacuous.
    gates.append(dynamic_range(y, floor=floor_of(res), name="measured lambda_ca across arms"))
    gates.append(distinct_units(arms, minimum=MIN_ARMS, name="ablation arms"))
    verdict = carries_verdict(gates, value=len(arms))

    span_pred = float(pred.max() - pred.min())
    span_meas = float(y.max() - y.min())
    ratio = span_pred / span_meas if span_meas else 0.0
    rho, p = stats.spearmanr(s_vals, y)
    parts.append(
        f"PRIMARY: mean field's predicted lambda spans {span_pred:.4f} across these {len(arms)} "
        f"interventions while the measured exponent spans {span_meas:.4f} -- a range ratio of "
        f"{ratio:.3f} against the {DISCRIMINATION_GATE} a predictor needs to be informative in "
        f"its target's units. Spearman(s, lambda) = {rho:+.4f} (p = {p:.3g}), reported as "
        f"description: with this little range it is not evidence in either direction.")

    # Directional, not blocking: failing to discriminate is the finding, not an inability to decide.
    dirn = directional(ratio - DISCRIMINATION_GATE, expect="increase", floor=0.0)

    if verdict.status == NOT_DECIDABLE:
        parts.append(
            f"NOT DECIDABLE: {verdict.reason}. This is F94's own defect and not a result -- a "
            f"predictor that cannot move as far as its target gives no evidence in either "
            f"direction, and a correlation quoted from one would be uninterpretable.")
        decided = False
    elif not dirn.usable:
        parts.append(
            f"NON-DISCRIMINATIVE UNDER INTERVENTION: {lev.reason} Handed the model's own exactly "
            f"measured s, annealed mean field predicts lambda in "
            f"[{pred.min():+.4f}, {pred.max():+.4f}] for EVERY arm -- for the intact lattice and "
            f"for one frozen at ignition {min(lam[a]['ignition_rate'] for a in arms):.3f} alike -- "
            f"while the measured exponent spans {y.min():+.4f} to {y.max():+.4f}. Every arm sits "
            f"far above the critical 1/r = {1/R:.4f}, so the theory calls all of them strongly "
            f"supercritical, including the ones whose damage dies. This is not a failure of "
            f"single-token sensitivity as such: it is that the ANNEALED mean has no room to move, "
            f"which is canalization.py's premise measured under intervention rather than across "
            f"training.")
        decided = True
    else:
        parts.append(
            f"DISCRIMINATES: mean field's predictions move with the interventions "
            f"(ratio {ratio:.3f}), so the theory is informative here and the correlation above "
            f"can be read as evidence. Gates: {verdict.reason}")
        decided = True

    parts.append(
        "BOUNDARY: one model, one checkpoint, one radius, one fixed ensemble. lambda_ca is read "
        "from F79/F80 rather than re-measured, so this inherits their geometry exactly and nothing "
        "here re-opens their numbers. A failure of ANNEALED mean field is not a failure of "
        "single-token sensitivity as such -- F94's rung 2 got 17 of 19 ECA rules right, missing "
        "rule 232, MAJORITY, the canonical canalizing function.")

    res["analysis"] = dict(
        n_arms=len(arms), range_ratio=round(ratio, 5), span_predicted=round(span_pred, 5),
        span_measured=round(span_meas, 5), leverage=lev.block(),
        spearman_rho=round(float(rho), 5), spearman_p=float(p),
        s_range=[round(float(s_vals.min()), 5), round(float(s_vals.max()), 5)],
        lambda_range=[round(float(y.min()), 5), round(float(y.max()), 5)],
        lambda_mf_range=[round(float(pred.min()), 5), round(float(pred.max()), 5)],
        critical=round(1 / R, 5), gates=[g.block() for g in gates],
        directional=dirn.block(), decided=decided,
        per_arm=[dict(arm=a, s=sblk[a]["s"], s_sd=sblk[a]["s_sd"],
                      lambda_mf=sblk[a]["lambda_mf"], lambda_measured=lam[a]["lambda_ca"],
                      ignition_rate=round(lam[a]["ignition_rate"], 4)) for a in arms])
    res["verdict"] = " ".join(parts)
    return res["verdict"]


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"s": {}}
    lam = measured_lambdas()
    res["lambda_measured"] = lam
    res["_preregistration"] = dict(
        base=BASE, step=STEP, r=R, T=T, n_ctx=N_CTX_S, sources=SOURCES,
        primary=f"range(lambda_MF(s)) / range(lambda_measured) across ablation arms, against the "
                f"{DISCRIMINATION_GATE} a predictor needs to be informative in its target's "
                f"units. A theory earns its keep by telling interventions apart",
        falsification="a range ratio below the gate: mean field predicts essentially ONE lambda "
                      "for every intervention, including ones that freeze the lattice",
        reported_not_registered="Spearman(s, lambda) is recorded as description; with a saturated "
                                "predictor it is not evidence in either direction (F94's defect)",
        not_decidable="measured lambda does not vary across arms (nothing to discriminate), or "
                      f"fewer than {MIN_ARMS} arms carry both quantities",
        ensemble="a single pool settled from the UNABLATED ring; the model varies, the ensemble "
                 "does not (F99's column design, breaking F96's circularity)",
        lambda_source="read from F79/F80, never re-measured; ignited runs only (F42)",
        boundary="one model, one checkpoint, one radius; a failure of ANNEALED mean field is not a "
                 "failure of single-token sensitivity as such",
        resumable="keyed by arm")

    todo = [a for a in sorted(lam) if a not in res["s"]]
    print(f"{len(res['s'])} cached, {len(todo)} arms to measure "
          f"(exact, no seeds)\n", flush=True)
    if todo:
        pool = settled_pool()
        rng = np.random.default_rng(0)
        for arm in todo:
            t0 = time.time()
            res["s"][arm] = s_for_arm(arm, pool, rng)
            r_ = res["s"][arm]
            print(f"  {arm:14s} s={r_['s']:.5f} +/- {r_['s_sd']:.5f}  "
                  f"lambda_MF={r_['lambda_mf']:+.4f}  lambda_meas={lam[arm]['lambda_ca']:+.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)

    print("\n  -> " + analyse(res))
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
