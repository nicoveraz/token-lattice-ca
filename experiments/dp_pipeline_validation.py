"""Can the DP extraction pipeline recover known exponents? Validate it on Domany-Kinzel (#82).

WHY THIS EXISTS. Phase 1 (dp_survival_scan.py) applied a survival/active-count fitting pipeline to
a language model and produced numbers. Nothing had ever checked that the pipeline can recover DP
exponents from data that provably has them. That is the project's own ladder principle -- validate
by reproduction before measuring the unknown -- applied to the ANALYSIS code rather than the
simulator, which is the half that had never been calibrated.

The check is affordable in a way the LM version is not. DK is pure numpy and fully vectorised:
512 replicas x 512 sites x 200 sweeps runs in 0.2s, so a million replicas costs about six minutes
on this machine with no GPU. The identical measurement on the LM costs ~152s per 16 replicas.
That gap is the whole argument for calibrating here first: if the pipeline cannot recover delta
and theta on DK, then no amount of GPU time spent on the LM would have produced a trustworthy
number, and the failure is in the estimator rather than in the physics.

WHY DK IS THE RIGHT CALIBRATION TARGET. The Domany-Kinzel automaton on its p2=0 line has a
directed-percolation transition whose exponents are the 1+1D DP values -- this is textbook, and
F38 already established that our DK implementation is bit-exact against an independent simulator.
So the answer is known, the simulator is trusted, and any discrepancy is attributable to the
fitting procedure alone.

PRE-REGISTERED BEFORE RUNNING:
  * Primary: at the published critical point, do the fitted delta and theta land within 20% of
    Jensen's values (delta=0.159464, theta=0.313686)?
      - yes -> the pipeline is calibrated; its LM numbers can be read as measurements.
      - no  -> the pipeline is the problem, phase 1's LM slopes are uninterpretable, and no GPU
               budget should be spent on the LM extraction until this is fixed.
  * Secondary: does the pipeline LOCATE the critical point? Scan p1 around the published value and
    check that the theta-crossing bracket contains it. That is exactly the operation phase 1 ran
    on temperature, so a failure here would invalidate phase 1's bracket too.
  * Hyperscaling theta = 1/z - 2*delta = 0.313685 (d=1) is reported as an internal check.
  * A sample-size ladder is run (64 -> 512 -> 4096 -> 32768 replicas) so the answer to "was phase
    1 simply under-sampled at 64?" is measured rather than asserted. Phase 1 used 64.

WHICH CRITICAL POINT. The DK p2=0 damage-spreading line is disputed in the literature -- 0.801(2)
(Zebende & Penna) vs 0.8087(5) (Hinrichsen et al.), a disagreement the paper already reports and
declines to adjudicate. This script scans a range covering both rather than assuming either, so
the located bracket is an output, not an input.

Writes results/dp_pipeline_validation.json. No model, no GPU, no network.
Usage:  .venv/bin/python experiments/dp_pipeline_validation.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time
import numpy as np

from dk import dk_run
from provenance import stamp, rel

DP = dict(delta=0.159464, theta=0.313686, z=1.580745)
P1_SCAN = [0.780, 0.795, 0.8087, 0.820, 0.835]     # spans both disputed published values
P1_CRIT = 0.8087                                   # Hinrichsen et al.; the primary test point
N_SITES, SWEEPS = 512, 200
LADDER = [64, 512, 4096, 32768]                    # phase 1 used 64
FIT_FROM = 5                                       # same transient cut as phase 1
TOL = 0.20                                         # pre-registered: within 20% of Jensen
OUT = str(_ROOT / "results" / "dp_pipeline_validation.json")


def survival(p1, replicas, seed, sweeps=SWEEPS, n=N_SITES):
    """P(t) and N(t) from a single-site seed, DP convention, batched over `replicas`."""
    P = np.zeros(sweeps + 1); Nt = np.zeros(sweeps + 1); done = 0
    chunk = 512
    while done < replicas:
        b = min(chunk, replicas - done)
        s0 = np.zeros((b, n), dtype=np.int8); s0[:, n // 2] = 1
        u = np.random.default_rng(seed + done).random(sweeps * n * b)
        a = np.asarray(dk_run(s0, u, p1=p1, p2=0.0, sweeps=sweeps))   # (sweeps+1, b, n)
        cnt = a.sum(axis=2)
        P += (cnt > 0).sum(axis=1); Nt += cnt.sum(axis=1)
        done += b
    return P / replicas, Nt / replicas


def _slope(t, y):
    """Log-log slope, the SAME estimator phase 1 used, so this validates that code path."""
    ok = y > 0
    if ok.sum() < 4:
        return None, None
    lt, ly = np.log(t[ok]), np.log(y[ok])
    c = np.polyfit(lt, ly, 1)
    r2 = 1 - np.sum((ly - np.polyval(c, lt)) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-12)
    return float(c[0]), float(r2)


def main():
    res = {"_preregistration": dict(
        dp_targets=DP, p1_scan=P1_SCAN, p1_crit=P1_CRIT, N=N_SITES, sweeps=SWEEPS,
        ladder=LADDER, fit_from=FIT_FROM, tolerance=TOL,
        primary="do delta and theta land within 20% of Jensen at the published critical point?",
        secondary="does the theta-crossing bracket contain the published critical point?",
        failure_means="the pipeline is the problem; phase 1's LM slopes are uninterpretable")}

    print(f"DP pipeline validation on Domany-Kinzel (no model, no GPU)")
    print(f"targets: delta={DP['delta']}, theta={DP['theta']}, "
          f"hyperscaling 1/z-2delta={1/DP['z'] - 2*DP['delta']:.6f}\n")

    # --- 1. sample-size ladder at the published critical point ----------------------------
    print(f"=== sample-size ladder at p1={P1_CRIT} (phase 1 used 64 replicas) ===")
    print(f"  {'replicas':>9} {'-delta':>9} {'R2':>7} {'+theta':>9} {'R2':>7} {'secs':>7}")
    ladder = {}
    for m in LADDER:
        t0 = time.time()
        P, Nt = survival(P1_CRIT, m, seed=1000)
        t = np.arange(len(P), dtype=float); msk = t >= FIT_FROM
        sd, r2d = _slope(t[msk], P[msk])
        st, r2t = _slope(t[msk], Nt[msk])
        dt = time.time() - t0
        ladder[str(m)] = dict(delta=(None if sd is None else round(-sd, 4)),
                              r2_delta=(None if r2d is None else round(r2d, 4)),
                              theta=(None if st is None else round(st, 4)),
                              r2_theta=(None if r2t is None else round(r2t, 4)),
                              secs=round(dt, 1))
        print(f"  {m:>9} {-sd:>9.4f} {r2d:>7.4f} {st:>+9.4f} {r2t:>7.4f} {dt:>7.1f}")

    big = ladder[str(LADDER[-1])]
    d_err = abs(big["delta"] - DP["delta"]) / DP["delta"]
    t_err = abs(big["theta"] - DP["theta"]) / DP["theta"]
    print(f"\n  at {LADDER[-1]} replicas: delta off by {d_err*100:.1f}%, theta off by {t_err*100:.1f}% "
          f"(tolerance {TOL*100:.0f}%)")

    # --- 2. can it LOCATE the critical point? ----------------------------------------------
    print(f"\n=== theta across p1 (the same bracket operation phase 1 ran on temperature) ===")
    print(f"  {'p1':>7} {'+theta':>9} {'-delta':>9} {'P(end)':>8}")
    scan = {}
    for p1 in P1_SCAN:
        P, Nt = survival(p1, 4096, seed=2000)
        t = np.arange(len(P), dtype=float); msk = t >= FIT_FROM
        sd, _ = _slope(t[msk], P[msk]); st, _ = _slope(t[msk], Nt[msk])
        scan[str(p1)] = dict(theta=round(st, 4), delta=round(-sd, 4), P_end=round(float(P[-1]), 4))
        print(f"  {p1:>7} {st:>+9.4f} {-sd:>9.4f} {P[-1]:>8.4f}")
    pts = sorted((float(k), v["theta"]) for k, v in scan.items())
    bracket = None
    for (a, ta), (b, tb) in zip(pts, pts[1:]):
        if ta < DP["theta"] <= tb:
            bracket = (a, b)
    contains = bool(bracket and bracket[0] <= P1_CRIT <= bracket[1])
    print(f"\n  theta crosses {DP['theta']:+.4f} in {bracket};  contains published "
          f"{P1_CRIT}? {contains}")

    calibrated = d_err <= TOL and t_err <= TOL
    if calibrated and contains:
        verdict = (f"PIPELINE CALIBRATED: at {LADDER[-1]} replicas it recovers delta and theta "
                   f"to within {max(d_err, t_err)*100:.1f}% of Jensen on a system whose answer is "
                   f"known, and its bracket contains the published critical point. Phase 1's LM "
                   f"slopes are readable as measurements, and GPU budget for the LM extraction "
                   f"would buy signal rather than noise.")
    elif calibrated:
        verdict = (f"EXPONENTS OK, LOCATION NOT: the fitted values are within tolerance at the "
                   f"known critical point, but the theta-crossing bracket {bracket} does not "
                   f"contain it. Phase 1's LM BRACKET is therefore the untrustworthy part, not "
                   f"its slopes.")
    else:
        verdict = (f"PIPELINE NOT CALIBRATED: at {LADDER[-1]} replicas on data whose exponents "
                   f"are known, delta is off by {d_err*100:.1f}% and theta by {t_err*100:.1f}%. "
                   f"Phase 1's LM slopes are uninterpretable and no GPU budget should be spent on "
                   f"the LM extraction until the estimator is fixed.")
    print(f"\n  -> {verdict}")

    res["ladder_at_critical"] = ladder
    res["p1_scan"] = scan
    res["theta_crossing_bracket"] = list(bracket) if bracket else None
    res["bracket_contains_published_pc"] = contains
    res["delta_rel_error"] = round(d_err, 4)
    res["theta_rel_error"] = round(t_err, 4)
    res["calibrated"] = bool(calibrated)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Validation of the DP extraction pipeline against Domany-Kinzel, where the exponents are "
        "known to be the 1+1D DP values and F38 already established our DK simulator is bit-exact "
        "against an independent implementation. So any discrepancy here is attributable to the "
        "FITTING procedure, which had never been calibrated -- phase 1 applied it to a language "
        "model and reported slopes without ever checking it can recover a known answer. DK is "
        "pure numpy, so this runs at literature sample sizes in minutes with no GPU, whereas the "
        "identical measurement on the LM costs ~152s per 16 replicas. The sample-size ladder "
        "answers 'was phase 1 simply under-sampled at 64?' by measurement. The p1 scan spans both "
        "disputed published values for the p2=0 line -- 0.801(2) and 0.8087(5) -- so the located "
        "bracket is an output rather than an assumption.")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {rel(OUT)}")


if __name__ == "__main__":
    main()
