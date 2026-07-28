"""Is the damage-spreading transition directed percolation? At a geometry that can answer (#82).

WHAT THIS DECIDES. The paper measures lambda_ca, whose sign separates a frozen phase (a one-site
perturbation dies) from a chaotic one (it spreads). This asks whether that sign change is a
CRITICAL POINT WITH EXPONENTS rather than a smooth crossover with a threshold chosen by eye. If
the transition is directed percolation, lambda_ca stops being a fitted heuristic and becomes an
order parameter for a known universality class, and the edge-of-chaos language acquires exponents
instead of metaphor. If it is not DP, that is a negative from an instrument that has been shown to
work -- which is the standard this project holds itself to after the audit.

WHY N=192 OVER 120 SWEEPS, AND NOT THE CHEAPER GRID. Phase 2 (dp_narrow_bracket) ran N=96 over 40
sweeps and reported "not in the DP class". That verdict was withdrawn (F56): the tolerance behind
it was measured on Domany-Kinzel at N=512 over 200 sweeps, and at the geometry actually used the
same estimator recovers DK's delta to only 21.2 +/- 10.0% -- on data that provably IS directed
percolation. The threshold rejected DK itself, so it could not have been rejecting the model.

This geometry is the cheapest on the scanned ladder that DEMONSTRABLY recovers the known answer,
where "demonstrably" means the deviation plus its own seed-to-seed spread clears the 20% that
dp_pipeline_validation pre-registered:

    N=96  sweeps=40      delta 21.2 +/-10.0%   theta  7.0 +/- 6.6%   fails   (phase 2)
    N=96  sweeps=200     delta  8.7 +/- 8.5%   theta 18.2 +/- 4.0%   fails
    N=192 sweeps=80      delta 13.1 +/- 7.5%   theta 10.5 +/- 5.3%   fails
    N=192 sweeps=120     delta  9.8 +/- 8.1%   theta  9.2 +/- 5.6%   PASSES  <- this run
    N=192 sweeps=200     delta  5.4 +/- 8.9%   theta  8.4 +/- 5.9%   passes, 1.7x the cost

The gate is re-measured inline here rather than trusted from that table, and it is evaluated on DK
alone -- blind to the LM numbers -- so it cannot be tuned to the answer.

THE TEMPERATURE GRID, AND WHY MOVING IT IS NOT GRID-SHOPPING. Phase 2 ran {0.400, 0.425, 0.450}.
Two changes, both registered here before the first run:

  * 0.400 is DROPPED. Its theta fit had R^2 = 0.0066 -- the active count is flat noise, not a
    power law. It is in the absorbing phase and contributes no slope to fit.
  * 0.475 and 0.500 are ADDED, because delta's crossing is not bracketed by the old grid and
    theta's is. Phase 2 gave delta = 0.4573, 0.2789, 0.2024 at T = 0.400, 0.425, 0.450 -- monotone,
    decelerating, and still above Jensen's 0.159464 at the top of the grid; extrapolating the
    decay puts its crossing near T ~ 0.48-0.49. theta already crosses inside (0.425, 0.450).

    Extending a grid after a null is what p-hacking looks like, so the distinction has to be
    stated, as it was for #87. This is not searching a window that already contains the phenomenon
    until something turns up. It is widening a window that provably fails to bracket one of the two
    crossings the test compares -- and a test of whether two crossings coincide is vacuous when one
    of them lies outside the grid. The extension is registered before any run, not after a null.

THE PRIMARY TEST IS BRACKET OVERLAP, NOT PER-POINT TOLERANCE. At a genuine DP critical point delta
and theta reach their DP values at the SAME temperature. Phase 2 asked whether some grid point had
both exponents within tolerance, which conflates two things: whether the crossings coincide, and
whether a grid point happens to sit near them. Here each exponent's crossing temperature is
estimated by bootstrap over replicas (resample, refit, interpolate the crossing), and the question
is whether the two intervals OVERLAP. That is the physical statement, and it is robust to a coarse
grid in a way a per-point test is not. Resampling replicas is legitimate only because of the
per_replica visit order described below -- under the old shared order the seed was the unit.

PRE-REGISTERED BEFORE RUNNING:
  * Gate: if the inline DK calibration at N=192/120 does not clear 20% including its spread, the
    result is NOT DECIDABLE and neither DP nor not-DP may be claimed. Checked first.
  * Primary: do the bootstrap 90% intervals for T_c(delta) and T_c(theta) overlap?
      - overlap  -> DP-CONSISTENT; quote the exponents at the overlap with the measured bias.
      - disjoint -> NOT DP; the two exponents demand different critical points.
      - either crossing unbracketed by the grid -> report that, do not extrapolate past it.
  * Secondary: is there a grid temperature where both exponents sit within twice the measured
    bias? Reported for continuity with phase 2, not used to decide.
  * Hyperscaling: theta = 1/z - 2*delta with DP's z is reported as an internal check on the
    fitted pair. z is not measured here, so this constrains the pair rather than testing it.
  * A null is publishable. The instrument being validated is the point, not the sign.
  * Exponents are quoted with bootstrap intervals over REPLICAS, which is legitimate only
    under per_replica ordering. If that flag is ever removed, the interval must revert to a
    seed-level bootstrap or it understates the error ~8x (F57).

EVERY REPLICA GETS ITS OWN VISIT ORDER (F57). The first attempt at this run returned zero damage
in 20 of 20 cells across all four temperatures, and the cause was not the model. The AR rule is
causal-left, so damage seeded at site j survives its first sweep only if j+1 or j+2 is visited
before j; otherwise j resamples against an identical context with the same uniform, heals, and
the run is absorbed. That is 1/3 of visit orders -- and `lattice.run` drew ONE order per sweep for
the whole batch, so it killed all 64 replicas at once instead of a third of them. Predicting
deaths from the permutation alone matched observation exactly (8/8 seeds at N=96, 5/5 at N=192).

The consequence outlived that run. Phase 2 pooled 512 replicas as independent when the quantity
deciding each outcome was drawn once per batch, so the real independent unit was the seed. Re-read
with the order as the unit, phase 2's T=0.450 gives delta = 0.2074 +/- 0.0373 and theta = 0.4075
+/- 0.0789 -- 1.3 and 1.2 standard errors from Jensen. Consistent with directed percolation. The
"27% discrepancy" that F56 spent its effort explaining was an error bar computed 8x too small.

So this run uses `order="per_replica"`, opt-in in lattice.py with the shared default untouched so
no existing number moves. Each replica now draws its own order, making 512 replicas 512
independent draws; the twins still share an order stream, because CRN coupling requires the pair
to be visited in the same sequence. This also repairs a mismatch against the calibration that was
never noticed: Domany-Kinzel is synchronous with genuinely independent replicas, so the DK ladder
was measuring a precision the LM run could not have had.

BATCH SIZE STAYS AT 64, ON ARITHMETIC RATHER THAN A NEW MEASUREMENT. A probe of B in
{64, 96, 128, 192} at this N was attempted and produced nothing usable: its output was block
buffered through a pipe and lost when the run was killed after stalling at high B with swap
nearly exhausted. That failure is worth recording as the negative it is -- B >= 128 at N=192 was
not viable on this machine -- but it is not a measurement, and the plan should not pretend it is.
The decision rests instead on the two points measured at N=96, where per-run time fits
t(B) = 83 + 4.42*B seconds: the fixed 83 s is already amortised by B=64 (5.41 s per replica),
and B=96 would return 5.28 and B=128 5.07. Two to six percent is not worth the instability that
probe demonstrated, so this runs at the same B as phase 2 -- which also keeps the two geometries
differing in geometry alone.

SEED-MAJOR ORDER. The loop runs seed-outer, temperature-inner, so an interruption leaves every
temperature with the same number of seeds rather than some temperatures complete and others empty.
At ~18 h that is the difference between a readable partial answer and none. Phase 2 ran
temperature-major and was interrupted repeatedly.

RESUMABLE BY CONSTRUCTION. Every completed run is written immediately, keyed by (T, seed); a
re-run skips what is already there and prints how much it found.

Writes results/dp_class_n192.json.
Usage:  caffeinate -dimsu .venv/bin/python experiments/dp_class_n192.py
        (safe to interrupt and re-run -- it resumes)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from dp_calibration import DP, CAL_TOL, calibrate, print_ladder, slope as _slope

BASE, REVISION = "EleutherAI/pythia-410m", "step143000"
TEMPS = [0.425, 0.450, 0.475, 0.500]   # brackets BOTH crossings; see the grid note above
SEEDS = [41, 42, 43, 44, 45, 46, 47, 48]
N, B, R = 192, 64, 2
SETTLE, SWEEPS = 8, 120
FIT_FROM = 5
REPLICAS = B * len(SEEDS)
BOOT = 2000                            # bootstrap resamples of the seed set
CI = 90                                # percent, two-sided
CAL_GRID = [(96, 40), (96, 200), (192, 80), (192, 120), (192, 200), (384, 200)]
OUT = str(_ROOT / "results" / "dp_class_n192.json")


def trajectory(rule, T, seed):
    """Per-replica damaged-site counts over time from a single-site seed. (sweeps, B).

    Identical protocol to dp_narrow_bracket, so the only difference between the two runs is the
    geometry under test. Both replicas share model, init, update order and uniform stream; only
    the one flipped site differs, which is the project's CRN coupling.
    """
    from ar_ca import run
    base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none",
               init="random", seed=seed, order="per_replica")["final"]
    flipped = base.copy()
    flipped[:, N // 2] = np.random.default_rng(seed).choice(rule.init_pool, size=B)
    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
    # F57: one visit order per REPLICA, so B replicas are B independent draws of the thing that
    # decides whether the seed ignites. The twins must still share their order exactly, or the
    # CRN coupling breaks and every site diverges for bookkeeping reasons -- hence the explicit
    # stream, tiled across the twin halves the same way u2 is.
    perm = np.argsort(np.random.default_rng(seed + 3).random((SWEEPS, B, N)), axis=2)
    perm2 = np.concatenate([perm, perm], axis=1)
    init2 = np.concatenate([base, flipped], axis=0)
    snaps = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
                init_state=init2, seed=seed + 2, u_stream=u2,
                order="per_replica", order_stream=perm2)["snaps"]
    # drop the pre-dynamics state: `snaps` opens with the initial lattice, so keeping it would
    # index the un-evolved configuration as t=1 and misalign the time axis against the DK
    # calibration, which fits t=1..sweeps of POST-update states
    return (snaps[1:, :B] != snaps[1:, B:]).sum(axis=2)


def _fit(counts):
    """(delta, theta) from pooled per-replica counts, the shared estimator."""
    t = np.arange(1, counts.shape[0] + 1, dtype=float)
    m = t >= FIT_FROM
    sd, r2d = _slope(t[m], (counts > 0).mean(axis=1)[m])
    st, r2t = _slope(t[m], counts.mean(axis=1)[m])
    return (None if sd is None else -sd), r2d, st, r2t


def _crossing(temps, vals, target, rising):
    """Temperature where a monotone exponent curve crosses its DP value, linearly interpolated.

    Returns None when the grid does not bracket the crossing -- which is reported, never
    extrapolated past, because an extrapolated crossing is the grid's shape and not a measurement.
    """
    for (ta, va), (tb, vb) in zip(zip(temps, vals), list(zip(temps, vals))[1:]):
        if va is None or vb is None:
            continue
        if (rising and va < target <= vb) or (not rising and va > target >= vb):
            if vb == va:
                return float(ta)
            return float(ta + (tb - ta) * (target - va) / (vb - va))
    return None


def _bootstrap(by_seed, rng):
    """90% intervals for T_c(delta) and T_c(theta) by resampling REPLICAS.

    Under order='per_replica' every replica carries its own visit order, its own settled base
    state and its own uniform stream, so the replica is the independent unit and 512 of them are
    512 draws. That was NOT true before F57: one visit order decided the entire batch, the real
    independent unit was the seed, and pooling 512 as though independent shrank the error bars
    by enough to turn a 1.3-sigma agreement with DP into an apparent 27% discrepancy. Resampling
    seeds here would now be the mirror-image error -- throwing away the independence the fix
    bought and quoting an interval ~8x too wide.
    """
    pooled = {}
    for T in TEMPS:
        cs = [by_seed[(T, s)] for s in SEEDS if (T, s) in by_seed]
        if cs:
            pooled[T] = np.concatenate(cs, axis=1)
    dxs, txs = [], []
    for _ in range(BOOT):
        dv, tv = [], []
        for T in TEMPS:
            c = pooled.get(T)
            if c is None:
                dv.append(None); tv.append(None); continue
            d, _, th, _ = _fit(c[:, rng.integers(0, c.shape[1], c.shape[1])])
            dv.append(d); tv.append(th)
        cd = _crossing(TEMPS, dv, DP["delta"], rising=False)
        ct = _crossing(TEMPS, tv, DP["theta"], rising=True)
        if cd is not None:
            dxs.append(cd)
        if ct is not None:
            txs.append(ct)
    lo, hi = (100 - CI) / 2, 100 - (100 - CI) / 2
    def iv(xs):
        if len(xs) < BOOT * 0.5:      # crossing unbracketed in most resamples -> not measured
            return None
        return [round(float(np.percentile(xs, lo)), 4), round(float(np.percentile(xs, hi)), 4)]
    return iv(dxs), iv(txs), len(dxs) / BOOT, len(txs) / BOOT


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=BASE, revision=REVISION, temps=TEMPS, seeds=SEEDS, N=N, B=B, r=R,
        settle=SETTLE, sweeps=SWEEPS, fit_from=FIT_FROM, dp_targets=DP,
        replicas_per_temperature=REPLICAS, bootstrap=BOOT, ci_percent=CI,
        geometry_justification="cheapest geometry on the ladder that recovers DK's known "
                               "exponents within 20% INCLUDING its seed spread (F56)",
        grid_justification="0.400 dropped (theta fit R^2=0.007, absorbing phase); 0.475/0.500 "
                           "added because delta's crossing is not bracketed by the old grid "
                           "while theta's is -- registered before the first run, not after a null",
        primary="do the bootstrap 90% intervals for T_c(delta) and T_c(theta) overlap?",
        gate=f"if the inline DK calibration at N={N}/{SWEEPS} does not clear {CAL_TOL:.0f}% "
             f"including its spread, the result is NOT DECIDABLE and no DP claim is licensed",
        order="seed-major, so an interruption leaves every temperature equally sampled",
        visit_order="per_replica (F57): each replica draws its own, so 512 replicas are 512 "
                    "independent draws; twins share an order stream to preserve CRN",
        bootstrap_unit="replica -- valid only because of the per_replica order",
        resumable="every completed run is saved immediately and keyed by (T, seed)")
    runs = res["runs"]
    from ar_ca import ARRule
    rule = ARRule(BASE, revision=REVISION)
    todo = [(T, sd) for sd in SEEDS for T in TEMPS]        # seed-major
    done0 = len([v for v in runs.values() if "counts" in v])
    print(f"DP class test at N={N}, {SWEEPS} sweeps: {len(todo)} runs "
          f"({len(TEMPS)} temps x {len(SEEDS)} seeds), B={B} -> {REPLICAS} replicas per "
          f"temperature", flush=True)
    if done0:
        print(f"  resuming: {done0}/{len(todo)} already complete", flush=True)
    try:
        for k, (T, sd) in enumerate(todo, 1):
            key = f"T{T}_s{sd}"
            if key in runs:
                continue
            t0 = time.time()
            traj = trajectory(rule, T, sd)
            runs[key] = dict(T=T, seed=sd, N=N, B=B, sweeps=SWEEPS,
                             counts=traj.astype(int).tolist(),
                             secs=round(time.time() - t0, 1))
            alive = (traj > 0).mean(axis=1)
            print(f"[{k}/{len(todo)}] {key}: P(end)={alive[-1]:.3f} "
                  f"sites(end)={traj.mean(axis=1)[-1]:.2f} ({runs[key]['secs']}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)      # save-per-run: this is the resume point
            del traj, alive
            try: torch.mps.empty_cache()
            except Exception: pass
            gc.collect()
    finally:
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    have = len([v for v in runs.values() if "counts" in v])
    if have < len(todo):
        print(f"\npartial: {have}/{len(todo)} -- re-run this command to continue", flush=True)
        json.dump(res, open(OUT, "w"), indent=1); return
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = [v for v in res["runs"].values() if "counts" in v]
    by_seed = {(v["T"], v["seed"]): np.array(v["counts"]) for v in runs}

    print(f"\n=== {REPLICAS} replicas per temperature, fitted from sweep {FIT_FROM} ===")
    print(f"  {'T':>7} {'reps':>6} {'-delta':>9} {'R2':>7} {'+theta':>9} {'R2':>7} {'P(end)':>8}")
    out, dv, tv = {}, [], []
    for T in TEMPS:
        cs = [by_seed[(T, s)] for s in SEEDS if (T, s) in by_seed]
        if not cs:
            dv.append(None); tv.append(None); continue
        c = np.concatenate(cs, axis=1)
        d, r2d, th, r2t = _fit(c)
        dv.append(d); tv.append(th)
        P_end = float((c > 0).mean(axis=1)[-1])
        out[str(T)] = dict(
            replicas=int(c.shape[1]), P_end=round(P_end, 4),
            delta=(None if d is None else round(d, 4)),
            r2_delta=(None if r2d is None else round(r2d, 4)),
            theta=(None if th is None else round(th, 4)),
            r2_theta=(None if r2t is None else round(r2t, 4)),
            delta_dev_pct=(None if d is None else round(abs(d - DP["delta"]) / DP["delta"] * 100, 1)),
            theta_dev_pct=(None if th is None else round(abs(th - DP["theta"]) / DP["theta"] * 100, 1)),
            # hyperscaling with DP's z: constrains the pair, does not test it (z is not measured)
            theta_from_hyperscaling=(None if d is None
                                     else round(1 / DP["z"] - 2 * d, 4)))
        # a None slope is possible (an all-dead temperature has no power law to fit) and must not
        # crash the print AFTER the whole run has been collected
        def col(x, w, spec):
            return format(x, spec) if x is not None else f"{'--':>{w}}"
        print(f"  {T:>7} {c.shape[1]:>6} {col(d, 9, '9.4f')} {col(r2d, 7, '7.4f')} "
              f"{col(th, 9, '+9.4f')} {col(r2t, 7, '7.4f')} {P_end:>8.4f}")

    secs = [v["secs"] for v in runs if "secs" in v]
    cal = calibrate(N, SWEEPS, REPLICAS, FIT_FROM, CAL_GRID,
                    float(np.median(secs)) if secs else 2076.0, len(TEMPS) * len(SEEDS))
    here = cal["at_run_geometry"]
    print_ladder(cal, N, SWEEPS)

    d_iv, t_iv, d_frac, t_frac = _bootstrap(by_seed, np.random.default_rng(20260727))
    print(f"\n=== PRIMARY: do the two crossings agree on one T_c? ({BOOT} seed bootstraps) ===")
    print(f"  T_c from delta crossing {DP['delta']}: "
          f"{d_iv if d_iv else 'NOT BRACKETED BY THE GRID'}  (bracketed in {d_frac*100:.0f}%)")
    print(f"  T_c from theta crossing {DP['theta']}: "
          f"{t_iv if t_iv else 'NOT BRACKETED BY THE GRID'}  (bracketed in {t_frac*100:.0f}%)")
    overlap = bool(d_iv and t_iv and d_iv[0] <= t_iv[1] and t_iv[0] <= d_iv[1])
    if d_iv and t_iv:
        print(f"  intervals overlap? {overlap}")

    tol_d, tol_t = here["delta_pct"] * 2, here["theta_pct"] * 2
    both = [k for k, v in out.items()
            if v["delta_dev_pct"] is not None and v["delta_dev_pct"] <= tol_d
            and v["theta_dev_pct"] is not None and v["theta_dev_pct"] <= tol_t]
    print(f"\n  SECONDARY (phase-2 style, not used to decide): grid temperatures with both "
          f"exponents inside {tol_d:.0f}%/{tol_t:.0f}%: {both or 'none'}")

    if not cal["geometry_decides"]:
        cp = cal["cheapest_passing"]
        verdict = (f"NOT DECIDABLE AT THIS GEOMETRY: the inline calibration at N={N}/{SWEEPS} "
                   f"recovers DK's known delta to {here['delta_pct']}+/-{here['delta_sd_pct']}% "
                   f"and theta to {here['theta_pct']}+/-{here['theta_sd_pct']}%, which does not "
                   f"demonstrate {CAL_TOL:.0f}%. The ladder said this geometry would pass; "
                   f"re-measured here it does not, so no DP claim is licensed. "
                   + (f"N={cp['N']} over {cp['sweeps']} sweeps does, at a projected "
                      f"{cp['projected_hours']} h." if cp else "No scanned geometry reaches it."))
    elif not (d_iv and t_iv):
        miss = "delta" if not d_iv else "theta"
        verdict = (f"INCONCLUSIVE -- {miss}'s crossing is not bracketed by T in {TEMPS} in a "
                   f"majority of bootstraps. The geometry is adequate but the temperature grid "
                   f"does not contain the crossing, so the two cannot be compared. Extend the "
                   f"grid in the direction {miss} is moving; do not extrapolate past it.")
    elif overlap:
        lo, hi = max(d_iv[0], t_iv[0]), min(d_iv[1], t_iv[1])
        verdict = (f"DP-CONSISTENT: delta and theta reach their DP values at overlapping "
                   f"temperatures, T_c in [{lo:.4f}, {hi:.4f}]. delta's {CI}% interval is {d_iv} "
                   f"and theta's is {t_iv}. The estimator was shown at this geometry to recover "
                   f"DK's known exponents to {here['delta_pct']}+/-{here['delta_sd_pct']}% and "
                   f"{here['theta_pct']}+/-{here['theta_sd_pct']}%, so the agreement is a "
                   f"measurement rather than a fit-window artifact. These are the first LM "
                   f"exponents worth quoting against directed percolation, with that bias stated.")
    else:
        verdict = (f"NOT DIRECTED PERCOLATION: delta and theta demand different critical "
                   f"temperatures -- delta's {CI}% interval is {d_iv}, theta's is {t_iv}, and "
                   f"they are disjoint. At a DP critical point both reach their DP values at the "
                   f"same T. The estimator was shown at this geometry to recover DK's known "
                   f"exponents to {here['delta_pct']}+/-{here['delta_sd_pct']}% and "
                   f"{here['theta_pct']}+/-{here['theta_sd_pct']}%, so unlike phase 2 this "
                   f"separation is not attributable to the fit window. What class it IS remains "
                   f"open; that needs the finite-size scaling half of #82 (nu_perp, z).")
    print(f"\n  -> {verdict}")

    res["analysis"] = out
    res["calibration_at_run_geometry"] = cal
    res["crossing"] = dict(delta_ci=d_iv, theta_ci=t_iv,
                           delta_bracketed_frac=round(d_frac, 4),
                           theta_bracketed_frac=round(t_frac, 4),
                           overlap=overlap, ci_percent=CI, bootstraps=BOOT)
    res["secondary_both_within_tolerance"] = both
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Does the token-lattice CA's damage-spreading transition sit in the directed percolation "
        "universality class? Run at N=192 over 120 sweeps -- the cheapest geometry on the scanned "
        "ladder that demonstrably recovers Domany-Kinzel's KNOWN exponents within the 20% "
        "dp_pipeline_validation pre-registered, counting the seed-to-seed spread. Phase 2 ran "
        "N=96/40, where the same estimator misses DK's delta by a fifth, and its 'not in the DP "
        "class' verdict was withdrawn as a fit-window artifact (F56). The temperature grid drops "
        "0.400 (theta fit R^2=0.007, absorbing phase) and adds 0.475/0.500 because delta's "
        "crossing was not bracketed by the phase-2 grid while theta's was -- a test of whether two "
        "crossings coincide is vacuous when one lies outside the grid. The primary test is whether "
        "the bootstrap intervals for the two crossing temperatures OVERLAP, which is the physical "
        "statement and is robust to a coarse grid; the phase-2 per-point tolerance test is "
        "reported alongside but does not decide. The calibration gate is evaluated on DK alone, "
        "blind to the LM numbers, so it cannot be tuned to the answer.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
