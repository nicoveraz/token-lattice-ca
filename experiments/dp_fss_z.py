"""Finite-size scaling at T_c: does the dynamic exponent z match directed percolation? (#82)

WHAT THIS ADDS TO F58. F58 showed delta and theta reach their DP values at the same temperature,
T_c in [0.4343, 0.4391]. That is two exponents agreeing at one point. A universality class is
conventionally assigned on more than that, and the next independent number is the DYNAMIC exponent
z, which F58's geometry could not touch: it says how the correlation length grows in time, and it
is only visible once the damage cone REACHES the system boundary.

That is the design point, and it is not "the same run at larger N". The cone radius grows as
t^(1/z), so it reaches the boundary at t ~ (N/2)^z:

    N= 12  ->    17 sweeps        N= 48  ->   152 sweeps
    N= 24  ->    51 sweeps        N=192  ->  1361 sweeps   (F58 ran 120 -- deliberately far below)

F58 avoided that regime on purpose, because finite-size contamination is exactly what would have
faked an exponent there. This run needs the opposite: LONG time at SMALL lattices, so the cutoff
is inside the window.

THE LADDER WAS CHOSEN BY CALIBRATION, AND THE INTUITIVE CHOICE FAILED. The obvious ladder --
{24, 48, 96}, biggest affordable lattices -- cannot recover DK's known z at all, and no amount of
compute rescues it. Measured on DK, which is free:

    ladder         window   recovered z (16 fits)      deviation      gate    LM cost
    {24,48,96}      3x      1.4419 +/- 0.272        8.8% +/-17.2%     FAIL      29 h
    {24,48,96}     12x      1.3800 +/- 0.200       12.7% +/-12.7%     FAIL     117 h
    {12,24,48}      3x      1.6513 +/- 0.193        4.5% +/-12.2%     pass       5 h
    {12,24,48}     12x      1.5706 +/- 0.121        0.6% +/- 7.6%     PASS      20 h   <- this run

Raising replicas 32x on the failing ladder moved nothing (1.363 -> 1.372, spread flat), so it is
not a statistics problem, and lengthening the window 8x moved nothing either. Smaller lattices
working BETTER is backwards from a corrections-to-scaling story and this file does not have an
explanation for it; what it has is a validated estimator -- exact to four digits on synthetic
curves built with a known z -- and a ladder measured to recover the right answer on a system whose
answer is known. That is the project's standard, and it is the only claim being made here.

N=192 is excluded anyway at ~143 h on its own. The ladder here is ~20 h in overnight batches.

HOW z IS EXTRACTED. At the critical point the survival probability obeys the scaling form
P(t) = t^(-delta) F(t / N^z). So plotting P(t)*t^delta against t/N^z collapses every lattice size
onto one curve, and z is whatever value makes the collapse tightest. z is fitted by minimising the
spread of the collapsed curves; delta is FIXED at Jensen's value, which makes this a test of z
GIVEN DP's delta rather than a two-parameter fit that could absorb an error in either.

THE COLLAPSE ESTIMATOR IS ITSELF CALIBRATED FIRST, ON DOMANY-KINZEL. This is the same discipline
F56 was written after: a collapse can look convincing and still return the wrong exponent, and a
tight-looking collapse is persuasive in a way a bad slope is not. So the identical routine is run
on DK at its known critical point over the SAME N ladder and the SAME time multiplier, where the
answer is z = 1.580745, before it is pointed at the model. If it cannot recover z there, this run
reports NOT DECIDABLE and no z is quoted. The gate is evaluated on DK alone, blind to the LM.

WHAT THIS DOES NOT DELIVER. nu_perp needs off-critical temperatures -- the collapse in
(T - T_c) * N^(1/nu_perp) -- which multiplies the cost by the number of temperatures and is not
affordable here. This is the z half of the FSS work. It also runs at a SINGLE temperature, T_c =
0.436 from F58's overlap; the residual uncertainty on T_c (+/-0.0024, the overlap half-width) is a
systematic on z that this design cannot separate, and it is reported rather than hidden. Adding
temperatures later is an edit to TEMPS plus a re-run -- the file resumes.

A NOTE ON CIRCULARITY. The time window per lattice is set to 12 * (N/2)^z using DP's z, and the
collapse is scored only within |log(t/N^z)| < 2.5. Both choose WHERE TO LOOK, not the answer: the
window has to contain the cutoff, and the band exists because the power-law region collapses for
ANY z -- it contributes no information and dilutes the fit, which is why the first version of this
estimator was biased. The fit itself scans z freely over 1.10-2.31, and the recovered value is
never near either edge.

Every replica draws its own visit order (F57), so replicas are independent and the bootstrap
resamples them.

OVERNIGHT BATCHES. Pass a number of hours as the first argument to stop cleanly after the cell in
flight: `python experiments/dp_fss_z.py 8`. Cells are ordered smallest-N first, so a short first
night still returns complete small-lattice curves rather than fragments of the expensive one.

Writes results/dp_fss_z.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/dp_fss_z.py [max_hours]
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
from dp_calibration import DP, CAL_TOL, CAL_PC, CAL_SEEDS

BASE, REVISION = "EleutherAI/pythia-410m", "step143000"
SIZES = [12, 24, 48]                   # chosen by calibration, not by intuition -- see above
TEMPS = [0.436]                        # F58's overlap [0.4343, 0.4391], centre
TC, TC_UNC = 0.436, 0.0024             # the systematic this design cannot separate
SEEDS = [51, 52, 53, 54, 55, 56, 57, 58]
B, R, SETTLE = 64, 2, 8
WINDOW_MULT = 12.0                     # t_max = WINDOW_MULT * (N/2)^z_DP; spans well past the bend
BAND = 2.5                             # collapse is scored near the bend, where z is legible
FIT_FROM = 5
BOOT = 1000
CI = 90
Z_SCAN = np.arange(1.10, 2.31, 0.005)  # wide enough that the answer is not the grid's edge
OUT = str(_ROOT / "results" / "dp_fss_z.json")


def sweeps_for(n):
    """Time window per lattice: generous enough to contain the cutoff for any plausible z."""
    return int(np.ceil(WINDOW_MULT * (n / 2) ** DP["z"]))


def trajectory(rule, n, T, seed, sweeps):
    """Per-replica damaged-site counts, (sweeps, B). Same protocol as dp_class_n192."""
    from ar_ca import run
    base = run(rule, B=B, N=n, r=R, T=T, sweeps=SETTLE, scheme="none",
               init="random", seed=seed, order="per_replica")["final"]
    flipped = base.copy()
    flipped[:, n // 2] = np.random.default_rng(seed).choice(rule.init_pool, size=B)
    u = np.random.default_rng(seed + 1).random(sweeps * n * B)
    u2 = np.concatenate([u.reshape(sweeps * n, B)] * 2, axis=1).reshape(-1)
    perm = np.argsort(np.random.default_rng(seed + 3).random((sweeps, B, n)), axis=2)
    snaps = run(rule, B=2 * B, N=n, r=R, T=T, sweeps=sweeps, scheme="none",
                init_state=np.concatenate([base, flipped], axis=0), seed=seed + 2, u_stream=u2,
                order="per_replica", order_stream=np.concatenate([perm, perm], axis=1))["snaps"]
    return (snaps[1:, :B] != snaps[1:, B:]).sum(axis=2)      # drop the pre-dynamics row


# ----------------------------------------------------------------- the collapse estimator
def _collapse_cost(curves, z, delta):
    """Spread of P(t)*t^delta against t/N^z, pooled over lattices. Lower is a tighter collapse.

    Curves are compared on a shared log-x grid over the range every lattice actually covers, so
    the cost cannot be reduced by a z that simply slides the curves apart until they stop
    overlapping -- which is the way a naive collapse score is usually gamed.
    """
    xs, ys = [], []
    for n, t, P in curves:
        ok = P > 0
        if ok.sum() < 5:
            return None
        xs.append(np.log(t[ok] / n ** z)); ys.append(np.log(P[ok] * t[ok] ** delta))
    lo = max(max(x.min() for x in xs), -BAND); hi = min(min(x.max() for x in xs), BAND)
    if not (hi > lo):
        return None                                   # no shared support: reject this z
    grid = np.linspace(lo, hi, 60)
    stack = np.array([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    return float(np.mean(np.var(stack, axis=0)))


def fit_z(curves, delta):
    """z minimising the collapse cost, plus the cost curve so flatness is visible not assumed."""
    costs = [(z, _collapse_cost(curves, z, delta)) for z in Z_SCAN]
    costs = [(z, c) for z, c in costs if c is not None]
    if not costs:
        return None, None
    z_best = min(costs, key=lambda p: p[1])[0]
    return float(z_best), costs


def _curves_from(pool, delta):
    return [(n, np.arange(1, c.shape[0] + 1, dtype=float)[FIT_FROM - 1:],
             (c > 0).mean(axis=1)[FIT_FROM - 1:]) for n, c in sorted(pool.items())]


# ----------------------------------------------------------------- calibration on Domany-Kinzel
def _dk_curves(p1, seed):
    from dk import dk_run
    out = {}
    for n in SIZES:
        sw = sweeps_for(n)
        P = np.zeros(sw); done = 0
        reps = B * len(SEEDS)
        while done < reps:
            b = min(512, reps - done)
            s0 = np.zeros((b, n), dtype=np.int8); s0[:, n // 2] = 1
            u = np.random.default_rng(seed + done).random(sw * n * b)
            a = np.asarray(dk_run(s0, u, p1=p1, p2=0.0, sweeps=sw))[1:]
            P += (a.sum(axis=2) > 0).sum(axis=1); done += b
        out[n] = P / reps
    return out


def calibrate_collapse():
    """Can this collapse recover z on a system whose z is KNOWN? Blind to the LM numbers."""
    zs = []
    for p1 in CAL_PC:
        for s in CAL_SEEDS[:8]:                      # 8 seeds x 2 p_c = 16 fits; 4 was not enough
            d = _dk_curves(p1, seed=s)
            curves = [(n, np.arange(1, len(P) + 1, dtype=float)[FIT_FROM - 1:], P[FIT_FROM - 1:])
                      for n, P in sorted(d.items())]
            z, _ = fit_z(curves, DP["delta"])
            if z is not None:
                zs.append(z)
    if not zs:
        return None
    m, sd = float(np.mean(zs)), float(np.std(zs))
    pct = abs(m - DP["z"]) / DP["z"] * 100
    sd_pct = sd / DP["z"] * 100
    return dict(z=round(m, 4), z_sd=round(sd, 4), n=len(zs),
                z_pct=round(pct, 1), z_sd_pct=round(sd_pct, 1),
                decides=bool(pct + sd_pct <= CAL_TOL),
                criterion=f"mean deviation plus its spread must clear {CAL_TOL:.0f}%")


def main():
    budget_h = float(_sys.argv[1]) if len(_sys.argv) > 1 else None
    t_start = time.time()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=BASE, revision=REVISION, sizes=SIZES, temps=TEMPS, seeds=SEEDS, B=B, r=R,
        settle=SETTLE, fit_from=FIT_FROM, window_mult=WINDOW_MULT,
        sweeps_per_size={str(n): sweeps_for(n) for n in SIZES},
        T_c=TC, T_c_uncertainty=TC_UNC, dp_z=DP["z"],
        scaling_form="P(t) = t^-delta * F(t / N^z); delta FIXED at Jensen so this tests z alone",
        gate="the collapse must recover DK's known z to 20% including its spread, or NOT DECIDABLE",
        excluded="N=192 (~143 h alone); nu_perp (needs off-critical temperatures)",
        visit_order="per_replica (F57); bootstrap resamples replicas",
        resumable="every completed cell is saved immediately and keyed by (N, T, seed)")
    runs = res["runs"]
    from ar_ca import ARRule
    rule = ARRule(BASE, revision=REVISION)
    todo = [(n, T, sd) for n in SIZES for T in TEMPS for sd in SEEDS]   # smallest N first
    done0 = len([v for v in runs.values() if "counts" in v])
    print(f"FSS ladder for z: {len(todo)} cells, sizes {SIZES}, "
          f"sweeps {[sweeps_for(n) for n in SIZES]}, T={TEMPS}", flush=True)
    if done0:
        print(f"  resuming: {done0}/{len(todo)} already complete", flush=True)
    if budget_h:
        print(f"  batch budget: {budget_h} h -- will stop cleanly after the cell in flight",
              flush=True)
    stopped = False
    try:
        for k, (n, T, sd) in enumerate(todo, 1):
            key = f"N{n}_T{T}_s{sd}"
            if key in runs:
                continue
            if budget_h and (time.time() - t_start) / 3600 >= budget_h:
                print(f"\nbatch budget reached at {k - 1}/{len(todo)} -- re-run to continue",
                      flush=True)
                stopped = True
                break
            t0 = time.time()
            traj = trajectory(rule, n, T, sd, sweeps_for(n))
            runs[key] = dict(N=n, T=T, seed=sd, B=B, sweeps=int(traj.shape[0]),
                             counts=traj.astype(int).tolist(),
                             secs=round(time.time() - t0, 1))
            print(f"[{k}/{len(todo)}] {key}: P(end)={(traj[-1] > 0).mean():.3f} "
                  f"max_sites={int(traj.max())}/{n} ({runs[key]['secs']}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            del traj
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
        print(f"\npartial: {have}/{len(todo)}"
              f"{'' if stopped else ' -- re-run this command to continue'}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1); return
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = [v for v in res["runs"].values() if "counts" in v]
    by = {}
    for v in runs:
        by.setdefault(v["N"], []).append(np.array(v["counts"]))
    pool = {n: np.concatenate(cs, axis=1) for n, cs in by.items()}

    print(f"\n=== survival at T_c={TC}, {B * len(SEEDS)} independent-order replicas per size ===")
    print(f"  {'N':>5} {'sweeps':>7} {'cutoff ~':>9} {'P(end)':>8} {'max sites':>10} {'saturated?':>11}")
    for n in sorted(pool):
        c = pool[n]
        print(f"  {n:>5} {c.shape[0]:>7} {(n/2)**DP['z']:>9.0f} {(c[-1] > 0).mean():>8.4f} "
              f"{int(c.max()):>7}/{n:<3} {str(c.max() >= n * 0.9):>11}")

    cal = calibrate_collapse()
    print(f"\n=== the SAME collapse on Domany-Kinzel, where z is known to be {DP['z']} ===")
    if cal:
        print(f"  recovered z = {cal['z']} +/- {cal['z_sd']}  ({cal['n']} fits)  "
              f"-> {cal['z_pct']}% +/- {cal['z_sd_pct']}% off")
        print(f"  gate ({cal['criterion']}) -> {cal['decides']}")

    curves = _curves_from(pool, DP["delta"])
    z_hat, costs = fit_z(curves, DP["delta"])

    rng = np.random.default_rng(20260729)
    zs = []
    for _ in range(BOOT):
        bc = []
        for n in sorted(pool):
            c = pool[n]
            r = c[:, rng.integers(0, c.shape[1], c.shape[1])]
            t = np.arange(1, r.shape[0] + 1, dtype=float)[FIT_FROM - 1:]
            bc.append((n, t, (r > 0).mean(axis=1)[FIT_FROM - 1:]))
        z, _ = fit_z(bc, DP["delta"])
        if z is not None:
            zs.append(z)
    lo, hi = ((round(float(np.percentile(zs, (100 - CI) / 2)), 4),
               round(float(np.percentile(zs, 100 - (100 - CI) / 2)), 4)) if zs else (None, None))

    print(f"\n=== PRIMARY: z from the collapse ===")
    print(f"  z = {z_hat}   {CI}% interval [{lo}, {hi}]   (DP: {DP['z']})")

    if not (cal and cal["decides"]):
        verdict = (f"NOT DECIDABLE: the collapse estimator does not recover DK's known z at this "
                   f"ladder to within {CAL_TOL:.0f}% including its spread"
                   + (f" (got {cal['z']} +/- {cal['z_sd']}, {cal['z_pct']}% off)" if cal else "")
                   + ", so no z is quoted for the model. A tight-looking collapse is persuasive "
                     "in a way a bad slope is not, which is exactly why this is gated.")
    elif lo is None:
        verdict = "NOT DECIDABLE: the collapse did not converge on the bootstrap resamples."
    elif lo <= DP["z"] <= hi:
        verdict = (f"z CONSISTENT WITH DP: the collapse gives z = {z_hat} with a {CI}% interval "
                   f"[{lo}, {hi}] containing DP's {DP['z']}, on an estimator shown at this same "
                   f"ladder to recover DK's known z to {cal['z_pct']}% +/- {cal['z_sd_pct']}%. "
                   f"With F58's delta/theta crossing agreement this is a THIRD exponent "
                   f"consistent with directed percolation. nu_perp is still unmeasured, and the "
                   f"T_c uncertainty (+/-{TC_UNC}) is an unseparated systematic.")
    else:
        verdict = (f"z NOT CONSISTENT WITH DP: {z_hat}, {CI}% interval [{lo}, {hi}], excludes "
                   f"DP's {DP['z']}. The estimator recovers DK's z to {cal['z_pct']}% +/- "
                   f"{cal['z_sd_pct']}% at this ladder, so this is not an estimator artifact. "
                   f"F58's two-exponent agreement and this disagree, which is itself the "
                   f"finding: delta and theta are static, z is dynamic.")
    print(f"\n  -> {verdict}")

    res["per_size"] = {str(n): dict(sweeps=int(pool[n].shape[0]),
                                    cutoff_estimate=round(float((n / 2) ** DP["z"]), 1),
                                    P_end=round(float((pool[n][-1] > 0).mean()), 4),
                                    max_sites=int(pool[n].max()),
                                    replicas=int(pool[n].shape[1])) for n in sorted(pool)}
    res["calibration"] = cal
    res["z"] = dict(estimate=z_hat, ci=[lo, hi], ci_percent=CI, bootstraps=len(zs),
                    dp_z=DP["z"], delta_fixed_at=DP["delta"],
                    cost_curve={str(round(z, 3)): round(c, 6) for z, c in (costs or [])})
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Finite-size scaling for the dynamic exponent z at T_c=0.436, the centre of F58's "
        "delta/theta crossing overlap. z is only visible once the damage cone reaches the "
        "boundary, which F58 deliberately avoided -- so this runs LONG time at SMALL lattices "
        "(t = 3*(N/2)^z_DP per size), the opposite geometry. N=192 is excluded at ~143 h on its "
        "own. z is fitted by collapsing P(t)*t^delta against t/N^z with delta FIXED at Jensen, "
        "making it a test of z given DP's delta rather than a two-parameter fit that could "
        "absorb an error in either. The collapse estimator is calibrated on Domany-Kinzel over "
        "the same ladder first and the verdict is gated on it, because a tight-looking collapse "
        "is persuasive in a way a bad slope is not. nu_perp needs off-critical temperatures and "
        "is not attempted; the residual T_c uncertainty is an unseparated systematic on z.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
