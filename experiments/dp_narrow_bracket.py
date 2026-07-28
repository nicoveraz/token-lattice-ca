"""Narrow the critical bracket to resolve the delta/theta disagreement (#82, phase 2).

WHAT PHASE 1 LEFT OPEN. dp_survival_scan.py bracketed the critical region at T in [0.40, 0.45],
where theta crosses the DP value +0.313686, but delta and theta disagreed about where T_c sits --
theta pointed inside that bracket, delta above 0.45. At a genuine DP critical point both agree
simultaneously, so that disagreement had to be resolved before any exponent could be quoted.

WHY IT IS A RESOLUTION PROBLEM, MEASURED NOT ASSUMED. dp_pipeline_validation.py ran the identical
fitting code on Domany-Kinzel, where the exponents are known to be the 1+1D DP values, and
produced a sample-size ladder:

    replicas   theta err   delta err
          64      24.4%       13.8%     <- what phase 1 used
         512       8.4%        8.7%
        4096       6.1%        6.1%
       32768       5.5%        8.6%

At 64 replicas theta carries ~24% error, which is the size of the disagreement phase 1 reported.
So the disagreement is resolution, and 512 replicas is the point where returns flatten -- 4096 and
32768 barely improve on it. That ladder is why this runs 512 rather than the 2000-20000 I first
guessed from the DP literature, and it is the difference between ~2.5 hours and 26.

BATCH SIZE. The measurement does sweeps*N sequential forward passes of batch 2B, so B trades
against GPU occupancy rather than against work. Measured on this machine: B=16 costs 9.63 s per
replica, B=64 costs 5.72 s -- a 1.68x improvement for free. B=128 did not finish inside the probe
window, so 64 is where this sits.

RESUMABLE BY CONSTRUCTION. Every completed run is written to the results file immediately and
keyed by (T, seed); a re-run skips whatever is already there. Re-running this exact command after
an interruption continues from the last completed cell, losing at most the one in flight. That is
not incidental -- the phase 1 scan was killed three times by the background runner and lost
nothing.

PRE-REGISTERED BEFORE RUNNING:
  * Primary: at 512 replicas per temperature, do delta and theta now agree on a single T_c inside
    [0.40, 0.45]?
      - agree      -> that T is the critical point, and the exponents there are the first LM
                      values worth quoting against DP.
      - still not  -> the disagreement is NOT resolution, and it becomes evidence that this
                      transition is not in the DP class. Both outcomes are reportable.
  * Reported against Jensen with the pipeline's DK-measured bias alongside, so a discrepancy of
    that size is not read as a physical result.
  * Hyperscaling theta = 1/z - 2*delta = 0.313685 is checked on the fitted pair.
  * lambda statistics are not involved here; this is survival and active-count only.

THE BIAS MUST BE MEASURED AT THIS RUN'S GEOMETRY, NOT IMPORTED FROM ANOTHER (amended after the
first full pass). The first version of this script hardcoded PIPELINE_BIAS = 8.6% / 5.5% from
dp_pipeline_validation, and declared "not in the DP class" when no temperature met twice those
tolerances. That comparison was invalid. The validation ran DK at N=512 over 200 sweeps; this runs
the LM at N=96 over 40. Re-running the identical estimator on DK at THIS geometry gives:

    DK at p_c, 512 replicas, fit from sweep 5, mean +/- spread over 20 seeds
      N=96  sweeps=40     delta 21.2 +/-10.0%   theta  7.0 +/- 6.6%   <- what this run uses
      N=96  sweeps=200    delta  8.7 +/- 8.5%   theta 18.2 +/- 4.0%
      N=192 sweeps=120    delta  9.8 +/- 8.1%   theta  9.2 +/- 5.6%   <- cheapest that decides
      N=192 sweeps=200    delta  5.4 +/- 8.9%   theta  8.4 +/- 5.9%

On a system that provably IS directed percolation, this geometry misses delta by a fifth, with a
spread half as large again. A tolerance of 17% therefore rejected DP on data known to be DP, which
makes the rejection an artifact of the fit window rather than a statement about the language
model. (The seed count matters as much as the geometry: at 5 seeds this table put N=192/80 inside
the gate and at 20 it does not, so the ladder is averaged over 20 -- DK is free.)

So the calibration is now computed INLINE at the run's own (N, sweeps, replicas, fit_from), and the
DP test is GATED on it: if the estimator cannot recover DK's known exponents to the 20% that
dp_pipeline_validation pre-registered, the geometry is not fit to decide and the LM result is
reported NOT DECIDABLE regardless of what its numbers say. That gate is evaluated on DK alone,
blind to the LM result. A ladder of alternative geometries is scanned so the output states which
geometry WOULD decide and what it would cost, rather than leaving "get more compute" unquantified.
DK is pure numpy, so all of this costs seconds and no GPU.

Writes results/dp_narrow_bracket.json.
Usage:  caffeinate -dimsu .venv/bin/python experiments/dp_narrow_bracket.py
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
TEMPS = [0.400, 0.425, 0.450]          # phase 1's theta-crossing bracket, plus its midpoint
SEEDS = [31, 32, 33, 34, 35, 36, 37, 38]
N, B, R = 96, 64, 2                    # B=64: 1.68x cheaper per replica than 16, measured
SETTLE, SWEEPS = 8, 40
FIT_FROM = 5
REPLICAS = B * len(SEEDS)
# The ladder is the set of geometries the gate is asked about; the gate itself, its tolerance,
# its seed count and its p_c handling all live in dp_calibration -- see F56 for why they are not
# duplicated here.
CAL_GRID = [(96, 40), (96, 80), (96, 200), (192, 80), (192, 120), (192, 200), (384, 200)]
OUT = str(_ROOT / "results" / "dp_narrow_bracket.json")


def trajectory(rule, T, seed):
    """Per-replica damaged-site counts over time from a single-site seed. (sweeps, B)."""
    from ar_ca import run
    base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none",
               init="random", seed=seed)["final"]
    flipped = base.copy()
    flipped[:, N // 2] = np.random.default_rng(seed).choice(rule.init_pool, size=B)
    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
    init2 = np.concatenate([base, flipped], axis=0)
    snaps = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
                init_state=init2, seed=seed + 2, u_stream=u2)["snaps"]
    return (snaps[:, :B] != snaps[:, B:]).sum(axis=2)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=BASE, revision=REVISION, temps=TEMPS, seeds=SEEDS, N=N, B=B, r=R,
        settle=SETTLE, sweeps=SWEEPS, fit_from=FIT_FROM, dp_targets=DP,
        replicas_per_temperature=B * len(SEEDS),
        sample_size_justification="dp_pipeline_validation ladder on DK: 512 replicas reaches "
                                  "~8% on both exponents and 4096/32768 barely improve on it",
        pipeline_bias="measured inline on DK at this run's own geometry, not imported",
        calibration_gate=f"if DK's known exponents are not recovered to {CAL_TOL:.0f}% at "
                         f"N={N}/{SWEEPS} sweeps, the geometry cannot decide and the LM result "
                         f"is NOT DECIDABLE regardless of its numbers",
        primary="do delta and theta agree on a single T_c inside [0.40, 0.45] at 512 replicas?",
        still_disagreeing_means="evidence against a DP reading, but ONLY if the gate passes",
        resumable="every completed run is saved immediately and keyed by (T, seed)")
    runs = res["runs"]
    from ar_ca import ARRule
    rule = ARRule(BASE, revision=REVISION)
    todo = [(t, s) for t in TEMPS for s in SEEDS]
    done0 = len([v for v in runs.values() if "counts" in v])
    print(f"DP bracket narrowing: {len(todo)} runs ({len(TEMPS)} temps x {len(SEEDS)} seeds), "
          f"B={B} -> {B * len(SEEDS)} replicas per temperature", flush=True)
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
    print(f"\n=== {B * len(SEEDS)} replicas per temperature, fitted from sweep {FIT_FROM} ===")
    print(f"  {'T':>7} {'reps':>6} {'-delta':>9} {'R2':>7} {'+theta':>9} {'R2':>7} {'P(end)':>8}")
    out = {}
    for T in TEMPS:
        cs = [np.array(v["counts"]) for v in runs if v["T"] == T]
        if not cs:
            continue
        c = np.concatenate(cs, axis=1)
        t = np.arange(1, c.shape[0] + 1, dtype=float)
        P = (c > 0).mean(axis=1); Nt = c.mean(axis=1)
        m = t >= FIT_FROM
        sd, r2d = _slope(t[m], P[m]); st, r2t = _slope(t[m], Nt[m])
        out[str(T)] = dict(
            replicas=int(c.shape[1]), P_end=round(float(P[-1]), 4),
            delta=(None if sd is None else round(-sd, 4)),
            r2_delta=(None if r2d is None else round(r2d, 4)),
            theta=(None if st is None else round(st, 4)),
            r2_theta=(None if r2t is None else round(r2t, 4)),
            delta_dev_pct=(None if sd is None else round(abs(-sd - DP["delta"]) / DP["delta"] * 100, 1)),
            theta_dev_pct=(None if st is None else round(abs(st - DP["theta"]) / DP["theta"] * 100, 1)))
        print(f"  {T:>7} {c.shape[1]:>6} {-sd:>9.4f} {r2d:>7.4f} {st:>+9.4f} {r2t:>7.4f} "
              f"{P[-1]:>8.4f}")

    secs = [v["secs"] for v in runs if "secs" in v]
    cal = calibrate(N, SWEEPS, REPLICAS, FIT_FROM, CAL_GRID,
                    float(np.median(secs)) if secs else 346.0, len(TEMPS) * len(SEEDS))
    here = cal["at_run_geometry"]
    print_ladder(cal, N, SWEEPS)

    # a temperature is "DP-consistent" when BOTH exponents sit within twice the pipeline's own bias
    # measured AT THIS GEOMETRY -- anything tighter claims precision the estimator lacks
    tol_d, tol_t = here["delta_pct"] * 2, here["theta_pct"] * 2
    both = [k for k, v in out.items()
            if v["delta_dev_pct"] is not None and v["delta_dev_pct"] <= tol_d
            and v["theta_dev_pct"] is not None and v["theta_dev_pct"] <= tol_t]
    dd = {k: v["delta_dev_pct"] for k, v in out.items()}
    tt = {k: v["theta_dev_pct"] for k, v in out.items()}

    if not cal["geometry_decides"]:
        cp = cal["cheapest_passing"]
        need = (f"N={cp['N']} over {cp['sweeps']} sweeps does ({cp['delta_pct']}+/-"
                f"{cp['delta_sd_pct']}% / {cp['theta_pct']}+/-{cp['theta_sd_pct']}%), at a "
                f"projected {cp['projected_hours']} h on this machine"
                if cp else "no geometry on the scanned ladder reaches it, so more replicas rather "
                           "than a bigger lattice is the next thing to try")
        verdict = (f"NOT DECIDABLE AT THIS GEOMETRY: on Domany-Kinzel, which provably IS directed "
                   f"percolation, this estimator at N={N} over {SWEEPS} sweeps recovers delta to "
                   f"{here['delta_pct']}+/-{here['delta_sd_pct']}% and theta to {here['theta_pct']}"
                   f"+/-{here['theta_sd_pct']}%, which does not demonstrate the "
                   f"{CAL_TOL:.0f}% that dp_pipeline_validation pre-registered. The LM deviations "
                   f"(delta {dd}, theta {tt}) are therefore not separable from the fit window, and "
                   f"the DP question cannot be answered either way from this run -- in particular "
                   f"the earlier 'not in the DP class' reading, which came from applying a "
                   f"tolerance measured at N=512/200 sweeps to a run at N={N}/{SWEEPS}, does not "
                   f"survive: that tolerance rejects DK itself at this geometry. {need}.")
    elif both:
        verdict = (f"DP-CONSISTENT at T in {both}: both exponents land within twice the pipeline's "
                   f"bias measured at this geometry ({tol_d:.0f}% delta, {tol_t:.0f}% theta), and "
                   f"that geometry recovers DK's known exponents to within {CAL_TOL:.0f}%. These "
                   f"are the first LM exponents worth quoting against DP, with that bias stated "
                   f"alongside.")
    else:
        verdict = (f"NO SINGLE T SATISFIES BOTH: delta deviations {dd}, theta deviations {tt}, "
                   f"against tolerances of {tol_d:.0f}% and {tol_t:.0f}% from the pipeline's bias "
                   f"at this geometry -- which does recover DK's known exponents to within "
                   f"{CAL_TOL:.0f}%, so the disagreement is not a fit-window artifact. It is "
                   f"evidence that this transition is not in the DP class.")
    print(f"\n  -> {verdict}")

    res["calibration_at_run_geometry"] = cal
    res["analysis"] = out
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Phase 2 of #82: narrow phase 1's theta-crossing bracket [0.40, 0.45] at 512 replicas per "
        "temperature and test whether delta and theta agree on one T_c. The sample size comes "
        "from dp_pipeline_validation's ladder on Domany-Kinzel, where the exponents are known: 512 "
        "reaches ~8% on both and 4096/32768 barely improve, so the 2000-20000 first guessed from "
        "the DP literature was 5-40x more than needed. Deviations are judged against twice the "
        "pipeline's own DK-measured bias rather than against Jensen directly, because claiming "
        "tighter agreement than the estimator demonstrably achieves on a known system would be "
        "reading noise. B=64 rather than 16 because the measurement is sequential in sweeps*N and "
        "B trades against GPU occupancy: 5.72 s per replica vs 9.63, measured. "
        "AMENDED after the first full pass: that bias was initially hardcoded from the validation "
        "run's geometry (N=512, 200 sweeps) while this runs at N=96 over 40, where the identical "
        "estimator recovers DK's delta to only ~33%. The first verdict therefore rejected DP using "
        "a tolerance that rejects DK itself. The bias is now measured inline at this run's own "
        "geometry and the DP test is gated on it -- a gate evaluated on DK alone, blind to the LM "
        "numbers -- with a geometry ladder reporting which configuration would decide and its "
        "projected cost. The 24 LM trajectories are unchanged; only the analysis was rerun.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
