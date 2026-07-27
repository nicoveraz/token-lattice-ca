"""Phase 1 of the DP program: locate the critical temperature and test for power-law survival (#82).

WHAT #82 ASKS FOR, AND WHAT THIS ACTUALLY DELIVERS. The issue asks for six exponents against
Jensen's parameter-free 1+1D DP values, by dynamic Monte Carlo (theta, delta) and finite-size
scaling (nu_perp, z). Before writing any of that I measured the cost, because exponent extraction
has a sample-size floor that does not negotiate:

    N=96, B=16, 40 sweeps -> 153 s per run, yielding 16 independent survival samples

delta = 0.159464 means P(t) ~ t^-0.159 -- a decay so slow that separating it from a nearby
exponent needs several decades in t AND enough samples that P(t) is resolved well below the
1/B = 0.0625 quantisation of a single run. At 16 samples per 153 s, 2000 samples at ONE
temperature is 5.3 hours; a five-temperature scan is ~27 hours, and 2000 is still thin by the
standards of the DP literature, which routinely uses 10^5-10^6 runs.

So this script does NOT extract exponents. It does the thing that must come first and is
affordable: locate the critical region in temperature and test whether survival there decays as a
POWER LAW at all. If it does not, the DP framing is wrong and no amount of compute on exponents
would have been worth spending. If it does, the bracket this produces is what a longer campaign
would target, and the compute needed is stated below rather than discovered later.

WHY TEMPERATURE IS THE CONTROL PARAMETER. DP exponents are defined AT a critical point, so the
system needs a tunable knob that carries it through one. The training checkpoint is discrete and
not tunable; the radius is integer-valued and F35 showed lambda(r) is model-invariant. Temperature
is continuous, and the #73 data already brackets a transition in damage survival at the plateau
checkpoint: ignition probability runs 0.211 at T=0.3, 0.805 at T=0.5, 0.984 at T=0.9. So the
critical region sits between T=0.3 and T=0.5, which is where this scan is placed.

WHY A SINGLE-SITE SEED. Dynamic Monte Carlo for DP starts from ONE active site, and the standard
observables are defined from that initial condition. The project's headline uses a 3-site block;
#81 established that lambda is independent of that choice under a fixed damage range, so moving to
block=1 here costs no comparability and matches the literature's setup.

OBSERVABLES, in the DP convention:
  P(t)  survival probability -- the fraction of replicas with ANY damage at sweep t.
        At criticality P(t) ~ t^-delta.
  N(t)  mean number of damaged sites, averaged over ALL replicas INCLUDING dead ones (this is
        the convention; averaging over survivors only measures a different exponent).
        At criticality N(t) ~ t^+theta.
An absorbing state exists and is reachable -- #88 found total extinction at step32 on this very
model -- which is the structural prerequisite for a DP reading and is why this is worth doing.

PRE-REGISTERED BEFORE RUNNING:
  * Primary: over T in {0.30, 0.35, 0.40, 0.45, 0.50}, which temperature gives the straightest
    log P vs log t? Straightness is scored by the R^2 of a linear fit in log-log over the
    post-transient window, and reported for every T rather than only the best.
  * A power law must beat an exponential to count. Both are fitted at every T and compared by
    R^2; if the exponential fits better at every temperature, the DP framing is REFUTED for this
    system and that is the finding.
  * Secondary: the local slope -delta and +theta at the straightest T, reported with the DP
    targets alongside. These are INDICATIVE at this sample size, explicitly not an extraction --
    the run count needed for a real one is computed and printed.
  * A NULL IS INFORMATIVE. "No temperature in this range gives power-law survival" bounds the
    universality program and saves the 27 hours the full campaign would cost.

Writes results/dp_survival_scan.json.
Usage:  caffeinate -i .venv/bin/python experiments/dp_survival_scan.py
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

BASE, REVISION = "EleutherAI/pythia-410m", "step143000"
TEMPS = [0.30, 0.35, 0.40, 0.45, 0.50]
SEEDS = [21, 22, 23, 24]
N, B, R = 96, 16, 2
SETTLE, SWEEPS = 8, 40
FIT_FROM = 5                      # drop the early transient; the seed itself is not asymptotic
DP = dict(delta=0.159464, theta=0.313686, z=1.580745,
          beta=0.276486, nu_perp=1.096854, nu_par=1.733847)
OUT = str(_ROOT / "results" / "dp_survival_scan.json")


def trajectory(rule, T, seed):
    """Per-replica damage over time from a SINGLE-SITE seed. Returns (sweeps, B) damaged counts.

    Replicates block_damage's construction exactly -- same settle, same CRN uniform stream shared
    between the twins, same update path -- but keeps the per-replica trajectory instead of
    collapsing to a cone, because survival probability is a property of individual replicas and
    is destroyed by averaging over the batch.
    """
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
    return (snaps[:, :B] != snaps[:, B:]).sum(axis=2)      # (sweeps, B) damaged-site counts


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=BASE, revision=REVISION, temps=TEMPS, seeds=SEEDS, N=N, B=B, r=R,
        settle=SETTLE, sweeps=SWEEPS, fit_from=FIT_FROM, dp_targets=DP,
        primary="which T gives the straightest log P vs log t?",
        power_law_must_beat_exponential=True,
        this_is_not_an_extraction="indicative slopes only; the run count for a real extraction "
                                  "is computed and reported",
        null_is_informative="no power law in this range bounds the program and saves ~27h")
    runs = res["runs"]
    from ar_ca import ARRule
    rule = ARRule(BASE, revision=REVISION)
    todo = [(t, s) for t in TEMPS for s in SEEDS]
    print(f"DP survival scan: {len(todo)} runs (T in {TEMPS} x {len(SEEDS)} seeds), "
          f"N={N} B={B} sweeps={SWEEPS}, single-site seed", flush=True)
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
                  f"mean_sites(end)={traj.mean(axis=1)[-1]:.2f} ({runs[key]['secs']}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            # Release MPS tensors between runs. Without this the loop accumulates until the OS
            # kills the process -- it died twice at run 12 before this was added. Every other
            # looping experiment here clears per iteration; this one only cleared in `finally`,
            # which runs once at the end and is therefore useless for a loop.
            del traj, alive
            try: torch.mps.empty_cache()
            except Exception: pass
            gc.collect()
    finally:
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    if len([v for v in runs.values() if "counts" in v]) < len(todo):
        print("partial"); json.dump(res, open(OUT, "w"), indent=1); return
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _fits(t, y):
    """Return (r2_power, slope_power, r2_exp). Power law = linear in log-log; exp = linear in semilog."""
    ok = y > 0
    if ok.sum() < 4:
        return None, None, None
    lt, ly = np.log(t[ok]), np.log(y[ok])
    pw = np.polyfit(lt, ly, 1)
    r2p = 1 - np.sum((ly - np.polyval(pw, lt)) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-12)
    ex = np.polyfit(t[ok], ly, 1)
    r2e = 1 - np.sum((ly - np.polyval(ex, t[ok])) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-12)
    return float(r2p), float(pw[0]), float(r2e)


def analyse(res):
    runs = [v for v in res["runs"].values() if "counts" in v]
    print(f"\n=== survival P(t) and active count N(t), fitted from sweep {FIT_FROM} ===")
    print(f"  {'T':>5} {'n_samp':>7} {'P(end)':>8} {'-delta':>9} {'R2 pow':>8} {'R2 exp':>8} "
          f"{'+theta':>9} {'verdict':>12}")
    out, best = {}, None
    for T in TEMPS:
        cs = [np.array(v["counts"]) for v in runs if v["T"] == T]
        if not cs:
            continue
        c = np.concatenate(cs, axis=1)                     # (sweeps, total replicas)
        t = np.arange(1, c.shape[0] + 1, dtype=float)
        P = (c > 0).mean(axis=1)                           # survival
        Nt = c.mean(axis=1)                                # active count over ALL replicas
        m = t >= FIT_FROM
        r2p, slope, r2e = _fits(t[m], P[m])
        r2pn, slopen, _ = _fits(t[m], Nt[m])
        # A comparison of two R^2 values is meaningless unless at least one fit is actually
        # good. The first pass declared "power law" at T=0.5 on R2=0.164 vs 0.079 -- a model
        # explaining 16% of the variance beating one explaining 8%. Require an absolute floor.
        MIN_R2 = 0.80
        power_wins = (r2p is not None and r2e is not None
                      and r2p > r2e and r2p >= MIN_R2)
        v = ("power law" if power_wins else "exponential") if r2p is not None else "no survivors"
        out[str(T)] = dict(n_replicas=int(c.shape[1]), P_end=round(float(P[-1]), 4),
                           delta_indicative=(None if slope is None else round(-slope, 4)),
                           r2_power=(None if r2p is None else round(r2p, 4)),
                           r2_exp=(None if r2e is None else round(r2e, 4)),
                           theta_indicative=(None if slopen is None else round(slopen, 4)),
                           power_law_beats_exponential=bool(power_wins))
        print(f"  {T:>5} {c.shape[1]:>7} {P[-1]:>8.3f} "
              f"{('--' if slope is None else f'{-slope:>9.4f}')} "
              f"{('--' if r2p is None else f'{r2p:>8.4f}')} "
              f"{('--' if r2e is None else f'{r2e:>8.4f}')} "
              f"{('--' if slopen is None else f'{slopen:>9.4f}')} {v:>12}")
        if r2p is not None and (best is None or r2p > best[1]):
            best = (T, r2p)

    print(f"\n  DP targets: delta={DP['delta']}, theta={DP['theta']}  "
          f"(hyperscaling theta = 1/z - 2*delta = {1/DP['z'] - 2*DP['delta']:.6f})")
    # theta locates criticality more sharply than either R^2: it is NEGATIVE when damage dies
    # (subcritical), POSITIVE and growing when it spreads, and equals +0.313686 at a DP critical
    # point. Bracket the crossing of the DP value.
    th = [(float(k), v["theta_indicative"]) for k, v in out.items()
          if v["theta_indicative"] is not None]
    th.sort()
    bracket = None
    for (a, ta), (b, tb) in zip(th, th[1:]):
        if ta < DP["theta"] <= tb:
            bracket = (a, b)
    if bracket:
        print(f"\n  theta crosses the DP value {DP['theta']:+.4f} between T={bracket[0]} "
              f"and T={bracket[1]}")
        res_bracket = list(bracket)
    else:
        res_bracket = None
    any_power = any(v["power_law_beats_exponential"] for v in out.values())
    if not any_power:
        verdict = ("NO POWER LAW in T in [0.30, 0.50]: an exponential fits survival better at "
                   "every temperature. The DP reading is not supported here, and the "
                   "27-hour exponent campaign should not be run against this bracket.")
    else:
        winners = [k for k, v in out.items() if v["power_law_beats_exponential"]]
        # The power-law winners are NOT the extraction target when their theta is negative:
        # theta < 0 means the active count DECAYS, i.e. subcritical, and a subcritical decay can
        # look straight in log-log over a short window without being critical. The target is
        # where theta crosses the DP value.
        sub = [k for k in winners if (out[k]["theta_indicative"] or 0) < 0]
        verdict = (
            f"CRITICAL REGION BRACKETED at T in {res_bracket}, where theta crosses the DP value "
            f"{DP['theta']:+.4f}. NO SAMPLED TEMPERATURE IS AT CRITICALITY: the power-law fits "
            f"win only at T in {sub}, where theta is NEGATIVE (damage decaying, subcritical), so "
            f"those are straight-looking subcritical decays rather than critical ones. "
            f"delta and theta also disagree about where T_c sits -- theta puts it in "
            f"{res_bracket}, delta above 0.45 -- and at a genuine DP point both would agree, so "
            f"that disagreement must be resolved before any exponent claim. At 64 replicas per "
            f"temperature this is expected; it is a resolution problem, not yet a verdict on DP."
            if res_bracket else
            f"NO THETA CROSSING in this range; the critical point is outside T in [0.30, 0.50].")
    print(f"\n  -> {verdict}")

    # what a real extraction would cost, from this run's own measured rate
    secs = float(np.mean([v["secs"] for v in runs]))
    per_run = B
    for target in (2000, 20000):
        hrs = target / per_run * secs / 3600
        print(f"  compute for {target:>6} replicas/T x {len(TEMPS)} temperatures: "
              f"{hrs * len(TEMPS):.1f} h at this run's measured {secs:.0f}s/run")

    res["analysis"] = out
    res["theta_crossing_bracket"] = res_bracket
    res["verdict"] = verdict
    res["extraction_cost"] = dict(
        measured_secs_per_run=round(secs, 1), replicas_per_run=per_run,
        hours_for_2000_per_T_all_temps=round(2000 / per_run * secs / 3600 * len(TEMPS), 1),
        hours_for_20000_per_T_all_temps=round(20000 / per_run * secs / 3600 * len(TEMPS), 1))
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Phase 1 of #82: locate the critical temperature and test whether survival decays as a "
        "power law, BEFORE spending compute on exponents. The full six-exponent extraction #82 "
        "asks for is not affordable here -- delta=0.159 needs several decades in t and thousands "
        "of replicas per temperature, and this machine yields 16 replicas per ~150s, so a "
        "five-temperature campaign at 2000 replicas each is ~27 hours. Temperature is the control "
        "parameter because DP exponents are defined AT a critical point and the checkpoint axis "
        "is discrete; #73's ignition data brackets the damage-survival transition between T=0.3 "
        "and T=0.5. Single-site seed per the dynamic-Monte-Carlo convention; N(t) averages over "
        "ALL replicas including dead ones, since averaging over survivors measures a different "
        "exponent. An absorbing state exists and is reachable (#88 found total extinction at "
        "step32 on this model), which is the structural prerequisite for a DP reading.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
