"""Bracket the critical region for a SECOND model family (#61, light version; claim E).

WHY A SECOND FAMILY, AND WHY THE LIGHT VERSION. Everything measured in the universality program
so far is `pythia-410m` at r=2: F58's critical point, F59's z, F60's ladder anomaly. Four Pythia
sizes and three lattice sizes are replications *within one training recipe* — one corpus, one
tokenizer, one optimiser schedule — so none of them separates "a property of trained language
models" from "a property of how Pythia was trained". Until a second family is measured, the word
*universality* does not belong in a title.

#61 as filed is heavier than this: it demands fine-spaced public intermediate checkpoints and an
independently published emergence curve, because it is scoped to the *emergence* question (#58).
Asking only "does a critical point exist in another model at all?" needs a different FINAL model
and nothing else. That is what this runs. Full #61 remains open for the emergence axis.

THE MODEL. `gpt2-medium` (355M) against `pythia-410m`: comparable size, but a genuinely different
recipe — WebText rather than the Pile, a different tokenizer, a different optimiser schedule, and
a different lab. `ARRule` is family-agnostic (AutoTokenizer / AutoModelForCausalLM), so the
measurement path is literally the same code, which is the only way the comparison means anything.

WHAT IS AND IS NOT EXPECTED TO MATCH. T_c is **non-universal** — it is a property of the
particular model, like a critical temperature, and there is no reason for two families to share
it. A smoke test already shows gpt2-medium spreading damage at T=0.4, where pythia-410m is
subcritical, so its transition sits lower. What *should* match, if the transition is a genuine
universality class rather than a model-specific curiosity, are the EXPONENTS. So this run does not
compare temperatures; it locates gpt2-medium's own critical region so the exponent measurement can
be pointed at it.

THIS GEOMETRY CANNOT QUOTE AN EXPONENT, AND DOES NOT TRY. N=96 over 40 sweeps is the geometry F56
retracted a verdict over: on Domany-Kinzel the same estimator recovers delta to only ~20% there.
That is fine for *bracketing* — locating where damage stops dying and starts spreading is a
coarse, robust question — and it is emphatically not fine for a number. The exponent run that
follows uses the F58 geometry (N=192, 120 sweeps) at whatever bracket this returns.

Every replica draws its own visit order (F57), so a third of runs are not silently killed by a
shared permutation.

PRE-REGISTERED BEFORE RUNNING:
  * Primary: locate the bracket [T_lo, T_hi] where the surviving-damage fraction P(end) crosses
    from near-zero to substantial, i.e. where the transition sits.
  * Expected: a bracket BELOW pythia-410m's T_c ~= 0.436, from the smoke test. If instead no
    bracket is found anywhere in the scanned range, that is reportable: it would mean gpt2-medium
    has no comparable transition, which is itself an answer to claim E.
  * NOT tested here: any exponent, and any comparison of T_c between families. T_c is
    non-universal; comparing it would be a category error.

Writes results/dp_scan_gpt2.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/dp_scan_gpt2.py
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
from dp_calibration import DP

BASE = "gpt2-medium"                   # 355M, WebText -- a different recipe from pythia-410m
REFERENCE = "EleutherAI/pythia-410m"   # what it is being compared against, for the record
TEMPS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
SEEDS = [71, 72]                       # a bracket needs coverage, not precision
N, B, R = 96, 64, 2
SETTLE, SWEEPS = 8, 40
OUT = str(_ROOT / "results" / "dp_scan_gpt2.json")


def trajectory(rule, T, seed):
    """Per-replica damaged-site counts, (sweeps, B). Identical protocol to dp_class_n192."""
    from ar_ca import run
    base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none",
               init="random", seed=seed, order="per_replica")["final"]
    flipped = base.copy()
    flipped[:, N // 2] = np.random.default_rng(seed).choice(rule.init_pool, size=B)
    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
    perm = np.argsort(np.random.default_rng(seed + 3).random((SWEEPS, B, N)), axis=2)
    snaps = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
                init_state=np.concatenate([base, flipped], axis=0), seed=seed + 2, u_stream=u2,
                order="per_replica", order_stream=np.concatenate([perm, perm], axis=1))["snaps"]
    return (snaps[1:, :B] != snaps[1:, B:]).sum(axis=2)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=BASE, compared_against=REFERENCE, temps=TEMPS, seeds=SEEDS, N=N, B=B, r=R,
        settle=SETTLE, sweeps=SWEEPS, replicas_per_temperature=B * len(SEEDS),
        primary="locate the bracket where P(end) crosses from near-zero to substantial",
        not_tested="any exponent, and any T_c comparison between families -- T_c is non-universal",
        geometry_caveat="N=96/40 is the geometry F56 retracted a verdict over (~20% delta error "
                        "on DK). Adequate for bracketing, never for a number.",
        visit_order="per_replica (F57)",
        resumable="every completed run is saved immediately and keyed by (T, seed)")
    runs = res["runs"]
    from ar_ca import ARRule
    rule = ARRule(BASE)
    todo = [(T, sd) for T in TEMPS for sd in SEEDS]
    done0 = len([v for v in runs.values() if "counts" in v])
    print(f"Bracket scan for {BASE}: {len(todo)} runs ({len(TEMPS)} temps x {len(SEEDS)} seeds), "
          f"{B * len(SEEDS)} replicas per temperature", flush=True)
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
            print(f"[{k}/{len(todo)}] {key}: P(end)={(traj[-1] > 0).mean():.3f} "
                  f"sites(end)={traj.mean(axis=1)[-1]:.2f} ({runs[key]['secs']}s)", flush=True)
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
        print(f"\npartial: {have}/{len(todo)} -- re-run this command to continue", flush=True)
        json.dump(res, open(OUT, "w"), indent=1); return
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = [v for v in res["runs"].values() if "counts" in v]
    print(f"\n=== {BASE}: where does damage stop dying and start spreading? ===")
    print(f"  {'T':>7} {'reps':>6} {'P(end)':>9} {'sites(end)':>11} {'max sites':>10}")
    out = {}
    for T in TEMPS:
        cs = [np.array(v["counts"]) for v in runs if v["T"] == T]
        if not cs:
            continue
        c = np.concatenate(cs, axis=1)
        P_end = float((c[-1] > 0).mean())
        out[str(T)] = dict(replicas=int(c.shape[1]), P_end=round(P_end, 4),
                           sites_end=round(float(c[-1].mean()), 3),
                           max_sites=int(c.max()))
        print(f"  {T:>7} {c.shape[1]:>6} {P_end:>9.4f} {c[-1].mean():>11.3f} {int(c.max()):>10}")

    # the bracket: the first adjacent pair straddling a substantial surviving fraction
    ts = sorted(float(k) for k in out)
    LO, HI = 0.05, 0.20                       # near-zero vs substantial, stated not tuned
    bracket = None
    for a, b in zip(ts, ts[1:]):
        if out[str(a)]["P_end"] < LO <= out[str(b)]["P_end"] or \
           (out[str(a)]["P_end"] < HI <= out[str(b)]["P_end"]):
            bracket = (a, b); break
    print(f"\n  bracket (P(end) crossing {LO}-{HI}): {bracket}")

    if bracket is None:
        allhigh = all(v["P_end"] >= HI for v in out.values())
        alllow = all(v["P_end"] < LO for v in out.values())
        where = ("ABOVE the scanned range -- every temperature already spreads" if allhigh else
                 "BELOW the scanned range -- every temperature dies" if alllow else
                 "not bracketed by adjacent points; P(end) is not monotone here")
        verdict = (f"NO BRACKET IN {TEMPS}: the transition for {BASE} lies {where}. Extend the "
                   f"scan in that direction before attempting exponents. Reportable either way: a "
                   f"second family with no comparable transition would itself answer claim E.")
    else:
        verdict = (f"BRACKETED at T in {list(bracket)} for {BASE}, against pythia-410m's "
                   f"T_c ~= 0.436 (F58). The two need NOT agree -- T_c is non-universal -- so this "
                   f"is a pointer for the exponent run, not a comparison. Next: the F58 protocol "
                   f"(N=192, 120 sweeps, per_replica) on a grid spanning this bracket, to test "
                   f"whether delta and theta cross at a common T_c here too. Only THAT tests "
                   f"whether the class is shared.")
    print(f"\n  -> {verdict}")

    res["analysis"] = out
    res["bracket"] = list(bracket) if bracket else None
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        f"Bracket scan for a SECOND model family ({BASE} vs {REFERENCE}), the light version of "
        "#61: a different final model, no intermediate checkpoints, aimed only at claim E -- does "
        "a critical point exist outside Pythia? Everything in the universality program so far is "
        "one training recipe, so the word universality is not yet earned. T_c is non-universal and "
        "is NOT compared across families; what should match, if the class is shared, are the "
        "exponents, and this run only locates where to measure them. The geometry (N=96/40) is the "
        "one F56 retracted a verdict over and is used for bracketing only, never for a number. "
        "Per-replica visit orders (F57).")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
