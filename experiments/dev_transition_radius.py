"""Does the developmental transition survive at a radius where the degeneracy is absent? (#99)

WHY THIS IS THE FIRST THING TO RUN, NOT THE FOURTH. The developmental transition (F39/F46) is the
only model-facing claim still standing after F26-F29 (the cross-level negative), F35 (real
generation absorbs no error) and F62-F70 (the AR critical point was a probe artifact). It is
measured at r=2 -- the SAME two-token window F69 showed carries the degeneracy, where r=2 -> r=3
drops top-1 by 52 points (74% -> 22%). Every other proposal in `critical_analysis.md` EXTENDS that
claim; this one DEFENDS it, and each of the others costs more.

THE EXISTING DEFENCE IS SOUND AND DOES NOT ANSWER THIS. Two arguments are already in the paper:
the developmental work runs at T=0.7 where F69 measured no attractor at ANY radius (11-36% top-1),
and the construction is held fixed across checkpoints so a change ACROSS checkpoints cannot come
from the apparatus. Both concern whether the artifact CONTAMINATES the measurement. Neither shows
the effect still EXISTS at a radius where the degeneracy is absent by construction. That has never
been run.

IDENTICAL PROTOCOL, NOT A COPY. `measure` and `bh_fdr` are imported from dev_transition_phase3 and
the F42 filters (`lambda_of` ignited-only, `dnorm_of` all-runs) from lyapunov. STEPS, PRE and SEEDS
are imported too, so the grid cannot drift from the arm it is compared against. Nothing about the
measurement path differs from F39's except `r`.

THE r=2 ARM IS NOT RE-RUN. `results/dev_transition_phase3.json` already holds N=48, B=16, T=0.7,
r=2 at these exact steps and seeds -- that IS the flagship. Re-measuring it would only add noise.
Disclosed asymmetry: those records predate `mean_damage`, so `run_ignited` takes its D_norm
fallback for r=2 and the mean_damage path for r>=3. That is the adapter's documented purpose, but
it means the two arms' ignition filters read different fields.

A PILOT CELL BEFORE THE GRID, because the result it hints at is the one at stake:

    step512, seed 21, N=48    r=2  lambda = -0.1632  D_norm 0.0599  ignition 0.125
                              r=3  lambda = +0.1906  D_norm 0.4258  ignition 0.562

step512 is in the pre-registered PRE set -- the "before" side of the crossing. At r=2 it is
sub-critical, as F39 reports. At r=3 that single cell is already super-critical. If that holds
across seeds, there may be no crossing at r=3 because the system is super-critical THROUGHOUT.
One cell, one seed; it is a reason to run the grid, not a result.

PRE-REGISTERED:
  Primary.    Does the sub->super crossing survive at r=3? Same test as F39: two-sided
              Mann-Whitney U over RUN-level values, PRE (steps 256, 512) against PLATEAU, per
              metric, with the seed as the independent unit (F57).
  Null.       No crossing at r=3. Then the developmental transition is RADIUS-CONFINED and must
              be scoped to the two-token window in the paper's own words rather than stated as a
              fact about training. THAT IS A GOOD RESULT and a cheap one -- far better to find it
              here than to have a reviewer find it.
  Third arm.  If the PRE set is already super-critical at r=3 (pilot hint), the honest reading is
              not "no transition" but "the transition is a property of the r=2 window": at a wider
              window the lattice is spreading at every checkpoint measured. Report which of the
              two it is, because they differ in what they say about the model.
  Kill.       If lambda is undefined in most cells because damage never ignites (F42), the
              comparison is not decidable at this geometry -- report NOT DECIDABLE and widen N or
              B rather than reading the estimator floor (F40) as a measurement.
  Never pool. r=2 and r>=3 answer different questions and are never combined; F69's whole point is
              that they are different regimes.

Writes results/dev_transition_radius.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/dev_transition_radius.py
        (safe to interrupt and re-run -- resumes, keyed per (r, step, seed))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from scipy import stats

from provenance import stamp, rel
# The identical protocol, imported rather than copied -- cross-radius comparison is only
# meaningful if the measurement path is literally the same code (dev_transition_scale's rule).
from dev_transition_phase3 import measure, bh_fdr, BASE, STEPS, PRE, SEEDS, T
from lyapunov import lambda_of, dnorm_of, run_ignited

RADII = [3, 4]                     # r=3 is primary; r=4 only matters if r=3 is ambiguous
N, B = 48, 16                      # one size first (#99); N=96 only if r=3 replicates
PLATEAU = [s for s in STEPS if s not in PRE]
REF = str(_ROOT / "results" / "dev_transition_phase3.json")     # the r=2 arm, not re-run
OUT = str(_ROOT / "results" / "dev_transition_radius.json")


def _step(s):
    return int(str(s).replace("step", ""))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        base=BASE, steps=list(STEPS), pre=sorted(PRE), plateau=PLATEAU, seeds=list(SEEDS),
        radii=RADII, N=N, B=B, T=T,
        reference_arm="results/dev_transition_phase3.json (r=2, N=48, B=16, T=0.7) -- NOT re-run",
        primary="does the sub->super crossing survive at r=3? Mann-Whitney U over run-level "
                "values, PRE vs PLATEAU, seed as the independent unit (F57)",
        null="no crossing at r=3 -> the developmental transition is RADIUS-CONFINED and must be "
             "scoped to the two-token window. A NULL IS A GOOD RESULT.",
        third_arm="if PRE is ALREADY super-critical at r=3, the reading is 'the transition is a "
                  "property of the r=2 window', not 'no transition' -- report which",
        kill="lambda undefined in most cells (F42) -> NOT DECIDABLE at this geometry",
        f42="lambda over IGNITED runs only; D_norm over ALL runs (lyapunov.lambda_of/dnorm_of)",
        never_pool="r=2 and r>=3 are different regimes (F69) and are never combined",
        resumable="keyed by (r, step, seed)")
    runs = res["runs"]

    todo = [(r, s, sd) for r in RADII for s in STEPS for sd in SEEDS
            if f"r{r}|{s}|s{sd}" not in runs]
    print(f"{len(runs)} cached, {len(todo)} to run "
          f"(~{len(todo)*140/3600:.1f} h at the pilot's 140 s/cell)\n", flush=True)

    for r, step, seed in todo:
        key = f"r{r}|{step}|s{seed}"
        t0 = time.time()
        lam, dn, md, ig = measure(step, N, B, seed, r=r)
        runs[key] = dict(r=r, step=_step(step), revision=step, seed=seed, N=N, B=B, T=T,
                         lambda_ca=lam, D_norm=dn, mean_damage=md, ignition_prob=ig,
                         secs=round(time.time() - t0, 1))
        print(f"  r={r} {step:<10} s={seed}  lambda={lam:+.4f}  D_norm={dn:.4f}  "
              f"ign={ig:.3f}  {time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)          # checkpoint every cell

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _arm(runs, r):
    pre = [v for v in runs if v["r"] == r and v["revision"] in PRE]
    plat = [v for v in runs if v["r"] == r and v["revision"] in PLATEAU]
    return pre, plat


def _test(pre, plat):
    """Mann-Whitney U on run-level values, with the F42 filter applied per metric."""
    out = {}
    for name, pick in (("lambda_ca", lambda_of), ("D_norm", dnorm_of)):
        a, b = pick(pre), pick(plat)
        if len(a) < 3 or len(b) < 3:
            out[name] = dict(n_pre=len(a), n_plateau=len(b), p=None,
                             reason="too few usable runs")
            continue
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        out[name] = dict(n_pre=len(a), n_plateau=len(b),
                         median_pre=round(float(sorted(a)[len(a)//2]), 4),
                         median_plateau=round(float(sorted(b)[len(b)//2]), 4),
                         p=float(u.pvalue),
                         pre_all_negative=bool(all(x < 0 for x in a)),
                         plateau_all_positive=bool(all(x > 0 for x in b)))
    return out


def analyse(res):
    runs = list(res["runs"].values())
    out, parts, pvals, labels = {}, [], [], []

    # the r=2 reference arm, read not re-run
    try:
        ref = [v for v in json.load(open(REF))["runs"].values() if v.get("N") == N]
        for v in ref:
            v.setdefault("revision", f"step{v['step']}")
            v["r"] = 2
        out["r2|reference"] = _test(*_arm(ref, 2))
    except (OSError, KeyError) as e:
        out["r2|reference"] = dict(error=str(e))

    for r in RADII:
        pre, plat = _arm(runs, r)
        if len(pre) + len(plat) < len(STEPS) * len(SEEDS):
            out[f"r{r}"] = dict(incomplete=True, n=len(pre) + len(plat),
                                need=len(STEPS) * len(SEEDS))
            continue
        t = _test(pre, plat)
        ignited = sum(1 for v in pre + plat if run_ignited(v))
        t["_ignited"] = f"{ignited}/{len(pre)+len(plat)}"
        out[f"r{r}"] = t
        for m in ("lambda_ca", "D_norm"):
            if t[m].get("p") is not None:
                pvals.append(t[m]["p"]); labels.append(f"r{r}|{m}")

    if pvals:
        for lab, q in zip(labels, bh_fdr(pvals)):
            r, m = lab.split("|")
            out[r][m]["p_BH"] = float(q)

    r3 = out.get("r3", {})
    if r3.get("incomplete") or "lambda_ca" not in r3:
        verdict = (f"INCOMPLETE -- r=3 has {r3.get('n', 0)}/{r3.get('need', 48)} cells. "
                   f"Absence of data is not absence of effect; this file is a checkpoint.")
    else:
        lam = r3["lambda_ca"]
        ign = int(r3["_ignited"].split("/")[0]) / int(r3["_ignited"].split("/")[1])
        if ign < 0.5:
            verdict = (f"NOT DECIDABLE at r=3: only {r3['_ignited']} runs ignited, so lambda is "
                       f"undefined in most cells (F42) and the estimator floor (F40) would be "
                       f"read as a measurement. Widen N or B before concluding.")
        elif lam.get("p_BH") is not None and lam["p_BH"] <= 0.05:
            verdict = (f"THE CROSSING SURVIVES AT r=3: lambda_ca pre {lam['median_pre']:+.4f} -> "
                       f"plateau {lam['median_plateau']:+.4f}, p_BH={lam['p_BH']:.2g} "
                       f"(n={lam['n_pre']}/{lam['n_plateau']} ignited runs). The developmental "
                       f"transition is NOT confined to the two-token window F69 condemned.")
        elif lam.get("pre_all_negative") is False and lam.get("plateau_all_positive"):
            verdict = (f"RADIUS-CONFINED, AND THE PRE SET IS ALREADY SUPER-CRITICAL at r=3: "
                       f"lambda pre {lam['median_pre']:+.4f}, plateau "
                       f"{lam['median_plateau']:+.4f}, p_BH="
                       f"{lam.get('p_BH', float('nan')):.2g}. The lattice spreads at EVERY "
                       f"checkpoint measured, so what F39 detects at r=2 is a property of the "
                       f"two-token window rather than of training. The headline must be scoped "
                       f"to that window.")
        else:
            verdict = (f"NO CROSSING AT r=3: lambda pre {lam['median_pre']:+.4f} -> plateau "
                       f"{lam['median_plateau']:+.4f}, p_BH={lam.get('p_BH', float('nan')):.2g}. "
                       f"The developmental transition is RADIUS-CONFINED and must be scoped to "
                       f"the two-token window rather than stated as a fact about training. This "
                       f"is the pre-registered null and it is a good result.")

    print(f"\n  -> {verdict}")
    res["analysis"] = out
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Does F39/F46's developmental transition survive at a radius where the F62-F70 degeneracy "
        "is absent? Measured at r=3 and r=4 against the r=2 arm in dev_transition_phase3.json, "
        "which is READ not re-run. Identical protocol: measure() and bh_fdr() are imported from "
        "dev_transition_phase3 and the F42 filters from lyapunov, so nothing but r differs. A "
        "null here is a good result -- it scopes the only surviving model-facing claim to the "
        "window it was measured in, which is far better found here than by a reviewer (#99).")


if __name__ == "__main__":
    main()
