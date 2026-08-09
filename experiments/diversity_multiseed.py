"""Settled-ring diversity, measured properly: multiple seeds, error bars, and a gate.

WHY THIS EXISTS. F111 claimed lambda_ca is a function of the settled ring's diversity, and its
motivating correlation (rho = +0.943 across the developmental grid) used SINGLE-SEED diversity
values taken from transplant_s: 8, 24, 41, 193, 191, 188 distinct tokens. A later run at the same
geometry with only the seed changed returned 13, 20, 21, 193, 214, 196 -- the high-diversity cells
reproduce within ~12%, but the three LOW-diversity cells, which are exactly where F111's signal
lives, differ by 49-62%.

That is not noise around a stable quantity. In a near-frozen ring, whether eight or thirteen distinct
tokens survive is a matter of which few happened to, and a single draw of that is not a measurement
of the model. F111's dissociation leg is unaffected -- the temperature grid averaged 3 seeds x 16
replicas per cell -- but the correlation that motivated the whole finding rests on single draws.

WHAT THIS MEASURES. The same settle at the same geometry as transplant_s (B=8, N=48, r=2, T=0.7,
30 sweeps, per-replica visit order), across the developmental grid, with SEEDS seeds per checkpoint.
Reports the pooled distinct-token count -- the statistic F111 used -- with its across-seed spread,
plus the per-replica mean for contrast.

PRE-REGISTERED:
  PRIMARY   the across-seed standard deviation of pooled diversity at each checkpoint, and whether
            diversity's span across checkpoints clears RANGE_K times its own seed floor. F111's
            correlation is only interpretable if it does.
  RECOMPUTE rho(diversity, lambda_ca) against the SEED-AVERAGED diversity, with a bootstrap over
            seeds giving a confidence interval on rho. F111 reported a point estimate from single
            draws; this replaces it or withdraws it.
  KILL      diversity's seed floor is comparable to its across-checkpoint span in the dip -> the
            low-diversity cells cannot carry a correlation, F111's motivating observation is
            withdrawn, and only its temperature dissociation survives.
  BOUNDARY  one family, one temperature, one radius. This re-grounds F111's premise; it does not
            re-test its conclusion.

Writes results/diversity_multiseed.json.  Resumable per (checkpoint, seed).
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from ranking import rank as _rank
from provenance import stamp, rel
from meanfield_lambda import lambda_measured
from gatecheck import dynamic_range, carries_verdict
from gatecheck.cohort import cohort_complete

OUT = str(_ROOT / "results" / "diversity_multiseed.json")
MODEL = "EleutherAI/pythia-410m"
STEPS = [128, 256, 512, 1000, 2000, 4000]
R, N, B, SWEEPS, T = 2, 48, 8, 30, 0.7      # transplant_s's settle geometry, unchanged
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
RANGE_K = 2.0
BOOT = 4000
# The single-seed values F111's correlation was built on, kept for the comparison.
F111_SINGLE = {128: 8, 256: 24, 512: 41, 1000: 193, 2000: 191, 4000: 188}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        model=MODEL, steps=STEPS, seeds=SEEDS, r=R, N=N, B=B, sweeps=SWEEPS, T=T,
        range_k=RANGE_K, bootstrap=BOOT, f111_single_seed=F111_SINGLE,
        primary="across-seed sd of pooled diversity per checkpoint, and whether diversity's span "
                "clears RANGE_K x its own seed floor",
        recompute="rho(diversity, lambda_ca) against SEED-AVERAGED diversity, with a bootstrap CI",
        kill="seed floor comparable to the across-checkpoint span in the dip -> F111's motivating "
             "observation is withdrawn and only its temperature dissociation survives",
        boundary="re-grounds F111's premise; does not re-test its conclusion")
    from ar_ca import ARRule, run
    for st in STEPS:
        if all(f"step{st}|s{sd}" in res["cells"] for sd in SEEDS):
            continue
        rule = ARRule(MODEL, revision=f"step{st}")
        for sd in SEEDS:
            k = f"step{st}|s{sd}"
            if k in res["cells"]:
                continue
            t0 = time.time()
            fin = run(rule, B=B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
                      seed=sd, order="per_replica")["final"]
            per = [int(len(np.unique(fin[b]))) for b in range(B)]
            vals, cnts = np.unique(fin.reshape(-1), return_counts=True)
            res["cells"][k] = dict(step=st, seed=sd, pooled_distinct=int(len(vals)),
                                   per_replica_mean=round(float(np.mean(per)), 3),
                                   top_share=round(float(cnts.max() / cnts.sum()), 4),
                                   secs=round(time.time() - t0, 1))
            json.dump(res, open(OUT, "w"), indent=1)
        ps = [res["cells"][f"step{st}|s{sd}"]["pooled_distinct"] for sd in SEEDS]
        print(f"  step{st:<6} pooled distinct per seed {ps}  mean={np.mean(ps):.1f} "
              f"sd={np.std(ps):.1f}  (F111 used {F111_SINGLE[st]})", flush=True)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    rows = {}
    for st in STEPS:
        v = [res["cells"][f"step{st}|s{sd}"]["pooled_distinct"]
             for sd in SEEDS if f"step{st}|s{sd}" in res["cells"]]
        if v:
            rows[st] = dict(n=len(v), values=v, mean=round(float(np.mean(v)), 3),
                            sd=round(float(np.std(v)), 3), single_seed=F111_SINGLE.get(st))
    coh = cohort_complete([str(s) for s in STEPS], [str(s) for s in rows], unit="checkpoint")
    parts = [f"COHORT: {coh.reason}"]
    print(f"\n  {'step':>6} {'n':>3} {'mean':>8} {'sd':>7} {'F111 used':>10} {'|dev|':>7}")
    for st, v in rows.items():
        dev = abs(v["single_seed"] - v["mean"]) / max(v["mean"], 1)
        print(f"  {st:>6} {v['n']:>3} {v['mean']:>8.1f} {v['sd']:>7.1f} {v['single_seed']:>10} "
              f"{dev:>6.0%}")
    if not coh.complete or len(rows) < 5:
        res["analysis"] = dict(rows=rows, cohort=coh.block())
        res["verdict"] = " ".join(parts) + " INCOMPLETE."
        res["_analysis_provenance"] = stamp(__file__); print(f"\n  -> {res['verdict']}"); return

    means = np.array([rows[s]["mean"] for s in STEPS])
    sds = np.array([rows[s]["sd"] for s in STEPS])
    floor = float(np.mean(sds)) / np.sqrt(len(SEEDS))
    lev = dynamic_range(means, floor=floor, k=RANGE_K, name="pooled diversity across checkpoints")
    meas = lambda_measured()
    ca = np.array([meas[s][0] for s in STEPS])
    rk = lambda x: _rank(x)
    rho = float(np.corrcoef(rk(means), rk(ca))[0, 1])
    g = np.random.default_rng(3)
    boots = []
    for _ in range(BOOT):
        d = np.array([float(np.mean(g.choice(rows[s]["values"], size=len(rows[s]["values"]))))
                      for s in STEPS])
        boots.append(np.corrcoef(rk(d), rk(ca))[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    dip = [s for s in STEPS if rows[s]["mean"] < 60]
    dip_sd = float(np.mean([rows[s]["sd"] for s in dip])) if dip else 0.0
    dip_span = (max(rows[s]["mean"] for s in dip) - min(rows[s]["mean"] for s in dip)) if len(dip) > 1 else 0.0
    parts.append(
        f"SEED STABILITY: across-seed sd of pooled diversity is {sds.tolist()} against means "
        f"{means.tolist()}. In the low-diversity cells (mean < 60) the sd averages {dip_sd:.1f} "
        f"against a within-dip span of {dip_span:.1f}, a ratio of {dip_sd/max(dip_span,1e-9):.2f}. "
        f"The single-seed values F111 used deviate from these means by up to "
        f"{max(abs(rows[s]['single_seed']-rows[s]['mean'])/max(rows[s]['mean'],1) for s in STEPS):.0%}.")
    v = carries_verdict([lev], value=rho)
    parts.append(
        f"RECOMPUTED PRIMARY: with seed-averaged diversity, rho(diversity, lambda_ca) = {rho:+.3f} "
        f"with a bootstrap 95% CI of [{lo:+.3f}, {hi:+.3f}] over {BOOT} resamples. F111 reported "
        f"{'+0.943'} from single draws. {lev.reason}"
        + ("" if v.status == "DECIDED" else f" NOT DECIDABLE: {v.reason}"))
    survives = bool(v.status == "DECIDED" and lo > 0)
    parts.append(
        "F111's motivating observation SURVIVES re-grounding: the correlation holds with seed "
        "averaging and its CI excludes zero, so the single-seed values were noisy but not "
        "misleading."
        if survives else
        "F111's motivating observation is WITHDRAWN as stated: with honest seed averaging the "
        "correlation no longer clears its own gate, so the rho = +0.943 was an artifact of single "
        "draws of a seed-unstable quantity. F111's temperature dissociation is unaffected and "
        "remains the only support for the reduction.")
    parts.append(
        "BOUNDARY: one family, one temperature, one radius. This re-grounds F111's premise and does "
        "not re-test its conclusion; the dissociation grid averaged 3 seeds x 16 replicas per cell "
        "and is untouched by this.")
    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rows=rows, cohort=coh.block(), rho=round(rho, 4),
                           rho_ci=[round(float(lo), 4), round(float(hi), 4)],
                           seed_floor=round(floor, 4), leverage=lev.block(),
                           survives=survives, f111_rho=0.943)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Re-measures settled diversity with 8 seeds after a control found single-seed "
                    "values deviating 49-62% in the low-diversity regime, which is where F111's "
                    "motivating correlation lives.")


if __name__ == "__main__":
    main()
