"""Is lambda_ca a function of the settled ring's DIVERSITY? The temperature dissociation.

THE OBSERVATION. Across the developmental grid, the number of distinct tokens in the settled ring
tracks lambda_ca at rho = +0.943 (exact permutation p = 0.0167): 8, 24, 41, 193, 191, 188 distinct
against lambda -0.093, -0.019, +0.068, +0.192, +0.156, +0.172. That is the tightest correlate
lambda_ca has ever had -- F86's external anchor is 0.833, F99's non-circular column 0.771.

WHY IT IS NOT YET AN EXPLANATION, and this is stated first because the project's failure mode is
believing a correlate too early. Both quantities rise monotonically with training step, so at n=6
any two monotone functions of time correlate near 1 and a permutation test on the ranking cannot
separate that. The correlation is also CIRCULAR in F96's sense: the settled ring is produced by the
dynamics whose exponent it would explain.

THE DISSOCIATION, which the existing grid already supports. `dev_transition_temp.json` measured
lambda_ca at 2 checkpoints x 4 temperatures x 8 seeds. Temperature moves lambda MORE than training
does at fixed weights -- at step256 it runs -0.2372 (T=0.3) to +0.3002 (T=1.1). So diversity and
lambda can be compared WITHIN a fixed model, where there is no training-time trend to confound
them. All that is missing is the diversity axis, which is one settle per cell and no damage run.

PRE-REGISTERED:
  PRIMARY   do all 8 (diversity, lambda) points fall on ONE curve regardless of checkpoint? Fit
            lambda ~ f(diversity) pooled, and per-checkpoint. If pooling does not inflate the
            residual beyond the seed floor, diversity is the explanatory variable and training step
            is incidental to it -- lambda_ca would be a function of the state, not of the model.
  KILL      the two checkpoints form SEPARATE curves -> training matters beyond diversity, the
            rho = 0.943 across the developmental grid is a shared time trend, and this closes.
  CONTROL   the stored lambda values are re-used unchanged from dev_transition_temp.json; nothing
            here re-measures them, so the pairing cannot be tuned.
  BOUNDARY  even a pass is DEFLATIONARY rather than mechanistic: it would reduce lambda_ca to a
            property of the settled state, which is a reduction of one measured quantity to
            another and not a named mechanism. That is worth having and must not be oversold.

Writes results/diversity_explanandum.json.
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
from gatecheck import dynamic_range, carries_verdict
from gatecheck.cohort import cohort_complete

OUT = str(_ROOT / "results" / "diversity_explanandum.json")
SRC = _ROOT / "results" / "dev_transition_temp.json"
MODEL = "EleutherAI/pythia-410m"
R, N, B, SWEEPS = 2, 48, 16, 30
SEEDS = [21, 22, 23]


def grid():
    d = json.load(open(SRC))["runs"]
    g = {}
    for v in d.values():
        g.setdefault((v["T"], v["step"]), []).append(v["lambda_ca"])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in g.items()}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    G = grid()
    res["_preregistration"] = dict(
        model=MODEL, r=R, N=N, B=B, sweeps=SWEEPS, seeds=SEEDS,
        source="dev_transition_temp.json -- lambda values re-used UNCHANGED, never re-measured",
        cells=[f"T{t}_step{s}" for t, s in sorted(G)],
        primary="do all (diversity, lambda) points fall on ONE curve regardless of checkpoint?",
        kill="separate curves per checkpoint -> training matters beyond diversity and the "
             "rho=0.943 across the developmental grid is a shared time trend",
        boundary="even a pass is DEFLATIONARY: a reduction of lambda_ca to a property of the "
                 "settled state, not a named mechanism")
    from ar_ca import ARRule, run
    for (T, step) in sorted(G):
        k = f"T{T}_step{step}"
        if k in res["cells"]:
            continue
        t0 = time.time()
        rule = ARRule(MODEL, revision=f"step{step}")
        ds, ts = [], []
        for sd in SEEDS:
            fin = run(rule, B=B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
                      seed=sd, order="per_replica")["final"]
            for b in range(B):
                vals, cnts = np.unique(fin[b], return_counts=True)
                ds.append(len(vals)); ts.append(cnts.max() / cnts.sum())
        lam, sd_, n = G[(T, step)]
        res["cells"][k] = dict(T=T, step=step, lambda_ca=round(lam, 5), lambda_sd=round(sd_, 5),
                               n_lambda=n, distinct=round(float(np.mean(ds)), 3),
                               distinct_sd=round(float(np.std(ds)), 3),
                               top_share=round(float(np.mean(ts)), 4),
                               secs=round(time.time() - t0, 1))
        c = res["cells"][k]
        print(f"  {k:<18} distinct={c['distinct']:>6.2f}  top={c['top_share']:.3f}  "
              f"lam={c['lambda_ca']:+.4f}  ({c['secs']:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    cs = sorted(res["cells"].values(), key=lambda c: (c["step"], c["T"]))
    G = grid()
    coh = cohort_complete([f"T{t}_step{s}" for t, s in sorted(G)],
                          [f"T{c['T']}_step{c['step']}" for c in cs], unit="cell")
    parts = [f"COHORT: {coh.reason}"]
    print(f"\n  {'cell':<18} {'T':>5} {'distinct':>9} {'top share':>10} {'lambda':>9}")
    for c in cs:
        print(f"  T{c['T']}_step{c['step']:<10} {c['T']:>5.1f} {c['distinct']:>9.2f} "
              f"{c['top_share']:>10.3f} {c['lambda_ca']:>+9.4f}")
    if not coh.complete or len(cs) < 6:
        res["analysis"] = dict(cohort=coh.block()); res["verdict"] = " ".join(parts) + " INCOMPLETE."
        res["_analysis_provenance"] = stamp(__file__); print(f"\n  -> {res['verdict']}"); return

    x = np.array([c["distinct"] for c in cs]); y = np.array([c["lambda_ca"] for c in cs])
    steps = np.array([c["step"] for c in cs])
    floor = float(np.mean([c["lambda_sd"] for c in cs])) / np.sqrt(8)
    # pooled fit vs per-checkpoint fits, on log diversity (the range spans an order of magnitude)
    lx = np.log(np.maximum(x, 1.0))
    pooled = np.polyval(np.polyfit(lx, y, 1), lx)
    r_pool = float(np.sqrt(np.mean((y - pooled) ** 2)))
    per = np.zeros_like(y)
    for s in np.unique(steps):
        m = steps == s
        per[m] = np.polyval(np.polyfit(lx[m], y[m], 1), lx[m]) if m.sum() > 1 else y[m]
    r_per = float(np.sqrt(np.mean((y - per) ** 2)))
    collapses = bool(r_pool <= r_per + floor)
    rk = lambda v: _rank(v)
    rho_within = [float(np.corrcoef(rk(x[steps == s]), rk(y[steps == s]))[0, 1])
                  for s in np.unique(steps) if (steps == s).sum() > 2]
    parts.append(
        f"WITHIN-MODEL, which is the part free of the training-time trend: at fixed weights, "
        f"diversity and lambda_ca rank-correlate at {rho_within} across the four temperatures "
        f"(one value per checkpoint). Temperature moves lambda from {y.min():+.4f} to {y.max():+.4f} "
        f"with the model held fixed.")
    parts.append(
        f"PRIMARY: pooling both checkpoints onto ONE diversity->lambda curve leaves a residual of "
        f"{r_pool:.4f}; fitting them separately leaves {r_per:.4f}, against a lambda seed floor of "
        f"{floor:.4f}. "
        + (f"Pooling costs less than the floor, so the two checkpoints lie on the SAME curve: "
           f"lambda_ca is a function of the settled state's diversity, and the training step enters "
           f"only through the diversity it produces. That reduces the developmental transition to a "
           f"property of the state rather than leaving it an unexplained event."
           if collapses else
           f"Pooling costs MORE than the floor, so the checkpoints lie on SEPARATE curves. Training "
           f"matters beyond the diversity it produces, and the rho = 0.943 measured across the "
           f"developmental grid is a shared time trend rather than a functional relation. This "
           f"route closes."))
    parts.append(
        "BOUNDARY: even the positive reading is DEFLATIONARY, not mechanistic -- it reduces "
        "lambda_ca to another measured property of the same settled system, in the way temperature "
        "reduces to mean kinetic energy. It names no circuit and no training event. lambda values "
        "are re-used unchanged from dev_transition_temp.json, so the pairing could not be tuned.")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(cohort=coh.block(), pooled_residual=round(r_pool, 5),
                           per_checkpoint_residual=round(r_per, 5), lambda_seed_floor=round(floor, 5),
                           collapses=collapses, rho_within=rho_within,
                           distinct=[c["distinct"] for c in cs], lam=[c["lambda_ca"] for c in cs])
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Tests whether lambda_ca is a function of the settled ring's diversity, using "
                    "temperature at fixed weights to dissociate diversity from the training-time "
                    "trend that makes the rho=0.943 across checkpoints uninterpretable on its own.")


if __name__ == "__main__":
    main()
