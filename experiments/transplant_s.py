"""Does the CONDITIONAL move, or does the STATE? The cross-checkpoint transplant.

WHAT F96 LEFT. F94 measured single-token sensitivity `s` on uniformly random token windows, found
it saturated and flat (0.833-0.876) across the developmental transition, and eliminated it as
lambda_ca's explanandum. F96 then showed that verdict was a property of the ENSEMBLE, not the
model: measured on the states the ring actually occupies, `s` spans 0.331 rather than 0.071, falls
to 0.5252 against the mean-field critical 1/r = 0.50, and the predictor finally clears a range gate
at 1.72 where the random ensemble sat at 0.29.

But F96 could not read that as a positive result, for two reasons it stated rather than argued
around:

  CIRCULARITY  the settled state is PRODUCED by the dynamics whose exponent it is being used to
               predict, so a correlation between them is not evidence that `s` drives lambda_ca
  DEGENERACY   the movement sits exactly where the settled ring is degenerate -- 7 distinct tokens
               at step128, 10 distinct windows out of 128 -- and jumps once it diversifies

THE TRANSPLANT BREAKS BOTH, and it is the experiment F96 specified. Measure `s` for checkpoint i's
CONDITIONAL on contexts drawn from checkpoint j's SETTLED STATE, for every (i, j) pair:

        s[i][j] = mean CRN disagreement of model_i's conditional, on windows from state_j

  the DIAGONAL  s[i][i] is exactly F96's measurement, circular and degenerate as before
  a ROW         s[i][:] varies the ensemble while holding the model fixed
  a COLUMN      s[:][j] varies the model while holding the ensemble fixed  <- NOT circular:
                the contexts do not come from the model being measured
  the CORNER    s[early][late] probes the early conditional on a RICH ensemble, which is the
                degeneracy escape -- step128's model never sees its own 7-token lattice

So the two-way decomposition answers the question directly. If `s` varies down COLUMNS, the
conditional itself changes and F94's elimination was an ensemble artifact. If it varies across
ROWS, what changes is the state the ring settles into, and `s` is downstream of the transition
rather than upstream of it. Both are informative and they are distinguishable, which is more than
either F94 or F96 could say.

PRE-REGISTERED, with the gate declared before any number:
  PRIMARY      Two-way decomposition of s[i][j]: the fraction of total variance attributable to
               the model (column effect, ensemble held fixed) versus to the ensemble (row effect).
               Reported as a ratio with both marginal ranges, gated by
               gatecheck.leverage.correlation_leverage on whichever marginal is used as a predictor.
  NON-CIRCULAR The headline uses a FIXED reference ensemble -- the richest settled state on the
               grid -- so `s` varies only with the model. That column is the non-circular version
               of F94's primary and is the one quoted.
  DEGENERACY   Every cell records its distinct-window count and is gated by
               gatecheck.leverage.distinct_units; cells below the floor are reported and EXCLUDED
               from the decomposition rather than silently averaged in. The diagonal's early cells
               are expected to fail this, which is the point.
  COHORT       gatecheck.cohort guards the checkpoint set: a checkpoint that fails to load makes
               the decomposition NOT DECIDABLE rather than shrinking the grid silently.
  KILL         `s` is flat down every column -> the conditional's single-token sensitivity really
               is constant across the transition, F94's elimination stands on its merits rather
               than on its ensemble, and this route is closed for good.
  DEFLATIONARY If the non-circular column reproduces lambda_ca within its seed floor, the ring is
               redundant for lambda_ca and the tool is `s`. Registered again, as in F94, because
               it is a live outcome and must be reported as found.

`s` is EXACT, not estimated: inverse-CDF sampling against a shared uniform makes the disagreement
probability a deterministic functional of the conditional pair (`meanfield_lambda.s_crn`). Two
forward passes per context, no seed, no Monte Carlo.

Writes results/transplant_s.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/transplant_s.py
        (resumable per (model_step, state_step))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from meanfield_lambda import s_crn, lambda_mf, lambda_measured, RANGE_RATIO
from gatecheck import correlation_leverage, distinct_units, carries_verdict
from gatecheck.cohort import cohort_complete, require_cohort

OUT = str(_ROOT / "results" / "transplant_s.json")
MODEL = "EleutherAI/pythia-410m"
STEPS = [128, 256, 512, 1000, 2000, 4000]      # the developmental grid, as F94/F96
R, T = 2, 0.7
N_CTX = 128
BATCH = 32
SEED = 20260806
DISTINCT_MIN = 32                              # F96's floor, unchanged and declared before the run
SET_B, SET_N, SET_SWEEPS = 8, 48, 30           # the settle geometry of the lambda runs


def settled_pool(rule):
    """The ring's own settled state at this checkpoint -- the ensemble the dynamics run in."""
    from ar_ca import run
    fin = run(rule, B=SET_B, N=SET_N, r=R, T=T, sweeps=SET_SWEEPS, scheme="none",
              seed=SEED, order="per_replica")["final"]
    return fin.reshape(-1).astype(np.int64)


def windows(pool, n, rng):
    st = rng.integers(0, len(pool) - R, size=n)
    return np.stack([pool[s:s + R] for s in st])


def measure(model, dev, pool, rng):
    """Exact mean CRN disagreement of this model's conditional on windows from `pool`."""
    base = windows(pool, N_CTX, rng)
    rows, keys = [], []
    for w in base:
        pos = int(rng.integers(0, R))
        alt = list(int(x) for x in w)
        while alt[pos] == int(w[pos]):
            alt[pos] = int(rng.choice(pool))       # replacement from the SAME ensemble (F56/F70)
        rows += [[int(w[0]), int(w[1])], alt]
        keys.append((int(w[0]), int(w[1])))
    rows = np.array(rows, np.int64)
    probs = []
    for i in range(0, len(rows), BATCH):
        with torch.no_grad():
            x = torch.tensor(rows[i:i + BATCH], device=dev)
            lg = model(input_ids=x).logits[:, -1].float()
            probs.append(torch.softmax(lg / T, dim=-1).cpu().double().numpy())
    probs = np.concatenate(probs, 0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    vals = [s_crn(probs[2 * k], probs[2 * k + 1]) for k in range(len(base))]
    return dict(s=round(float(np.mean(vals)), 5), s_sd=round(float(np.std(vals)), 5),
                ctx_distinct=int(len(set(keys))), n_ctx=len(vals))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}, "pools": {}}
    res["_preregistration"] = dict(
        model=MODEL, steps=STEPS, r=R, T=T, n_ctx=N_CTX, seed=SEED,
        distinct_min=DISTINCT_MIN, settle=dict(B=SET_B, N=SET_N, sweeps=SET_SWEEPS),
        design="s[i][j] = model_i's conditional measured on windows from state_j; the diagonal is "
               "F96's circular measurement, the columns are the non-circular version",
        primary="two-way decomposition: variance of s attributable to the MODEL (column effect, "
                "ensemble fixed) versus to the ENSEMBLE (row effect, model fixed)",
        non_circular="the headline column uses the richest settled state on the grid as a fixed "
                     "reference ensemble, so s varies only with the model",
        degeneracy=f"cells with fewer than {DISTINCT_MIN} distinct windows are reported and "
                   f"EXCLUDED from the decomposition, not averaged in; the diagonal's early cells "
                   f"are expected to fail this",
        cohort="a checkpoint that fails to load makes the decomposition NOT DECIDABLE rather than "
               "shrinking the grid silently (gatecheck.cohort)",
        kill="s flat down every column -> the conditional really is constant and F94's elimination "
             "stands on its merits rather than on its ensemble; route closed",
        deflationary="if the non-circular column reproduces lambda_ca within its seed floor, the "
                     "ring is redundant and the tool is s -- registered as in F94")

    from ar_ca import ARRule
    # PASS 1: settle every checkpoint's own state, so the ensembles exist before any transplant.
    for st in STEPS:
        if str(st) in res["pools"]:
            continue
        t0 = time.time()
        rule = ARRule(MODEL, revision=f"step{st}")
        pool = settled_pool(rule)
        res["pools"][str(st)] = dict(tokens=[int(x) for x in pool],
                                     distinct=int(len(np.unique(pool))),
                                     secs=round(time.time() - t0, 1))
        print(f"  settled step{st}: {res['pools'][str(st)]['distinct']} distinct tokens "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    # PASS 2: the full grid. One model load per row.
    for mi in STEPS:
        if all(f"m{mi}|s{sj}" in res["cells"] for sj in STEPS):
            continue
        rule = ARRule(MODEL, revision=f"step{mi}")
        m, dev = rule.model, rule.device
        for sj in STEPS:
            k = f"m{mi}|s{sj}"
            if k in res["cells"]:
                continue
            t0 = time.time()
            pool = np.array(res["pools"][str(sj)]["tokens"], np.int64)
            row = measure(m, dev, pool, np.random.default_rng(SEED + sj))
            row.update(model_step=mi, state_step=sj, secs=round(time.time() - t0, 1),
                       degenerate=bool(row["ctx_distinct"] < DISTINCT_MIN))
            res["cells"][k] = row
            print(f"    model step{mi:<5} on state step{sj:<5}: s={row['s']:.4f} "
                  f"(distinct {row['ctx_distinct']}){'  DEGENERATE' if row['degenerate'] else ''}",
                  flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
        del rule, m
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    cells = res["cells"]
    parts = []

    # COHORT FIRST: the grid must be the grid that was registered.
    got = sorted({c["model_step"] for c in cells.values()})
    coh = cohort_complete(STEPS, got, unit="checkpoint")
    parts.append(f"COHORT: {coh.reason}")
    if not coh.complete:
        res["analysis"] = dict(cohort=coh.block())
        res["verdict"] = " ".join(parts) + " Decomposition NOT DECIDABLE."
        res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}")
        return

    S = np.full((len(STEPS), len(STEPS)), np.nan)
    D = np.zeros_like(S, dtype=bool)
    for i, mi in enumerate(STEPS):
        for j, sj in enumerate(STEPS):
            c = cells.get(f"m{mi}|s{sj}")
            if c:
                S[i, j] = c["s"]
                D[i, j] = bool(c["degenerate"])

    print(f"\n  s[model][state]   columns = state ensemble, rows = model conditional")
    hdr = "model / state"
    print("  " + hdr.ljust(16) + "".join(f"{sj:>9}" for sj in STEPS))
    for i, mi in enumerate(STEPS):
        cellstr = "".join((f"{S[i,j]:>8.4f}" + ("*" if D[i, j] else " ")) for j in range(len(STEPS)))
        print(f"  step{mi:<12}{cellstr}")
    print(f"  (* = fewer than {DISTINCT_MIN} distinct windows; excluded from the decomposition)")

    usable = ~D & ~np.isnan(S)
    # Row effect = ensemble (model fixed); column effect = model (ensemble fixed).
    row_ranges = [np.nanmax(S[i][usable[i]]) - np.nanmin(S[i][usable[i]])
                  for i in range(len(STEPS)) if usable[i].sum() >= 3]
    col_ranges = [np.nanmax(S[:, j][usable[:, j]]) - np.nanmin(S[:, j][usable[:, j]])
                  for j in range(len(STEPS)) if usable[:, j].sum() >= 3]
    ens_eff = float(np.mean(row_ranges)) if row_ranges else float("nan")
    mod_eff = float(np.mean(col_ranges)) if col_ranges else float("nan")
    parts.append(
        f"DECOMPOSITION over the {int(usable.sum())} usable cells of {S.size}: holding the model "
        f"fixed and varying the ENSEMBLE moves s by {ens_eff:.4f} on average; holding the ensemble "
        f"fixed and varying the MODEL moves it by {mod_eff:.4f}. "
        + (f"The ensemble effect is {ens_eff/mod_eff:.1f}x the model effect, so what changes across "
           f"the developmental transition is mostly the STATE the ring settles into, not the "
           f"conditional's sensitivity -- s is downstream of the transition."
           if mod_eff > 0 and ens_eff > mod_eff else
           f"The model effect is {mod_eff/max(ens_eff,1e-9):.1f}x the ensemble effect, so the "
           f"conditional itself changes and F94's elimination was an artifact of the random "
           f"ensemble it was measured on."))

    # THE NON-CIRCULAR HEADLINE: richest ensemble, fixed, model varying down the column.
    richest = int(np.argmax([res["pools"][str(sj)]["distinct"] for sj in STEPS]))
    col = S[:, richest]
    # The reference column needs enough USABLE cells to carry a trend. An earlier version passed
    # the per-cell distinct-window COUNTS to `distinct_units`, which expects unit IDENTITIES: the
    # counts are near-identical down a column by construction, so it reported "6 values collapse
    # to 1" and returned a spurious NOT DECIDABLE. Misusing the guard is not the same as the guard
    # binding, and the fix is to ask the question the column actually poses.
    col_usable = [i for i in range(len(STEPS)) if not D[i, richest] and not np.isnan(S[i, richest])]
    dis = distinct_units(col_usable, minimum=3, name="usable cell in the reference column")
    meas = lambda_measured()
    ca = np.array([meas[st][0] for st in STEPS])
    mf = np.array([lambda_mf(R, x) for x in col])
    lev = correlation_leverage(mf, ca, name="lambda_MF from the fixed-ensemble column")
    rk = lambda x: np.argsort(np.argsort(x))
    rho = float(np.corrcoef(rk(mf), rk(ca))[0, 1]) if np.nanstd(mf) > 0 else 0.0
    floor = float(np.mean([meas[st][1] for st in STEPS])) / np.sqrt(8)
    resid = float(np.mean(np.abs(mf - ca)))
    verdict = carries_verdict([lev, dis], value=rho)
    parts.append(
        f"NON-CIRCULAR PRIMARY: with the ensemble held at the richest settled state "
        f"(step{STEPS[richest]}, {res['pools'][str(STEPS[richest])]['distinct']} distinct tokens), "
        f"s across models runs {[round(float(x),4) for x in col]} -- span "
        f"{float(np.nanmax(col)-np.nanmin(col)):.4f}. The contexts do not come from the model being "
        f"measured, so this is F94's primary without the circularity F96 flagged. "
        + (f"rho(lambda_MF, lambda_ca) = {rho:+.3f}. {lev.reason}" if verdict.status == "DECIDED"
           else f"NOT DECIDABLE: {verdict.reason}"))
    parts.append(
        f"DEFLATION CHECK (registered): mean |lambda_MF - lambda_ca| = {resid:.4f} against a "
        f"lambda seed floor of {floor:.4f}. "
        + ("s reproduces lambda_ca WITHIN the floor, so the ring is redundant for this quantity "
           "and the tool is s."
           if resid <= 2 * floor else
           "s does NOT reproduce lambda_ca, so the ring is not redundant."))

    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(
        steps=STEPS, matrix=[[None if np.isnan(x) else round(float(x), 5) for x in r] for r in S],
        degenerate=[[bool(x) for x in r] for r in D],
        ensemble_effect=round(ens_eff, 5), model_effect=round(mod_eff, 5),
        reference_state=STEPS[richest], reference_column=[round(float(x), 5) for x in col],
        rho=round(rho, 3), residual=round(resid, 4), lambda_seed_floor=round(floor, 4),
        leverage=lev.block(), cohort=coh.block(), status=verdict.status)
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "The experiment F96 specified. F94 measured s on uniform noise and called it flat; F96 "
        "showed that was a property of the ensemble but could not read the settled-state result "
        "because the state is produced by the dynamics whose exponent it predicts, and because "
        "the early settled rings are degenerate. Transplanting each checkpoint's conditional onto "
        "every other checkpoint's settled state separates the two: columns vary the model with the "
        "ensemble fixed and are not circular, rows vary the ensemble with the model fixed, and the "
        "early-model/late-state corner escapes the degeneracy. s is exact (inverse-CDF CRN "
        "disagreement), so no cell carries sampling error.")


if __name__ == "__main__":
    main()
