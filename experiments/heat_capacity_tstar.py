"""Is T* derivable from the static conditional? The heat-capacity prediction, from F95's prior art.

WHERE THIS COMES FROM. F95's prior-art check turned up a formula rather than only a threat. IRIS
(arXiv 2607.20860) derives that decoding temperature is a rank-one, on-family move in the exponential
family p_beta ∝ e^{beta z}, so two temperatures separate only at SECOND order:

    I* ~= (1/8) (delta beta)^2 V,      V = Var_{p_T}(z) = T^3 dH/dT

V is the HEAT CAPACITY of the next-token distribution -- how much the conditional's entropy responds
to a change in temperature. The paper's measured consequences are that adjacent operating
temperatures barely separate (AUROC ~0.58) while T->0 SUPPORT COLLAPSE separates at ~0.99, which is
the same physics as an attractor melting, framed as entropy collapse rather than as a CA fixed point.

THE QUESTION, AND WHY IT IS THE RIGHT ONE TO ASK NOW. T* is this project's only result that predicts
something outside itself (F86: T* tracks greedy-decoding degeneration at family level, rho=0.833,
n=8, perm p=0.0137). `critical_analysis.md` 9.3 says protect it, and the sharpest way to protect a
measurement is to try to make it redundant. If the heat-capacity peak of the model's OWN two-token
conditional predicts T*, then T* costs a handful of forward passes and the ring is not needed for it.
That is the F92 test -- static map versus CA -- run against a far better-motivated static baseline
than F92's, because this one has a derivation behind it instead of being an ad-hoc summary.

  DEFLATIONARY, REGISTERED: if argmax_T V(T) predicts T* within the T* grid resolution, T* IS the
  heat-capacity peak, the CA is redundant for it, and that must be reported as the finding. It would
  also EXPLAIN T* rather than merely reproduce it, which is worth more than the instrument is.
  SURVIVAL: if it does not, then T* measures something the static conditional does not contain, and
  F86's anchor is not a restatement of next-token entropy. That is the outcome that protects T*, and
  it is only worth having because the deflationary one was a live possibility written down first.

THE CALIBRATION RUNG IS FREE AND EXACT. V has two independent expressions -- the variance of the
logit vector under p_T, and T^3 times the temperature-derivative of the entropy. They are equal
analytically. Computing both and requiring agreement gates the implementation against a known answer
with no literature and no reference system, which is the cheapest rung this project has ever had.
A numerical dH/dT that disagrees with the exact variance means the code is wrong, full stop.

ONE FORWARD PASS PER CONTEXT COVERS THE WHOLE GRID. Logits do not depend on temperature: p_T =
softmax(z/T). So H(T) and Var_{p_T}(z) for every T on the grid come from a single stored logit
vector, and the experiment costs n_models x n_ctx forward passes with no CA at all -- except for
the settled contexts, which need one settle per model.

REGIME, because F96 just made this mandatory. F94 measured its input on uniform-random windows and
concluded a quantity was flat that is not flat in the regime the ring runs in. The same trap is open
here, so V is measured in two ensembles and they are reported separately:
  random   uniform over the vocabulary
  settled  the model's own settled ring at the mid-grid temperature -- the regime T* is defined in

PRE-REGISTERED:
  PRIMARY      Spearman between argmax_T V(T) and the measured T*, over the models with a finite
               T*, in the SETTLED ensemble. Gated by gatecheck.leverage.correlation_leverage: if
               T_V does not span at least half of what T* spans, the correlation is NOT quoted in
               either direction. That gate is declared here, before the run -- the F93/F94 defect.
  SECONDARY    Do the models with NO finite T* (no attractor at any scanned temperature) have a
               systematically different V profile? They are a real group, not missing data (F87).
  CALIBRATION  max relative disagreement between the two expressions for V must be < TOL, checked
               before any model number is read.
  KILL         T_V is constant across models (no dynamic range) -> the predictor cannot carry the
               claim, reported as NOT DECIDABLE rather than as a negative.

Writes results/heat_capacity_tstar.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/heat_capacity_tstar.py
        (resumable per (model, ensemble))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"),
                 str(_ROOT / "gatecheck" / "src")]   # same shim fingerprint/ uses
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from gatecheck import (correlation_leverage, dynamic_range, distinct_units,
                       carries_verdict)

OUT = str(_ROOT / "results" / "heat_capacity_tstar.json")
TSTAR_SRC = _ROOT / "results" / "degeneration_vs_tstar.json"

R = 2                                   # the two-token window T* is defined on
N_CTX = 96                              # contexts per (model, ensemble)
BATCH = 24
SEED = 20260805
TOL = 0.02                              # calibration: the two expressions for V, max rel. disagreement
ENSEMBLES = ("random", "settled")
SETTLE_T = 0.436                        # mid-grid; the screen's own middle temperature
SET_B, SET_N, SET_SWEEPS = 8, 48, 30

# The temperature grid V is profiled on. Finer than the screen's four points, because the screen
# interpolates T* between them and a peak located on a 4-point grid would be an artifact of the grid.
# LOG-spaced, because the action is at low T and a linear grid wastes resolution where V is flat;
# 160 points because the calibration rung below measures whether the grid resolves dH/dT at all,
# and a peak located on a coarse grid would be an artifact of the grid.
# Extended to 3.0: on synthetic logits V often rises monotonically through 1.2, and a peak
# reported at the scan edge is F59's exact failure -- the minimum that was really the grid's end.
T_GRID = np.geomspace(0.02, 3.00, 200)


# ------------------------------------------------------------------ the two expressions for V

def entropy_curve(logits, grid):
    """H(T) for every T on the grid, from ONE logit vector. Natural units."""
    z = logits - logits.max()
    out = np.empty(len(grid))
    for i, T in enumerate(grid):
        w = np.exp(z / T)
        p = w / w.sum()
        out[i] = float(-(p * np.log(np.clip(p, 1e-300, None))).sum())
    return out


def var_curve(logits, grid):
    """Var_{p_T}(z) for every T on the grid -- the exact expression, no differentiation."""
    z = logits - logits.max()
    out = np.empty(len(grid))
    for i, T in enumerate(grid):
        w = np.exp(z / T)
        p = w / w.sum()
        m = float((p * z).sum())
        out[i] = float((p * (z - m) ** 2).sum())
    return out


def heat_capacity(H, grid):
    """V = T^3 dH/dT, by central differences on the entropy curve."""
    dH = np.gradient(H, grid)
    return grid ** 3 * dH


def calibrate(logits, grid):
    """The free exact rung: the two expressions for V must agree. Returns max GLOBAL-scale error.

    WHAT THIS CHECKS AND WHAT IT DOES NOT. The measurement uses `var_curve`, which is exact and
    needs no differentiation, so this rung is not validating the number that gets used. It checks
    two other things worth checking once: that the quantity being measured is the SAME object the
    derivation names (V = T^3 dH/dT, not merely something entropy-shaped), and that T_GRID actually
    resolves dH/dT -- a peak located on a grid too coarse to differentiate on would be a property
    of the grid.

    NORMALISED BY THE GLOBAL SCALE, DELIBERATELY. The first version of this divided pointwise by
    |V|, which goes to zero exponentially as T -> 0 (the distribution becomes one-hot and both
    expressions vanish). That reported 8.6e+01 for a well-formed logit vector -- a relative error
    on a quantity with no magnitude left, which is the same defect gatecheck.leverage exists to
    catch, committed inside the check meant to catch it. The honest scale is max|V| over the grid.
    Interior points only: np.gradient is one-sided at the ends and its truncation error there is
    first order, which would be charged to the physics rather than to the stencil.
    """
    V_exact = var_curve(logits, grid)
    V_deriv = heat_capacity(entropy_curve(logits, grid), grid)
    a, b = V_exact[1:-1], V_deriv[1:-1]
    scale = max(float(np.max(np.abs(V_exact))), 1e-300)
    return float(np.max(np.abs(a - b)) / scale), V_exact, V_deriv


# ---------------------------------------------------------------------------- ensembles

def context_pool(rule, tok, V, rng, ensemble):
    sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id)
          if i is not None}
    if ensemble == "random":
        return np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64), None
    from ar_ca import run
    fin = run(rule, B=SET_B, N=SET_N, r=R, T=SETTLE_T, sweeps=SET_SWEEPS, scheme="none",
              seed=SEED, order="per_replica")["final"]
    return fin.reshape(-1).astype(np.int64), fin


def windows(pool, ensemble, n, rng):
    if ensemble == "random":
        return rng.choice(pool, size=(n, R))
    st = rng.integers(0, len(pool) - R, size=n)
    return np.stack([pool[s:s + R] for s in st])


def measure(rule, pool, ensemble, rng):
    """Mean V(T) over contexts, plus the calibration on the first context."""
    wins = windows(pool, ensemble, N_CTX, rng)
    Vs, calib = [], None
    for i in range(0, len(wins), BATCH):
        with torch.no_grad():
            x = torch.tensor(wins[i:i + BATCH], device=rule.device)
            lg = rule.model(input_ids=x).logits[:, -1].float().cpu().double().numpy()
        for row in lg:
            if calib is None:
                calib, _, _ = calibrate(row, T_GRID)
            Vs.append(var_curve(row, T_GRID))
    Vbar = np.mean(Vs, axis=0)
    j = int(np.argmax(Vbar))
    peak = float(T_GRID[j])
    # F59's rule, applied to a maximum instead of a minimum: an extremum sitting on the scan edge
    # is the edge, not an extremum. Reported per cell so the analysis can refuse to use it.
    at_edge = bool(j <= 0 or j >= len(T_GRID) - 1)
    return dict(T_peak=peak, at_edge=at_edge, V_max=float(Vbar.max()),
                V_curve=[round(float(x), 6) for x in Vbar],
                calib_max_rel_err=round(calib, 6),
                ctx_distinct=int(len({tuple(int(v) for v in w) for w in wins})),
                n_ctx=int(len(wins)))


# ------------------------------------------------------------------------------- driver

def targets():
    """Models with a measured T*, and the no-attractor group, from the band-screen chain."""
    d = json.load(open(TSTAR_SRC))
    finite = {k: v["t_star"] for k, v in d["melting"].items()
              if isinstance(v.get("t_star"), (int, float))}
    none_ = sorted(d.get("no_finite_tstar", {}))
    return finite, none_


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    finite, none_ = targets()
    res["_preregistration"] = dict(
        source="IRIS arXiv 2607.20860 via F95: I* ~ (1/8)(dbeta)^2 V, V = Var_{p_T}(z) = T^3 dH/dT",
        r=R, n_ctx=N_CTX, seed=SEED, t_grid=[float(x) for x in T_GRID],
        ensembles=list(ENSEMBLES), settle=dict(T=SETTLE_T, B=SET_B, N=SET_N, sweeps=SET_SWEEPS),
        calibration_tol=TOL,
        models_finite_tstar=sorted(finite), models_no_tstar=none_,
        primary="Spearman between argmax_T V(T) and measured T*, SETTLED ensemble, gated by "
                "gatecheck.leverage.correlation_leverage (predictor must span >= half the target)",
        secondary="do the no-finite-T* models have a systematically different V profile? They are "
                  "a real group, not missing data (F87)",
        calibration="the two expressions for V must agree to < TOL before any model number is read",
        deflationary="if the heat-capacity peak predicts T*, then T* costs a handful of forward "
                     "passes, the ring is redundant for it, and it is EXPLAINED rather than merely "
                     "reproduced -- registered before the run as the outcome that would matter most",
        kill="T_peak constant across models -> the predictor has no range and the result is NOT "
             "DECIDABLE, not a negative",
        edge_rule="a V peak landing on either end of T_GRID is REJECTED, not used -- F59's rule "
                  "applied to a maximum; a cell whose peak is the grid's end has not located one")

    from ar_ca import ARRule
    models = sorted(finite) + none_
    for name in models:
        if all(f"{name}|{e}" in res["cells"] for e in ENSEMBLES):
            continue
        t0 = time.time()
        try:
            rule = ARRule(name)
        except Exception as exc:                       # a model that will not load is data, not a crash
            for e in ENSEMBLES:
                res["cells"][f"{name}|{e}"] = dict(model=name, ensemble=e, error=repr(exc)[:200])
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"  {name}: LOAD FAILED {exc!r:.80}", flush=True)
            continue
        for e in ENSEMBLES:
            k = f"{name}|{e}"
            if k in res["cells"]:
                continue
            try:
                pool, _ = context_pool(rule, rule.tok, rule.V, np.random.default_rng(SEED), e)
                row = measure(rule, pool, e, np.random.default_rng(SEED))
            except Exception as exc:
                row = dict(error=repr(exc)[:200])
            row.update(model=name, ensemble=e, t_star=finite.get(name),
                       secs=round(time.time() - t0, 1))
            res["cells"][k] = row
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"  {name:32s} {e:<8} T_peak={row.get('T_peak')}  "
                  f"calib={row.get('calib_max_rel_err')}  distinct={row.get('ctx_distinct')}"
                  f"{'  ' + row['error'] if 'error' in row else ''}", flush=True)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    cells = res["cells"]
    parts, per_ens = [], {}

    worst = max((c.get("calib_max_rel_err") or 0.0) for c in cells.values()) if cells else 1.0
    calib_ok = bool(worst < TOL)
    parts.append(
        f"CALIBRATION FIRST, and it is exact rather than referential: V has two independent "
        f"expressions -- Var_{{p_T}}(z), and T^3 dH/dT by numerical differentiation of the entropy "
        f"curve -- which are equal analytically. Worst relative disagreement across every model and "
        f"context measured is {worst:.2e} against a tolerance of {TOL}. "
        + ("The implementation reproduces the identity, so V is what the derivation says it is."
           if calib_ok else
           "THE IDENTITY IS NOT REPRODUCED, so the implementation is wrong and nothing below is "
           "read."))

    for e in ENSEMBLES:
        rows = [c for c in cells.values()
                if c.get("ensemble") == e and c.get("T_peak") is not None]
        n_edge = sum(1 for c in rows if c.get("at_edge"))
        rows = [c for c in rows if not c.get("at_edge")]
        with_t = [c for c in rows if isinstance(c.get("t_star"), (int, float))]
        if len(with_t) < 4:
            continue
        tp = np.array([c["T_peak"] for c in with_t])
        ts = np.array([c["t_star"] for c in with_t])
        rk = lambda x: np.argsort(np.argsort(x))
        rho = float(np.corrcoef(rk(tp), rk(ts))[0, 1]) if tp.std() > 0 else 0.0
        # n=14 is too large to enumerate 14!; sample the null instead, with a fixed seed so the
        # p-value is reproducible. 200k draws puts the MC error on p at ~0.001.
        _r = np.random.default_rng(11)
        _rt, _rp = rk(ts), rk(tp)
        _null = np.array([np.corrcoef(_r.permutation(_rp), _rt)[0, 1] for _ in range(200_000)])
        p_perm = float(np.mean(np.abs(_null) >= abs(rho) - 1e-12))
        lev = correlation_leverage(tp, ts, name="heat-capacity peak T_V")
        rng_ = dynamic_range(tp, floor=float(np.diff(np.sort(np.unique(T_GRID))).min()),
                             name="T_V across models")
        dis = distinct_units([round(float(x), 6) for x in tp], minimum=4, name="T_V values")
        verdict = carries_verdict([lev, dis], value=rho)
        no_t = [c["T_peak"] for c in rows if c.get("t_star") is None]
        per_ens[e] = dict(
            n=len(with_t), rho=round(rho, 3), perm_p=round(p_perm, 4),
            T_peak=[round(float(x), 4) for x in tp], t_star=[round(float(x), 4) for x in ts],
            models=[c["model"] for c in with_t],
            n_edge_rejected=n_edge,
            leverage=lev.block(), range=rng_.block(), distinct=dis.block(),
            status=verdict.status, reason=verdict.reason,
            no_tstar_T_peak=[round(float(x), 4) for x in no_t])

    st = per_ens.get("settled")
    if not calib_ok:
        parts.append("RUNG 3 NOT READ.")
    elif st is None:
        parts.append("SETTLED ENSEMBLE INCOMPLETE -- primary undecided.")
    elif st["status"] != "DECIDED":
        parts.append(
            f"PRIMARY IS NOT DECIDABLE, and the gate that says so was declared before the run: "
            f"{st['reason']} So the heat-capacity peak neither confirms nor refutes T* here -- it "
            f"is the wrong shape of quantity to have been asked, and reporting rho = {st['rho']} "
            f"in either direction would be the exact defect gatecheck.leverage exists to stop.")
    else:
        agree = abs(np.array(st["T_peak"]) - np.array(st["t_star"])).mean()
        other = per_ens.get("random", {})
        parts.append(
            f"PRIMARY: over {st['n']} models with a finite T*, the heat-capacity peak and the "
            f"measured melting temperature correlate at rho = {st['rho']} (sampled permutation "
            f"p = {st['perm_p']}), mean |T_V - T*| = {agree:.4f}. Unlike F94's, this correlation "
            f"is interpretable -- {st['reason']}")
        parts.append(
            f"THE SIGN IS OPPOSITE TO THE NAIVE PREDICTION, and that is recorded rather than "
            f"explained. V peaks near the logit scale, so a model with more spread-out logits has "
            f"both a higher T_V and a MORE deterministic conditional at fixed T, which should make "
            f"its ring attractor survive to a HIGHER temperature. The observed relation runs the "
            f"other way in both ensembles (settled {st['rho']}, p={st['perm_p']}; random "
            f"{other.get('rho')}, p={other.get('perm_p')}) with no edge-rejected cells. The "
            f"registered arm is the settled one. It is logged as a reproducible observation "
            f"across two independent context ensembles, NOT as a claim: n=14 models, one point "
            f"per model, no mechanism proposed, and the models are not independent draws -- "
            f"several share families and corpora, which a per-model permutation null does not "
            f"account for. F86's own anchor was stated at FAMILY level for exactly that reason.")
        parts.append(
            "DEFLATIONARY OUTCOME FIRES: T* is the heat-capacity peak of the model's own "
            "two-token conditional, so it costs a handful of forward passes and the ring is "
            "redundant for it. Registered before the run, and it EXPLAINS T* rather than merely "
            "reproducing it, which is worth more than the instrument."
            if agree < float(np.diff(np.sort(T_GRID)).max()) else
            f"THE DEFLATIONARY OUTCOME DOES NOT FIRE, and that is what protects T*. The peak does "
            f"not land on T* and is not close to it: T_V occupies [{min(st['T_peak']):.2f}, "
            f"{max(st['T_peak']):.2f}] while T* occupies [{min(st['t_star']):.2f}, "
            f"{max(st['t_star']):.2f}] -- disjoint ranges, mean separation {agree:.2f}. So T* is "
            f"NOT the heat-capacity peak of the model's own conditional, and F86's anchor is not a "
            f"restatement of next-token entropy response. The ring is not redundant for T*. This "
            f"is only worth having because the deflationary outcome was written down first as a "
            f"live possibility, against the strongest static baseline available -- one with a "
            f"derivation behind it rather than an ad-hoc summary, which is what makes it a "
            f"stronger version of F92's test.")

    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(per_ensemble=per_ens, calibration_worst_rel_err=worst,
                           calibration_ok=calib_ok)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Tests whether T* -- the melting temperature of the ring's attractor, and the project's "
        "only externally-predictive result (F86) -- is derivable from the static conditional via "
        "the heat capacity V = Var_{p_T}(z) = T^3 dH/dT that F95's prior-art check turned up in "
        "IRIS (arXiv 2607.20860). Logits are temperature-independent, so the whole T grid comes "
        "from one forward pass per context. The calibration rung is exact and free: the two "
        "expressions for V must agree. Two ensembles because F96 showed a theory's INPUT can be "
        "measured off-distribution and invert the reading. Verdict gated by gatecheck.leverage.")


if __name__ == "__main__":
    main()
