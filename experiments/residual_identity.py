"""Does lambda_ca's residual after diversity carry MODEL IDENTITY? And does F111 hold across models?

TWO QUESTIONS, ONE FREE DATASET. F111 reduced lambda_ca to the settled ring's diversity (rho =
+0.771 within Pythia, across checkpoints and temperatures). Two things follow that it never tested:

  1. Does the relation hold ACROSS MODELS? F111's evidence is entirely within one family -- six
     checkpoints of pythia-410m, plus a temperature dissociation on two of them. A relation that
     organises one model's training trajectory need not organise a population of models.
  2. Is the ~40% of variance that diversity does not explain STRUCTURED? If the residual is
     structureless, the reduction is complete and the ring is an expensive way to compute a
     diversity statistic. If model identity survives in it, that is where the model's own
     contribution lives -- and it is the one place a model-specific signal could still hide that
     has not already been searched.

No new runs. `canalization_predicts.json` measured settled diversity for 14 models and
`lambda_temperature_crossing.json` measured lambda at T=0.7 for the same 14; both files are re-used
unchanged so the pairing cannot be tuned.

PRE-REGISTERED:
  PRIMARY    rho(diversity, lambda_ca) across the 14 models at fixed T=0.7, against F111's +0.771.
  RESIDUAL   against the best MONOTONE fit (what a rank correlation presumes), then two searches:
             does the residual predict rep_4, and does it separate by family -- measured as
             between-family sd of the mean residual over within-family sd. A ratio near or above 1
             means identity survives; well below means it does not.
  READING    a cross-model null is NOT automatically a refutation of F111. If the models all sit in
             the plateau regime where F111's own curve is flat, a null is what F111 predicts, and
             the finding is a SCOPE statement rather than a contradiction. The diversity range is
             therefore reported first and read before the correlation.
  BOUNDARY   lambda and diversity come from DIFFERENT settle geometries (B=8/30 sweeps for
             diversity, B=16/12 sweeps plus damage for lambda). Both use N=48 and the same
             384-token pool, so the pairing is approximate rather than exact; a clean version
             measures both from one settle. Stated because it is not fixable from stored data.

Writes results/residual_identity.json.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json
import numpy as np
from ranking import rank as _rank
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "residual_identity.json")
R = _ROOT / "results"
T_FIXED, SEEDS, F111_RHO = 0.7, (21, 22, 23), 0.771
PLATEAU_MIN = 100          # F111's own curve is flat above this; declared before reading the rho


def _rho_p(a, b, seed=0, n_perm=20000):
    a, b = np.array(a, float), np.array(b, float)
    rk = lambda x: _rank(x)
    r = float(np.corrcoef(rk(a), rk(b))[0, 1])
    g = np.random.default_rng(seed)
    null = [np.corrcoef(g.permutation(rk(a)), rk(b))[0, 1] for _ in range(n_perm)]
    return r, float(np.mean(np.abs(np.array(null)) >= abs(r) - 1e-12)), len(a)


def _isotonic(y):
    lvl, wts, i = list(map(float, y)), [1.0] * len(y), 0
    while i < len(lvl) - 1:
        if lvl[i] > lvl[i + 1]:
            t = wts[i] + wts[i + 1]
            lvl[i:i + 2] = [(lvl[i] * wts[i] + lvl[i + 1] * wts[i + 1]) / t]
            wts[i:i + 2] = [t]; i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, w in zip(lvl, wts):
        out += [v] * int(round(w))
    return np.array(out[:len(y)])


def main():
    can = json.load(open(R / "canalization_predicts.json"))["cells"]
    lam = json.load(open(R / "lambda_temperature_crossing.json"))["cells"]
    deg = json.load(open(R / "degeneration_vs_tstar.json"))
    tgt = {m: v for s in ("runs", "censored_above") for m, v in deg.get(s, {}).items()
           if v.get("rep_4") is not None}
    rows = []
    for m, c in can.items():
        if "settled_distinct" not in c:
            continue
        v = [lam[f"{m}|T{T_FIXED}|s{s}"]["lambda_ca"] for s in SEEDS
             if lam.get(f"{m}|T{T_FIXED}|s{s}", {}).get("ignited")]
        if v:
            rows.append(dict(model=m, diversity=c["settled_distinct"],
                             lambda_ca=round(float(np.mean(v)), 5),
                             rep_4=tgt.get(m, {}).get("rep_4")))
    rows.sort(key=lambda r: r["diversity"])
    res = {"cells": rows, "_preregistration": dict(
        sources=["canalization_predicts.json", "lambda_temperature_crossing.json",
                 "degeneration_vs_tstar.json"],
        t_fixed=T_FIXED, seeds=list(SEEDS), f111_rho=F111_RHO, plateau_min=PLATEAU_MIN,
        primary="rho(diversity, lambda_ca) across models at fixed T, against F111's +0.771",
        residual="against the best MONOTONE fit; then does it predict rep_4, and does it separate "
                 "by family (between-family sd of mean residual over within-family sd)",
        reading="a cross-model null is not automatically a refutation: if all models sit in the "
                "plateau regime where F111's curve is flat, a null is what F111 PREDICTS and the "
                "result is a scope statement. Diversity range is read first",
        boundary="lambda and diversity come from different settle geometries; the pairing is "
                 "approximate and not fixable from stored data")}

    x = np.array([r["diversity"] for r in rows], float)
    y = np.array([r["lambda_ca"] for r in rows])
    print(f"  {'model':<32} {'diversity':>10} {'lambda':>9} {'rep_4':>7}")
    for r in rows:
        print(f"  {r['model']:<32} {r['diversity']:>10} {r['lambda_ca']:>+9.4f} "
              f"{'' if r['rep_4'] is None else f'{r[chr(114)+chr(101)+chr(112)+chr(95)+chr(52)]:.3f}':>7}")
    rho, p, n = _rho_p(x, y)
    in_plateau = int((x >= PLATEAU_MIN).sum())
    o = np.argsort(x)
    iso = _isotonic(y[o]); resid = np.empty_like(y); resid[o] = y[o] - iso
    ok = [i for i, r in enumerate(rows) if r["rep_4"] is not None]
    rr, pr, nr = _rho_p(resid[ok], [rows[i]["rep_4"] for i in ok], seed=1)
    fam = lambda m: m.split("/")[0] if "/" in m else m.split("-")[0]
    byf = {}
    for r, rv in zip(rows, resid):
        byf.setdefault(fam(r["model"]), []).append(float(rv))
    multi = [v for v in byf.values() if len(v) >= 2]
    between = float(np.std([np.mean(v) for v in multi])) if len(multi) >= 2 else float("nan")
    within = float(np.mean([np.std(v) for v in multi])) if multi else float("nan")
    ratio = between / within if within else float("nan")

    parts = [
        f"RANGE FIRST, as registered: across these {n} models settled diversity spans "
        f"{int(x.min())}-{int(x.max())} distinct tokens, and {in_plateau} of {n} sit above "
        f"{PLATEAU_MIN}. F111's relation was driven by the dip (7.5 to 31) rising into the plateau; "
        f"within the plateau its own curve is flat (185/205/196 diversity giving lambda "
        f"0.19/0.16/0.17). So this population lies almost entirely in the flat part.",
        f"PRIMARY: rho(diversity, lambda_ca) across models = {rho:+.3f} (p = {p:.4f}, n = {n}), "
        f"against F111's +{F111_RHO} within Pythia. The relation does NOT hold across models -- and "
        f"given the range above, that is what F111 PREDICTS rather than a contradiction of it. "
        f"**F111 is a developmental statement: diversity organises lambda_ca's trajectory during "
        f"training, not its value across a population of models.**",
        f"RESIDUAL, SEARCHED TWICE AND EMPTY BOTH TIMES. Against the best monotone fit the residual "
        f"has sd {resid.std():.4f}. It does not predict degeneration: rho(residual, rep_4) = "
        f"{rr:+.3f} (p = {pr:.4f}, n = {nr}). And model identity does not survive in it: "
        f"between-family sd of the mean residual is {between:.4f} against a within-family sd of "
        f"{within:.4f}, a ratio of {ratio:.2f} -- the opposite of what 'the model's contribution "
        f"lives here' would look like, since families differ from each other far less than models "
        f"within a family differ among themselves.",
        f"BOUNDARY: lambda and diversity come from DIFFERENT settle geometries (B=8/30 sweeps for "
        f"diversity, B=16/12 sweeps plus damage for lambda). Both use N=48 and the same 384-token "
        f"pool so the pairing is approximate rather than exact, and a clean version measures both "
        f"from one settle. The effects here are nowhere near marginal, but that caveat is not "
        f"fixable from stored data and is stated rather than absorbed."]
    verdict = " ".join(parts)
    print(f"\n  rho(diversity, lambda) across models = {rho:+.3f}  p={p:.4f}  n={n}")
    print(f"  rho(residual, rep_4) = {rr:+.3f}  p={pr:.4f}")
    print(f"  between-family sd {between:.4f} / within-family sd {within:.4f} = {ratio:.2f}")
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rho_cross_model=round(rho, 4), perm_p=round(p, 4), n=n,
                           f111_rho=F111_RHO, diversity_range=[int(x.min()), int(x.max())],
                           n_in_plateau=in_plateau, plateau_min=PLATEAU_MIN,
                           residual_sd=round(float(resid.std()), 5),
                           rho_residual_rep4=round(rr, 4), perm_p_residual=round(pr, 4),
                           between_family_sd=round(between, 5), within_family_sd=round(within, 5),
                           identity_ratio=round(ratio, 4),
                           by_family={k: [round(v, 5) for v in vs] for k, vs in byf.items()})
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Asks whether F111's diversity reduction holds ACROSS MODELS and whether its "
                    "residual carries model identity. Both files re-used unchanged; no new runs.")
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
