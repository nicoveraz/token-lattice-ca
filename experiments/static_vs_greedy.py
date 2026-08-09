"""Regenerate F92's deflation table into a results file. It never had one.

WHY THIS EXISTS. F92 quotes a four-row table -- the static argmax-map predictors and T\*, each
correlated against greedy degeneration on the same 8 finite-T\* families -- and F119's reconciliation
found that **no results file backed three of its four rows**. Only `T* vs rep_4 = 0.833` traced, as
`tstar_second_target.json -> analysis.tstar_vs_greedy_same_rows`. The static rows (`fix -0.119`,
`cyc +0.119`, `modal +0.595`) and the 15-family parenthetical came from a run whose output was never
persisted, so they could not be re-derived and could not be screened for F119's tie bug.

That is a provenance gap in a headline deflation: F92 is what licenses "the static map carries no
information about degeneration; the CA-derived T\* does", which is one of the two legs holding up
F86. A number that cannot be regenerated cannot be defended.

NO NEW MEASUREMENT. Every input already exists in `results/tstar_second_target.json -> analysis.rows`
(15 families, each with `tstar`, `greedy`, `fix`, `cyc`, `modal`, `top1`). This script recomputes the
correlations from those rows with the CORRECTED tie-aware ranking and an exact permutation null, and
writes them down. Nothing is measured that was not measured before; what changes is that the table
now exists on disk and is re-derivable.

PRE-REGISTERED:
  RUNG      `rho(tstar, greedy)` on the 8 finite-T* rows must reproduce the stored
            `tstar_vs_greedy_same_rows` = 0.833 to within rounding. That scalar IS backed, so it
            pins this script's row selection and ranking to the run F92 actually reported. A
            mismatch means the rows here are not F92's rows and nothing else may be read.
  PRIMARY   the same four predictors against `greedy`, on the same 8 rows, with exact permutation p.
            F92's claim is comparative -- the static predictors are weak where T* is strong -- so
            the table is the finding, not any single rho.
  SECONDARY the all-15-family version, which F92 quotes parenthetically.
  BOUNDARY  n = 8 families. This is a re-derivation of a recorded result, not new evidence for it;
            if the numbers differ from F92's, F92's are the ones that were never backed.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import itertools, json

import numpy as np
from ranking import spearman
from provenance import stamp, rel

SRC = str(_ROOT / "results" / "tstar_second_target.json")
OUT = str(_ROOT / "results" / "static_vs_greedy.json")
TARGET = "greedy"
PREDICTORS = ["fix", "cyc", "modal", "top1", "tstar"]
STATIC = ["fix", "cyc", "modal", "top1"]          # the argmax map read as a level
RUNG_EXPECT = 0.833
RUNG_TOL = 0.002


EXACT_MAX = 9          # 9! = 362880 enumerable; 15! is 1.3e12 and is not
N_SAMPLED = 100_000


def rho_p(a, b, seed=0):
    """Spearman with a permutation p: exact below EXACT_MAX, randomly SAMPLED above.

    The sampling must be random. `islice(permutations(x), N)` looks like a sample and is not -- it
    takes the lexicographically FIRST N orderings, which are all near-identical and produce a null
    concentrated around one value. That bug once cost this project a quoted p of 0.051 whose correct
    value was 0.0079, so the fallback here draws with a seeded Generator instead.
    """
    r = spearman(a, b)
    if not np.isfinite(r):
        return float("nan"), float("nan"), None
    a = np.asarray(a, float)
    if len(a) <= EXACT_MAX:
        null = [spearman(list(p), b) for p in itertools.permutations(a)]
        mode = "exact"
    else:
        g = np.random.default_rng(seed)
        null = [spearman(g.permutation(a), b) for _ in range(N_SAMPLED)]
        mode = f"sampled({N_SAMPLED})"
    null = np.array([v for v in null if np.isfinite(v)])
    return float(r), float(np.mean(np.abs(null) >= abs(r) - 1e-12)), mode


def table(rows):
    out = {}
    for k in PREDICTORS:
        # T* is CENSORED for families whose ladder never crossed the threshold -- those rows carry
        # None, not zero. Dropping them per-predictor keeps every other predictor on all 15 rows,
        # which is what F92's parenthetical reports; coercing them would invent a measurement.
        pairs = [(r[k], r[TARGET]) for r in rows if r.get(k) is not None]
        if len(pairs) < 3:
            out[k] = dict(rho=None, perm_p=None, null=None, span=None, n=len(pairs))
            continue
        vals = [v for v, _ in pairs]
        tgt = [t for _, t in pairs]
        r, p, mode = rho_p(vals, tgt)
        out[k] = dict(rho=(None if not np.isfinite(r) else round(r, 4)),
                      perm_p=(None if not np.isfinite(p) else round(p, 4)),
                      null=mode, n=len(pairs), span=round(float(max(vals) - min(vals)), 4))
    return out


def analyse(res):
    prim, sec = res["primary"], res["secondary"]
    parts = []
    got = prim["tstar"]["rho"]
    ok = got is not None and abs(got - RUNG_EXPECT) <= RUNG_TOL
    parts.append(
        f"RUNG (pins these rows to F92's): rho(tstar, {TARGET}) on the {res['n_primary']} "
        f"finite-T* families is {got}, against the stored tstar_vs_greedy_same_rows = {RUNG_EXPECT}. "
        + ("Reproduced, so the row selection and ranking match the run F92 reported and the static "
           "rows below are the same comparison."
           if ok else
           "NOT reproduced. These are not F92's rows, so nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, tstar_rho=got)
        res["verdict"] = " ".join(parts)
        return
    worst = max(abs(prim[k]["rho"]) for k in STATIC)
    parts.append(
        "PRIMARY: on the same 8 rows, "
        + ", ".join(f"{k} {prim[k]['rho']:+.3f} (p={prim[k]['perm_p']:.3f})" for k in STATIC)
        + f", against T* {prim['tstar']['rho']:+.3f} (p={prim['tstar']['perm_p']:.3f}). "
        + (f"Every static predictor is weaker than T*, the strongest reaching |rho| = {worst:.3f}. "
           f"F92's deflation is re-derived and now has a results file."
           if worst < abs(prim["tstar"]["rho"]) else
           f"A static predictor reaches |rho| = {worst:.3f}, at or above T*. F92's deflation does "
           f"NOT re-derive and the finding must be revisited."))
    parts.append(
        "SECONDARY (all 15 families, T* censored for 7): "
        + ", ".join(f"{k} {sec[k]['rho']:+.3f}" for k in STATIC) + ".")
    parts.append(
        f"BOUNDARY: n = {res['n_primary']} families, and this is a RE-DERIVATION of a recorded "
        f"result from stored rows -- no new measurement. Where these numbers differ from F92's "
        f"quoted table, F92's are the ones that were never backed by a file.")
    res["analysis"] = dict(rung_passes=True, tstar_rho=got, primary=prim, secondary=sec,
                           strongest_static=round(worst, 4))
    res["verdict"] = " ".join(parts)


def main():
    rows = json.load(open(SRC))["analysis"]["rows"]
    fin = [r for r in rows if r.get("tstar") is not None and r.get(TARGET) is not None]
    allr = [r for r in rows if r.get(TARGET) is not None]
    res = dict(_preregistration=dict(
        source=rel(SRC), target=TARGET, predictors=PREDICTORS,
        rung=f"rho(tstar,{TARGET}) on the finite-T* rows must reproduce the stored "
             f"tstar_vs_greedy_same_rows = {RUNG_EXPECT} +/- {RUNG_TOL}; a mismatch means these are "
             f"not F92's rows and stops the read",
        primary="the four static argmax-map predictors vs greedy degeneration on those same rows",
        note="no new measurement -- every input is already in the source file. This exists because "
             "F92's table had no results file behind it (F119).",
    ))
    res["n_primary"], res["n_secondary"] = len(fin), len(allr)
    res["families"] = [r["fam"] for r in fin]
    res["primary"], res["secondary"] = table(fin), table(allr)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  {'predictor':<10} {'rho(8)':>9} {'p':>8}    {'rho(15)':>9}")
    for k in PREDICTORS:
        a, b = res["primary"][k], res["secondary"][k]
        sec = "n/a" if b["rho"] is None else f"{b['rho']:+.4f}"
        print(f"  {k:<10} {a['rho']:>+9.4f} {a['perm_p']:>8.4f}    {sec:>9}")
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
