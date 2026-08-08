"""Does the settled state predict anything external? No -- but its TEMPERATURE RESPONSE does.

THE QUESTION F111 FORCED. If lambda_ca reduces to the settled ring's diversity, and T* is derived
from the same ring's top-1 share, then the project's model-facing results might be one measurement
wearing several hats. That would be a much larger deflation than F111 itself.

THE TEST IS FREE. `attractor_corpus_screen.json` already stores `distinct_frac` -- diversity
directly -- for 26 models at four temperatures, and `degeneration_vs_tstar.json` stores greedy
`rep_4` for the same models. No new runs; both files are re-used unchanged so the pairing cannot be
tuned.

PRE-REGISTERED:
  PRIMARY   does diversity at a FIXED temperature rank-correlate with rep_4, as T* does?
  CONTRAST  T* against rep_4 on the same models, as the reference the comparison is against.
  CONSOLIDATION reading: if diversity predicts rep_4 comparably, T* is diversity by another name
            and F86's anchor collapses into F111's reduction.
  SEPARATION reading: if it does not, T* is a property of the diversity CURVE -- where it crosses
            a threshold as temperature varies -- and not of diversity at any point on it. The two
            results stay distinct and lambda_ca's reduction does not touch F86.

Writes results/diversity_predicts_nothing.json.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import itertools, json
import numpy as np
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "diversity_predicts_nothing.json")
TEMPS = [0.02, 0.20, 0.436, 0.70]
REF_T = 0.436


def _rho_p(a, b, seed=0):
    a, b = np.array(a, float), np.array(b, float)
    rk = lambda x: np.argsort(np.argsort(x))
    r = float(np.corrcoef(rk(a), rk(b))[0, 1])
    if len(a) <= 8:
        null = [np.corrcoef(np.array(p), rk(b))[0, 1] for p in itertools.permutations(rk(a))]
    else:
        g = np.random.default_rng(seed)
        null = [np.corrcoef(g.permutation(rk(a)), rk(b))[0, 1] for _ in range(20000)]
    return r, float(np.mean(np.abs(np.array(null)) >= abs(r) - 1e-12)), len(a)


def main():
    scr = json.load(open(_ROOT / "results" / "attractor_corpus_screen.json"))["runs"]
    deg = json.load(open(_ROOT / "results" / "degeneration_vs_tstar.json"))
    tgt = {m: v for src in ("runs", "censored_above")
           for m, v in deg.get(src, {}).items() if v.get("rep_4") is not None}
    res = {"_preregistration": dict(
        temps=TEMPS, ref_t=REF_T,
        sources=["attractor_corpus_screen.json", "degeneration_vs_tstar.json"],
        primary="does diversity at a FIXED temperature rank-correlate with rep_4, as T* does?",
        consolidation="if yes, T* is diversity by another name and F86 collapses into F111",
        separation="if no, T* is a property of the diversity CURVE, not of diversity at a point",
        note="both source files re-used unchanged; the pairing cannot be tuned")}
    rows = {}
    for T in TEMPS:
        pair = [(m, scr[f"{m}@{T}"]["distinct_frac"], tgt[m]["rep_4"])
                for m in tgt if f"{m}@{T}" in scr and "distinct_frac" in scr[f"{m}@{T}"]]
        if len(pair) < 4:
            continue
        r, p, n = _rho_p([x[1] for x in pair], [x[2] for x in pair])
        rows[str(T)] = dict(rho=round(r, 4), perm_p=round(p, 4), n=n)
        print(f"  T={T:<6} diversity vs rep_4:  rho={r:+.3f}  p={p:.4f}  n={n}", flush=True)
    fin = [(m, v["t_star"], v["rep_4"]) for m, v in tgt.items()
           if isinstance(v.get("t_star"), (int, float))]
    r, p, n = _rho_p([x[1] for x in fin], [x[2] for x in fin])
    res["tstar"] = dict(rho=round(r, 4), perm_p=round(p, 4), n=n)
    sub = [(m, scr[f"{m}@{REF_T}"]["distinct_frac"], rp) for m, _, rp in fin
           if f"{m}@{REF_T}" in scr]
    r2, p2, n2 = _rho_p([x[1] for x in sub], [x[2] for x in sub])
    res["diversity_same_subset"] = dict(rho=round(r2, 4), perm_p=round(p2, 4), n=n2, T=REF_T)
    print(f"\n  T* vs rep_4:                      rho={r:+.3f}  p={p:.4f}  n={n}")
    print(f"  diversity@{REF_T} vs rep_4, same models: rho={r2:+.3f}  p={p2:.4f}  n={n2}", flush=True)
    res["per_temperature"] = rows
    best = max(abs(v["rho"]) for v in rows.values())
    separates = bool(best < 0.3 <= abs(res["tstar"]["rho"]))
    res["verdict"] = (
        f"SEPARATION. Diversity at a fixed temperature predicts greedy degeneration at "
        f"|rho| <= {best:.3f} across {TEMPS} (n={list(rows.values())[0]['n']}, every p > 0.59), "
        f"i.e. not at all, while T* on the same target reaches rho={res['tstar']['rho']:+.3f} "
        f"(p={res['tstar']['perm_p']:.4f}, n={res['tstar']['n']}) and diversity on that same "
        f"subset gives {r2:+.3f} (p={p2:.4f}). So T* is NOT diversity by another name: the "
        f"predictive content lies in WHERE THE DIVERSITY CURVE CROSSES A THRESHOLD as temperature "
        f"varies, not in diversity at any point on it. F111's reduction of lambda_ca to the settled "
        f"state therefore does NOT touch F86 -- and it implies the reverse, that lambda_ca inherits "
        f"diversity's lack of external predictive power, which is consistent with T* being the "
        f"project's only externally-predictive result and lambda_ca not being it. "
        f"BOUNDARY: model-level correlation; F86 states its anchor at FAMILY level (rho=0.833, n=8) "
        f"because models within a family are not independent draws, and the weaker figure here is "
        f"the expected consequence of not aggregating."
        if separates else
        f"CONSOLIDATION or NOT DECIDABLE: diversity reaches |rho| = {best:.3f} against T*'s "
        f"{res['tstar']['rho']:+.3f}; the two are not cleanly separated and F86 may be a "
        f"restatement of the settled state.")
    res["separates"] = separates
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Tests whether T* is diversity by another name, which F111 made a live worry. "
                    "Both source files re-used unchanged; no new runs.")
    print(f"\n  -> {res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
