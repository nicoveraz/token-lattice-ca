"""Does the FULL-VOCABULARY lattice rank models construction-independently? The test that matters.

WHAT FORCED THIS. The sub-alphabet invariance test returned zero on every readout: branching -0.050,
s_near +0.000, s_far -0.037, distinct -0.028, top1 +0.111, over 18 constructions, with a PASSING
seed-stability rung (0.71-0.97) and live lattices (median branching 1.046, 66/106 supercritical,
2 of 108 frozen). So in that family the model ordering is reproducible at a FIXED construction and
scrambled ACROSS constructions -- the instrument was ranking lattices, not models.

WHY THAT DID NOT SETTLE IT, AND WHY THIS DOES. Every headline result in this project -- the lambda_ca
training curve, F63/F64's corpus discrimination, T*/F86 -- runs on the FULL vocabulary. The
sub-alphabet family is known to be a bad construction on independent grounds (F109: no live regime
anywhere on its grid; F123: its far-position behaviour is controlled by the selection rule). A
failure there could be a fact about sub-alphabets rather than about the instrument. This runs the
identical design on the construction the claims actually use.

THE STAKES ARE SYMMETRIC, which is why it is worth running rather than arguing about:
  scrambled -> every full-vocabulary model comparison in the project is construction-relative, and
               the deflationary reading is close to complete
  invariant -> the sub-alphabet family is simply a bad construction, the main line is safe, and the
               contrast between the two families is itself the finding

CONSTRUCTIONS. Radius x temperature, the two knobs every full-vocabulary result fixes silently:
r in {2, 3, 4} x T in {0.7, 1.0, 1.3} = 9. No sub-alphabet, no selection rule -- the lattice runs on
the model's own vocabulary, which is what makes this the paper's construction rather than a variant.

READOUTS. lambda_ca is the headline and the one the paper's claims rest on; the others are carried so
a scrambled lambda cannot be blamed on a noisy estimator while some other quantity holds.

PRE-REGISTERED:
  RUNG      SEED STABILITY at fixed construction, two seeds, threshold RUNG_MIN. A readout that
            cannot rank models reproducibly at a fixed construction cannot be asked about invariance
            across them, and a scrambled result would otherwise be unreadable. Nothing below is read
            for a readout that fails it.
  PRIMARY   per readout, mean pairwise Spearman between the model-rankings different constructions
            produce. >= CONCORDANT is model-attributable; <= SCRAMBLED is construction-dominated.
  CONTRAST  the sub-alphabet numbers are quoted alongside, because the comparison between families
            is the point and neither number means much alone.
  BOUNDARY  three models is a 3-point ranking -- this identifies a scrambled readout far more
            confidently than it certifies an invariant one. One lattice size, one settle length.
            Ignited-only cells for lambda, since lambda on an unignited run is a sentinel not a
            measurement (lyapunov.DEAD_DAMAGE_FLOOR).
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import itertools, json, time

import numpy as np, torch
from ranking import spearman
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "fullvocab_invariance.json")
SUB = str(_ROOT / "results" / "construction_invariance.json")
MODELS = ["EleutherAI/pythia-410m", "gpt2", "gpt2-large"]
RADII = [2, 3, 4]
TEMPS = [0.7, 1.0, 1.3]
N, B, SETTLE, SWEEPS, BLOCK = 48, 16, 12, 22, 3
SEEDS = [20260810, 20260811]
# ignition is NOT carried: block_damage returns only cone and mean_damage, and the cone
# is already averaged over replicas, so per-replica ignition is not recoverable from it.
# An all-nan column would fail the rung and read as a finding about the readout.
READOUTS = ["lambda_ca", "mean_damage", "distinct", "top1"]
RUNG_MIN, CONCORDANT, SCRAMBLED = 0.6, 0.6, 0.3


def cell(rule, r, T, seed):
    from ar_probe import block_damage
    from lyapunov import lyap_from_cone, is_dead_damage_floor
    from dev_transition_phase3 import FIT_KW
    from ar_ca import run
    d = block_damage(rule, T=T, r=r, block=BLOCK, B=B, N=N, settle=SETTLE, sweeps=SWEEPS, seed=seed)
    lam = float(lyap_from_cone(d["cone"], N, **FIT_KW)[0])
    settled = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme="none", init="random",
                  seed=seed)["final"]
    pool = settled.reshape(-1)
    vals, cnt = np.unique(pool, return_counts=True)
    dead = bool(is_dead_damage_floor(lam))
    return dict(lambda_ca=(float("nan") if dead else lam),
                mean_damage=float(np.mean(d["cone"][-1])),
                distinct=float(len(vals)), top1=float(cnt.max() / cnt.sum()),
                dead_floor=dead)


def rankings(cells, seed, readout):
    out = {}
    for con in {c["construction"] for c in cells.values()}:
        vals = []
        for m in MODELS:
            k = f"{m}|{con}|s{seed}"
            if k not in cells:
                vals = None; break
            vals.append(cells[k][readout])
        if vals is not None:
            out[con] = vals
    return out


def analyse(res):
    cells, parts = res["cells"], []
    rung, primary = {}, {}
    for ro in READOUTS:
        a, b = rankings(cells, SEEDS[0], ro), rankings(cells, SEEDS[1], ro)
        agree = [spearman(a[c], b[c]) for c in sorted(set(a) & set(b))
                 if all(np.isfinite(x) for x in a[c] + b[c])]
        agree = [v for v in agree if np.isfinite(v)]
        rung[ro] = round(float(np.mean(agree)), 4) if agree else None
    ok = [ro for ro, v in rung.items() if v is not None and v >= RUNG_MIN]
    parts.append(
        "RUNG (seed stability at FIXED construction): "
        + ", ".join(f"{ro}={rung[ro]}" for ro in READOUTS)
        + f". {len(ok)} of {len(READOUTS)} readouts clear {RUNG_MIN}"
        + (f" -- {ok}; only these are asked the primary question."
           if ok else ". NONE clears it, so the construction question is undefined here and nothing "
                      "below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, seed_agreement=rung)
        res["verdict"] = " ".join(parts); return
    for ro in ok:
        r0 = rankings(cells, SEEDS[0], ro)
        live = [c for c in sorted(r0) if all(np.isfinite(x) for x in r0[c])]
        ps = [spearman(r0[x], r0[y]) for x, y in itertools.combinations(live, 2)]
        ps = [v for v in ps if np.isfinite(v)]
        primary[ro] = dict(mean_rho=round(float(np.mean(ps)), 4) if ps else None,
                           n_pairs=len(ps), n_constructions=len(live))
    inv = [ro for ro, v in primary.items() if v["mean_rho"] is not None
           and v["mean_rho"] >= CONCORDANT]
    scr = [ro for ro, v in primary.items() if v["mean_rho"] is not None
           and v["mean_rho"] <= SCRAMBLED]
    lam = primary.get("lambda_ca", {}).get("mean_rho")
    parts.append(
        "PRIMARY, agreement between the model-rankings different constructions produce: "
        + ", ".join(f"{ro}={primary[ro]['mean_rho']:+.3f}" for ro in primary)
        + f" over up to {max(v['n_constructions'] for v in primary.values())} constructions. "
        + (f"MODEL-ATTRIBUTABLE at {CONCORDANT}: {inv}. "
           if inv else f"NONE reaches {CONCORDANT}. ")
        + (f"lambda_ca -- the readout the paper's claims rest on -- scores {lam:+.3f}, "
           + ("so it ranks models the same way however the lattice is built and the full-vocabulary "
              "line is NOT construction-relative."
              if lam is not None and lam >= CONCORDANT else
              "so the headline readout is construction-dependent on the full vocabulary too, and "
              "every model comparison in this project is relative to the construction it was "
              "measured at."
              if lam is not None and lam <= SCRAMBLED else
              "which is between the registered thresholds -- undetermined, and it says so rather "
              "than picking a side.")
           if lam is not None else "lambda_ca did not survive the rung and is not read."))
    try:
        sub = json.load(open(SUB))["analysis"]["primary"]
        parts.append(
            "CONTRAST with the SUB-ALPHABET family, which is the point of running this: there "
            + ", ".join(f"{k}={v['mean_rho']:+.3f}" for k, v in sub.items())
            + ". Neither number means much alone; the comparison between families is what "
              "distinguishes 'the instrument ranks lattices' from 'sub-alphabets are a bad "
              "construction'.")
    except Exception:
        pass
    parts.append(
        f"BOUNDARY: {len(MODELS)} models, so each ranking has 3 points and one swap moves rho a long "
        f"way -- this identifies a SCRAMBLED readout far more confidently than it certifies an "
        f"invariant one. N={N}, one settle length, radii {RADII}, temperatures {TEMPS}. lambda on an "
        f"unignited run is a sentinel rather than a measurement and is carried as nan.")
    res["analysis"] = dict(rung_passes=True, seed_agreement=rung, readouts_asked=ok,
                           primary=primary, invariant=inv, scrambled=scr, lambda_rho=lam)
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, radii=RADII, temps=TEMPS, N=N, B=B, settle=SETTLE, sweeps=SWEEPS,
        block=BLOCK, seeds=SEEDS, readouts=READOUTS, rung_min=RUNG_MIN,
        concordant=CONCORDANT, scrambled=SCRAMBLED, sub_alphabet_reference=rel(SUB),
        rung="model ordering must agree across two seeds at fixed construction",
        primary="mean pairwise Spearman between the model-rankings different constructions give",
        stakes="scrambled -> full-vocabulary model comparisons are construction-relative; "
               "invariant -> the sub-alphabet family is simply a bad construction")
    from ar_ca import ARRule
    for m in MODELS:
        rule = ARRule(m)
        for r in RADII:
            for T in TEMPS:
                for sd in SEEDS:
                    key = f"{m}|r{r}.T{T}|s{sd}"
                    if key in res["cells"]:
                        continue
                    t0 = time.time()
                    c = cell(rule, r, T, sd)
                    c.update(model=m, construction=f"r{r}.T{T}", r=r, T=T, seed=sd,
                             secs=round(time.time() - t0, 1))
                    res["cells"][key] = c
                    print(f"  {key:<40} lambda={c['lambda_ca']:+.4f} "
                          f"dmg={c['mean_damage']:.3f} distinct={c['distinct']:.0f} "
                          f"top1={c['top1']:.3f} ({c['secs']:.0f}s)", flush=True)
                    json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
