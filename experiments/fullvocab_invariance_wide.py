"""Does lambda_ca have ANY cross-model signal on a set that actually spans families?

WHAT F128 LEFT OPEN. On three mid-size English LMs (pythia-410m, gpt2, gpt2-large) the across-model
spread in lambda_ca was 0.010-0.031 while the construction moved it 0.68 -- a ~30x ratio -- so the
seed-stability rung failed and the invariance question could not be asked. That is either a fact
about LAMBDA_CA or a fact about THAT TRIO, and three similar models cannot tell the two apart.

THE MODEL SET IS THE WHOLE POINT HERE. Ten models across six families and four architecture classes,
all locally cached: Pythia (31m/160m/410m), GPT-2 (base/medium/large), OPT, BLOOM, and two
NON-ATTENTION models -- Mamba (state-space) and RWKV (linear recurrence). F64 found architecture is
where this instrument's largest effects live (RWKV has no attractor at all), so if lambda_ca carries
cross-model information anywhere, a set containing non-attention architectures is where it shows.

THE QUESTION ORDER IS REVERSED FROM F126/F128, deliberately. Those asked "is the ranking
construction-invariant?" and discovered too late that there was no reliable ranking to begin with.
Here the prior question is asked FIRST and gates the rest.

PRE-REGISTERED:
  PRIMARY   SIGNAL, before invariance. Per construction, the across-model spread in lambda_ca must
            exceed the across-seed spread by NOISE_FACTOR. This is F128's failure turned into the
            registered question: a readout whose model differences are smaller than its own noise
            carries no cross-model information, and no amount of construction-averaging creates any.
            Reported as the fraction of constructions with real signal.
  RUNG      seed stability at fixed construction, the same gate F128 failed. Retained because it is
            the operational form of the primary and because passing it here while F128 failed is
            itself informative about the model set.
  SECONDARY invariance -- mean pairwise agreement between the model-rankings different constructions
            produce -- asked ONLY for readouts whose signal gate passes. Asking it of a noise
            ranking is what F126 nearly did.
  BOUNDARY  ten models is a 10-point ranking, far better than F128's three, but they differ in size
            as well as family, so a cross-model signal here would not by itself be architecture
            rather than scale. One lattice size, four constructions.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import itertools, json, time

import numpy as np, torch
from ranking import spearman
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "fullvocab_invariance_wide.json")
NARROW = str(_ROOT / "results" / "fullvocab_invariance.json")
MODELS = ["EleutherAI/pythia-31m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
          "gpt2", "gpt2-medium", "gpt2-large",
          "facebook/opt-350m", "bigscience/bloom-560m",
          "state-spaces/mamba-130m-hf", "RWKV/rwkv-4-169m-pile"]
RADII = [2, 3]
TEMPS = [0.7, 1.0]
N, B, SETTLE, SWEEPS, BLOCK = 48, 16, 12, 22, 3
SEEDS = [20260810, 20260811]
READOUTS = ["lambda_ca", "mean_damage", "distinct", "top1"]
NOISE_FACTOR = 2.0
RUNG_MIN, CONCORDANT, SCRAMBLED = 0.6, 0.6, 0.3


def rankings(cells, seed, readout):
    out = {}
    for con in {c["construction"] for c in cells.values()}:
        vals = []
        for m in MODELS:
            k = f"{m}|{con}|s{seed}"
            if k not in cells or not np.isfinite(cells[k][readout]):
                vals = None; break
            vals.append(cells[k][readout])
        if vals is not None:
            out[con] = vals
    return out


def analyse(res):
    cells, parts = res["cells"], []
    signal, rung, primary = {}, {}, {}
    for ro in READOUTS:
        a, b = rankings(cells, SEEDS[0], ro), rankings(cells, SEEDS[1], ro)
        shared = sorted(set(a) & set(b))
        live = []
        for c in shared:
            spread = float(max(a[c]) - min(a[c]))
            noise = float(np.mean([abs(x - y) for x, y in zip(a[c], b[c])]))
            if spread >= NOISE_FACTOR * noise:
                live.append(c)
        signal[ro] = dict(n_with_signal=len(live), n_constructions=len(shared),
                          constructions=live,
                          ratios={c: round(float((max(a[c]) - min(a[c]))
                                                 / max(np.mean([abs(x - y) for x, y in zip(a[c], b[c])]),
                                                       1e-12)), 2) for c in shared})
        agree = [spearman(a[c], b[c]) for c in shared]
        rung[ro] = round(float(np.mean([v for v in agree if np.isfinite(v)])), 4) if agree else None
    lam = signal.get("lambda_ca", {})
    lam_rung = rung.get("lambda_ca")
    # THREE OUTCOMES, NOT TWO. A first version branched on "signal on a majority of constructions"
    # and printed "only 2 of 4" -- true but misleading, because the two that pass are BOTH r=2 and
    # pass strongly (3.46, 2.18) while the two that fail are r=3, where seed noise explodes. Worse,
    # it collapsed the case that actually occurred: a spread that EXCEEDS noise while producing NO
    # reproducible ordering. A spread you cannot rank is not a model measurement, and calling it
    # either "signal" or "no signal" loses the distinction that matters.
    has_spread = lam.get("n_with_signal", 0) > 0
    has_rank = lam_rung is not None and lam_rung >= RUNG_MIN
    parts.append(
        "PRIMARY (SIGNAL, asked before invariance because F128 discovered too late that there was "
        "no ranking to be invariant about): per construction, the across-model spread must exceed "
        f"the across-seed spread by {NOISE_FACTOR}x. "
        + ", ".join(f"{ro} {signal[ro]['n_with_signal']}/{signal[ro]['n_constructions']}"
                    for ro in READOUTS)
        + ". lambda_ca spread/noise by construction: "
        + ", ".join(f"{c}={v}" for c, v in lam.get("ratios", {}).items()) + ". "
        + ("lambda_ca has NO spread above its own noise at any construction, so model identity moves "
           "it less than the seed does and there is nothing to rank."
           if not has_spread else
           f"lambda_ca's spread exceeds noise on {lam['n_with_signal']} of "
           f"{lam['n_constructions']} constructions -- both at r=2, where seed noise is smallest; "
           f"the r=3 constructions fail because noise grows faster than spread. "
           + ("The ordering is also seed-stable, so this is a usable cross-model ranking."
              if has_rank else
              f"BUT THE ORDERING IS NOT REPRODUCIBLE: seed stability is {lam_rung}, far below "
              f"{RUNG_MIN}. So there is a real spread and NO usable ranking inside it -- the "
              f"separation comes from a few models sitting slightly high rather than from an order "
              f"the readout can reproduce. A spread that cannot be ranked is not a model "
              f"measurement.")))
    parts.append(
        "RUNG (seed stability at fixed construction, the gate F128 failed at 0.167): "
        + ", ".join(f"{ro}={rung[ro]}" for ro in READOUTS) + f", threshold {RUNG_MIN}.")
    ok = [ro for ro in READOUTS
          if rung[ro] is not None and rung[ro] >= RUNG_MIN
          and signal[ro]["n_with_signal"] > signal[ro]["n_constructions"] / 2]
    if ok:
        for ro in ok:
            r0 = rankings(cells, SEEDS[0], ro)
            live = signal[ro]["constructions"]
            ps = [spearman(r0[x], r0[y]) for x, y in itertools.combinations(live, 2)]
            ps = [v for v in ps if np.isfinite(v)]
            primary[ro] = dict(mean_rho=round(float(np.mean(ps)), 4) if ps else None,
                               n_pairs=len(ps), n_constructions=len(live))
        inv = [ro for ro, v in primary.items() if v["mean_rho"] is not None
               and v["mean_rho"] >= CONCORDANT]
        parts.append(
            "SECONDARY (invariance, asked only of readouts with both a stable ranking and real "
            "signal): "
            + ", ".join(f"{ro}={primary[ro]['mean_rho']:+.3f}" for ro in primary) + ". "
            + (f"MODEL-ATTRIBUTABLE: {inv}." if inv else
               f"None reaches {CONCORDANT}, so even where a ranking exists it is "
               f"construction-relative."))
    else:
        parts.append(
            "SECONDARY: not asked. No readout has BOTH a seed-stable ranking and cross-model signal "
            "on a majority of constructions, and asking whether a noise ranking is "
            "construction-invariant is the error F126 came close to making.")
    parts.append(
        f"BOUNDARY: {len(MODELS)} models over six families and four architecture classes, including "
        f"two non-attention (Mamba, RWKV) -- F64 found architecture is where this instrument's "
        f"largest effects live. They differ in SIZE as well as family, so a cross-model signal here "
        f"would not by itself be architecture rather than scale. N={N}, radii {RADII}, temperatures "
        f"{TEMPS}.")
    res["analysis"] = dict(signal=signal, seed_agreement=rung, primary=primary,
                           readouts_asked=ok, noise_factor=NOISE_FACTOR)
    res["verdict"] = " ".join(parts)


def main():
    from fullvocab_invariance import cell
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, radii=RADII, temps=TEMPS, N=N, B=B, settle=SETTLE, sweeps=SWEEPS,
        block=BLOCK, seeds=SEEDS, readouts=READOUTS, noise_factor=NOISE_FACTOR,
        rung_min=RUNG_MIN, concordant=CONCORDANT, narrow_reference=rel(NARROW),
        primary="SIGNAL first: across-model spread must exceed across-seed spread by NOISE_FACTOR",
        secondary="invariance, asked only where a stable ranking with real signal exists",
        follows="F128: on three similar models lambda_ca's across-model spread was 0.010-0.031 "
                "against a construction range of 0.68")
    from ar_ca import ARRule
    for m in MODELS:
        try:
            rule = ARRule(m)
        except Exception as e:
            print(f"  {m}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
            continue
        for r in RADII:
            for T in TEMPS:
                for sd in SEEDS:
                    key = f"{m}|r{r}.T{T}|s{sd}"
                    if key in res["cells"]:
                        continue
                    t0 = time.time()
                    try:
                        c = cell(rule, r, T, sd)
                    except Exception as e:
                        print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                        continue
                    c.update(model=m, construction=f"r{r}.T{T}", r=r, T=T, seed=sd,
                             secs=round(time.time() - t0, 1))
                    res["cells"][key] = c
                    print(f"  {key:<44} lambda={c['lambda_ca']:+.4f} dmg={c['mean_damage']:.3f} "
                          f"distinct={c['distinct']:.0f} ({c['secs']:.0f}s)", flush=True)
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
