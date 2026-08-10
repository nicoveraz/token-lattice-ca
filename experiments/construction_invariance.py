"""Does ANY readout rank models the same way regardless of how the lattice is built?

THE QUESTION F123/F124 FORCE. Branching moved from 0.887 to 1.593 -- subcritical to firmly
supercritical -- with the model, weights and temperature held fixed and only the alphabet and window
changed. If the construction has that much dynamic range, then any single-construction number is a
fact about the apparatus until shown otherwise. That is F56-F66's lesson arriving from a second
direction: there the critical point belonged to the probe's OOD degeneracy, here it moves ~80% under
ordinary design choices.

The salvage, if there is one, is INVARIANCE. A readout that ranks models the same way across many
constructions is measuring the model; one whose ranking scrambles is measuring the lattice. This has
never been tested -- every comparison in the project fixes one construction and varies the model.

DESIGN. `construction` is a RULE (alphabet x selection mode x radius), not a fixed token set: each
model's arms are built from its OWN tokenizer and its OWN marginal, because comparing identical ids
across different vocabularies would be comparing different things. Every model therefore meets the
same recipe, not the same tokens.

  models         three, two families, all locally cached
  constructions  3 alphabets x 3 selection modes x 2 radii = 18
  readouts       branching, s_near, s_far, settled diversity, top-1 share

PRE-REGISTERED:
  RUNG      SEED STABILITY, checked before anything else. Each cell is measured at two seeds, and
            the model ordering under each readout must agree between them on at least
            RUNG_MIN of constructions. If seed noise alone scrambles the ranking, the construction
            question is undefined and nothing below is read. This is the known-answer check: a
            readout that cannot rank models reproducibly at FIXED construction cannot be asked
            whether it ranks them reproducibly across constructions.
  PRIMARY   per readout, the mean pairwise Spearman correlation between the model-rankings produced
            by different constructions. Registered reading: >= CONCORDANT names the readout
            model-attributable; <= SCRAMBLED names it construction-dominated; between is
            undetermined and says so.
  BOUNDARY  three models is a 3-point ranking, so a single swap moves rho by a lot; this can
            identify a scrambled readout far more confidently than it can certify an invariant one.
            One temperature, one lattice size.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import itertools, json, time

import numpy as np, torch
from ranking import spearman
from subalphabet import pick_tokens, make_sampler, sub_init, BINARY, COLOURS, DIGITS
from selection_mode import marginal, matched_ids
from window_ladder import s_at
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "construction_invariance.json")
MODELS = ["EleutherAI/pythia-410m", "gpt2", "gpt2-large"]
ALPHABETS = [("binary", BINARY), ("colours", COLOURS), ("digits", DIGITS)]
MODES = ["semantic", "freq_matched", "uniform"]
RADII = [2, 3]
T, N, B, SETTLE = 0.7, 48, 16, 12
SEEDS = [20260810, 20260811]
READOUTS = ["branching", "s_near", "s_far", "distinct", "top1"]
RUNG_MIN = 0.6
CONCORDANT, SCRAMBLED = 0.6, 0.3


def cell(rule, ids, r, seed):
    from ar_ca import run
    rng = np.random.default_rng(seed)
    smp = make_sampler(ids, None)
    settled = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme="none",
                  init_state=sub_init(ids, B, N, rng), seed=seed, sampler=smp)["final"]
    pool = settled.reshape(-1)
    sp = [s_at(rule, ids, pool, r, p, np.random.default_rng(seed + p)) for p in range(r)]
    vals, cnt = np.unique(pool, return_counts=True)
    # a frozen ring returns nan from s_at; carry it rather than summing it to a number
    if not all(np.isfinite(v) for v in sp):
        return dict(branching=float("nan"), s_far=float("nan"), s_near=float("nan"),
                    distinct=float(len(vals)), top1=float(cnt.max() / cnt.sum()),
                    s_pos=None, frozen=True)
    return dict(branching=float(sum(sp)), s_far=float(sp[0]), s_near=float(sp[-1]),
                distinct=float(len(vals)), top1=float(cnt.max() / cnt.sum()),
                s_pos=[round(v, 5) for v in sp])


def rankings(cells, seed, readout):
    """{construction: [readout value per model]} at one seed."""
    out = {}
    for con in {c["construction"] for c in cells.values()}:
        vals = []
        for m in MODELS:
            k = f"{m}|{con}|s{seed}"
            if k not in cells:
                vals = None; break
            vals.append(cells[k][readout])
        if vals:
            out[con] = vals
    return out


def analyse(res):
    cells, parts = res["cells"], []
    rung, primary = {}, {}
    for ro in READOUTS:
        a, b = rankings(cells, SEEDS[0], ro), rankings(cells, SEEDS[1], ro)
        shared = sorted(set(a) & set(b))
        agree = [spearman(a[c], b[c]) for c in shared
                 if all(np.isfinite(x) for x in a[c] + b[c])]
        agree = [v for v in agree if np.isfinite(v)]
        rung[ro] = round(float(np.mean(agree)), 4) if agree else None
    ok = [ro for ro, v in rung.items() if v is not None and v >= RUNG_MIN]
    parts.append(
        "RUNG (seed stability at FIXED construction, checked before the construction question): "
        + ", ".join(f"{ro}={rung[ro]}" for ro in READOUTS) + f". {len(ok)} of {len(READOUTS)} "
        f"readouts rank the models consistently across two seeds at threshold {RUNG_MIN}"
        + (f" -- {ok}. Only these are asked the primary question; a readout that cannot rank models "
           f"reproducibly at fixed construction cannot be asked about invariance across them."
           if ok else ". NONE clears it, so the construction question is undefined and nothing "
                      "below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, seed_agreement=rung)
        res["verdict"] = " ".join(parts); return
    for ro in ok:
        r0 = rankings(cells, SEEDS[0], ro)
        cons = sorted(r0)
        live = [c for c in cons if all(np.isfinite(x) for x in r0[c])]
        ps = [spearman(r0[x], r0[y]) for x, y in itertools.combinations(live, 2)]
        ps = [v for v in ps if np.isfinite(v)]
        primary[ro] = dict(mean_rho=round(float(np.mean(ps)), 4) if ps else None,
                           n_pairs=len(ps), n_constructions=len(live),
                           n_frozen=len(cons) - len(live))
    best = max(primary, key=lambda k: primary[k]["mean_rho"])
    inv = [ro for ro, v in primary.items() if v["mean_rho"] >= CONCORDANT]
    scr = [ro for ro, v in primary.items() if v["mean_rho"] <= SCRAMBLED]
    parts.append(
        "PRIMARY, mean pairwise agreement between the model-rankings different constructions "
        "produce: "
        + ", ".join(f"{ro}={primary[ro]['mean_rho']:+.3f}" for ro in primary)
        + f" over {primary[best]['n_constructions']} constructions. "
        + (f"MODEL-ATTRIBUTABLE at the registered {CONCORDANT} threshold: {inv}. These rank models "
           f"the same way however the lattice is built, so they are candidates for a claim about "
           f"language models rather than about this apparatus."
           if inv else
           f"NONE reaches the registered {CONCORDANT} threshold. No readout ranks models "
           f"consistently across constructions, so on this evidence the instrument characterises "
           f"LATTICES rather than language models, and any single-construction comparison is a "
           f"fact about the apparatus.")
        + (f" CONSTRUCTION-DOMINATED at or below {SCRAMBLED}: {scr}." if scr else ""))
    parts.append(
        f"BOUNDARY: {len(MODELS)} models, so each ranking has 3 points and one swap moves rho a "
        f"long way -- this identifies a SCRAMBLED readout far more confidently than it certifies an "
        f"invariant one. One temperature (T={T}), one lattice size (N={N}), radii {RADII}. "
        f"Constructions are RULES applied to each model's own tokenizer and marginal, not shared "
        f"token ids, since identical ids across different vocabularies would not be comparable.")
    res["analysis"] = dict(rung_passes=True, seed_agreement=rung, readouts_asked=ok,
                           primary=primary, invariant=inv, scrambled=scr)
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, alphabets=[a for a, _ in ALPHABETS], modes=MODES, radii=RADII,
        T=T, N=N, B=B, settle=SETTLE, seeds=SEEDS, readouts=READOUTS,
        rung_min=RUNG_MIN, concordant=CONCORDANT, scrambled=SCRAMBLED,
        rung="model ordering must agree across two seeds at fixed construction, else the "
             "construction question is undefined",
        primary="mean pairwise Spearman between the model-rankings different constructions give",
        reading=f">= {CONCORDANT} model-attributable; <= {SCRAMBLED} construction-dominated")
    from ar_ca import ARRule
    for m in MODELS:
        rule = ARRule(m)
        g = np.random.default_rng(SEEDS[0])
        marg = marginal(rule, g)
        exclude = set()
        sem = {}
        for name, words in ALPHABETS:
            ids, _, _ = pick_tokens(rule.tok, words)
            sem[name] = ids
            exclude |= set(int(i) for i in ids)
        V = len(marg)
        for name, _ in ALPHABETS:
            s_ids = sem[name]
            arms = dict(semantic=s_ids,
                        freq_matched=matched_ids(s_ids, marg, g, exclude),
                        uniform=np.array(sorted(g.choice(V, size=len(s_ids), replace=False)),
                                         dtype=np.int64))
            for mode, ids in arms.items():
                for r in RADII:
                    for sd in SEEDS:
                        key = f"{m}|{name}.{mode}.r{r}|s{sd}"
                        if key in res["cells"]:
                            continue
                        t0 = time.time()
                        c = cell(rule, ids, r, sd)
                        c.update(model=m, construction=f"{name}.{mode}.r{r}", seed=sd,
                                 k=int(len(ids)), ids=[int(i) for i in ids],
                                 secs=round(time.time() - t0, 1))
                        res["cells"][key] = c
                        print(f"  {key:<52} branching={c['branching']:.4f} "
                              f"distinct={c['distinct']:.0f} top1={c['top1']:.3f}", flush=True)
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
