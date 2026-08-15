"""The domain axis on BASE models: one BOS token against none. Paper 2's missing cohort.

WHY THIS IS THE GAP. F144 and F145 established the domain as the dominant variable — a chat template
moves a fixed-point class, a model ranking and a scalar value further than the model does — but both
are INSTRUCT-ONLY, because base models have no chat template. Every other finding in this project
rests on base models, so the domain axis has never been run on the cohort the programme is built on.

BOS IS THE DOMAIN CHANGE BASE MODELS CAN HAVE, and it is not a substitute chosen for convenience:
arXiv:2608.10986 already reports that prepending a single BOS token moves a frozen fraction from
74.4% to 24.1%. That is this project's earliest domain result, published, and never followed up on
the share. `ar_ca.run` has carried `scheme="bos"` since the beginning — the arm exists and has not
been used for this question.

AND IT IS CHEAP, which is why it can cover the whole grid where F145 could not. A chat template adds
~35 tokens to every forward pass and cost F145 an 11x tax that forced it down to two temperatures and
five models. BOS adds ONE token. The full share_invariance grid is therefore affordable: ten models,
six constructions, two seeds.

PRE-REGISTERED:
  RUNG      with scheme="none" this must reproduce share_invariance's stored top1 BIT-IDENTICALLY
            for the same (model, construction, seed). Same geometry, same seeds, same rule, so any
            difference means this is not F130's measurement and nothing below is read.
  SIGNAL    the BOS arm must itself have across-model spread above across-seed noise, per
            construction. Without it the BOS ranking is noise and the agreement below is
            uninterpretable in either direction -- the trap F137/F138 fell into and F145 avoided.
  PRIMARY   Spearman between the model rankings the two domains produce, per construction.
            Registered reading: >= CONCORDANT on a majority of constructions means the share's
            ranking survives a one-token domain change on base models; below it means the domain
            reorders base models too, and F145's result is not an artefact of instruct models or of
            long prefixes.
  SECONDARY the shift in VALUES, for comparability with F135 and F145.
  BOUNDARY  ten base models, ONE domain change of ONE token. A null here would bound the domain
            effect to larger prefixes rather than establishing domain-invariance in general.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from gatecheck import pack_state, has_state, spearman, STATE_KEY

OUT = str(_ROOT / "results" / "share_bos.json")
RAW = _ROOT / "results" / "share_invariance.json"

# share_invariance's geometry, copied so top1 is F130's quantity and the rung can be exact
MODELS = ["EleutherAI/pythia-31m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
          "gpt2", "gpt2-medium", "gpt2-large",
          "facebook/opt-350m", "bigscience/bloom-560m",
          "state-spaces/mamba-130m-hf", "RWKV/rwkv-4-169m-pile"]
RADII = [2, 3]
TEMPS = [0.02, 0.2, 0.7]
N, B, SETTLE = 48, 16, 30
SEEDS = [20260810, 20260811]
RUNG_TOL = 1e-12
CONCORDANT = 0.6
NOISE_FACTOR = 2.0


def cell(rule, r, T, seed, scheme):
    from ar_ca import run
    settled = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme=scheme, init="random",
                  seed=seed)["final"]
    pool = settled.reshape(-1)
    vals, cnt = np.unique(pool, return_counts=True)
    return dict(top1=float(cnt.max() / cnt.sum()), distinct=float(len(vals)),
                rep2=float(np.mean(settled[:, :-1] == settled[:, 1:])),
                **{STATE_KEY: pack_state(settled, stride_axis=0,
                                         note="settled lattice, (replica, site)")})


def analyse(res):
    cells = res["cells"]
    raw_ref = json.load(open(RAW))["cells"] if RAW.exists() else {}
    parts, analysis = [], {}

    errs = []
    for k, v in cells.items():
        if not k.endswith("|rawcheck"):
            continue
        r = raw_ref.get(k.replace("|rawcheck", ""))
        if r:
            errs.append(abs(v["top1"] - r["top1"]))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst <= RUNG_TOL
    parts.append(
        f"RUNG (scheme='none' reproduces share_invariance): worst error {worst:.2e} across "
        f"{len(errs)} cells (tolerance {RUNG_TOL:g}). "
        + ("Bit-identical, so the BOS arm differs from F130's measurement in the DOMAIN and in "
           "nothing else."
           if ok else "NOT reproduced -- this is not F130's measurement and nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    cons = [f"r{r}.T{T}" for r in RADII for T in TEMPS]
    have = [m for m in MODELS
            if all(f"{m}|{c}|s{s}|bos" in cells for c in cons for s in SEEDS)]
    analysis["models"] = have
    if len(have) < 4:
        parts.append(f"Only {len(have)} models complete -- a rank correlation on fewer cannot fail "
                     f"informatively, so the PRIMARY is not read.")
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts)
        return

    sig, rhos, shifts = [], {}, {}
    for c in cons:
        a = [cells[f"{m}|{c}|s{SEEDS[0]}|bos"]["top1"] for m in have]
        b = [cells[f"{m}|{c}|s{SEEDS[1]}|bos"]["top1"] for m in have]
        spread = float(max(a) - min(a))
        noise = float(np.mean([abs(x - y) for x, y in zip(a, b)]))
        sig.append((c, spread, noise, noise > 0 and spread >= NOISE_FACTOR * noise))
        rawv = [float(np.mean([raw_ref[f"{m}|{c}|s{s}"]["top1"] for s in SEEDS])) for m in have]
        bosv = [float(np.mean([cells[f"{m}|{c}|s{s}|bos"]["top1"] for s in SEEDS])) for m in have]
        rhos[c] = round(float(spearman(rawv, bosv)), 4)
        shifts[c] = round(float(np.mean(np.abs(np.array(bosv) - np.array(rawv)))), 4)
    n_sig = sum(1 for *_x, o in sig if o)
    analysis.update(signal=[dict(construction=c, spread=round(s, 4), seed_noise=round(n, 4),
                                 passes=bool(o)) for c, s, n, o in sig],
                    rank_agreement=rhos, mean_abs_shift=shifts)
    parts.append(
        f"SIGNAL in the BOS arm: {n_sig} of {len(sig)} constructions carry across-model spread above "
        f"across-seed noise ("
        + ", ".join(f"{c} {s:.3f}/{n:.3f}" for c, s, n, _o in sig) + "). "
        + ("The BOS ranking is a ranking, so the agreement below is interpretable."
           if n_sig > len(sig) / 2 else
           "The BOS ranking is mostly noise, so the agreement below is uninterpretable in either "
           "direction and the PRIMARY is NOT_DECIDABLE."))
    good = [c for c, r in rhos.items() if r >= CONCORDANT]
    parts.append(
        "PRIMARY, agreement between the RAW and BOS model rankings: "
        + ", ".join(f"{c}: {r:+.3f}" for c, r in sorted(rhos.items())) + ". "
        + (f"At or above {CONCORDANT} on {len(good)} of {len(rhos)} constructions: the share's "
           f"ranking SURVIVES a one-token domain change on base models. Combined with F145, that "
           f"bounds the domain effect on rankings to LARGER prefixes rather than to any conditioning "
           f"at all -- a sharper claim than either run alone."
           if len(good) > len(rhos) / 2 else
           f"Below {CONCORDANT} on {len(rhos) - len(good)} of {len(rhos)} constructions: ONE token "
           f"is enough to reorder base models, so F145's result is neither an instruct-model effect "
           f"nor a long-prefix effect, and the domain axis reaches the cohort every other finding "
           f"in this project rests on."))
    parts.append(
        "SECONDARY, mean |shift| in top1 between domains: "
        + ", ".join(f"{c}: {v:.4f}" for c, v in sorted(shifts.items()))
        + ". For comparability with F135 (0.1327 on a chat scaffold) and F145 (0.1696 at T=0.02).")
    parts.append(
        f"BOUNDARY: {len(have)} base models, {len(cons)} constructions, N={N}, B={B}, "
        f"settle={SETTLE}, and ONE domain change of ONE token. A null here bounds the domain effect "
        f"to larger prefixes; it does not establish domain-invariance in general, and BOS is a "
        f"special token whose embedding is trained differently from ordinary text.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, radii=RADII, temps=TEMPS, N=N, B=B, settle=SETTLE, seeds=SEEDS,
        rung_tol=RUNG_TOL, concordant=CONCORDANT, noise_factor=NOISE_FACTOR,
        geometry="copied from share_invariance so top1 is F130's quantity",
        domain="scheme='bos' -- one BOS token, the arm ar_ca has carried since the start and the "
               "domain change arXiv:2608.10986 measured at 74.4% -> 24.1% on a frozen fraction",
        rung="scheme='none' must reproduce share_invariance bit-identically",
        primary="Spearman between the raw and BOS model rankings, per construction",
        why="F144/F145 are instruct-only because base models have no chat template, so the domain "
            "axis has never touched the cohort every other finding rests on")
    if "--analyse" not in _sys.argv:
        from ar_ca import ARRule
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for m in MODELS:
            need = [(r, T, s) for r in RADII for T in TEMPS for s in SEEDS
                    if not has_state(res["cells"].get(f"{m}|r{r}.T{T}|s{s}|bos", {}))]
            if not need:
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            try:
                rule = ARRule(m)
            except Exception as e:
                print(f"  {m}: LOAD FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
            rk = f"{m}|r{RADII[0]}.T{TEMPS[0]}|s{SEEDS[0]}|rawcheck"
            if rk not in res["cells"]:
                c = cell(rule, RADII[0], TEMPS[0], SEEDS[0], "none")
                c.update(model=m, construction=f"r{RADII[0]}.T{TEMPS[0]}", seed=SEEDS[0],
                         arm="rawcheck")
                res["cells"][rk] = c
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<34} RUNG none top1={c['top1']:.4f}", flush=True)
            for r, T, s in need:
                k = f"{m}|r{r}.T{T}|s{s}|bos"
                t0 = time.time()
                c = cell(rule, r, T, s, "bos")
                c.update(model=m, construction=f"r{r}.T{T}", r=r, T=T, seed=s, arm="bos")
                res["cells"][k] = c
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {k:<52} top1={c['top1']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            done += 1
            del rule
            gc.collect()
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
