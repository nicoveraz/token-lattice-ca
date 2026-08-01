"""Gate 2 — controlled pairs: does the battery read a manipulation it was never tuned on?

WHAT GATE 2 IS FOR. Gate 0 found the CA signature coheres within families; Gate 1 found the cheap
static baseline cannot reproduce the one corpus separation we have. Both are about *coherence* and
both used models that differ in many ways at once. A fingerprint claim needs the other thing:
pairs that differ in ONE registered respect, where the manipulation is known and the expectation
was written down first.

  gpt-neo-125M vs gpt2        corpus, with the tokenizer held IDENTICAL. Registered expectation:
                              replicates F64's split beyond within-family spread. This is the only
                              pair with a registered direction, and it is the one Gate 1 showed the
                              static baseline fails on (0.5x within-range against the CA's 2.4x).
  gpt2 vs distilgpt2          distillation. Registered as UNCERTAIN -- a discovery either way.
  Qwen2.5-0.5B vs -Instruct   post-training. Separation expected, DIRECTION UNREGISTERED, so a
                              gap in either direction counts and the sign is reported not claimed.
  pythia-160m vs -deduped     corpus deduplication only. Registered as a DISCOVERY PROBE AND NOT A
                              REQUIREMENT: dedup is a far weaker manipulation than a different
                              corpus, these two are the SAME family, so their gap is a within-family
                              gap by construction and a null here costs the program nothing.
  Cerebras-GPT-111M           not a pair -- a third Pile family, retried after HTTP-401. The corpus
                              direction currently rests on 2 independent families (F68's
                              pseudoreplication hazard arriving a third time); this moves it to 3,
                              against the ~21 the author's own power note calls for.

THE BATTERY AND THE BASELINE ARE COMPUTED ON THE SAME RUNS, deliberately, so that K1 stays
evaluable on every model this gate touches rather than only on Gate 1's set. If the static
conditional separates a pair the CA misses, that is evidence for K1 and it must be able to surface
here too.

WHAT WOULD FIRE K2. Neither the neo/gpt2 replication nor any of the distillation/instruct pairs
separating beyond within-family spread. Then corpus and post-training inference die, F64 stays what
it is today -- a control inside a negative-result argument -- and this folds back into paper 2 as a
paragraph. Per PROGRAM.md 7 that outcome costs ~2 GPU-days and buys avoiding a much longer illusion.

THE DENOMINATOR IS THE HONEST PART. The prereg's statistic is "gap over worst within-family range",
measured on the 26-model screen. That reference exists for the top-1 features because the screen
measured them; it does NOT exist for the two features this gate adds (radius drop, BOS drop), which
were never run screen-wide. Those are therefore reported against the pooled 4-seed within-model
spread, labelled as a DIFFERENT denominator, and never silently mixed with the prereg statistic.

Writes fingerprint/gate2.json.
Usage:  caffeinate -dimsu .venv/bin/python -u fingerprint/gate2.py
        (resumable, keyed by (model, arm, seed))
"""
import collections
import json
import os
import pathlib
import statistics
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path[:0] = [str(ROOT / "gatecheck" / "src"), str(_HERE.parent),
                str(ROOT / "src"), str(ROOT / "experiments")]

import torch  # noqa: E402
from gatecheck import save_results, verify_block  # noqa: E402
from evidence_falloff import t_star, SCREEN_TEMPS  # noqa: E402  one implementation, imported
from gate1 import (argmax_census, conditional_stats, by_length, BASELINE_FEATURES,  # noqa: E402
                   perm_p_stable)
from reanalysis import FAMILY, load as load_ca  # noqa: E402

OUT = _HERE.parent / "gate2.json"
PREREG = _HERE.parent / "prereg.json"

# Frozen in prereg.json; restated here so a drift between the two is visible rather than silent.
N, B, SETTLE, R = 96, 16, 16, 2
SEEDS = [101, 102, 103, 104]              # seeds_per_model = 4
R_ALT = 3                                 # the r=2 -> r=3 drop
TSTAR_THRESH = 0.40

PAIRS = [
    ("corpus", "EleutherAI/gpt-neo-125M", "gpt2",
     "corpus differs, tokenizer identical; registered: replicates F64 beyond within-family spread"),
    ("distillation", "gpt2", "distilgpt2",
     "distillation; registered as UNCERTAIN -- a discovery either way"),
    ("post_training", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct",
     "post-training; separation expected, DIRECTION UNREGISTERED"),
    ("dedup", "EleutherAI/pythia-160m", "EleutherAI/pythia-160m-deduped",
     "dedup only, same family; DISCOVERY PROBE, not a requirement"),
]
EXTRA = [("cerebras/Cerebras-GPT-111M", "third Pile family; retried after HTTP-401")]


def settle_top1(rule, T, r, scheme, seed):
    """One settled ring's top-1 share and dominant token -- the screen's own measurement."""
    from ar_ca import run
    s = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme=scheme,
            init="random", seed=seed, order="per_replica")["final"]
    tops, toks = [], collections.Counter()
    for row in s:
        c = collections.Counter(row.tolist())
        tops.append(c.most_common(1)[0][1] / N)
        toks.update(c)
    return float(np.mean(tops)), int(toks.most_common(1)[0][0])


def measure_model(name, res, dev):
    """The frozen battery plus the Gate-1 baseline, on the same loaded model."""
    from ar_ca import ARRule
    from transformers import AutoTokenizer, AutoModelForCausalLM
    runs = res["runs"]
    keys = [f"{name}|T{T}|s{s}" for T in SCREEN_TEMPS for s in SEEDS]
    keys += [f"{name}|r{R_ALT}|s{s}" for s in SEEDS] + [f"{name}|bos|s{s}" for s in SEEDS]
    if all(k in runs for k in keys) and f"{name}|baseline" in runs:
        print(f"  {name}: already complete", flush=True)
        return True
    t0 = time.time()
    try:
        rule = ARRule(name)
    except Exception as e:
        print(f"  {name}: LOAD FAILED ({type(e).__name__}: {str(e)[:70]})", flush=True)
        runs[f"{name}|failed"] = {"model": name, "error": type(e).__name__}
        json.dump(res, open(OUT, "w"), indent=1)
        return False
    print(f"  {name}: loaded in {time.time()-t0:.0f}s", flush=True)

    for T in SCREEN_TEMPS:
        for s in SEEDS:
            k = f"{name}|T{T}|s{s}"
            if k in runs: continue
            a, tok = settle_top1(rule, T, R, "none", s)
            runs[k] = dict(model=name, arm="temp", T=T, seed=s, top1=round(a, 4), dominant=tok)
            json.dump(res, open(OUT, "w"), indent=1)
        vals = [runs[f"{name}|T{T}|s{s}"]["top1"] for s in SEEDS]
        print(f"     T={T:<6} top1={statistics.mean(vals):.3f}+-{statistics.pstdev(vals):.3f}",
              flush=True)
    for s in SEEDS:                                    # the radius arm, at the strongest T
        k = f"{name}|r{R_ALT}|s{s}"
        if k in runs: continue
        a, tok = settle_top1(rule, SCREEN_TEMPS[0], R_ALT, "none", s)
        runs[k] = dict(model=name, arm="radius", T=SCREEN_TEMPS[0], r=R_ALT, seed=s,
                       top1=round(a, 4), dominant=tok)
        json.dump(res, open(OUT, "w"), indent=1)
    for s in SEEDS:                                    # the one-token BOS prefix arm (F66)
        k = f"{name}|bos|s{s}"
        if k in runs: continue
        a, tok = settle_top1(rule, SCREEN_TEMPS[0], R, "bos", s)
        runs[k] = dict(model=name, arm="bos", T=SCREEN_TEMPS[0], seed=s,
                       top1=round(a, 4), dominant=tok)
        json.dump(res, open(OUT, "w"), indent=1)

    if f"{name}|baseline" not in runs:                 # Gate 1's static battery, same model
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).eval().to(
            dev, torch.float16 if dev != "cpu" else torch.float32)
        V = int(getattr(model.config, "vocab_size", len(tok)))
        special = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                               tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in special], dtype=np.int64)
        rng = np.random.default_rng(20260801)
        rec = dict(model=name, arm="baseline")
        rec.update(argmax_census(model, tok, dev, pool, rng))
        rec.update(conditional_stats(model, tok, dev, pool, rng))
        rec["by_length"] = by_length(model, tok, dev, rng)
        runs[f"{name}|baseline"] = rec
        json.dump(res, open(OUT, "w"), indent=1)
        del model
    del rule
    try: torch.mps.empty_cache()
    except Exception: pass
    print(f"  {name}: done in {time.time()-t0:.0f}s", flush=True)
    return True


def features(name, runs):
    """Collapse one model's runs into the frozen battery, with the 4-seed spread on every entry."""
    if any(k.startswith(f"{name}|failed") for k in runs):
        return None
    out = {}
    prof = {}
    for T in SCREEN_TEMPS:
        vs = [runs[f"{name}|T{T}|s{s}"]["top1"] for s in SEEDS if f"{name}|T{T}|s{s}" in runs]
        if not vs: return None
        out[f"top1_T{T}"] = round(statistics.mean(vs), 4)
        out[f"top1_T{T}_sd"] = round(statistics.pstdev(vs), 4)
        prof[f"{name}@{T}"] = {"top1_share": statistics.mean(vs)}
    ts = t_star(prof, name, thresh=TSTAR_THRESH)
    out["tstar"] = ts if isinstance(ts, (int, float)) else None
    out["tstar_state"] = ("finite" if isinstance(ts, (int, float))
                          else ("censored_above" if ts == "censored_above" else "none"))
    doms = collections.Counter(runs[f"{name}|T{SCREEN_TEMPS[0]}|s{s}"]["dominant"]
                               for s in SEEDS if f"{name}|T{SCREEN_TEMPS[0]}|s{s}" in runs)
    out["dominant_token"] = doms.most_common(1)[0][0]
    for arm, lab in (("r%d" % R_ALT, "radius_drop"), ("bos", "bos_drop")):
        vs = [runs[f"{name}|{arm}|s{s}"]["top1"] for s in SEEDS if f"{name}|{arm}|s{s}" in runs]
        if vs:
            out[lab] = round(out[f"top1_T{SCREEN_TEMPS[0]}"] - statistics.mean(vs), 4)
            out[f"{lab}_sd"] = round(statistics.pstdev(vs), 4)
    b = runs.get(f"{name}|baseline")
    if b:
        for f in BASELINE_FEATURES:
            out[f] = b[f]
    return out


# Features whose screen-wide within-family range EXISTS (the screen measured top-1 at these temps).
# The two arms this gate adds were never run screen-wide, so they get a different denominator and
# are labelled as such rather than being compared to a number that does not describe them.
PREREG_DENOM = [f"top1_T{T}" for T in SCREEN_TEMPS]
NEW_DENOM = ["radius_drop", "bos_drop"]
CA_FEATURES = PREREG_DENOM + NEW_DENOM


def within_family_ranges(feat):
    """Worst within-family range on the 26-model screen -- Gate 0's denominator, for top-1 features."""
    ca = load_ca()
    T = feat.replace("top1_T", "")
    idx = {str(t): i for i, t in enumerate(SCREEN_TEMPS)}
    if T not in idx:
        return None
    rng = {}
    for f in {r["family"] for r in ca.values()}:
        vs = [r["profile"][idx[T]] for r in ca.values()
              if r["family"] == f and r["profile"][idx[T]] is not None]
        if len(vs) >= 2:
            rng[f] = max(vs) - min(vs)
    return max(rng.values()) if rng else None


def analyse(res):
    runs = res["runs"]
    feats = {}
    for _, a, b, _ in PAIRS:
        for m in (a, b):
            if m not in feats:
                feats[m] = features(m, runs)
    for m, _ in EXTRA:
        feats[m] = features(m, runs)
    out = {"features": feats}

    pairs_out, separated = {}, []
    for kind, a, b, note in PAIRS:
        fa, fb = feats.get(a), feats.get(b)
        if not (fa and fb):
            pairs_out[kind] = {"skipped": "a member failed to load", "a": a, "b": b}
            continue
        rows = {}
        for f in CA_FEATURES + BASELINE_FEATURES:
            if f not in fa or f not in fb:
                continue
            gap = abs(fa[f] - fb[f])
            seed_sd = max(fa.get(f + "_sd", 0.0), fb.get(f + "_sd", 0.0))
            row = {"a": fa[f], "b": fb[f], "gap": round(gap, 4),
                   "worst_seed_sd": round(seed_sd, 4),
                   "gap_over_seed_sd": round(gap / seed_sd, 2) if seed_sd > 1e-9 else None}
            if f in PREREG_DENOM:
                w = within_family_ranges(f)
                row["worst_within_family_range"] = round(w, 4) if w else None
                row["gap_over_worst_within"] = round(gap / w, 2) if w else None
                row["denominator"] = "prereg: worst within-family range on the 26-model screen"
            else:
                row["denominator"] = ("pooled 4-seed within-model spread -- this feature was never "
                                      "run screen-wide, so the prereg denominator does not exist "
                                      "for it and is NOT substituted")
            rows[f] = row
        ratios = [r["gap_over_worst_within"] for r in rows.values()
                  if r.get("gap_over_worst_within") is not None]
        best = max(ratios) if ratios else None
        ca_best = max([r["gap_over_worst_within"] for f, r in rows.items()
                       if f in PREREG_DENOM and r.get("gap_over_worst_within") is not None] or [0])
        base_best = max([r["gap_over_seed_sd"] for f, r in rows.items()
                         if f in BASELINE_FEATURES and r.get("gap_over_seed_sd") is not None] or [0])
        sep = bool(best is not None and best > 1.0)
        if sep:
            separated.append(kind)
        pairs_out[kind] = {"a": a, "b": b, "note": note, "features": rows,
                           "best_gap_over_worst_within": best,
                           "separates_beyond_within_family": sep,
                           "ca_best": ca_best, "baseline_best_over_seed_sd": base_best}
    out["pairs"] = pairs_out

    # -- K2, exactly as frozen ------------------------------------------------------------
    required = [k for k in ("corpus", "distillation", "post_training") if k in pairs_out]
    fired = not any(pairs_out[k].get("separates_beyond_within_family") for k in required)
    out["K2"] = {"fired": bool(fired), "required_pairs": required, "separated": separated,
                 "rule": ("K2 fires if NEITHER the neo/gpt2 replication NOR any of "
                          "{distilgpt2, instruct-sibling} separates beyond within-family spread; "
                          "the dedup pair is a discovery probe and cannot fire or prevent K2")}

    parts = []
    c = pairs_out.get("corpus", {})
    if c.get("features"):
        parts.append(
            f"CORPUS (gpt-neo-125M vs gpt2, tokenizer identical): best CA separation "
            f"{c['best_gap_over_worst_within']}x the worst within-family range on the screen. "
            + ("REPLICATES -- the registered expectation holds and F64's split survives a "
               "4-seed-per-side measurement." if c["separates_beyond_within_family"] else
               "DOES NOT REPLICATE at 4 seeds per side, which is the pair the whole corpus "
               "direction rests on."))
    for kind in ("distillation", "post_training", "dedup"):
        p = pairs_out.get(kind)
        if not p or not p.get("features"):
            continue
        parts.append(
            f"{kind.upper().replace('_', '-')} ({p['a'].split('/')[-1]} vs "
            f"{p['b'].split('/')[-1]}): best {p['best_gap_over_worst_within']}x -> "
            + ("SEPARATES." if p["separates_beyond_within_family"] else "no separation.")
            + (" Registered as a discovery probe on the same family, so a null here costs the "
               "program nothing." if kind == "dedup" else ""))
    if fired:
        parts.append(
            "K2 FIRES: no required pair separates beyond within-family spread. Corpus and "
            "post-training inference die, F64 stays a control inside a negative-result argument, "
            "and the program folds back into paper 2 as a paragraph -- which PROGRAM.md 7 "
            "registered as costing ~2 GPU-days to avoid a much longer illusion.")
    else:
        parts.append(
            f"K2 DOES NOT FIRE: {', '.join(separated)} separate(s) beyond within-family spread, so "
            f"the capability survives its controlled-pair test and Gate 3 (the API port) is the "
            f"next decision. Per plan_paper2's caution the third-paper decision point is HERE, "
            f"after Gate 2 -- not before.")
    # K1 re-checked on this gate's own models, per the prereg's "computed together" requirement
    k1_rows = [(k, p["ca_best"], p["baseline_best_over_seed_sd"]) for k, p in pairs_out.items()
               if p.get("features")]
    beats = [k for k, ca, bs in k1_rows if bs >= ca and ca > 0]
    parts.append(
        f"K1 re-checked on Gate 2's own runs (battery and baseline computed together, as frozen): "
        + (f"the static baseline matches or beats the CA on {', '.join(beats)}."
           if beats else "the static baseline beats the CA on NO pair here, consistent with "
                         "Gate 1.") +
        " Note the two use different denominators and this is a direction check, not a K1 verdict.")
    parts.append(
        "Family is the independent unit (F68). The dedup pair is WITHIN one family by construction, "
        "so its gap is a within-family gap and is reported as a probe, never as separation.")
    out["gate2_verdict"] = " ".join(parts)
    return out


def main():
    block = json.load(open(PREREG))
    if not verify_block(block):
        raise SystemExit("prereg.json failed its own hash check -- refusing to run against it")
    print("  prereg block verifies: True", flush=True)
    res = json.load(open(OUT)) if OUT.exists() else {"runs": {}}
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    todo = [m for _, a, b, _ in PAIRS for m in (a, b)] + [m for m, _ in EXTRA]
    seen = set()
    for m in todo:
        if m in seen:
            continue
        seen.add(m)
        measure_model(m, res, dev)
    out = analyse(res)
    out["runs"] = res["runs"]
    out["_prereg_sha256"] = block["sha256"]
    print("\n  ->", out["gate2_verdict"])
    save_results(OUT, out, script=__file__, root=ROOT, prereg=block,
                 independent_unit="family", forbid_paths=True)
    print("\nwrote fingerprint/gate2.json")


if __name__ == "__main__":
    main()
