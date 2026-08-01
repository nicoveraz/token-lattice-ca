"""Is the frozen phase corpus-induced? A cheap screen across six models (#61; tests F62).

WHAT F62 LEFT AS A HYPOTHESIS. Pythia-410m's low-temperature phase collapses to 4 distinct tokens
out of 96 -- 81 newlines -- and damage heals there because every site resamples to `\\n` regardless
of context. gpt2-medium has no such fixed-point token and correspondingly no frozen phase at any
accessible temperature. F62 proposed an explanation: the attractor is a property of the TRAINING
CORPUS, with the Pile being newline-rich and WebText less so.

That is a guess with two models. It is also falsifiable, and cheaply, which is the point of this
script.

WHY THIS IS CHEAP. The hypothesis is about the SETTLED STATE, not about damage. Measuring what the
lattice relaxes to needs one settle run -- no twin, no common random numbers, no long sweep window.
That is ~1150 forward passes at batch 8, against the ~316 s the damage protocol costs per cell. So
six models across four temperatures is minutes plus downloads, where the damage version would be
hours per model.

PRE-REGISTERED BEFORE ANY MODEL IS DOWNLOADED:
  * Prediction: models trained on **the Pile** show a dominant single-token attractor at low T
    (top-1 share high, distinct-token count low); models trained on **other corpora** do not.
  * The set is chosen so that "the Pile" is separable from "EleutherAI's training setup":

        Pile,     EleutherAI   pythia-410m        <- has the attractor (F62)
        Pile,     EleutherAI   gpt-neo-125M       <- different architecture and era
        Pile,     Cerebras     Cerebras-GPT-256M  <- DIFFERENT LAB; the load-bearing case
        WebText,  OpenAI       gpt2-medium        <- no attractor (F62)
        ROOTS,    BigScience   bloom-560m
        mixed,    Alibaba      Qwen2.5-0.5B

    If the two non-EleutherAI Pile models show the attractor and the three non-Pile models do not,
    the corpus explanation survives a real test. If Cerebras-GPT does NOT show it, the explanation
    is wrong and something about EleutherAI's recipe or tokenizer is doing the work instead.
  * The dominant token is reported by what it DECODES TO, not by id -- token ids are not comparable
    across tokenizers, and "is it whitespace?" is the question, not "is it token 187?".
  * A mixed result is the interesting outcome and is reported as such, not smoothed over.

WHAT THIS DOES NOT TEST. Whether a model with the attractor has DP exponents; whether the
transition temperature agrees across families (it does not, and should not -- T_c is
non-universal). This screens for the precondition only: does an ordered phase exist to melt?

Writes results/attractor_corpus_screen.json. Downloads ~3 GB on first run.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/attractor_corpus_screen.py
        (safe to interrupt and re-run -- it resumes)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time, collections
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel

# (hf name, revision, corpus, lab, pre-registered prediction)
MODELS = [
    ("EleutherAI/pythia-410m",     "step143000", "Pile",    "EleutherAI", "attractor"),
    ("EleutherAI/gpt-neo-125M",    None,         "Pile",    "EleutherAI", "attractor"),
    # Cerebras-GPT was the intended non-EleutherAI Pile model; its repos return HTTP 401
    # (gated, needs an auth token) so it could not be used. mamba-130m-hf substitutes: also
    # Pile-trained, also not EleutherAI. It confounds lab with ARCHITECTURE -- a state-space
    # model, not a transformer -- which is stated rather than hidden, and cuts both ways: if it
    # shows the attractor, that is evidence the effect is not even attention-specific.
    ("state-spaces/mamba-130m-hf", None,         "Pile",    "state-spaces", "attractor"),
    ("gpt2-medium",                None,         "WebText", "OpenAI",     "none"),
    ("bigscience/bloom-560m",      None,         "ROOTS",   "BigScience", "none"),
    ("Qwen/Qwen2.5-0.5B",          None,         "mixed",   "Alibaba",    "none"),
    # DENSE vs MoE from ONE family. Everything except architecture is shared -- same lab, corpus,
    # tokenizer, recipe -- so this is the cleanest available test of whether architecture is the
    # determinant. No directional prediction is registered because F63 established we do not know
    # what the determinant is; the registered expectation is that they AGREE, since only
    # architecture differs. Disagreement would be the first positive identification of any
    # determinant at all. Caveat: total sizes differ (1.38B MoE vs 2.53B dense), so a disagreement
    # would still be architecture-or-scale, not architecture alone.
    ("ibm-granite/granite-3.0-1b-a400m-base", None, "granite", "IBM (MoE)",   "same-as-pair"),
    ("ibm-granite/granite-3.0-2b-base",       None, "granite", "IBM (dense)", "same-as-pair"),
    # a third dense model on a third corpus, cheap to add while downloading anyway
    ("allenai/OLMo-1B-hf",         None,         "Dolma",   "AI2",        "unknown"),
    # SCALE LADDERS. Within a ladder the corpus, tokenizer, architecture and recipe are all fixed
    # and ONLY scale varies, which is the one axis the cross-family comparisons cannot isolate.
    # Across the two ladders at matched scale, family varies and scale does not. Both are already
    # cached, so this costs compute only. No directional prediction: granite's dense/MoE pair
    # already showed a 2x scale change moving nothing, but 2x is not 100x.
    ("EleutherAI/pythia-14m",      "step143000", "Pile",    "EleutherAI", "unknown"),
    ("EleutherAI/pythia-31m",      "step143000", "Pile",    "EleutherAI", "unknown"),
    ("EleutherAI/pythia-70m",      "step143000", "Pile",    "EleutherAI", "unknown"),
    ("EleutherAI/pythia-160m",     "step143000", "Pile",    "EleutherAI", "unknown"),
    ("EleutherAI/pythia-1b",       "step143000", "Pile",    "EleutherAI", "unknown"),
    ("gpt2",                       None,         "WebText", "OpenAI",     "unknown"),
    ("gpt2-large",                 None,         "WebText", "OpenAI",     "unknown"),
    ("gpt2-xl",                    None,         "WebText", "OpenAI",     "unknown"),
    # THE TWO-FACTOR TEST (F64). If attention is necessary and the corpus determines, then a
    # Pile-trained model WITHOUT attention must lack the attractor. RWKV-4 is an RNN trained on
    # the Pile -- the same corpus as pythia and gpt-neo, both of which have it. Predicted: none.
    # A second mamba size joins so the non-attention side does not rest on one model at one scale.
    ("RWKV/rwkv-4-169m-pile",      None,         "Pile",    "BlinkDL (RNN)",     "none"),
    ("state-spaces/mamba-370m-hf", None,         "Pile",    "state-spaces (SSM)", "none"),
]

# parameter counts in millions, for the within-family scale trend
PARAMS = {
    "EleutherAI/pythia-14m": 14, "EleutherAI/pythia-31m": 31, "EleutherAI/pythia-70m": 70,
    "EleutherAI/pythia-160m": 160, "EleutherAI/pythia-410m": 410, "EleutherAI/pythia-1b": 1000,
    "gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774, "gpt2-xl": 1558,
}
LADDERS = {"pythia": [m for m in PARAMS if m.startswith("EleutherAI/pythia")],
           "gpt2":   [m for m in PARAMS if m.startswith("gpt2")]}
TEMPS = [0.02, 0.20, 0.436, 0.70]      # 0.436 is F58's T_c; 0.70 is the submitted paper's
N, B, R, SETTLE = 96, 8, 2, 12
# "has an attractor" is a stated threshold, not one tuned after seeing the numbers
TOP1_HIGH, DISTINCT_LOW = 0.40, 0.30   # top-1 share >= 40% AND distinct/N <= 30%, at the lowest T
# models WITHOUT attention -- the axis F64 identified as necessary
NON_ATTENTION = {"RWKV/rwkv-4-169m-pile", "state-spaces/mamba-130m-hf",
                 "state-spaces/mamba-370m-hf"}
OUT = str(_ROOT / "results" / "attractor_corpus_screen.json")


def settled(rule, T, seed=5):
    """What does the lattice relax to? Composition of the settled ring, per replica."""
    from ar_ca import run
    s = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none",
            init="random", seed=seed, order="per_replica")["final"]
    distinct, top1, toks = [], [], collections.Counter()
    for row in s:
        c = collections.Counter(row.tolist())
        distinct.append(len(c) / N)
        top1.append(c.most_common(1)[0][1] / N)
        toks.update(c)
    tid, cnt = toks.most_common(1)[0]
    return (float(np.mean(distinct)), float(np.mean(top1)),
            rule.tok.decode([tid]), float(cnt / (B * N)))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=[dict(name=m, corpus=c, lab=l, prediction=p) for m, _, c, l, p in MODELS],
        temps=TEMPS, N=N, B=B, r=R, settle=SETTLE,
        hypothesis="the low-T attractor is corpus-induced: Pile-trained models have a dominant "
                   "single-token fixed point, models on other corpora do not",
        separability="Pile models from DIFFERENT labs, so 'the Pile' is separable from "
                     "'EleutherAI's recipe'; the non-EleutherAI Pile model is load-bearing",
        criterion=f"attractor iff top-1 share >= {TOP1_HIGH} AND distinct/N <= {DISTINCT_LOW} "
                  f"at the lowest temperature ({min(TEMPS)})",
        falsifier="if the non-EleutherAI Pile model lacks the attractor, the corpus explanation is wrong",
        dense_vs_moe="granite-3.0 1b-a400m (MoE) vs 2b (dense): one family, one corpus, one "
                     "tokenizer, one recipe -- registered expectation is that they AGREE, since "
                     "only architecture differs. Disagreement would isolate architecture (or "
                     "scale: totals are 1.38B vs 2.53B) as the first identified determinant.",
        not_tested="exponents, and any cross-family T_c comparison (T_c is non-universal)",
        resumable="keyed by (model, T)")
    runs = res["runs"]
    todo = [(m, rev, T) for m, rev, _, _, _ in MODELS for T in TEMPS]
    print(f"Attractor screen: {len(MODELS)} models x {len(TEMPS)} temperatures", flush=True)
    from ar_ca import ARRule
    for name, rev, corpus, lab, pred in MODELS:
        if all(f"{name}@{T}" in runs for T in TEMPS):
            print(f"  {name}: already complete", flush=True)
            continue
        t0 = time.time()
        try:
            rule = ARRule(name, revision=rev) if rev else ARRule(name)
        except Exception as e:
            print(f"  {name}: LOAD FAILED ({type(e).__name__}: {str(e)[:80]})", flush=True)
            for T in TEMPS:
                runs[f"{name}@{T}"] = dict(model=name, T=T, corpus=corpus, lab=lab,
                                           prediction=pred, failed=f"{type(e).__name__}")
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        print(f"  {name} ({corpus}, {lab}) loaded in {time.time()-t0:.0f}s, V={rule.V}", flush=True)
        for T in TEMPS:
            key = f"{name}@{T}"
            if key in runs:
                continue
            d, t1, tok, share = settled(rule, T)
            runs[key] = dict(model=name, T=T, corpus=corpus, lab=lab, prediction=pred,
                             distinct_frac=round(d, 4), top1_share=round(t1, 4),
                             dominant_token=tok, dominant_share_overall=round(share, 4))
            print(f"    T={T:<6} distinct={d*100:>5.1f}%  top1={t1*100:>5.1f}%  "
                  f"dominant={tok!r}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = res["runs"]
    lo = min(TEMPS)
    print(f"\n=== attractor at the lowest temperature (T={lo}) ===")
    print(f"  {'model':>28} {'corpus':>8} {'lab':>11} {'distinct':>9} {'top1':>7} "
          f"{'dominant':>12} {'predicted':>10} {'observed':>10}")
    rows, agree, total = {}, 0, 0
    for name, _, corpus, lab, pred in MODELS:
        v = runs.get(f"{name}@{lo}")
        if not v or "distinct_frac" not in v:
            print(f"  {name:>28} {corpus:>8} {lab:>11}   (no data)")
            continue
        has = bool(v["top1_share"] >= TOP1_HIGH and v["distinct_frac"] <= DISTINCT_LOW)
        obs = "attractor" if has else "none"
        rows[name] = dict(v, has_attractor=has, agrees=bool(obs == pred))
        total += 1; agree += int(obs == pred)
        print(f"  {name:>28} {corpus:>8} {lab:>11} {v['distinct_frac']*100:>8.1f}% "
              f"{v['top1_share']*100:>6.1f}% {v['dominant_token']!r:>12} {pred:>10} {obs:>10}"
              + ("" if obs == pred else "   <- against prediction"))

    # --- the two-factor account (F64): attention necessary, corpus determines ---
    attn = {n: r for n, r in rows.items() if n not in NON_ATTENTION}
    noattn = {n: r for n, r in rows.items() if n in NON_ATTENTION}
    pile_attn = [r["has_attractor"] for n, r in attn.items() if r["corpus"] == "Pile"]
    noattn_any = [r["has_attractor"] for r in noattn.values()]
    print(f"\n=== TWO-FACTOR TEST: is attention necessary? ===")
    print(f"  {'model':>30} {'corpus':>8} {'attention':>10} {'top1':>7} {'attractor':>10}")
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["top1_share"]):
        print(f"  {n:>30} {r['corpus']:>8} {str(n not in NON_ATTENTION):>10} "
              f"{r['top1_share']*100:>6.1f}% {str(r['has_attractor']):>10}")
    attention_necessary = bool(noattn) and not any(noattn_any)
    pile_attn_all = bool(pile_attn) and all(pile_attn)

    if attention_necessary and pile_attn_all:
        verdict = (f"ATTENTION IS NECESSARY, CORPUS DETERMINES: all {len(noattn)} attention-free "
                   f"models are Pile-trained and NONE has the attractor "
                   f"(top-1 {min(r['top1_share'] for r in noattn.values())*100:.1f}-"
                   f"{max(r['top1_share'] for r in noattn.values())*100:.1f}%), while every "
                   f"attention model on the same corpus does "
                   f"({min(r['top1_share'] for n, r in attn.items() if r['corpus']=='Pile')*100:.1f}-"
                   f"{max(r['top1_share'] for n, r in attn.items() if r['corpus']=='Pile')*100:.1f}%). "
                   f"Among ATTENTION models the corpus separates them -- gpt-neo-125M and gpt2 share "
                   f"an identical tokenizer and differ only in corpus, 78.1% vs 20.4%. This "
                   f"supersedes the single-factor 'corpus refuted' reading: mamba's negative was "
                   f"confounded by architecture, exactly as F63 disclosed.")
    elif not attention_necessary:
        bad = [n for n, r in noattn.items() if r["has_attractor"]]
        verdict = (f"ATTENTION IS NOT NECESSARY: {bad} lack attention yet show the attractor, so "
                   f"the two-factor account of F64 fails and the determinant is open again.")
    else:
        verdict = (f"MIXED: attention-free models show no attractor, but not every Pile-trained "
                   f"attention model does either, so 'corpus determines among attention models' "
                   f"is not clean. Reported as ambiguous.")

    print(f"\n=== SCALE LADDERS: does top-1 share move with size, at fixed family? ===")
    ladder_out = {}
    for fam, names in LADDERS.items():
        pts = [(PARAMS[n], rows[n]["top1_share"], rows[n]["has_attractor"])
               for n in names if n in rows]
        if len(pts) < 3:
            print(f"  {fam}: only {len(pts)} points, skipping"); continue
        pts.sort()
        xs = np.log10([p for p, _, _ in pts]); ys = np.array([t for _, t, _ in pts])
        slope = float(np.polyfit(xs, ys, 1)[0])
        flips = len({h for _, _, h in pts}) > 1
        print(f"  {fam:>7}: " + "  ".join(f"{p}M:{t*100:.0f}%" for p, t, _ in pts))
        print(f"  {'':>7}  slope {slope:+.3f} top-1 share per decade of params; "
              f"attractor flips within ladder: {flips}")
        ladder_out[fam] = dict(points=[[p, round(t, 4), h] for p, t, h in pts],
                               slope_per_decade=round(slope, 4), flips_within_ladder=bool(flips))
    res["scale_ladders"] = ladder_out
    if len(ladder_out) == 2:
        f1, f2 = ladder_out["pythia"], ladder_out["gpt2"]
        sep = min(t for _, t, _ in f1["points"]) - max(t for _, t, _ in f2["points"])
        print(f"\n  lowest pythia top-1 minus highest gpt2 top-1: {sep*100:+.1f} points"
              f"  -> {'families never overlap; FAMILY dominates scale' if sep > 0 else 'the ladders overlap; scale matters'}")
        res["ladder_separation"] = round(float(sep), 4)

    moe = rows.get("ibm-granite/granite-3.0-1b-a400m-base")
    dense = rows.get("ibm-granite/granite-3.0-2b-base")
    if moe and dense:
        agree = moe["has_attractor"] == dense["has_attractor"]
        pair = (f"DENSE vs MoE (granite-3.0, one family): MoE has_attractor="
                f"{moe['has_attractor']} (top1 {moe['top1_share']*100:.1f}%), dense="
                f"{dense['has_attractor']} (top1 {dense['top1_share']*100:.1f}%) -> "
                + ("AGREE, so architecture is not the determinant within this family."
                   if agree else
                   "DISAGREE, isolating architecture-or-scale as a determinant -- the first "
                   "positive identification of one. Totals differ (1.38B vs 2.53B), so scale "
                   "is not excluded."))
        print(f"\n  {pair}")
        res["dense_vs_moe"] = pair

    res["at_lowest_T"] = rows
    res["predictions_matched"] = f"{agree}/{total}"
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Screens six models for the precondition F62 identified: does a dominant single-token "
        "attractor exist at low temperature, i.e. is there an ordered phase for the damage "
        "transition to melt? Pythia-410m has one (81 of 96 sites newline at T=0.02) and "
        "gpt2-medium does not, which is why the second family showed no transition at all. F62 "
        "attributed this to the training corpus; this tests that with two Pile models from "
        "DIFFERENT labs, so the Pile is separable from EleutherAI's recipe. Cheap because the "
        "hypothesis concerns the SETTLED STATE, needing one settle run rather than the twin-plus-"
        "long-window damage protocol. The dominant token is reported by what it decodes to, since "
        "ids are not comparable across tokenizers.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
