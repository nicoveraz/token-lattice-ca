"""How OFTEN does a 9-token prefix push fixed-point structure UP? F152's two up-shifts, sampled.

WHAT F152 LEFT UNFINISHED. It found the only two up-shifts in the programme that survive a gate:
`Falcon3-1B-Base` +0.693 under one BOS token, and `Minerva-3B-base` +0.458 under `shak@9` while the
SAME model at the SAME length fell to 0.000 under `corpus@9`. Both models are bidirectional from a
single raw value, which no floor/ceiling account can produce, and that finding overturned F151.

BUT EACH UP-SHIFT WAS FOUND BY EXACTLY ONE TEXT. Minerva's evidence is two prose samples of the same
length pointing opposite ways -- a text sample size of TWO. F148 was built precisely to stop claims
resting on single draws, and letting F152's stand would repeat the defect F148 exists to catch. This
samples the text axis properly on the two models that showed the effect.

PRE-REGISTERED:
  RUNG       the two texts already measured (corpus@9, shak@9) must reproduce domain_base's stored
             fixed_point_fraction EXACTLY for both models at both seeds. Same census, same prefixes.
  ANTI-VACUITY  both models were shown readable in F152 (headroom on both sides exceeds tolerance);
             it is re-checked here rather than assumed, using each model's own raw seed noise.
  PRIMARY    the FRACTION of 9-token texts producing a robust UP-shift, per model, each judged
             against its own seed noise. Registered readings:
               a substantial minority go up -> bidirectionality is a real and reasonably common
                 property of these models, and F152's claim stands with a rate attached instead of
                 an anecdote.
               only the originally-found text goes up -> the up-shift is a single-text curiosity.
                 F152's bidirectionality survives as an existence result (one text IS enough to
                 refute a universal) but the finding must carry that caveat prominently, and any
                 impression that "prefixes often raise structure" is wrong.
               most go up -> the DOWN arms were the unusual ones, and the tendency claim
                 ("conditioning usually destroys structure") is itself a sampling artefact of which
                 texts happened to be used across the programme. This would be the largest revision
                 yet and would reach back to every domain finding.
  SECONDARY  TEXT x MODEL: the same ten texts run on both models. If the same texts go up on both,
             the effect belongs to the TEXT and might be predictable from it. If different texts go
             up on each, it is an interaction and no property of the text alone can explain it.
             This is answerable with this design, unlike a predictor study.
  NOT TESTED surface features of the text (punctuation, casing, dialogue-vs-prose, mid-word starts)
             as predictors of sign. Declared HERE, before the numbers: ten texts cannot fail such a
             test informatively, and computing one would manufacture a result. F149's refusal.
  BOUNDARY   two models, ONE length (9 tokens), two prose sources. A rate over ten texts is a rate
             over THESE ten texts.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from gate1 import argmax_census, CORPUS
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_templated import _Prefixed

OUT = str(_ROOT / "results" / "domain_text_sample.json")
BASE = _ROOT / "results" / "domain_base.json"
SHAKE = _ROOT / "data" / "shakespeare.txt"

MODELS = ["sapienzanlp/Minerva-3B-base-v1.0", "tiiuae/Falcon3-1B-Base"]
LENGTH = 9                       # the length at which F152 found Minerva going both ways
N_CORPUS = 4                     # disjoint 9-token chunks of gate1's CORPUS
SHAKE_OFFSETS = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85)   # 0.10 reproduces F152's shak@9 (the RUNG)
MIN_SHIFT = 4.0 / N_STARTS
NOISE_FACTOR = 2.0


def texts(tok):
    """Ten 9-token prefixes, every one selected by OFFSET and never by content."""
    out = {}
    cids = tok(CORPUS, add_special_tokens=False)["input_ids"]
    for j in range(N_CORPUS):
        ch = cids[j * LENGTH:(j + 1) * LENGTH]
        if len(ch) == LENGTH:
            out[f"c{j}"] = [int(t) for t in ch]
    raw = SHAKE.read_text(encoding="utf-8", errors="replace")
    for j, frac in enumerate(SHAKE_OFFSETS):
        p = int(len(raw) * frac)
        nl = raw.find("\n", p)
        ids = tok(raw[(nl + 1 if nl != -1 else p):][:4000], add_special_tokens=False)["input_ids"]
        if len(ids) >= LENGTH:
            out[f"s{j}"] = [int(t) for t in ids[:LENGTH]]
    return out


def _pair(runs, m, key):
    ks = [f"{m}|s{cs}|{key}" for cs in CENSUS_SEEDS]
    if not all(k in runs for k in ks):
        return None
    v = [runs[k]["fixed_point_fraction"] for k in ks]
    return float(np.mean(v)), float(abs(v[0] - v[1]))


def analyse(res):
    runs, parts, analysis = res["runs"], [], {}
    base = json.load(open(BASE))["runs"] if BASE.exists() else {}

    # RUNG: c0 is CORPUS[0:9] = domain_base's corpus@9; s0 is shakespeare@0.10 = its shak@9.
    errs = []
    for m in MODELS:
        for mine, theirs in (("c0", "corpus@9"), ("s0", "shak@9")):
            for cs in CENSUS_SEEDS:
                a, b = runs.get(f"{m}|s{cs}|{mine}"), base.get(f"{m}|s{cs}|{theirs}")
                if a and b:
                    errs.append(abs(a["fixed_point_fraction"] - b["fixed_point_fraction"]))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst == 0.0
    parts.append(
        f"RUNG (c0/s0 reproduce domain_base's corpus@9/shak@9): {len(errs)} cells, worst error "
        f"{worst:.2e}. "
        + ("Identical, so this is F152's measurement with only the text varying."
           if ok else "NOT reproduced -- nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    rows = {}
    for m in MODELS:
        r = base.get(f"{m}|s{CENSUS_SEEDS[0]}|raw")
        if not r:
            continue
        rv = [base[f"{m}|s{cs}|raw"]["fixed_point_fraction"] for cs in CENSUS_SEEDS]
        raw, raw_n = float(np.mean(rv)), float(abs(rv[0] - rv[1]))
        per = {}
        for k in sorted({x.split("|")[2] for x in runs
                         if x.startswith(f"{m}|") and len(x.split("|")) == 3}):
            p = _pair(runs, m, k)
            if p is None:
                continue
            v, n = p
            tol = max(MIN_SHIFT, NOISE_FACTOR * max(n, raw_n))
            d = v - raw
            per[k] = dict(value=round(v, 4), shift=round(d, 4), tol=round(tol, 4),
                          dir="up" if d > tol else ("down" if d < -tol else "flat"))
        if per:
            rows[m] = dict(raw=round(raw, 4), raw_seed_noise=round(raw_n, 4), texts=per)
    analysis["rows"] = rows
    if not rows:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + " No model complete yet."
        return

    flat = [m for m, r in rows.items()
            if min(r["raw"], 1 - r["raw"]) <= max(MIN_SHIFT, NOISE_FACTOR * r["raw_seed_noise"])]
    parts.append(
        f"ANTI-VACUITY (re-checked, not assumed): {len(flat)} of {len(rows)} models lack room to "
        f"move both ways" + (f" -- {[m.split('/')[-1] for m in flat]}, excluded." if flat
                             else ", so both can show either direction."))
    readable = [m for m in rows if m not in flat]

    ups_by_model = {}
    for m in readable:
        per = rows[m]["texts"]
        u = [k for k, v in per.items() if v["dir"] == "up"]
        d = [k for k, v in per.items() if v["dir"] == "down"]
        f = [k for k, v in per.items() if v["dir"] == "flat"]
        ups_by_model[m] = set(u)
        analysis.setdefault("counts", {})[m] = dict(n=len(per), up=len(u), down=len(d), flat=len(f),
                                                    up_texts=sorted(u))
    parts.append(
        f"PRIMARY, fraction of {LENGTH}-token texts producing a robust UP-shift: "
        + "; ".join("{} raw {:.3f}: {}/{} up ({}), {} down, {} flat".format(
            m.split("/")[-1], rows[m]["raw"], analysis["counts"][m]["up"],
            analysis["counts"][m]["n"], ",".join(analysis["counts"][m]["up_texts"]) or "-",
            analysis["counts"][m]["down"], analysis["counts"][m]["flat"]) for m in readable)
        + ". ")
    tot_up = sum(analysis["counts"][m]["up"] for m in readable)
    tot = sum(analysis["counts"][m]["n"] for m in readable)
    only_orig = all(analysis["counts"][m]["up_texts"] in ([], ["s0"]) for m in readable)
    parts.append(
        (f"Most texts go up ({tot_up} of {tot}), so the DOWN arms across the programme were the "
         f"unusual ones and the 'conditioning usually destroys structure' tendency is itself a "
         f"sampling artefact of which texts were used. This reaches back to every domain finding."
         if tot_up > tot / 2 else
         f"ONLY the originally-found text goes up. F152's bidirectionality survives as an EXISTENCE "
         f"result -- one text is enough to refute a universal -- but it is a single-text curiosity "
         f"and must be reported as one. Any impression that prefixes often RAISE structure is wrong."
         if only_orig and tot_up > 0 else
         f"A minority of texts go up ({tot_up} of {tot}), so bidirectionality is a real and "
         f"reasonably common property of these models rather than an anecdote, and F152's claim now "
         f"carries a rate instead of a single draw."
         if tot_up else
         "NO text produces an up-shift here, which CONTRADICTS the RUNG-verified F152 cells and "
         "means the analysis is wrong, not the data -- investigate before reading anything."))

    if len(readable) == 2:
        a, b = readable
        sa, sb = ups_by_model[a], ups_by_model[b]
        both, either = sa & sb, sa | sb
        analysis["text_by_model"] = dict(up_on_both=sorted(both), up_on_either=sorted(either))
        parts.append(
            f"SECONDARY, TEXT x MODEL over the same ten texts: up on both models {sorted(both) or '-'}"
            f", up on exactly one {sorted(either - both) or '-'}. "
            + ("Every up-text is shared, so the effect belongs to the TEXT and may be predictable "
               "from it -- worth a targeted follow-up."
               if both and not (either - both) else
               "No text raises both models, so the up-shift is an INTERACTION of text and weights: "
               "no property of the text alone can explain it, and a predictor study on text features "
               "would be looking in the wrong place."
               if not both else
               "Some up-texts are shared and some are not, so the effect is partly the text and "
               "partly the model -- neither a pure text property nor a pure interaction."))
    parts.append(
        f"NOT TESTED, as declared before the run: surface features of the text as predictors of "
        f"sign. {tot} text-model units over {len(readable)} models cannot fail such a test "
        f"informatively, and computing one would manufacture a result.")
    parts.append(
        f"BOUNDARY: {len(readable)} models, ONE length ({LENGTH} tokens), {N_CORPUS} CORPUS chunks "
        f"and {len(SHAKE_OFFSETS)} Shakespeare offsets, all selected by OFFSET and never by content. "
        f"A rate over ten texts is a rate over THESE ten texts, and both sources are English prose "
        f"or verse -- not a survey of what can precede a loop.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=MODELS, length=LENGTH, n_corpus=N_CORPUS, shake_offsets=list(SHAKE_OFFSETS),
        n_starts=N_STARTS, census_seeds=CENSUS_SEEDS, min_shift=MIN_SHIFT,
        noise_factor=NOISE_FACTOR,
        rung="c0/s0 must reproduce domain_base's corpus@9/shak@9 exactly",
        primary="the FRACTION of 9-token texts producing a robust up-shift, per model",
        secondary="TEXT x MODEL -- do the SAME texts raise both models (a text property) or "
                  "different ones (an interaction)",
        not_tested="text surface features as predictors of sign -- ten texts cannot fail that test "
                   "informatively, declared before the numbers",
        why="F152's two up-shifts were each found by exactly ONE text; Minerva's evidence is a text "
            "sample of size TWO, and F148 exists to stop claims resting on single draws")
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for m in MODELS:
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
            except Exception as e:
                print(f"  {m}: TOK FAILED ({type(e).__name__})", flush=True)
                continue
            tx = texts(tok)
            if all(f"{m}|s{cs}|{k}" in res["runs"] for k in tx for cs in CENSUS_SEEDS):
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            try:
                model = AutoModelForCausalLM.from_pretrained(m).eval().to(
                    dev, torch.float16 if dev != "cpu" else torch.float32)
            except Exception as e:
                res["runs"][f"{m}|failed"] = dict(model=m, error=type(e).__name__)
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: LOAD FAILED ({type(e).__name__})", flush=True)
                continue
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            print(f"  {m:<34} {len(tx)} texts at {LENGTH} tok", flush=True)
            for k, pre in tx.items():
                for cs in CENSUS_SEEDS:
                    key = f"{m}|s{cs}|{k}"
                    if key in res["runs"]:
                        continue
                    c = argmax_census(_Prefixed(model, pre), tok, dev, pool,
                                      np.random.default_rng(cs), n_starts=N_STARTS)
                    c.update(cls=classify(c), model=m, census_seed=cs, text=k,
                             n_prefix_tokens=len(pre))
                    res["runs"][key] = c
                    json.dump(res, open(OUT, "w"), indent=1)
                    print(f"  {m:<34} {k:<4} s={cs} cls={c['cls']:<11} "
                          f"fix={c['fixed_point_fraction']:.3f}", flush=True)
            done += 1
            print(f"  {m:<34} model done in {time.time()-t0:.0f}s", flush=True)
            del model
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
