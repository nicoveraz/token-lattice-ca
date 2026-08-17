"""Are F151's INSTRUCT models raisable by any text, or is the instruct cohort genuinely different?

WHAT IS LEFT UNTESTED. F151 ran ten domain arms on `Qwen2.5-1.5B-Instruct` and `gemma-2-2b-it` and
got 18 of 18 DOWN. That became paper 2's headline and was overturned by F152 on base models. But
these two were never re-run with SAMPLED text -- and sampling the text axis is exactly what broke
every subsequent claim:

    F152  "18 of 18, without exception"          died on a wider MODEL set
    F153  "no text raises two models"            died on a wider CORPUS
    F154  "bidirectionality is a model property" died on a wider TEXT CLASS (F155)

Three for three, each an apparent PROPERTY dissolving into an undersampled interaction. F151's arms
used F148's prose ensemble -- chunks of one paragraph plus Shakespeare -- with nothing structural in
it, which is the precise gap that F155 showed matters.

WHY THE NULL WOULD BE THE INTERESTING OUTCOME HERE. Every other factor named in this programme has
been withdrawn within one run of naming it. If these two instruct models resist raising across twelve
texts including six structural ones -- when `SmolLM-1.7B` went 0.562 -> 0.990 under legal boilerplate
and every base model tested is now known to be raisable -- that would be the FIRST factor to survive
widening: a real instruct-vs-base cohort difference. It would also give paper 2 something it does not
currently have, which is a positive structural claim rather than a catalogue of what cannot be
predicted.

THE TEXTS ARE F155'S, UNCHANGED, so the two cohorts are compared on identical prefixes: six
STRUCTURAL rows (newline/markup/digit density >= 0.15 over the first 200 chars, the Pile's p95) and
six PROSE rows (<= 0.02), taken in index order and never read for content.

PRE-REGISTERED:
  RUNG       `Qwen2.5-1.5B-Instruct` already has F154's `p1`, which is F155's `t0`. It must reproduce
             text_interaction's stored value EXACTLY. Same prefix, same seeds.
  ANTI-VACUITY  headroom on both sides must exceed each model's own tolerance, from its stored raw
             seed noise. Verified, not assumed.
  PRIMARY    is EITHER model raised by ANY of the twelve texts? Registered readings:
               any robust up-shift -> F151's 18-of-18 was a text-sampling artefact on the instruct
                 cohort too, exactly as on base models. No cohort difference; the programme's
                 "no clean factors" pattern goes four for four.
               neither raised -> the FIRST factor in this programme to survive widening. Instruct
                 models would resist raising where every base model tested does not, and that is a
                 positive claim paper 2 can make. It would need its own replication before being
                 leaned on, and the boundary below says why.
  SECONDARY  structural vs prose up-rate within each model, for comparison with F155's base-model
             numbers on the SAME twelve texts.
  BOUNDARY   TWO instruct models. A null over two models and twelve texts is not "instruct models
             are unraisable" -- it is "these two resisted these twelve", and the programme's own
             history says that distinction is where claims die. `gemma-2-2b-it` is also the only
             FRAGMENTED model in the domain work, so cohort and class are confounded in it.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from provenance import stamp, rel
from gate1 import argmax_census
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_templated import _Prefixed
from structural_text import pick_rows, texts, LENGTH      # identical prefixes to F155

OUT = str(_ROOT / "results" / "instruct_raisable.json")
TI = _ROOT / "results" / "text_interaction.json"
SCREEN = _ROOT / "results" / "midrange_screen.json"
ST = _ROOT / "results" / "structural_text.json"

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "google/gemma-2-2b-it"]
MIN_SHIFT = 4.0 / N_STARTS
NOISE_FACTOR = 2.0


def _raw_of(m):
    src = json.load(open(SCREEN))["runs"]
    ks = [f"{m}|s{cs}" for cs in CENSUS_SEEDS]
    if not all(k in src for k in ks):
        return None
    v = [src[k]["fixed_point_fraction"] for k in ks]
    return float(np.mean(v)), float(abs(v[0] - v[1]))


def analyse(res):
    runs, parts, analysis = res["runs"], [], {}
    ti = json.load(open(TI))["runs"] if TI.exists() else {}

    errs = []
    m0 = "Qwen/Qwen2.5-1.5B-Instruct"
    for cs in CENSUS_SEEDS:
        a, b = runs.get(f"{m0}|s{cs}|t0"), ti.get(f"{m0}|s{cs}|p1")
        if a and b:
            errs.append(abs(a["fixed_point_fraction"] - b["fixed_point_fraction"]))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst == 0.0
    parts.append(
        f"RUNG (t0 is F154's p1 on Qwen2.5-1.5B-Instruct): {len(errs)} cells, worst error "
        f"{worst:.2e}. " + ("Identical." if ok else "NOT reproduced -- nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    rows, excluded = {}, []
    for m in MODELS:
        r = _raw_of(m)
        if r is None:
            continue
        raw, raw_n = r
        if min(raw, 1 - raw) <= max(MIN_SHIFT, NOISE_FACTOR * raw_n):
            excluded.append((m, round(raw, 3)))
            continue
        per = {}
        for k in sorted({x.split("|")[2] for x in runs
                         if x.startswith(f"{m}|") and len(x.split("|")) == 3}):
            ks = [f"{m}|s{cs}|{k}" for cs in CENSUS_SEEDS]
            if not all(x in runs for x in ks):
                continue
            v = [runs[x]["fixed_point_fraction"] for x in ks]
            mu, n = float(np.mean(v)), float(abs(v[0] - v[1]))
            tol = max(MIN_SHIFT, NOISE_FACTOR * max(n, raw_n))
            d = mu - raw
            per[k] = dict(value=round(mu, 4), shift=round(d, 4), tol=round(tol, 4),
                          dir="up" if d > tol else ("down" if d < -tol else "flat"))
        if per:
            rows[m] = dict(raw=round(raw, 4), raw_seed_noise=round(raw_n, 4), texts=per)
    analysis["rows"], analysis["excluded"] = rows, excluded
    parts.append(
        f"ANTI-VACUITY: {len(excluded)} model(s) lack room to move both ways"
        + (f" -- {excluded}, excluded." if excluded else ", so both can show either direction."))
    if not rows:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + " No model complete yet."
        return

    ups = {m: [k for k, v in r["texts"].items() if v["dir"] == "up"] for m, r in rows.items()}
    n_up = sum(len(u) for u in ups.values())
    n_tot = sum(len(r["texts"]) for r in rows.values())
    analysis["up_texts"] = ups
    parts.append(
        f"PRIMARY, is either INSTRUCT model raised by any of the twelve texts? "
        + "; ".join("{} raw {:.3f}: {} up of {} ({})".format(
            m.split("/")[-1], rows[m]["raw"], len(ups[m]), len(rows[m]["texts"]),
            ",".join(ups[m]) or "none") for m in rows)
        + f". Total {n_up} of {n_tot}. "
        + (f"RAISED. F151's 18-of-18 was a text-sampling artefact on the instruct cohort too, just "
           f"as F152 showed for base models. There is NO instruct-vs-base cohort difference here, "
           f"and the programme's 'no clean factors -- only text x weights pairs' pattern goes FOUR "
           f"for four."
           if n_up else
           f"NEITHER is raised by any of twelve texts, six of them structural -- the same texts that "
           f"took `SmolLM-1.7B` from 0.562 to 0.990. This is the FIRST factor in this programme to "
           f"survive a widening: instruct models resist raising where every base model tested does "
           f"not. It is a positive claim paper 2 can make, and it needs replication on more instruct "
           f"models before being leaned on -- two models is exactly the n at which F151 and F153 "
           f"both failed."))
    per_arm = {}
    for m, r in rows.items():
        s_up = sum(1 for k, v in r["texts"].items() if k[0] == "t" and v["dir"] == "up")
        s_n = sum(1 for k in r["texts"] if k[0] == "t")
        p_up = sum(1 for k, v in r["texts"].items() if k[0] == "r" and v["dir"] == "up")
        p_n = sum(1 for k in r["texts"] if k[0] == "r")
        per_arm[m] = dict(struct_up=s_up, struct_n=s_n, prose_up=p_up, prose_n=p_n)
    analysis["per_arm"] = per_arm
    parts.append(
        "SECONDARY, structural vs prose on the SAME twelve texts F155 used on base models: "
        + "; ".join("{} struct {}/{}, prose {}/{}".format(
            m.split("/")[-1], per_arm[m]["struct_up"], per_arm[m]["struct_n"],
            per_arm[m]["prose_up"], per_arm[m]["prose_n"]) for m in rows)
        + ". F155's base-model comparison: Falcon3-1B 4/6 vs 1/6, Minerva 3/6 vs 1/6, "
          "Qwen1.5 0/6 vs 1/6, SmolLM 1/6 vs 0/6.")
    parts.append(
        f"BOUNDARY: TWO instruct models, twelve texts, ONE length ({LENGTH} tokens), one corpus. A "
        f"null here is 'these two resisted these twelve', NOT 'instruct models are unraisable' -- "
        f"and this programme's history is that the distinction is where claims die. `gemma-2-2b-it` "
        f"is also the only FRAGMENTED model in the domain work, so cohort and class are confounded "
        f"in it.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    _keep = {k: v for k, v in res.get("_preregistration", {}).items()
             if k in ("struct_rows", "prose_rows")}
    res["_preregistration"] = dict(
        models=MODELS, length=LENGTH, n_starts=N_STARTS, census_seeds=CENSUS_SEEDS,
        texts="F155's twelve, unchanged, so the cohorts are compared on IDENTICAL prefixes",
        rung="t0 is F154's p1 and must reproduce it on Qwen2.5-1.5B-Instruct",
        primary="is either instruct model raised by ANY of the twelve texts",
        interesting_null="if neither is raised, that is the FIRST factor in this programme to "
                         "survive a widening -- a real instruct-vs-base cohort difference",
        why="F151's arms used F148's prose ensemble with nothing structural in it, and sampling the "
            "text axis is what broke F152, F153 and F154 in turn")
    res["_preregistration"].update(_keep)
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        ds, struct, prose = pick_rows()
        res["_preregistration"]["struct_rows"] = struct
        res["_preregistration"]["prose_rows"] = prose
        print(f"  structural rows {struct}\n  prose rows      {prose}", flush=True)
        done = 0
        for m in MODELS:
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
            except Exception as e:
                print(f"  {m}: TOK FAILED ({type(e).__name__})", flush=True)
                continue
            tx = texts(tok, ds, struct, prose)
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
            print(f"  {m:<30} {len(tx)} texts at {LENGTH} tok", flush=True)
            for k, pre in tx.items():
                for cs in CENSUS_SEEDS:
                    key = f"{m}|s{cs}|{k}"
                    if key in res["runs"]:
                        continue
                    c = argmax_census(_Prefixed(model, pre), tok, dev, pool,
                                      np.random.default_rng(cs), n_starts=N_STARTS)
                    c.update(cls=classify(c), model=m, census_seed=cs, text=k,
                             n_prefix_tokens=len(pre), arm="structural" if k[0] == "t" else "prose")
                    res["runs"][key] = c
                    json.dump(res, open(OUT, "w"), indent=1)
                    print(f"  {m:<30} {k:<4} s={cs} cls={c['cls']:<11} "
                          f"fix={c['fixed_point_fraction']:.3f}", flush=True)
            done += 1
            print(f"  {m:<30} model done in {time.time()-t0:.0f}s", flush=True)
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
