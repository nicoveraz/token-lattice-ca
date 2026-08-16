"""Is the domain effect UNIDIRECTIONAL on BASE models too? F151's claim, on four times the cohort.

WHY THIS RUN. F151 found 18 of 18 domain arms moving DOWN on two instruct models screened to have
headroom in both directions, and that result became paper 2's headline. It rests on TWO models, and
F151's own boundary says two cannot establish that no model is ever bidirectional. This is the
cheapest available strengthening: F143 already censused 17 base models, and EIGHT of them are
already mid-range, so the screen that cost F150 a day of downloads is free here.

    llm-jp-3-1.8b 0.776   starcoder2-3b 0.724   SmolLM-1.7B 0.562   Qwen1.5-1.8B 0.510
    pythia-410m   0.458   pythia-410m-deduped 0.427   Minerva-3B 0.328   Falcon3-1B-Base 0.214

They also spread across the whole band rather than clustering, and they are THE COHORT THE PROGRAMME
RESTS ON -- F63/F64/F130 are base-model results, and the domain axis has touched base models exactly
once (F146, BOS only).

THE HONEST LIMITATION, STATED BEFORE THE RUN. Base models have no chat template, and the template was
the arm producing the largest shifts in F151 (-0.714 on gemma-2-2b-it). So this tests direction under
BOS and PROSE only. A null here does not extend the template result; it extends the DIRECTION claim,
which is what is under test.

PREFIX LENGTHS ARE GROUNDED, NOT PICKED. With no template there is no natural length, so instead of
choosing one, both extremes of the measured instruct template lengths are used: 9 tokens (gemma's)
and 29 (Qwen2.5's). That also gives a second, weaker read on F147's non-monotonicity within each
model, at no extra cost.

PRE-REGISTERED:
  RUNG       the raw arm must reproduce argmax_census_hardened's stored fixed_point_fraction EXACTLY
             at both census seeds. Same census, same seeds -- any difference and nothing is read.
  ANTI-VACUITY  headroom is CHECKED, not assumed, even though these models were selected on it: a
             model whose room on either side is below its own tolerance is excluded and named. F149's
             failure was exactly a cohort assumed to be able to move.
  PRIMARY    does ANY arm on ANY model move UP beyond its own seed noise? Registered readings:
               zero up-shifts -> F151's unidirectional claim extends to base models, taking the
                 direction result to ten models across two cohorts, and it becomes about as well
                 supported as this instrument can make it.
               any robust up-shift -> the unidirectional claim is FALSIFIED, and paper 2's revised
                 thesis needs revising again. This is the outcome that would matter most, and it is
                 why the run is worth doing before the paper is written rather than after.
  SECONDARY  magnitude text-dependence at FIXED length on base models: the spread of prose arms
             within a model at 9 and at 29 tokens. F148/F151 found this spans most of the statistic's
             range on instruct models; it is now the load-bearing half of the thesis, so it should be
             checked on the other cohort.
  MATCHED PAIR  pythia-410m (0.458) vs pythia-410m-deduped (0.427) differ only in corpus
             deduplication. A free within-pair comparison that no other pair in the project offers.
  BOUNDARY   no chat-template arm exists for base models, so the largest-effect domain is untested
             here; two prefix lengths and two prose sources are not a survey of text.
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

OUT = str(_ROOT / "results" / "domain_base.json")
REF = _ROOT / "results" / "argmax_census_hardened.json"
SHAKE = _ROOT / "data" / "shakespeare.txt"

MODELS = ["llm-jp/llm-jp-3-1.8b", "bigcode/starcoder2-3b", "HuggingFaceTB/SmolLM-1.7B",
          "Qwen/Qwen1.5-1.8B", "EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped",
          "sapienzanlp/Minerva-3B-base-v1.0", "tiiuae/Falcon3-1B-Base"]
PAIR = ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped")

# Grounded in the measured instruct template lengths rather than chosen: gemma's 9 and Qwen2.5's 29.
LENGTHS = (9, 29)
SHAKE_OFFSET = 0.10          # same fixed offset F148 used for shak0
MIN_SHIFT = 4.0 / N_STARTS
NOISE_FACTOR = 2.0


def arms_for(tok):
    """raw, bos, and {corpus,shak} x {9,29}. Offset-selected texts, as in F148."""
    out = {"raw": None}
    b = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if b is not None:
        out["bos"] = [int(b)]
    cids = tok(CORPUS, add_special_tokens=False)["input_ids"]
    raw = SHAKE.read_text(encoding="utf-8", errors="replace")
    p = int(len(raw) * SHAKE_OFFSET)
    nl = raw.find("\n", p)
    sids = tok(raw[(nl + 1 if nl != -1 else p):][:4000], add_special_tokens=False)["input_ids"]
    for n in LENGTHS:
        if len(cids) >= n:
            out[f"corpus@{n}"] = [int(t) for t in cids[:n]]
        if len(sids) >= n:
            out[f"shak@{n}"] = [int(t) for t in sids[:n]]
    return out


def _pair(runs, m, key):
    ks = [f"{m}|s{cs}|{key}" for cs in CENSUS_SEEDS]
    if not all(k in runs for k in ks):
        return None
    v = [runs[k]["fixed_point_fraction"] for k in ks]
    return float(np.mean(v)), float(abs(v[0] - v[1])), [runs[k]["cls"] for k in ks]


def analyse(res):
    runs, parts, analysis = res["runs"], [], {}
    ref = json.load(open(REF))["runs"] if REF.exists() else {}

    errs = []
    for m in MODELS:
        for cs in CENSUS_SEEDS:
            a, b = runs.get(f"{m}|s{cs}|raw"), ref.get(f"{m}|s{cs}")
            if a and b:
                errs.append(abs(a["fixed_point_fraction"] - b["fixed_point_fraction"]))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst == 0.0
    parts.append(
        f"RUNG (raw arm reproduces argmax_census_hardened): {len(errs)} cells, worst error "
        f"{worst:.2e}. "
        + ("Identical, so this is F143's measurement with only the domain varying."
           if ok else "NOT reproduced -- nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    rows = {}
    for m in MODELS:
        r = _pair(runs, m, "raw")
        if r is None:
            continue
        raw_v, raw_n, raw_cls = r
        moves = {}
        for key in sorted({k.split("|")[2] for k in runs
                           if k.startswith(f"{m}|") and len(k.split("|")) == 3
                           and k.split("|")[2] != "raw"}):
            p = _pair(runs, m, key)
            if p is None:
                continue
            v, n, cls = p
            tol = max(MIN_SHIFT, NOISE_FACTOR * max(n, raw_n))
            d = v - raw_v
            moves[key] = dict(value=round(v, 4), shift=round(d, 4), seed_noise=round(n, 4),
                              tol=round(tol, 4), cls=cls[0] if cls[0] == cls[1] else "UNSTABLE",
                              dir="up" if d > tol else ("down" if d < -tol else "flat"))
        if moves:
            rows[m] = dict(raw=round(raw_v, 4), raw_seed_noise=round(raw_n, 4),
                           raw_cls=raw_cls[0] if raw_cls[0] == raw_cls[1] else "UNSTABLE",
                           moves=moves)
    analysis["rows"] = rows
    if not rows:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + " No model complete yet."
        return

    flat = [(m, r["raw"]) for m, r in rows.items()
            if min(r["raw"], 1.0 - r["raw"]) <= max(MIN_SHIFT, NOISE_FACTOR * r["raw_seed_noise"])]
    readable = [m for m in rows if m not in {x[0] for x in flat}]
    parts.append(
        f"ANTI-VACUITY (checked, not assumed): {len(flat)} of {len(rows)} models lack room to move "
        f"both ways beyond tolerance"
        + (f" -- {[(m.split('/')[-1], round(v, 3)) for m, v in flat]}, excluded."
           if flat else ", so every model below could show either direction."))

    ups, downs, flats = [], [], []
    for m in readable:
        for k, v in rows[m]["moves"].items():
            (ups if v["dir"] == "up" else downs if v["dir"] == "down" else flats).append(
                (m, k, v["shift"], v["tol"]))
    n_arms = len(ups) + len(downs) + len(flats)
    analysis.update(n_up=len(ups), n_down=len(downs), n_flat=len(flats), n_arms=n_arms,
                    up_detail=[dict(model=m, arm=k, shift=s, tol=t) for m, k, s, t in ups])
    parts.append(
        f"PRIMARY, direction of every arm on {len(readable)} BASE models ({n_arms} arms): "
        f"{len(downs)} DOWN, {len(ups)} UP, {len(flats)} flat. "
        + (f"UP-SHIFTS FOUND -- {[(m.split('/')[-1], k, s) for m, k, s, _t in ups]}. F151's "
           f"unidirectional claim is FALSIFIED on base models, and paper 2's revised thesis needs "
           f"revising again: conditioning does not only destroy fixed-point structure."
           if ups else
           f"NOT ONE arm moves up. Combined with F151's 18 of 18 on instruct models, that is "
           f"{n_arms + 18} arms across {len(readable) + 2} models spanning both cohorts, every one "
           f"DOWN, on models selected to be able to move either way. The domain effect is "
           f"unidirectional: conditioning destroys fixed-point structure."))
    parts.append(
        "Per model: "
        + "; ".join("{} raw {:.3f} [{}] -> ".format(
            m.split("/")[-1], rows[m]["raw"], rows[m]["raw_cls"])
            + ", ".join(f"{k} {v['shift']:+.3f} ({v['dir']})" for k, v in rows[m]["moves"].items())
            for m in readable) + ".")

    spreads = []
    for m in readable:
        for n in LENGTHS:
            vs = [v["value"] for k, v in rows[m]["moves"].items() if k.endswith(f"@{n}")]
            if len(vs) >= 2:
                spreads.append((m, n, round(max(vs) - min(vs), 4), round(min(vs), 4),
                                round(max(vs), 4)))
    if spreads:
        analysis["magnitude_spread"] = [dict(model=m, length=n, spread=s, lo=lo, hi=hi)
                                        for m, n, s, lo, hi in spreads]
        big = [x for x in spreads if x[2] >= 0.2]
        parts.append(
            "SECONDARY, magnitude text-dependence at FIXED length (two prose sources per length): "
            + "; ".join(f"{m.split('/')[-1]}@{n} spans [{lo:.3f}, {hi:.3f}] = {s:.3f}"
                        for m, n, s, lo, hi in spreads) + ". "
            + (f"{len(big)} of {len(spreads)} (model, length) cells span >= 0.20 on text alone, so "
               f"the magnitude is strongly text-dependent on base models too -- the load-bearing "
               f"half of the thesis holds on the other cohort."
               if big else
               "No cell spans >= 0.20, so on base models the magnitude is much less text-dependent "
               "than F148/F151 found on instruct models, and that difference needs explaining."))

    a, b = PAIR
    if a in rows and b in rows:
        parts.append(
            f"MATCHED PAIR, pythia-410m vs -deduped (identical but for corpus deduplication): raw "
            f"{rows[a]['raw']:.3f} vs {rows[b]['raw']:.3f}; arms "
            + "; ".join(f"{k} {rows[a]['moves'][k]['shift']:+.3f} vs "
                        f"{rows[b]['moves'][k]['shift']:+.3f}"
                        for k in rows[a]["moves"] if k in rows[b]["moves"])
            + ". The only pair in the project differing in ONE pretraining choice.")
    parts.append(
        f"BOUNDARY: {len(rows)} base models, {N_STARTS} starts, {len(CENSUS_SEEDS)} census seeds. "
        f"NO chat-template arm exists for base models, so the largest-effect domain in F151 is "
        f"untested here -- this extends the DIRECTION claim, not the template result. Two prefix "
        f"lengths ({LENGTHS}) and two prose sources are not a survey of text.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=MODELS, lengths=list(LENGTHS), shake_offset=SHAKE_OFFSET, n_starts=N_STARTS,
        census_seeds=CENSUS_SEEDS, min_shift=MIN_SHIFT, noise_factor=NOISE_FACTOR, matched_pair=PAIR,
        rung="the raw arm must reproduce argmax_census_hardened exactly at both seeds",
        primary="does ANY arm on ANY base model move UP beyond its own seed noise",
        falsifier="a single robust up-shift falsifies F151's unidirectional claim and forces another "
                  "revision of paper 2's thesis -- that is the outcome that would matter most",
        lengths_rationale="no chat template exists for base models, so both extremes of the measured "
                          "instruct template lengths are used (gemma 9, Qwen2.5 29) rather than one "
                          "arbitrary choice",
        limitation="base models have no chat template, the arm with the largest effect in F151, so "
                   "this extends the DIRECTION claim only",
        why="F151 is paper 2's headline and rests on two models; eight mid-range base models were "
            "already censused by F143, so the screen is free and the cohort is the one the whole "
            "programme rests on")
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
            arms = arms_for(tok)
            if all(f"{m}|s{cs}|{k}" in res["runs"] for k in arms for cs in CENSUS_SEEDS):
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            try:
                model = AutoModelForCausalLM.from_pretrained(m).eval().to(
                    dev, torch.float16 if dev != "cpu" else torch.float32)
            except Exception as e:
                res["runs"][f"{m}|failed"] = dict(model=m, error=type(e).__name__,
                                                  detail=str(e)[:120])
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: LOAD FAILED {type(e).__name__}", flush=True)
                continue
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            print(f"  {m:<36} {len(arms)} arms: {sorted(arms)}", flush=True)
            for key, pre in arms.items():
                for cs in CENSUS_SEEDS:
                    k = f"{m}|s{cs}|{key}"
                    if k in res["runs"]:
                        continue
                    target = model if pre is None else _Prefixed(model, pre)
                    c = argmax_census(target, tok, dev, pool, np.random.default_rng(cs),
                                      n_starts=N_STARTS)
                    c.update(cls=classify(c), model=m, census_seed=cs, arm=key,
                             n_prefix_tokens=0 if pre is None else len(pre))
                    res["runs"][k] = c
                    json.dump(res, open(OUT, "w"), indent=1)
                    print(f"  {m:<36} {key:<10} s={cs} cls={c['cls']:<11} "
                          f"fix={c['fixed_point_fraction']:.3f}", flush=True)
            done += 1
            print(f"  {m:<36} model done in {time.time()-t0:.0f}s", flush=True)
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
