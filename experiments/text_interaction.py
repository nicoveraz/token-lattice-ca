"""Does ANY text raise TWO models? F153's empty overlap, on 6 models and a real corpus.

WHY THIS IS THE RUN THAT MATTERS. F153's result -- that the up-sets of two models do not overlap at
all, so direction is a TEXT x WEIGHTS interaction and no prefix can be certified in advance -- is now
the most load-bearing claim in paper 2. It is what forecloses calibration, and it rests on TWO models
and TEN texts drawn from ONE paragraph plus Shakespeare.

THE PROJECT'S OWN TRACK RECORD SAYS THIS IS WHERE TO LOOK. Every claim in this programme has weakened
the first time n widened. F147's sign-flip died at two mid-range models; F151's "18 of 18, without
exception" died at nine. Both felt decisive when written. F153 is the narrowest evidence in the
current thesis, so it is the most likely thing to move, and finding that out now is worth more than
finding it out in review.

WHAT WIDENS. Models 2 -> 6, spanning raw 0.213 to 0.573 and including the instruct cohort. Texts 10
-> 12 but drawn mostly from THE PILE (10k rows of web, code, papers and books) instead of four chunks
of one paragraph -- F153's boundary named this exact gap. Sources are sampled at FIXED row indices
and offsets, never by content.

PRE-REGISTERED:
  RUNG       c0 and s0 are byte-identical to F153's for Minerva and Falcon3-1B and must reproduce
             their stored fixed_point_fraction EXACTLY at both seeds.
  ANTI-VACUITY  every model's headroom on both sides must exceed its own tolerance, computed from its
             stored raw seed noise. Models failing this are excluded and named -- F152 lost
             llm-jp-3-1.8b this way, because F150's band used the raw MEAN and ignored raw NOISE.
  PRIMARY    does ANY text produce a robust UP-shift on TWO OR MORE models? Registered readings:
               no such text -> F153's empty overlap survives a 3x wider model set and a real corpus.
                 Direction is an interaction, "structure-raising prefix" is not a coherent category,
                 and a text-feature predictor study is misconceived rather than underpowered. The
                 thesis's strongest clause is earned.
               such a text exists -> F153's empty overlap was a SMALL-SAMPLE ARTEFACT. Some texts do
                 raise structure across weights, text properties may be partly predictive after all,
                 and the "no prefix can be certified" clause must be withdrawn or heavily qualified.
                 This would be the fourth thesis revision and it is the outcome to watch for.
  SECONDARY  the per-model UP rate over a real corpus, extending F153's 3/10 and 1/10.
  TERTIARY   source (pile / shakespeare / corpus) as a predictor of sign: DESCRIPTIVE ONLY, declared
             here before the numbers. Six model clusters cannot fail such a test informatively. F153
             logged "all up-texts were Shakespeare" as a lead precisely because it could not be
             tested; the pile arm is what would eventually let it be.
  BOUNDARY   one prefix length (9 tokens), three sources, English only.
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
from gate1 import argmax_census, CORPUS
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_templated import _Prefixed

OUT = str(_ROOT / "results" / "text_interaction.json")
F153 = _ROOT / "results" / "domain_text_sample.json"
BASE = _ROOT / "results" / "domain_base.json"
SCREEN = _ROOT / "results" / "midrange_screen.json"
SHAKE = _ROOT / "data" / "shakespeare.txt"

# raw values and their source of truth, so anti-vacuity uses each model's OWN stored seed noise
MODELS = {
    "tiiuae/Falcon3-1B-Base":        BASE,      # 0.213, a known up-shifter (F153)
    "sapienzanlp/Minerva-3B-base-v1.0": BASE,   # 0.328, the other known up-shifter
    "EleutherAI/pythia-410m-deduped": BASE,     # 0.427, fast
    "Qwen/Qwen1.5-1.8B":             BASE,      # 0.510
    "HuggingFaceTB/SmolLM-1.7B":     BASE,      # 0.562
    "Qwen/Qwen2.5-1.5B-Instruct":    SCREEN,    # 0.573, extends to the instruct cohort
}
LENGTH = 9
N_CORPUS = 3
SHAKE_OFFSETS = (0.10, 0.40, 0.70)      # 0.10 reproduces F153's s0 -> the RUNG
PILE_ROWS = (7, 101, 503, 1201, 2609, 5417)     # fixed indices, chosen before any text was read
MIN_SHIFT = 4.0 / N_STARTS
NOISE_FACTOR = 2.0


def _pile_texts():
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    out = {}
    for j, r in enumerate(PILE_ROWS):
        t = ds[int(r)]["text"]
        if len(t) >= 400:
            out[f"p{j}"] = t[:4000]
    return out


def texts(tok):
    """Twelve 9-token prefixes. Every one selected by INDEX or OFFSET, never by content."""
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
    for k, t in _pile_texts().items():
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) >= LENGTH:
            out[k] = [int(x) for x in ids[:LENGTH]]
    return out


def _raw_of(m):
    src = json.load(open(MODELS[m]))["runs"]
    keys = ([f"{m}|s{cs}|raw" for cs in CENSUS_SEEDS] if MODELS[m] is BASE
            else [f"{m}|s{cs}" for cs in CENSUS_SEEDS])
    if not all(k in src for k in keys):
        return None
    v = [src[k]["fixed_point_fraction"] for k in keys]
    return float(np.mean(v)), float(abs(v[0] - v[1]))


def analyse(res):
    runs, parts, analysis = res["runs"], [], {}
    old = json.load(open(F153))["runs"] if F153.exists() else {}

    errs = []
    for m in ("sapienzanlp/Minerva-3B-base-v1.0", "tiiuae/Falcon3-1B-Base"):
        for t in ("c0", "s0"):
            for cs in CENSUS_SEEDS:
                a, b = runs.get(f"{m}|s{cs}|{t}"), old.get(f"{m}|s{cs}|{t}")
                if a and b:
                    errs.append(abs(a["fixed_point_fraction"] - b["fixed_point_fraction"]))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst == 0.0
    parts.append(
        f"RUNG (c0/s0 reproduce F153): {len(errs)} cells, worst error {worst:.2e}. "
        + ("Identical." if ok else "NOT reproduced -- nothing below is read."))
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
        tol_raw = max(MIN_SHIFT, NOISE_FACTOR * raw_n)
        if min(raw, 1 - raw) <= tol_raw:
            excluded.append((m, round(raw, 3), round(tol_raw, 3)))
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
        + (f" -- {[(m.split('/')[-1], v, t) for m, v, t in excluded]}, excluded."
           if excluded else ", so every model below can show either direction."))
    if len(rows) < 2:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + f" Only {len(rows)} model(s) complete -- the overlap "
        res["verdict"] += "question needs at least two and is not read."
        return

    up_by_text, counts = {}, {}
    for m, r in rows.items():
        u = [k for k, v in r["texts"].items() if v["dir"] == "up"]
        counts[m] = dict(n=len(r["texts"]), up=len(u),
                         down=sum(1 for v in r["texts"].values() if v["dir"] == "down"),
                         flat=sum(1 for v in r["texts"].values() if v["dir"] == "flat"),
                         up_texts=sorted(u))
        for t in u:
            up_by_text.setdefault(t, []).append(m)
    shared = {t: ms for t, ms in up_by_text.items() if len(ms) >= 2}
    analysis["counts"], analysis["up_by_text"] = counts, {t: sorted(ms) for t, ms in up_by_text.items()}
    analysis["shared_up_texts"] = {t: sorted(ms) for t, ms in shared.items()}

    parts.append(
        f"PRIMARY, does ANY text raise TWO OR MORE of the {len(rows)} models? "
        + (f"YES -- " + "; ".join(f"{t} raises {[x.split('/')[-1] for x in ms]}"
                                  for t, ms in sorted(shared.items()))
           + f". F153's empty overlap was a SMALL-SAMPLE ARTEFACT. Some texts DO raise structure "
             f"across different weights, so 'structure-raising prefix' is a partly coherent category "
             f"after all, text properties may be partly predictive, and the thesis clause that no "
             f"prefix can be certified in advance must be withdrawn or heavily qualified."
           if shared else
           f"NO. Across {sum(c['n'] for c in counts.values())} text-model units and "
           f"{len(up_by_text)} distinct up-texts, not one raises two models. F153's empty overlap "
           f"SURVIVES a {len(rows)}-model set and a real corpus: direction is a TEXT x WEIGHTS "
           f"interaction, 'structure-raising prefix' is not a coherent category, and a text-feature "
           f"predictor study is misconceived rather than underpowered."))
    parts.append(
        "SECONDARY, per-model UP rate over a real corpus: "
        + "; ".join("{} raw {:.3f}: {}/{} up ({})".format(
            m.split("/")[-1], rows[m]["raw"], counts[m]["up"], counts[m]["n"],
            ",".join(counts[m]["up_texts"]) or "-") for m in rows) + ".")

    by_src = {}
    for m, r in rows.items():
        for k, v in r["texts"].items():
            by_src.setdefault(k[0], {"up": 0, "n": 0})
            by_src[k[0]]["n"] += 1
            by_src[k[0]]["up"] += (v["dir"] == "up")
    analysis["by_source"] = by_src
    parts.append(
        "TERTIARY (DESCRIPTIVE ONLY, declared before the numbers -- "
        f"{len(rows)} model clusters cannot fail this test informatively): up-rate by source "
        + ", ".join(f"{k}={v['up']}/{v['n']}" for k, v in sorted(by_src.items()))
        + " (c=CORPUS, p=pile, s=shakespeare). F153 logged 'all up-texts were Shakespeare' as an "
          "untestable lead; this is the same lead with a real corpus beside it, and it is still not "
          "a test.")
    parts.append(
        f"BOUNDARY: {len(rows)} models, {sum(c['n'] for c in counts.values()) // max(len(rows), 1)} "
        f"texts, ONE length ({LENGTH} tokens), three English sources. An empty overlap over these "
        f"texts and models is an existence claim, not a proof that no text ever raises two models.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=list(MODELS), length=LENGTH, n_corpus=N_CORPUS,
        shake_offsets=list(SHAKE_OFFSETS), pile_rows=list(PILE_ROWS), n_starts=N_STARTS,
        census_seeds=CENSUS_SEEDS, min_shift=MIN_SHIFT, noise_factor=NOISE_FACTOR,
        rung="c0/s0 must reproduce domain_text_sample exactly for the two F153 models",
        primary="does ANY text produce a robust up-shift on TWO OR MORE models",
        falsifier="a single shared up-text withdraws the thesis clause that no prefix can be "
                  "certified in advance -- the fourth revision, and the outcome to watch for",
        tertiary="source as a predictor of sign is DESCRIPTIVE ONLY; six clusters cannot fail it",
        why="F153 is now the most load-bearing claim in paper 2 and rests on 2 models and 10 texts "
            "from one paragraph plus Shakespeare; every claim in this programme has weakened the "
            "first time n widened")
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
