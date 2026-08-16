"""Find instruct models with MID-RANGE raw fixed-point fractions. The prerequisite F149 identified.

WHAT F149 ESTABLISHED, AND WHY THIS IS THE NEXT RUN. M3b could not test whether the domain's
direction is predictable, because scoring the floor/ceiling baseline only where it COULD have been
wrong left one model and two units. Three of six models sit at raw 0.000 and cannot move down;
SmolLM2 has 0.021 of headroom against a 0.042 floor; Falcon3's raw disagrees across census seeds by
more than any of its domain shifts. The blocker is not the number of models -- six more models at
raw 0.000 buy exactly nothing. It is that the cohort is PINNED AT THE EXTREMES. A model can only
show that direction is a property of the interaction if it has room to move both ways.

THIS IS INSTRUMENT-BUILDING, NOT A HYPOTHESIS TEST, and that is registered here so the yield is not
later read as a result. The candidate list is what happens to be cached plus a hand-specified
extension; it is NOT a sample of instruct models, so "X% of instruct models are mid-range" is not
supported by anything below and must not be claimed. The only output that counts is the LIST.

CHEAP BY CONSTRUCTION. One census per candidate at one seed -- no domain arms, no second seed --
which is about a twentieth of what a full domain-gradient cell costs. Only candidates that land in
range earn a confirmation seed. Screening is meant to be run on many models and mostly discard them.

PRE-REGISTERED:
  RUNG       re-screening models already measured must reproduce argmax_census_instruct's stored
             fixed_point_fraction EXACTLY at the same seed. Same census, same seed, same rule -- any
             difference means this screen is not measuring F144's quantity and no candidate below is
             trusted.
  BAND       [0.2, 0.8], fixed here before any candidate is run. The band exists so a model has room
             to move by more than the 4/96 tolerance in BOTH directions; 0.2 leaves 0.2 below and
             0.8 leaves 0.2 above, both far outside census noise.
  PRIMARY    the LIST of candidates whose raw fixed_point_fraction lands in the band, and which have
             a chat template (a mid-range model with no template cannot carry the kind contrast that
             motivated the search, so it is recorded but flagged unusable for M2-style work).
  CONFIRM    a candidate in the band is only KEPT if a second census seed also lands in the band.
             One seed placing a model mid-range is not enough -- Falcon3 is the standing example of
             a raw value that moves 0.615 to 0.771 between seeds.
  SECONDARY  extra census seeds for Falcon3's raw arm specifically. F149 showed its tolerance (0.312)
             swamps every domain shift it has, which is why its apparent +0.146 template rise could
             not be read. More seeds on the raw arm is the cheapest way to recover a model already
             fully measured on every domain.
  BOUNDARY   a convenience cohort screened on one statistic at one temperature-free greedy map.
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
from gate1 import argmax_census
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_templated import template_ids

OUT = str(_ROOT / "results" / "midrange_screen.json")
REF = _ROOT / "results" / "argmax_census_instruct.json"

BAND = (0.2, 0.8)
SCREEN_SEED = CENSUS_SEEDS[0]
CONFIRM_SEED = CENSUS_SEEDS[1]

# Already measured -- these are the RUNG, not candidates.
KNOWN = ["google/gemma-1.1-2b-it", "tiiuae/Falcon3-3B-Instruct"]

# STRATUM A: instruct/chat models already in the local cache, so screening them costs no download.
CACHED = ["ibm-granite/granite-3.1-2b-instruct",
          "internlm/internlm2_5-1_8b-chat",
          "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
          "microsoft/Phi-4-mini-instruct",
          "microsoft/bitnet-b1.58-2B-4T-bf16"]

# STRATUM B: pre-specified extension, fixed HERE before any result is seen so the list cannot be
# grown toward whatever happens to land mid-range. Downloaded only if stratum A yields too few.
EXTENSION = ["Qwen/Qwen2.5-1.5B-Instruct",
             "HuggingFaceTB/SmolLM2-360M-Instruct",
             "meta-llama/Llama-3.2-1B-Instruct",
             "google/gemma-2-2b-it",
             "tiiuae/Falcon3-1B-Instruct",
             "allenai/OLMo-2-1124-7B-Instruct"]

CANDIDATES = CACHED + EXTENSION
# Falcon3's raw arm is seed-unstable; F149 could not read its template shift because of it.
EXTRA_RAW_SEEDS = [20260815, 771013]
EXTRA_RAW_MODEL = "tiiuae/Falcon3-3B-Instruct"


def census(model, tok, dev, seed):
    V = int(getattr(model.config, "vocab_size", len(tok)))
    sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id)
          if i is not None}
    pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
    c = argmax_census(model, tok, dev, pool, np.random.default_rng(seed), n_starts=N_STARTS)
    c["cls"] = classify(c)
    return c


def analyse(res):
    runs, parts = res["runs"], []
    ref = json.load(open(REF))["runs"] if REF.exists() else {}

    errs = []
    for m in KNOWN:
        a, b = runs.get(f"{m}|s{SCREEN_SEED}"), ref.get(f"{m}|s{SCREEN_SEED}")
        if a and b:
            errs.append((m, abs(a["fixed_point_fraction"] - b["fixed_point_fraction"])))
    worst = max((e for _m, e in errs), default=float("inf"))
    ok = bool(errs) and worst == 0.0
    parts.append(
        f"RUNG (known models reproduce argmax_census_instruct at the same seed): {len(errs)} "
        f"compared, worst error {worst:.2e}. "
        + ("Identical, so this screen measures F144's quantity."
           if ok else "NOT reproduced -- the screen is not F144's measurement and no candidate "
                      "below is trusted."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    screened, in_band, kept, no_template = [], [], [], []
    for m in CANDIDATES:
        a = runs.get(f"{m}|s{SCREEN_SEED}")
        if not a:
            continue
        v = a["fixed_point_fraction"]
        screened.append((m, v, a["cls"], a.get("has_template"), a.get("n_template_tokens")))
        if BAND[0] <= v <= BAND[1]:
            in_band.append(m)
            b = runs.get(f"{m}|s{CONFIRM_SEED}")
            if b and BAND[0] <= b["fixed_point_fraction"] <= BAND[1]:
                kept.append((m, v, b["fixed_point_fraction"], a.get("has_template")))
                if not a.get("has_template"):
                    no_template.append(m)
    res["analysis"] = dict(
        screened=[dict(model=m, fix=round(v, 4), cls=c, has_template=bool(t), n_template=n)
                  for m, v, c, t, n in screened],
        in_band=in_band, kept=[dict(model=m, seed0=round(a, 4), seed1=round(b, 4),
                                    has_template=bool(t)) for m, a, b, t in kept],
        band=list(BAND))
    # NO SILENT DROPS. A candidate that never loaded is UNSCREENED, not rejected, and leaving it out
    # of the count would make the screen's coverage look better than it is.
    failed = [(k.split("|")[0], v.get("error"), v.get("detail", "")[:60])
              for k, v in runs.items() if k.endswith("|failed")]
    attempted = len(screened) + len(failed)
    parts.append(
        f"SCREEN of {len(screened)} candidates at seed {SCREEN_SEED} (of {attempted} attempted): "
        + "; ".join(f"{m.split('/')[-1]} {v:.3f} ({c}{'' if t else ', NO TEMPLATE'})"
                    for m, v, c, t, _n in screened) + ". "
        + (f"{len(failed)} candidate(s) NEVER LOADED and are UNSCREENED, not rejected -- "
           + "; ".join(f"{m.split('/')[-1]} ({e})" for m, e, _d in failed)
           + ". These are environment failures (cache corruption, transformers-version config "
             "incompatibility), not properties of the models, so they remain open candidates and "
             "the screen's coverage is lower than the candidate list suggests. "
           if failed else ""))
    res["analysis"]["unscreened_load_failures"] = [
        dict(model=m, error=e, detail=d) for m, e, d in failed]
    parts.append(
        f"PRIMARY, candidates landing in the pre-registered band {BAND} on BOTH seeds: "
        + (", ".join(f"{m.split('/')[-1]} ({a:.3f}, {b:.3f}"
                     + (")" if t else ", NO TEMPLATE -- unusable for the kind contrast)")
                     for m, a, b, t in kept) if kept else "NONE")
        + f". {len(in_band)} of {len(screened)} landed in band on the screen seed, "
          f"{len(kept)} survived confirmation. "
        + (f"This is the cohort M3b needs -- models with room to move by more than tolerance in "
           f"BOTH directions."
           if kept else
           "The screen found nothing usable yet. That is informative about the CANDIDATES, not "
           "about instruct models: extreme raw values may simply be typical, in which case M3b's "
           "question may not be answerable on instruct models at all and would need a different "
           "cohort or a different statistic."))

    ex = [runs[f"{EXTRA_RAW_MODEL}|s{s}"]["fixed_point_fraction"]
          for s in [SCREEN_SEED, CONFIRM_SEED] + EXTRA_RAW_SEEDS
          if f"{EXTRA_RAW_MODEL}|s{s}" in runs]
    if len(ex) >= 3:
        sd = float(np.std(ex, ddof=1))
        res["analysis"]["falcon3_raw"] = dict(values=[round(v, 4) for v in ex],
                                              mean=round(float(np.mean(ex)), 4), sd=round(sd, 4))
        parts.append(
            f"SECONDARY, {EXTRA_RAW_MODEL.split('/')[-1]}'s raw arm on {len(ex)} census seeds: "
            + ", ".join(f"{v:.4f}" for v in ex)
            + f" (mean {np.mean(ex):.4f}, sd {sd:.4f}). "
            + (f"The spread is real, not a two-seed accident, so its tolerance genuinely swamps its "
               f"domain shifts and F149's refusal to read its +0.146 template rise was correct."
               if sd > 0.05 else
               f"With more seeds the raw value is tighter than the two-seed range suggested, so its "
               f"tolerance should be recomputed and its template shift may become readable."))
    parts.append(
        f"BOUNDARY: a CONVENIENCE cohort -- {len(CACHED)} cached models plus a pre-specified "
        f"extension -- screened on ONE statistic with {N_STARTS} starts. The yield is NOT a "
        f"population estimate and no rate may be quoted from it; the only output that counts is the "
        f"LIST of models that qualify.")
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        band=list(BAND), screen_seed=SCREEN_SEED, confirm_seed=CONFIRM_SEED,
        known_rung=KNOWN, cached=CACHED, extension=EXTENSION,
        extra_raw_seeds=EXTRA_RAW_SEEDS, extra_raw_model=EXTRA_RAW_MODEL, n_starts=N_STARTS,
        purpose="INSTRUMENT-BUILDING, not a hypothesis test -- the output is a LIST of usable "
                "models, and the yield is not a population estimate because the candidate list is "
                "convenience-cached plus a hand-specified extension",
        rung="known models must reproduce argmax_census_instruct exactly at the same seed",
        primary="candidates whose raw fixed_point_fraction lands in the band on BOTH seeds",
        why="F149 could not test direction-predictability because 5 of 6 models sit at a floor or "
            "ceiling; the blocker is the cohort's position, not its size")
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        only_cached = "--cached-only" in _sys.argv
        todo = KNOWN + CACHED + ([] if only_cached else EXTENSION)
        done = 0
        for m in todo:
            need = [SCREEN_SEED]
            if m == EXTRA_RAW_MODEL:
                need += [CONFIRM_SEED] + EXTRA_RAW_SEEDS
            have = [s for s in need if f"{m}|s{s}" in res["runs"]]
            a = res["runs"].get(f"{m}|s{SCREEN_SEED}")
            if a and BAND[0] <= a["fixed_point_fraction"] <= BAND[1]:
                need.append(CONFIRM_SEED)            # earned a confirmation seed
            if all(f"{m}|s{s}" in res["runs"] for s in need):
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
                model = AutoModelForCausalLM.from_pretrained(m).eval().to(
                    dev, torch.float16 if dev != "cpu" else torch.float32)
            except Exception as e:
                res["runs"][f"{m}|failed"] = dict(model=m, error=type(e).__name__,
                                                  detail=str(e)[:120])
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<44} LOAD FAILED {type(e).__name__}", flush=True)
                continue
            try:
                tids, _txt = template_ids(tok)
            except Exception:
                tids = None
            for s in need:
                k = f"{m}|s{s}"
                if k in res["runs"]:
                    continue
                c = census(model, tok, dev, s)
                c.update(model=m, census_seed=s, has_template=bool(tids),
                         n_template_tokens=len(tids) if tids else 0)
                res["runs"][k] = c
                json.dump(res, open(OUT, "w"), indent=1)
                flag = "" if tids else "  NO TEMPLATE"
                band = " <-- IN BAND" if BAND[0] <= c["fixed_point_fraction"] <= BAND[1] else ""
                print(f"  {m:<44} s={s} cls={c['cls']:<11} "
                      f"fix={c['fixed_point_fraction']:.3f}{flag}{band}", flush=True)
            done += 1
            print(f"  {m:<44} done in {time.time()-t0:.0f}s", flush=True)
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
