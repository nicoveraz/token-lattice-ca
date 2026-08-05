"""Harden F87's taxonomy: is funnel / none / fragmented a stable property of each model's map?

F87 classified 15 families' argmax maps from the band screen's census rider -- 24 starts, one
census seed per model. The class gaps are wide (fixed-point fraction 0.00 vs 0.83), but a
claimable taxonomy needs more than wide gaps on one draw. This run re-censuses every band
representative plus the duped/deduped pythia-410m pair at 4x the starts and TWO independent
census seeds, and classifies under a rule registered here, before the data.

  FUNNEL       fix >= 0.30 and modal >= 0.30      one dominant basin (pythia's case)
  NONE         fix <= 0.10                        the map wanders or cycles (gpt2's case)
  FRAGMENTED   fix >= 0.30 and modal < 0.20       many small basins (gemma-2's case in F87)
  BORDERLINE   anything else                      reported, never forced into a class

  STABLE       same class from both census seeds at 96 starts. The taxonomy is only claimed
               for models that are stable; an unstable model is the finding "the census is
               underpowered here", not a fourth class.

WHY THE DUPED/DEDUPED PAIR IS IN. F87's suggestive correlate is that fragmentation/absence
tracks distillation and annealed data recipes. The cheapest confound is the tokenizer or the
data-cleaning pass rather than the recipe: pythia-410m vs pythia-410m-deduped differ ONLY in
corpus deduplication (same tokenizer, same architecture, same schedule -- Gate 2's own dedup
pair logic, one scale up). If dedup alone changes the class, the recipe correlate is dead on
arrival; if not, it survives to be tested properly.

RECIPE LABELS are hand-entered from model cards and marked as such; the association is reported
descriptively (n is far too small for a test and none is run). The labels are data about the
DOCUMENTATION, not measurements.

Writes results/argmax_census_hardened.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/argmax_census_hardened.py
        (resumable per (model, census_seed))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from gate1 import argmax_census

OUT = str(_ROOT / "results" / "argmax_census_hardened.json")
BAND = _ROOT / "results" / "band_screen.json"
N_STARTS = 96
CENSUS_SEEDS = [20260803, 990017]
EXTRA = ["EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped"]   # the dedup confound pair

# From model cards, marked as documentation-derived labels, not measurements.
RECIPE = {
    "google/gemma-2-2b": "distilled", "meta-llama/Llama-3.2-3B": "pruned+distilled",
    "allenai/OLMo-2-0425-1B": "mid-trained/annealed", "tiiuae/Falcon3-1B-Base": "annealed-decay",
    "LiquidAI/LFM2-2.6B": "unclear", "sapienzanlp/Minerva-3B-base-v1.0": "from-scratch",
    "EleutherAI/polyglot-ko-1.3b": "from-scratch", "bigscience/bloom-3b": "from-scratch",
    "EleutherAI/gpt-neo-2.7B": "from-scratch", "HuggingFaceTB/SmolLM-1.7B": "from-scratch",
    "kyutai/helium-1-preview-2b": "from-scratch", "llm-jp/llm-jp-3-1.8b": "from-scratch",
    "Qwen/Qwen1.5-1.8B": "from-scratch", "stabilityai/stablelm-3b-4e1t": "from-scratch",
    "bigcode/starcoder2-3b": "from-scratch", "EleutherAI/pythia-410m": "from-scratch",
    "EleutherAI/pythia-410m-deduped": "from-scratch (deduped corpus)",
}


def classify(c):
    fix, modal = c["fixed_point_fraction"], c["modal_endpoint_share"]
    if fix >= 0.30 and modal >= 0.30:
        return "funnel"
    if fix <= 0.10:
        return "none"
    if fix >= 0.30 and modal < 0.20:
        return "fragmented"
    return "borderline"


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    band = json.load(open(BAND))
    models = sorted(set(band["_preregistration"]["representatives"].values())
                    & {v["model"] for v in band["families"].values()}) + EXTRA
    res["_preregistration"] = dict(
        models=models, n_starts=N_STARTS, census_seeds=CENSUS_SEEDS,
        classes="funnel: fix>=0.30 & modal>=0.30; none: fix<=0.10; fragmented: fix>=0.30 & "
                "modal<0.20; borderline otherwise",
        stability="same class from both seeds at 96 starts; unstable = underpowered, not a class",
        dedup_pair="pythia-410m vs -deduped: same tokenizer/arch/schedule, corpus dedup only -- "
                   "if dedup alone changes the class, the recipe correlate dies here",
        recipe_labels="hand-entered from model cards; documentation, not measurement; "
                      "reported descriptively, no test at this n")
    runs = res["runs"]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    for m in models:
        if all(f"{m}|s{cs}" in runs for cs in CENSUS_SEEDS):
            continue
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(m).eval().to(
                dev, torch.float16 if dev != "cpu" else torch.float32)
        except Exception as e:
            runs[f"{m}|failed"] = dict(model=m, error=type(e).__name__)
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"  {m}: LOAD FAILED ({type(e).__name__})", flush=True)
            continue
        V = int(getattr(model.config, "vocab_size", len(tok)))
        sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                          tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
        for cs in CENSUS_SEEDS:
            k = f"{m}|s{cs}"
            if k in runs: continue
            c = argmax_census(model, tok, dev, pool, np.random.default_rng(cs),
                              n_starts=N_STARTS)
            c["census_seed"] = cs
            c["cls"] = classify(c)
            runs[k] = c
            json.dump(res, open(OUT, "w"), indent=1)
        a, b = runs[f"{m}|s{CENSUS_SEEDS[0]}"], runs[f"{m}|s{CENSUS_SEEDS[1]}"]
        print(f"  {m:44s} {a['cls']:>10}/{b['cls']:<10} fix={a['fixed_point_fraction']:.2f}/"
              f"{b['fixed_point_fraction']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        del model
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = res["runs"]
    models = res["_preregistration"]["models"]
    rows, unstable = {}, []
    for m in models:
        a = runs.get(f"{m}|s{CENSUS_SEEDS[0]}")
        b = runs.get(f"{m}|s{CENSUS_SEEDS[1]}")
        if not (a and b):
            continue
        stable = a["cls"] == b["cls"]
        rows[m] = dict(cls=a["cls"] if stable else f"UNSTABLE({a['cls']}/{b['cls']})",
                       stable=stable, fix=[a["fixed_point_fraction"], b["fixed_point_fraction"]],
                       modal=[a["modal_endpoint_share"], b["modal_endpoint_share"]],
                       endpoints=[a["n_distinct_endpoints"], b["n_distinct_endpoints"]],
                       recipe=RECIPE.get(m, "unlabeled"))
        if not stable:
            unstable.append(m)
    print(f"\n  {'model':44s} {'class':>22} {'recipe'}")
    for m, v in rows.items():
        print(f"  {m:44s} {v['cls']:>22} {v['recipe']}")

    dd = {m: rows[m]["cls"] for m in EXTRA if m in rows}
    dedup_same = len(set(dd.values())) == 1 if len(dd) == 2 else None
    stable_rows = {m: v for m, v in rows.items() if v["stable"]}
    from collections import Counter
    by_recipe = Counter()
    for m, v in stable_rows.items():
        r = "modified" if any(x in v["recipe"] for x in ("distil", "anneal", "pruned")) \
            else ("from-scratch" if "scratch" in v["recipe"] else "other")
        by_recipe[(r, v["cls"])] += 1

    parts = [
        f"STABILITY: {len(stable_rows)}/{len(rows)} models keep their class across two census "
        f"seeds at {N_STARTS} starts"
        + (f"; unstable: {', '.join(m.split('/')[-1] for m in unstable)} -- reported as "
           f"underpowered, not forced into a class." if unstable else ". The taxonomy is a "
           f"stable property of each map at this resolution.")]
    if dedup_same is not None:
        parts.append(
            f"DEDUP CONFOUND: pythia-410m and -deduped are "
            + (f"the SAME class ({list(dd.values())[0]}) -- corpus deduplication alone does not "
               f"change the map's class, so the recipe correlate survives its cheapest confound."
               if dedup_same else
               f"DIFFERENT classes ({dd}) -- deduplication alone moves the class, and the recipe "
               f"correlate is confounded at its root. It dies here."))
    parts.append(
        "RECIPE ASSOCIATION (descriptive, documentation-derived labels, no test at this n): "
        + "; ".join(f"{r}/{c}: {n}" for (r, c), n in sorted(by_recipe.items())) + ".")
    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rows=rows, unstable=unstable, dedup_pair=dd,
                           dedup_same_class=dedup_same,
                           recipe_by_class={f"{r}|{c}": n for (r, c), n in by_recipe.items()})
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Hardens F87's funnel/none/fragmented taxonomy: 96 starts (4x the band census), two "
        "independent census seeds, class rule registered before the data, stability required "
        "before a class is claimed. The duped/deduped pythia-410m pair isolates the cheapest "
        "confound of the recipe correlate (corpus dedup alone, same tokenizer/arch/schedule). "
        "Recipe labels are hand-entered from model cards and are documentation, not measurement; "
        "the association is reported descriptively with no test at this n.")


if __name__ == "__main__":
    main()
