"""Does INSTRUCTION TUNING change a model's fixed-point class? The taxonomy's missing axis.

WHAT EXISTS AND WHAT DOES NOT. `argmax_census_hardened` classifies 17 models by the fixed-point
geometry of the argmax map -- funnel / none / fragmented / borderline -- and the class is stable
(17/17 across two census seeds at 96 starts). Its recipe association is the striking part: all 8
funnels are from-scratch, and no `modified` model (distilled, pruned, annealed) is a funnel. But
**every one of those 17 is a BASE model.** The recipe axis is from-scratch vs modified;
instruction tuning has never been on it.

WHY THAT AXIS IS WORTH MEASURING NOW. F140 found the attractor share does not correlate with
compliance on instruction-tuned models, and the paired comparison showed why in the sharpest
possible form: on the exact pair Llama-3.2-3B -> Llama-3.2-3B-Instruct, IFEval moves +60.5 points
while the share moves -0.03. Instruction tuning is enormous behaviourally and invisible to the
scalar. The class is a different object -- a shape, not a level -- and it is the natural place to
look for a structural effect the scalar cannot see.

THE DESIGN IS PAIRED, which is what makes it readable at this n. Each instruct model is compared
against a BASE model from the same family whose class is already in the census, so the comparison
holds the family fixed and varies the tuning. Two pairs are exact or near-exact and carry the
argument; the rest are confounded by generation or size and are reported as context, not evidence:

  EXACT       Llama-3.2-3B            -> Llama-3.2-3B-Instruct       (same pretrain, same size)
  NEAR-EXACT  stablelm-3b-4e1t        -> stablelm-zephyr-3b          (zephyr is tuned FROM 4e1t)
  LOOSE       SmolLM-1.7B             -> SmolLM2-1.7B-Instruct       (different generation)
              gemma-2-2b              -> gemma-1.1-2b-it             (different generation)
              Qwen1.5-1.8B            -> Qwen2.5-3B-Instruct         (generation AND size)
              Falcon3-1B-Base         -> Falcon3-3B-Instruct         (size)

EVERYTHING IS IMPORTED, NOT REIMPLEMENTED. `argmax_census` comes from `gate1` and `classify` from
`argmax_census_hardened`, so the thresholds, the start count and the class boundaries are literally
the same code that produced the base-model taxonomy. A reimplementation would make any class change
uninterpretable -- it could be the tuning or it could be the estimator.

PRE-REGISTERED:
  RUNG      the two census seeds must agree on each instruct model's class, as they did 17/17 for
            the base models. An unstable class is UNDERPOWERED, not a class, and is reported as
            such rather than as a change.
  PRIMARY   on the EXACT and NEAR-EXACT pairs, does the class change? Registered reading: a change
            on both is a structural effect of instruction tuning that the scalar share is blind to;
            no change on either says the fixed-point geometry is set in pretraining and survives
            tuning. Both outcomes are worth having, and the second is the one that would extend
            "recipe shapes the geometry" rather than contradict it.
  CONTEXT   the four loose pairs, reported as direction only -- generation and size are confounded
            with tuning in every one of them.
  PRIOR-ART GATE, OWED AND NOT YET CLEARED. "Instruction tuning changes fixed-point / attractor
            structure" is a claim about a literature this project has not searched. The
            MEASUREMENT stands on its own, but no novelty claim may be made from it until the
            F90/F91 check runs against instruction-tuning-and-degeneration specifically, which is
            a different literature from the distillation and pruning one already checked.
  BOUNDARY  six pairs, two of them clean. This is a paired probe, not a survey of instruction
            tuning, and the class is a coarse four-way label.
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
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS   # same code, same thresholds

OUT = str(_ROOT / "results" / "argmax_census_instruct.json")
BASE_CENSUS = _ROOT / "results" / "argmax_census_hardened.json"

# (base model already in the census, instruct model, how clean the pair is)
PAIRS = [
    ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", "exact"),
    ("stabilityai/stablelm-3b-4e1t", "stabilityai/stablelm-zephyr-3b", "near-exact"),
    ("HuggingFaceTB/SmolLM-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "loose: generation"),
    ("google/gemma-2-2b", "google/gemma-1.1-2b-it", "loose: generation"),
    ("Qwen/Qwen1.5-1.8B", "Qwen/Qwen2.5-3B-Instruct", "loose: generation+size"),
    ("tiiuae/Falcon3-1B-Base", "tiiuae/Falcon3-3B-Instruct", "loose: size"),
]


def base_classes():
    d = json.load(open(BASE_CENSUS))
    return {m: v["cls"] for m, v in d["analysis"]["rows"].items()}


def analyse(res):
    base = base_classes()
    runs = res["runs"]
    rows, parts = {}, []
    for b, i, kind in PAIRS:
        ks = [f"{i}|s{cs}" for cs in CENSUS_SEEDS]
        if not all(k in runs for k in ks):
            continue
        cls = [runs[k]["cls"] for k in ks]
        stable = cls[0] == cls[1]
        rows[i] = dict(base=b, base_cls=base.get(b), instruct_cls=cls[0] if stable else None,
                       both=cls, stable=stable, kind=kind,
                       fix=[round(runs[k]["fixed_point_fraction"], 4) for k in ks],
                       modal=[round(runs[k]["modal_endpoint_share"], 4) for k in ks],
                       endpoints=[runs[k]["n_distinct_endpoints"] for k in ks],
                       changed=(stable and base.get(b) is not None and cls[0] != base[b]))
    res["analysis"] = dict(rows=rows, n_pairs=len(rows))
    unstable = [m for m, r in rows.items() if not r["stable"]]
    parts.append(
        f"RUNG (class stability across two census seeds, as 17/17 base models achieved): "
        f"{len(rows) - len(unstable)} of {len(rows)} instruct models keep their class. "
        + ("Stable, so a class difference below is a difference and not resolution."
           if not unstable else
           f"UNSTABLE: {unstable} — for those the class is underpowered at {N_STARTS} starts and "
           f"is NOT read as a change."))
    clean = {m: r for m, r in rows.items() if r["kind"] in ("exact", "near-exact") and r["stable"]}
    if clean:
        parts.append(
            "PRIMARY (the clean pairs): "
            + "; ".join(f"{r['base'].split('/')[-1]} [{r['base_cls']}] -> "
                        f"{m.split('/')[-1]} [{r['instruct_cls']}]"
                        + ("  CHANGED" if r["changed"] else "  unchanged")
                        for m, r in clean.items())
            + ". "
            + ("Instruction tuning changes the fixed-point class on the clean pairs — a structural "
               "effect the scalar share is blind to (the same tuning moves IFEval +60 points and "
               "the share -0.03)."
               if all(r["changed"] for r in clean.values()) else
               "The fixed-point class SURVIVES instruction tuning on the clean pairs: the geometry "
               "is set in pretraining and behavioural tuning does not move it. That extends "
               "'recipe shapes the geometry' rather than contradicting it — the recipes that move "
               "the class are the ones that change pretraining, not the ones that change behaviour."
               if not any(r["changed"] for r in clean.values()) else
               "MIXED on the clean pairs, which at n=2 is the least readable outcome and is "
               "reported as such."))
    loose = {m: r for m, r in rows.items() if r["kind"].startswith("loose") and r["stable"]}
    if loose:
        parts.append(
            "CONTEXT (loose pairs — generation or size is confounded with tuning in every one, so "
            "these are direction only): "
            + "; ".join(f"{r['base'].split('/')[-1]} [{r['base_cls']}] -> "
                        f"{m.split('/')[-1]} [{r['instruct_cls']}]" for m, r in loose.items()) + ".")
    parts.append(
        f"PRIOR-ART GATE OWED: the measurement above stands, but 'instruction tuning changes "
        f"fixed-point structure' is a novelty claim against a literature this project has not "
        f"searched — instruction-tuning-and-degeneration, which is not the distillation/pruning "
        f"literature already checked. No novelty claim until F90/F91's check runs.")
    parts.append(
        f"BOUNDARY: {len(rows)} pairs, {len(clean)} of them clean; the rest confound generation or "
        f"size with tuning. {N_STARTS} starts, two census seeds, one greedy map per model. The "
        f"class is a coarse four-way label and a within-class shift would not show here.")
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        pairs=[dict(base=b, instruct=i, kind=k) for b, i, k in PAIRS],
        n_starts=N_STARTS, census_seeds=CENSUS_SEEDS,
        imported="argmax_census from gate1, classify/N_STARTS/CENSUS_SEEDS from "
                 "argmax_census_hardened -- identical code, so a class change cannot be the "
                 "estimator",
        rung="both census seeds must agree per model, as 17/17 base models did",
        primary="do the EXACT and NEAR-EXACT pairs change class under instruction tuning",
        prior_art="OWED: no novelty claim until the instruction-tuning-and-degeneration literature "
                  "is searched (F90/F91)",
        base_census=rel(str(BASE_CENSUS)))
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for _b, m, _k in PAIRS:
            if all(f"{m}|s{cs}" in res["runs"] for cs in CENSUS_SEEDS):
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
                res["runs"][f"{m}|failed"] = dict(model=m, error=type(e).__name__)
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: LOAD FAILED ({type(e).__name__}: {str(e)[:60]})", flush=True)
                continue
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            for cs in CENSUS_SEEDS:
                k = f"{m}|s{cs}"
                if k in res["runs"]:
                    continue
                c = argmax_census(model, tok, dev, pool, np.random.default_rng(cs),
                                  n_starts=N_STARTS)
                c["census_seed"] = cs
                c["model"] = m
                c["cls"] = classify(c)
                res["runs"][k] = c
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<44} s={cs}  cls={c['cls']:<11} fix={c['fixed_point_fraction']:.3f} "
                      f"modal={c['modal_endpoint_share']:.3f} endpts={c['n_distinct_endpoints']}",
                      flush=True)
            done += 1
            del model
            gc.collect()
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
