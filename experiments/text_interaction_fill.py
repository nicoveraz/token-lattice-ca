"""The 36-cell fill: three models on F154's twelve texts. One run, two verdicts.

WHAT THIS SETTLES, AND WHY THE ORDER OF EVENTS IS THE POINT.

F164 is NOT DECIDABLE at 59% coverage against a floor of 60% declared before the fit. The coverage
analysis identified the gap precisely: `pythia-410m`, `starcoder2-3b` and `llm-jp-3-1.8b` are absent
from every text-arm column. Filling them takes the bilinear matrix to ~75% -- clearing the gate with
margin rather than by the three cells that would scrape past it.

That cohort was chosen BLIND TO F162. The newline factor was born after the fill list existed, so it
now faces a pre-registered widening it had no hand in selecting -- which is this project's own stated
criterion for a factor that survives. The margins were measured and hashed first
(results/newline_margin_frozen.json, 72d5f1ce...) precisely so that ordering is checkable by someone
who was not here.

This script only produces census cells. It reads no prediction and computes no verdict: the joins
live in bilinear_rank1.py (F164) and newline_margin_verdict.py (the factor), both run afterwards.
Keeping the producer ignorant of the predictions is deliberate.

GEOMETRY IS F154's, UNCHANGED: the identical twelve offset-selected prefixes via
`text_interaction.texts`, 96 starts, the same two census seeds, the same `_Prefixed` wrapper. Nothing
about the estimator is re-chosen, so the new cells are commensurable with the 144 already stored.
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
from text_interaction import texts as ti_texts

OUT = str(_ROOT / "results" / "text_interaction_fill.json")
MODELS = ["EleutherAI/pythia-410m", "bigcode/starcoder2-3b", "llm-jp/llm-jp-3-1.8b"]


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        prereg_file="experiments/prereg_newline_margin.json",
        frozen_predictions="results/newline_margin_frozen.json",
        frozen_sha256_tier1="72d5f1ce12073ae2f99f20b3f4f25c25418918d130c6c3dae7b5fd2be7384ba5",
        models=MODELS, n_starts=N_STARTS, census_seeds=CENSUS_SEEDS,
        geometry="identical to F154: the same twelve offset-selected prefixes from "
                 "text_interaction.texts, the same _Prefixed wrapper, the same 96 starts and two "
                 "census seeds -- nothing about the estimator is re-chosen",
        purpose="fills the 36 never-measured cells the F164 coverage analysis identified, chosen "
                "BLIND to F162; settles F164's coverage gate and exposes the newline factor to a "
                "widening it did not pick",
        separation="this script produces cells only. It reads no prediction and computes no "
                   "verdict; the joins are run afterwards by bilinear_rank1.py and "
                   "newline_margin_verdict.py")
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
        tx = ti_texts(tok)
        want = [f"{m}|s{cs}|{k}" for k in tx for cs in CENSUS_SEEDS]
        if all(k in res["runs"] for k in want):
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
        print(f"  {m:<28} {len(tx)} texts", flush=True)
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
                print(f"  {m:<28} {k:<4} s={cs} cls={c['cls']:<11} "
                      f"phi={c['fixed_point_fraction']:.3f}", flush=True)
        done += 1
        print(f"  {m:<28} model done in {time.time()-t0:.0f}s", flush=True)
        del model
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    n = sum(1 for k in res["runs"] if len(k.split("|")) == 3)
    print(f"\n  {n}/72 census cells stored")
    print("wrote", rel(OUT))


if __name__ == "__main__":
    main()
