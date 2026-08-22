"""The decisive cell of the size-matched arm, run FIRST because it can kill E3.

WHY THIS IS A SEPARATE FILE. `size_matched_pile.py` is RUNNING as this is written, grinding through
mamba-370m at more than seven times RWKV's per-seed cost. Editing it to reorder the cohort would
leave the live job writing an end-of-run stamp over code it never executed -- the stale-provenance
trap this project has hit before and now guards against. So the decisive pair gets its own script and
its own results file, and the two runs cannot touch each other.

WHAT IS DECISIVE. PLAN.md's E3 rests on F172: gpt-neo-2.7B class NONE against pythia-410m FUNNEL, on
one corpus. F177's gate found that confounded -- 6.6x apart in size, while the published same-corpus
architecture contrasts are size-matched. `gpt-neo-125m` is the SAME architecture family as F172's
gpt-neo at a size matched to `pythia-160m`. KB in experiments/prereg_size_matched_pile.json, frozen
`d771f1e5...` before any cell existed: if the two land in the SAME class, F172's difference is
attributable to scale, and E3 is withdrawn as an architecture claim.

ESTIMATOR AND CLASS RULE ARE IMPORTED UNCHANGED, from the same modules the stored 17-model census
used, so these cells are directly comparable to it and to the 400M tier already measured.
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
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS

OUT = str(_ROOT / "results" / "size_matched_decisive.json")
PREREG = "experiments/prereg_size_matched_pile.json"
PAIR = [("EleutherAI/pythia-160m", "GPTNeoX"), ("EleutherAI/gpt-neo-125m", "GPTNeo")]


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res.update(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_size_matched_pile.sha256").read().strip(),
               _why="the decisive cell of PLAN.md E3's size-matched arm, split out so it could run "
                    "while size_matched_pile.py was still on mamba-370m",
               _estimator="gate1.argmax_census + argmax_census_hardened.classify, imported unchanged")
    runs = res["runs"]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    for m, fam in PAIR:
        if all(f"{m}|s{cs}" in runs for cs in CENSUS_SEEDS):
            continue
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", torch.float32)
        except Exception as e:
            runs[f"{m}|failed"] = dict(model=m, error=type(e).__name__)
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"  {m}: LOAD FAILED {type(e).__name__}", flush=True)
            continue
        V = int(getattr(model.config, "vocab_size", len(tok)))
        sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                          tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
        for cs in CENSUS_SEEDS:
            k = f"{m}|s{cs}"
            if k in runs:
                continue
            c = argmax_census(model, tok, "cpu", pool, np.random.default_rng(cs), n_starts=N_STARTS)
            c.update(census_seed=cs, cls=classify(c), model=m, tier="150M", family=fam)
            runs[k] = c
            json.dump(res, open(OUT, "w"), indent=1)
        a, b = runs[f"{m}|s{CENSUS_SEEDS[0]}"], runs[f"{m}|s{CENSUS_SEEDS[1]}"]
        print(f"  {m:<28} ({fam:<8}) {a['cls']:>10}/{b['cls']:<10} "
              f"fix={a['fixed_point_fraction']:.3f}/{b['fixed_point_fraction']:.3f} "
              f"modal={a['modal_endpoint_share']:.3f}/{b['modal_endpoint_share']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model
        gc.collect()

    rows = {}
    for m, fam in PAIR:
        cs = [runs[f"{m}|s{s}"] for s in CENSUS_SEEDS if f"{m}|s{s}" in runs]
        if len(cs) == len(CENSUS_SEEDS):
            rows[m] = dict(family=fam, classes=sorted({c["cls"] for c in cs}),
                           stable=len({c["cls"] for c in cs}) == 1,
                           phi=[round(c["fixed_point_fraction"], 4) for c in cs],
                           endpoint=[c["modal_endpoint_token"] for c in cs])
    res["pair"] = rows

    p, g = rows.get("EleutherAI/pythia-160m"), rows.get("EleutherAI/gpt-neo-125m")
    parts = [f"THE DECISIVE CELL of E3's size-matched arm. Both trained on The Pile, 160M vs 125M "
             f"(1.28x, inside the registered 1.5x), {N_STARTS} starts x {len(CENSUS_SEEDS)} seeds, "
             f"CPU float32, estimator and class rule imported unchanged. "]
    if not p or not p["stable"]:
        parts.append("KA FIRES: pythia-160m is missing or class-unstable, so the tier has no anchor "
                     "and the cell is NOT DECIDABLE.")
    elif not g or not g["stable"]:
        parts.append("NOT DECIDABLE: gpt-neo-125m is missing or class-unstable.")
    else:
        same = p["classes"][0] == g["classes"][0]
        res["KB_fires"] = bool(same)
        parts.append(f"pythia-160m (GPTNeoX) {p['classes'][0]} phi={np.mean(p['phi']):.3f}; "
                     f"gpt-neo-125m (GPTNeo) {g['classes'][0]} phi={np.mean(g['phi']):.3f}. ")
        if same:
            parts.append(
                "KB FIRES -- E3 IS WITHDRAWN AS AN ARCHITECTURE CLAIM. At matched size the two "
                "families land in the SAME class, so F172's gpt-neo-2.7B/pythia-410m difference is "
                "attributable to SCALE rather than architecture. Registered before the run and "
                "reported as registered.")
        else:
            parts.append(
                "THE CELL SURVIVES: the classes differ at matched size, so F172's difference is not "
                "a size effect. Per the registered boundary this licenses 'not explained by corpus "
                "or scale' and NOT 'caused by architecture' -- these families differ in schedule "
                "and hyperparameters too, and this design cannot separate those.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
