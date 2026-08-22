"""E3's size-matched Pile arm: does the class difference survive when scale is held fixed?

Registered in experiments/prereg_size_matched_pile.json (frozen `d771f1e5…` before any new cell).

WHY THIS RUN EXISTS. F172 found gpt-neo-2.7B class NONE against pythia-410m FUNNEL on one corpus,
and F177's gate found that reading confounded: the two differ by 6.6x in size, while the published
same-corpus architecture contrasts (arXiv:2404.19178 COLM 2024, arXiv:2410.06672 ICLR 2025) are
SIZE-MATCHED. The reviewer demand for a size-matched arm is fair, and this answers it on a readout
neither of those papers uses.

THE DECISIVE CELL is gpt-neo-125m against pythia-160m -- the same architecture family as F172's
gpt-neo, at a size matched to a pythia. KB in the prereg: if they land in the SAME class, F172's
architecture reading is refuted at matched size and E3 is withdrawn as an architecture claim. That
outcome is registered in advance precisely because it is the expensive one.

THE ESTIMATOR IS IMPORTED, NOT RESTATED. gate1.argmax_census and the classify() thresholds come
from experiments/argmax_census_hardened.py unchanged, so the new cells are directly comparable to
the stored 17 and cannot drift from them. pythia-410m is READ from the stored file rather than
re-measured, for the same reason.
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
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS   # unchanged rule and geometry

OUT = str(_ROOT / "results" / "size_matched_pile.json")
PREREG = "experiments/prereg_size_matched_pile.json"
STORED = _ROOT / "results" / "argmax_census_hardened.json"

TIERS = {
    "400M": ["EleutherAI/pythia-410m", "RWKV/rwkv-4-430m-pile", "state-spaces/mamba-370m-hf"],
    "150M": ["EleutherAI/pythia-160m", "RWKV/rwkv-4-169m-pile", "state-spaces/mamba-130m-hf",
             "EleutherAI/gpt-neo-125m"],
}
FAMILY = {"EleutherAI/pythia-410m": "GPTNeoX", "EleutherAI/pythia-160m": "GPTNeoX",
          "EleutherAI/gpt-neo-125m": "GPTNeo", "RWKV/rwkv-4-430m-pile": "RWKV",
          "RWKV/rwkv-4-169m-pile": "RWKV", "state-spaces/mamba-370m-hf": "Mamba",
          "state-spaces/mamba-130m-hf": "Mamba"}
REUSE = {"EleutherAI/pythia-410m"}          # already censused; re-measuring would invite drift


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration_file"] = PREREG
    res["_prereg_sha256"] = open(_ROOT / "experiments" / "prereg_size_matched_pile.sha256").read().strip()
    res["_estimator"] = "gate1.argmax_census + argmax_census_hardened.classify, both imported unchanged"
    runs = res["runs"]

    stored = json.load(open(STORED))["runs"]
    for m in REUSE:
        for cs in CENSUS_SEEDS:
            k = f"{m}|s{cs}"
            if k in stored and k not in runs:
                runs[k] = dict(stored[k], _reused_from="results/argmax_census_hardened.json")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cpu"
    for tier, models in TIERS.items():
        for m in models:
            if all(f"{m}|s{cs}" in runs for cs in CENSUS_SEEDS):
                continue
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
                model = AutoModelForCausalLM.from_pretrained(m).eval().to(dev, torch.float32)
            except Exception as e:
                runs[f"{m}|failed"] = dict(model=m, error=type(e).__name__, detail=str(e)[:200])
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
                c = argmax_census(model, tok, dev, pool, np.random.default_rng(cs),
                                  n_starts=N_STARTS)
                c.update(census_seed=cs, cls=classify(c), model=m, tier=tier,
                         family=FAMILY.get(m, "?"))
                runs[k] = c
                json.dump(res, open(OUT, "w"), indent=1)
            a, b = runs[f"{m}|s{CENSUS_SEEDS[0]}"], runs[f"{m}|s{CENSUS_SEEDS[1]}"]
            print(f"  [{tier}] {m:<32} {a['cls']:>10}/{b['cls']:<10} "
                  f"fix={a['fixed_point_fraction']:.3f}/{b['fixed_point_fraction']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del model
            gc.collect()

    _verdict(res, runs)


def _verdict(res, runs):
    def cells(m):
        return [runs[f"{m}|s{cs}"] for cs in CENSUS_SEEDS if f"{m}|s{cs}" in runs]

    table, unstable, failed = [], [], []
    for tier, models in TIERS.items():
        for m in models:
            if f"{m}|failed" in runs:
                failed.append((m, runs[f"{m}|failed"]["error"])); continue
            cs = cells(m)
            if len(cs) < len(CENSUS_SEEDS):
                failed.append((m, "incomplete")); continue
            classes = {c["cls"] for c in cs}
            row = dict(tier=tier, model=m, family=FAMILY.get(m, "?"),
                       classes=sorted(classes), stable=len(classes) == 1,
                       phi=[round(c["fixed_point_fraction"], 4) for c in cs],
                       modal=[round(c["modal_endpoint_share"], 4) for c in cs],
                       endpoint=[c["modal_endpoint_token"] for c in cs])
            table.append(row)
            if not row["stable"]:
                unstable.append(m)
    res["table"], res["unstable"], res["failed"] = table, unstable, failed

    parts = [f"SIZE-MATCHED PILE ARM. Corpus fixed (The Pile), scale fixed within tier, architecture "
             f"varied. {N_STARTS} starts x {len(CENSUS_SEEDS)} census seeds, CPU float32, estimator "
             f"and class rule imported unchanged. "]
    if failed:
        parts.append(f"KE: {len(failed)} model(s) unusable and named rather than dropped: {failed}. ")
    if unstable:
        parts.append(f"KC: class-unstable across seeds and excluded from every comparison: "
                     f"{[m.split('/')[-1] for m in unstable]}. ")

    ok = [r for r in table if r["stable"]]
    for tier in TIERS:
        rows = [r for r in ok if r["tier"] == tier]
        if not rows:
            continue
        parts.append(
            f"TIER {tier}: "
            + "; ".join(f"{r['model'].split('/')[-1]} ({r['family']}) {r['classes'][0]} "
                        f"phi={np.mean(r['phi']):.3f}" for r in rows) + ". ")
        klasses = {r["classes"][0] for r in rows}
        if len(klasses) == 1:
            parts.append(f"KD FIRES at {tier}: every architecture lands in the SAME class "
                         f"({klasses.pop()}). No heterogeneity at fixed corpus AND fixed size -- "
                         f"evidence FOR the invariance prior of arXiv:2510.24963 and against E3. ")
        else:
            parts.append(f"Heterogeneity at {tier}: {len(klasses)} distinct classes {sorted(klasses)} "
                         f"at one corpus and one weight class. ")

    # the decisive cell
    a = next((r for r in ok if r["model"] == "EleutherAI/pythia-160m"), None)
    b = next((r for r in ok if r["model"] == "EleutherAI/gpt-neo-125m"), None)
    if not a:
        parts.append("KA FIRES: pythia-160m is missing or unstable, so the 150M tier has no anchor "
                     "and the decisive cell is NOT DECIDABLE.")
    elif not b:
        parts.append("The decisive cell is NOT DECIDABLE: gpt-neo-125m is missing or unstable.")
    else:
        same = a["classes"][0] == b["classes"][0]
        res["decisive"] = dict(pythia_160m=a["classes"][0], gpt_neo_125m=b["classes"][0],
                               same_class=bool(same),
                               phi_pythia=round(float(np.mean(a["phi"])), 4),
                               phi_gptneo=round(float(np.mean(b["phi"])), 4))
        if same:
            parts.append(
                f"KB FIRES -- E3 DIES AS AN ARCHITECTURE CLAIM. At matched size, pythia-160m and "
                f"gpt-neo-125m are BOTH {a['classes'][0]} (phi {np.mean(a['phi']):.3f} vs "
                f"{np.mean(b['phi']):.3f}). F172's gpt-neo-2.7B/pythia-410m difference is therefore "
                f"attributable to scale, not architecture, and E3 must be withdrawn as registered.")
        else:
            parts.append(
                f"THE DECISIVE CELL SURVIVES: at matched size pythia-160m is {a['classes'][0]} "
                f"(phi {np.mean(a['phi']):.3f}) and gpt-neo-125m is {b['classes'][0]} "
                f"(phi {np.mean(b['phi']):.3f}). F172's class difference is NOT a size effect. "
                f"Per the registered boundary this means 'not explained by corpus or scale', never "
                f"'caused by architecture' mechanistically -- these families differ in schedule and "
                f"hyperparameters too.")
    parts.append("REFUSALS, registered before the numbers: no p-value; no generalisation beyond "
                 "these checkpoints; no adjustment of the class thresholds; no claim about WHY.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
