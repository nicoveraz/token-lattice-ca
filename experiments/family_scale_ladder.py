"""Is the Pythia/GPT-Neo split a stable FAMILY property, or an artefact of two checkpoints?

Registered in experiments/prereg_family_scale_ladder.json (frozen `49adb654…` before any new cell).

WHAT THIS DOES AND DOES NOT ANSWER. F178 showed the funnel/none split is not a size effect: it
reproduces at matched size. The alternative it could not exclude is that the PYTHIA RECIPE is simply
idiosyncratic. This run does not settle that -- settling it needs a second funnel from a different
family at fixed corpus, and no affordable candidate exists (GPT-J-6B is ~24GB in float32 on a 16GB
machine). What it settles is narrower and worth having: whether the split holds ACROSS SCALE within
each family, or whether F178 happened to pick two unrepresentative checkpoints. KH in the prereg
forbids reading a clean ladder as more than that.

Existing cells are REUSED from the files that produced them, never re-measured, so the ladder cannot
drift from the runs it extends.
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

OUT = str(_ROOT / "results" / "family_scale_ladder.json")
PREREG = "experiments/prereg_family_scale_ladder.json"

NEW = [("EleutherAI/pythia-70m", "Pythia", 70), ("EleutherAI/pythia-1b", "Pythia", 1000),
       ("EleutherAI/gpt-neo-1.3B", "GPT-Neo", 1300)]
REUSE = [("EleutherAI/pythia-160m", "Pythia", 160, "results/size_matched_decisive.json"),
         ("EleutherAI/pythia-410m", "Pythia", 410, "results/argmax_census_hardened.json"),
         ("EleutherAI/gpt-neo-125m", "GPT-Neo", 125, "results/size_matched_decisive.json"),
         ("EleutherAI/gpt-neo-2.7B", "GPT-Neo", 2700, "results/argmax_census_hardened.json")]


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res.update(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_family_scale_ladder.sha256").read().strip(),
               _estimator="gate1.argmax_census + argmax_census_hardened.classify, imported unchanged")
    runs = res["runs"]

    meta = {m: (fam, mb) for m, fam, mb in NEW}
    for m, fam, mb, src in REUSE:
        meta[m] = (fam, mb)
        stored = json.load(open(_ROOT / src))["runs"]
        for cs in CENSUS_SEEDS:
            k = f"{m}|s{cs}"
            if k in stored and k not in runs:
                runs[k] = dict(stored[k], model=m, family=fam, size_m=mb, _reused_from=src)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    for m, fam, mb in NEW:
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
            c.update(census_seed=cs, cls=classify(c), model=m, family=fam, size_m=mb)
            runs[k] = c
            json.dump(res, open(OUT, "w"), indent=1)
        a, b = runs[f"{m}|s{CENSUS_SEEDS[0]}"], runs[f"{m}|s{CENSUS_SEEDS[1]}"]
        print(f"  {m:<26} ({fam:<7} {mb:>4}M) {a['cls']:>8}/{b['cls']:<8} "
              f"fix={a['fixed_point_fraction']:.3f}/{b['fixed_point_fraction']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model
        gc.collect()

    _verdict(res, runs, meta)


def _verdict(res, runs, meta):
    ladder, unstable, failed = [], [], []
    for m, (fam, mb) in sorted(meta.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        if f"{m}|failed" in runs:
            failed.append(m); continue
        cs = [runs[f"{m}|s{s}"] for s in CENSUS_SEEDS if f"{m}|s{s}" in runs]
        if len(cs) < len(CENSUS_SEEDS):
            failed.append(m); continue
        klass = {c["cls"] for c in cs}
        row = dict(model=m, family=fam, size_m=mb, classes=sorted(klass), stable=len(klass) == 1,
                   phi=round(float(np.mean([c["fixed_point_fraction"] for c in cs])), 4),
                   modal=round(float(np.mean([c["modal_endpoint_share"] for c in cs])), 4),
                   endpoint=cs[0]["modal_endpoint_token"])
        ladder.append(row)
        if not row["stable"]:
            unstable.append(m)
    res["ladder"], res["unstable"], res["failed"] = ladder, unstable, failed

    parts = [f"FAMILY SCALE LADDER on The Pile. {N_STARTS} starts x {len(CENSUS_SEEDS)} seeds, CPU "
             f"float32, estimator and class rule imported unchanged; existing cells reused, not "
             f"re-measured. "]
    if failed:
        parts.append(f"Unusable and named: {failed}. ")
    if unstable:
        parts.append(f"KC: class-unstable and excluded: {[m.split('/')[-1] for m in unstable]}. ")

    ok = [r for r in ladder if r["stable"]]
    broke = []
    for fam in ("Pythia", "GPT-Neo"):
        rows = [r for r in ok if r["family"] == fam]
        if not rows:
            continue
        klasses = {r["classes"][0] for r in rows}
        parts.append(
            f"{fam} across {len(rows)} scales ({rows[0]['size_m']}M-{rows[-1]['size_m']}M): "
            + "; ".join(f"{r['size_m']}M {r['classes'][0]} phi={r['phi']:.3f}" for r in rows)
            + f" -- {'UNIFORM' if len(klasses)==1 else 'NOT uniform ' + str(sorted(klasses))}. ")
        if len(klasses) > 1:
            broke.append(fam)

    pyth = {r["classes"][0] for r in ok if r["family"] == "Pythia"}
    neo = {r["classes"][0] for r in ok if r["family"] == "GPT-Neo"}
    res["pythia_uniform_funnel"] = bool(pyth == {"funnel"})
    res["gptneo_uniform_none"] = bool(neo == {"none"})

    if "Pythia" in broke:
        parts.append("KF FIRES: Pythia is not funnel at every scale, so the split is a SCALE-WINDOW "
                     "property, not a family property. E3 must be restated accordingly and this is "
                     "the weaker claim.")
    elif "GPT-Neo" in broke:
        parts.append("KG FIRES: GPT-Neo funnels at some scale, so the contrast is scale-dependent "
                     "rather than family-level.")
    elif pyth == {"funnel"} and neo == {"none"}:
        parts.append(
            f"BOTH LADDERS ARE UNIFORM: Pythia funnels at every scale tested and GPT-Neo at none. "
            f"F178's split is therefore a stable property of the families across roughly an order of "
            f"magnitude in size, not an artefact of two checkpoints.")
    parts.append(
        "KH, registered before the run and binding whatever the ladders did: this does NOT settle "
        "whether the Pythia recipe is idiosyncratic. A uniform ladder shows the split is stable; it "
        "does not show it is architectural, recipe-driven, or general. Settling that needs a second "
        "funnel from a different family at fixed corpus, and no affordable candidate exists -- "
        "GPT-J-6B is ~24GB in float32 on a 16GB machine. "
        "REFUSALS: no p-value; no claim about WHAT differs between the families; no extrapolation "
        "beyond the scales tested.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
