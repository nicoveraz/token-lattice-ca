"""Paper 3, §5.1 pair search and §5.4 stability table. Zero forward passes.

WHY THESE TWO FIRST. PLAN.md §7 sequences the pair search before anything else because K1 can kill
the paper's primary exhibit at no cost: if a SECOND matched-corpus pair exists and its class differs
across the manipulation, E1 is dead. The stability table rides along because it is the same file
read the same way, and because §5.4 notes the cohort's class stability has been assumed and never
printed.

WHAT A "MATCHED-CORPUS PAIR" IS HERE, and the distinction is the whole result. Two models can share
a corpus in two very different senses:

  MANIPULATION PAIR -- same architecture, same size, same corpus, differing in one deliberate
  intervention. `pythia-410m` vs `pythia-410m-deduped` is this, and it is the only one: the
  deduplication is the exact manipulation the data-side account says should matter.

  CONFOUNDED SIBLING -- same corpus, but differing in architecture, size, or both. Informative about
  heterogeneity (E3) and NOT a test of any manipulation, because nothing is held fixed.

Collapsing the two would let a confounded comparison fire K1, which is written about manipulations.
They are reported in separate lists for that reason.

CORPUS LABELS ARE DOCUMENTATION-DERIVED, NOT MEASURED, and every row carries its basis. F90 already
flagged this for the recipe correlate ("the labels are documentation-derived rather than measured").
Models whose training corpus is undisclosed are marked UNKNOWN and are explicitly NOT evidence of
absence: an undisclosed pair would be invisible to this search, and the verdict says so.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import collections, json

import numpy as np

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "cohort_pairs_and_stability.json")
CENSUS = _ROOT / "results" / "argmax_census_hardened.json"

# Training-corpus attribution. DOCUMENTATION-DERIVED: model cards and papers, not measured here.
# `family` groups models the publishers describe as trained on the same corpus. UNKNOWN means the
# corpus is undisclosed or a proprietary mixture -- NOT that it differs from anything.
CORPUS = {
    "EleutherAI/pythia-410m":           ("the-pile",   "Pythia model card / paper: The Pile"),
    "EleutherAI/pythia-410m-deduped":   ("the-pile",   "Pythia: The Pile, DEDUPLICATED -- the manipulation"),
    "EleutherAI/gpt-neo-2.7B":          ("the-pile",   "GPT-Neo model card: The Pile"),
    "EleutherAI/polyglot-ko-1.3b":      ("ko-tunib",   "Polyglot-Ko: Korean corpus, not The Pile"),
    "bigscience/bloom-3b":              ("roots",      "BLOOM: the ROOTS corpus"),
    "bigcode/starcoder2-3b":            ("the-stack",  "StarCoder2: The Stack v2 (code)"),
    "allenai/OLMo-2-0425-1B":           ("dolma",      "OLMo 2: Dolma / OLMo-mix"),
    "HuggingFaceTB/SmolLM-1.7B":        ("smollm",     "SmolLM-Corpus (Cosmopedia, FineWeb-Edu, Python-Edu)"),
    "llm-jp/llm-jp-3-1.8b":             ("llmjp",      "llm-jp corpus (Japanese + English)"),
    "sapienzanlp/Minerva-3B-base-v1.0": ("minerva",    "Minerva: Italian + English mixture"),
    "stabilityai/stablelm-3b-4e1t":     ("UNKNOWN",    "documented as a MIXTURE that includes Pile components; "
                                                       "not the same corpus as a Pile-only model, so not a pair"),
    "meta-llama/Llama-3.2-3B":          ("UNKNOWN",    "training data undisclosed"),
    "google/gemma-2-2b":                ("UNKNOWN",    "training data undisclosed"),
    "Qwen/Qwen1.5-1.8B":                ("UNKNOWN",    "training data undisclosed"),
    "tiiuae/Falcon3-1B-Base":           ("UNKNOWN",    "training data undisclosed"),
    "kyutai/helium-1-preview-2b":       ("UNKNOWN",    "training data undisclosed"),
    "LiquidAI/LFM2-2.6B":               ("UNKNOWN",    "training data undisclosed"),
}

# The one deliberate manipulation available off the shelf in this cohort.
MANIPULATION_PAIRS = [("EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped", "deduplication")]


def per_model(runs):
    out = {}
    for key, v in runs.items():
        m = key.rsplit("|", 1)[0]
        h = sorted(v["endpoint_histogram"], key=lambda r: -int(r[2]))[0]
        out.setdefault(m, []).append(dict(
            seed=v["census_seed"], cls=v["cls"],
            phi=float(v["fixed_point_fraction"]),
            endpoint_id=int(h[0]), endpoint=h[1], endpoint_count=int(h[2]),
            n_distinct=int(v["n_distinct_endpoints"])))
    return {m: sorted(rows, key=lambda r: r["seed"]) for m, rows in out.items()}


def main():
    res = {"_plan": "paper3_arxiv/PLAN.md sections 5.1 (pair search) and 5.4 (stability table)",
           "_forward_passes": 0,
           "_corpus_labels": "DOCUMENTATION-DERIVED from model cards and papers, not measured. Each "
                             "row carries its basis. UNKNOWN means undisclosed or a proprietary "
                             "mixture, never 'differs from the others'."}
    runs = json.load(open(CENSUS))["runs"]
    models = per_model(runs)
    res["n_models"] = len(models)

    # ---------------- 5.4 stability ----------------
    stability, unstable_cls, unstable_end = [], [], []
    for m, rows in sorted(models.items()):
        cls_set = {r["cls"] for r in rows}
        end_set = {r["endpoint_id"] for r in rows}
        phis = [r["phi"] for r in rows]
        row = dict(model=m, n_seeds=len(rows),
                   classes=sorted(cls_set), class_stable=len(cls_set) == 1,
                   endpoints=sorted({r["endpoint"] for r in rows}),
                   endpoint_stable=len(end_set) == 1,
                   phi_mean=round(float(np.mean(phis)), 4),
                   phi_range=round(float(max(phis) - min(phis)), 4),
                   corpus=CORPUS.get(m, ("UNLABELLED", "not in the attribution table"))[0])
        stability.append(row)
        if not row["class_stable"]:
            unstable_cls.append(m)
        if not row["endpoint_stable"]:
            unstable_end.append(m)
    res["stability"] = stability
    res["class_unstable"] = unstable_cls
    res["endpoint_unstable"] = unstable_end

    # ---------------- 5.1 pair search ----------------
    fam = collections.defaultdict(list)
    for m in models:
        fam[CORPUS.get(m, ("UNLABELLED", ""))[0]].append(m)
    known = {k: sorted(v) for k, v in fam.items() if k not in ("UNKNOWN", "UNLABELLED") and len(v) > 1}
    res["same_corpus_groups"] = known
    res["unknown_corpus_models"] = sorted(fam.get("UNKNOWN", []) + fam.get("UNLABELLED", []))

    def cls_of(m):
        return sorted({r["cls"] for r in models[m]})

    manip = []
    for a, b, what in MANIPULATION_PAIRS:
        if a in models and b in models:
            manip.append(dict(a=a, b=b, manipulation=what, cls_a=cls_of(a), cls_b=cls_of(b),
                              same_class=cls_of(a) == cls_of(b),
                              phi_a=round(float(np.mean([r["phi"] for r in models[a]])), 4),
                              phi_b=round(float(np.mean([r["phi"] for r in models[b]])), 4),
                              endpoint_a=sorted({r["endpoint"] for r in models[a]}),
                              endpoint_b=sorted({r["endpoint"] for r in models[b]})))
    res["manipulation_pairs"] = manip

    # confounded siblings: same corpus, but NOT a controlled manipulation
    sib = []
    for corpus, ms in known.items():
        pairs = [(x, y) for i, x in enumerate(ms) for y in ms[i + 1:]]
        for x, y in pairs:
            if any({x, y} == {a, b} for a, b, _ in MANIPULATION_PAIRS):
                continue
            sib.append(dict(corpus=corpus, a=x, b=y, cls_a=cls_of(x), cls_b=cls_of(y),
                            same_class=cls_of(x) == cls_of(y),
                            confounds="architecture and/or size differ; nothing is held fixed, so "
                                      "this is not a test of any manipulation"))
    res["confounded_same_corpus_pairs"] = sib

    # ---------------- verdict ----------------
    parts = [f"ZERO FORWARD PASSES; {len(models)} models read from "
             f"results/argmax_census_hardened.json. "]

    parts.append(
        f"5.4 STABILITY: class is stable across seeds on {len(models) - len(unstable_cls)} of "
        f"{len(models)} models"
        + (f" (unstable: {[m.split('/')[-1] for m in unstable_cls]})" if unstable_cls else "")
        + f"; the MODAL ENDPOINT is stable on {len(models) - len(unstable_end)} of {len(models)} "
        f"(unstable: {[m.split('/')[-1] for m in unstable_end]}). The class assumption is now "
        f"printed rather than assumed. ")

    if not manip:
        parts.append("5.1: no manipulation pair found in the cohort. ")
    else:
        p = manip[0]
        parts.append(
            f"5.1 MANIPULATION PAIRS: exactly {len(manip)} in the cohort -- "
            f"{p['a'].split('/')[-1]} vs {p['b'].split('/')[-1]} ({p['manipulation']}). Classes "
            + ("AGREE" if p["same_class"] else "DIFFER")
            + f" ({p['cls_a']} vs {p['cls_b']}), phi {p['phi_a']} vs {p['phi_b']}, endpoints "
              f"{p['endpoint_a']} vs {p['endpoint_b']}. ")
        parts.append(
            "K1 DOES NOT FIRE: it is conditioned on a SECOND manipulation pair, and there is no "
            "second one. E1 survives as a single pair, which is exactly the small-n risk PLAN.md "
            "flagged -- one pair is an anecdote and the search for a second came back empty. "
            if len(manip) == 1 else "")

    diff_sib = [s for s in sib if not s["same_class"]]
    parts.append(
        f"CONFOUNDED SAME-CORPUS SIBLINGS: {len(sib)} pair(s), of which {len(diff_sib)} differ in "
        f"class"
        + (f" -- {[(s['a'].split('/')[-1], s['cls_a'], s['b'].split('/')[-1], s['cls_b']) for s in diff_sib]}"
           if diff_sib else "")
        + ". These are NOT manipulation pairs and cannot fire K1: architecture and size vary, so "
          "nothing is held fixed. They bear on E3 (heterogeneity under a shared corpus), which is "
          "the exhibit a uniformity thesis has to explain. ")

    parts.append(
        f"COVERAGE, stated as a bound rather than a result: {len(res['unknown_corpus_models'])} of "
        f"{len(models)} models have an undisclosed or mixture training corpus, so a second "
        f"manipulation pair could exist in this cohort and be invisible to a documentation-derived "
        f"search. Absence of a second pair here is not evidence that none exists.")

    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
