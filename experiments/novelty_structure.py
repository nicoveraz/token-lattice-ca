"""Does the CA produce structured novelty, or only recall and noise? (#93)

THE QUESTION, AND WHY ENTROPY ALONE CANNOT ANSWER IT. "Can a model create complexity, or only
recombine its corpus?" is not answerable by an entropy measurement: entropy is MAXIMISED BY NOISE.
Raise the temperature and you get maximal entropy and zero interest. The quantity that peaks at
"interesting" is excess entropy -- the mutual information between past and future -- which is low
for repetitive text, low for random text, and high for structured text. That is Crutchfield's
statistical complexity and it is the edge-of-chaos framing this project already cites.

It also cannot be estimated here. Block entropies over a 50,000-token vocabulary from a few
thousand sampled tokens are hopelessly undersampled; the estimate would measure the sample size,
not the system.

WHAT IS MEASURABLE INSTEAD: A TWO-AXIS PLANE.

    NOVELTY    fraction of word n-grams in the generated ring that appear in NEITHER reference
               corpus. Word-level, so it is comparable across tokenizers.
    STRUCTURE  per-token NLL of the decoded ring, scored by a DIFFERENT model than the one that
               generated it.

The question becomes falsifiable: as temperature sweeps, does the CA ever enter the region where
REAL TEXT sits -- unseen and predictable -- or does it pass straight from "seen and predictable"
to "unseen and unpredictable" with nothing in between?

    structure (NLL, low = predictable)
      low  |  repetitive recall  .        . REAL TEXT   <- the region that matters
           |  (low T)
           +--------------------------------- novelty
      high |                          . noise (high T)

FOUR DESIGN CONSTRAINTS, EACH THERE FOR A REASON

  1. **r >= 3, never r=2.** F69 showed the family-distinguishing degeneracy occupies r in {1,2}
     only. Measuring "creativity" in the regime F62-F66 proved is an out-of-distribution artifact
     would repeat the entire mistake on a far more embarrassing topic.
  2. **Both constructions.** The AR rule and the masked-LM rule, because F66/F67 showed they behave
     differently and a claim about "the model" must not rest on one probe.
  3. **The scorer is a DIFFERENT model.** Scoring generated text with its own generator is close to
     circular -- a model finds its own output predictable almost by construction. `gpt2-large` is
     from a third family (WebText) and never generates here.
  4. **Round-tripped references.** Real text is encoded and decoded through the GENERATOR's
     tokenizer before scoring, so the formatting artifacts of that round trip appear in the
     reference too and cancel in the comparison. Without this, BERT's lowercasing alone would look
     like a structure deficit.

PRE-REGISTERED, INCLUDING THE OVERCLAIM BOUNDARY:
  * Primary: is there a temperature at which the CA's (novelty, NLL) sits within the band spanned
    by real text on the NLL axis, while novelty stays above the recall baseline? That is
    "structured novelty".
  * Null: the trajectory goes from low-novelty/low-NLL straight to high-novelty/high-NLL, never
    approaching the real-text NLL band at elevated novelty. Then this construction recombines or
    randomises, and does not produce structured novelty. **A null is a perfectly good result.**
  * **WHAT A POSITIVE RESULT WOULD NOT MEAN.** It would mean: this construction produces token
    sequences that are unseen and predictable to an independent model. It would NOT mean the model
    "has new ideas". Novel n-grams are not ideas; semantic novelty is not measured here and cannot
    be measured this way. Any write-up saying otherwise is overclaiming, and this paragraph exists
    so that the boundary is on record before the numbers are.

Writes results/novelty_structure.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/novelty_structure.py
        (safe to interrupt and re-run -- it resumes)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, re, time, collections
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel

GENERATORS = [("ar",  "EleutherAI/pythia-410m", "step143000"),
              ("mlm", "bert-base-uncased",      None)]
SCORER = "gpt2-large"                  # third family, never generates -- avoids circularity
RADII = [3, 8]                         # r=2 excluded on purpose: F69 says it is degenerate
TEMPS = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
NGRAMS = [2, 3, 4]
N, B, SETTLE = 96, 16, 16
CORPUS_DOCS = 3000                     # reference for novelty
OUT = str(_ROOT / "results" / "novelty_structure.json")

_WORD = re.compile(r"[a-z0-9']+")


def words(text):
    return _WORD.findall(text.lower())


def ngram_sets(texts, ns=NGRAMS):
    out = {n: set() for n in ns}
    for t in texts:
        w = words(t)
        for n in ns:
            out[n].update(zip(*(w[i:] for i in range(n))))
    return out


def novelty(text, ref):
    """Fraction of word n-grams absent from the reference sets."""
    w = words(text)
    res = {}
    for n in NGRAMS:
        g = list(zip(*(w[i:] for i in range(n))))
        res[f"novel_{n}gram"] = (round(sum(x not in ref[n] for x in g) / len(g), 4)
                                 if g else None)
    res["n_words"] = len(w)
    return res


@torch.no_grad()
def nll(scorer_tok, scorer, text, dev, max_len=512):
    """Per-token NLL of `text` under a model that did NOT generate it."""
    ids = scorer_tok(text, return_tensors="pt", truncation=True,
                     max_length=max_len).input_ids.to(dev)
    if ids.shape[1] < 8:
        return None
    out = scorer(input_ids=ids, labels=ids)
    return float(out.loss.item())


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        generators=[g for _, g, _ in GENERATORS], scorer=SCORER, radii=RADII, temps=TEMPS,
        ngrams=NGRAMS, N=N, B=B, settle=SETTLE, corpus_docs=CORPUS_DOCS,
        primary="is there a T where NLL sits inside the real-text band while novelty stays above "
                "the recall baseline? that is structured novelty",
        null="trajectory goes low-novelty/low-NLL straight to high-novelty/high-NLL -- the "
             "construction recombines or randomises. A perfectly good result",
        overclaim_boundary="a positive result means the construction produces sequences that are "
                           "unseen and predictable to an INDEPENDENT model. It does NOT mean the "
                           "model has new ideas. Novel n-grams are not ideas and semantic novelty "
                           "is not measured here",
        why_r_ge_3="F69: the degeneracy occupies r in {1,2}; measuring creativity there would "
                   "repeat F62-F66's mistake",
        why_external_scorer="scoring output with its own generator is near-circular",
        why_roundtrip="real text is encoded/decoded through the GENERATOR's tokenizer so its "
                      "formatting artifacts appear in the reference and cancel",
        resumable="keyed by (construction, model, r, T)")
    runs = res["runs"]

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"loading reference corpus ({CORPUS_DOCS} docs) ...", flush=True)
    docs = [t for t in load_dataset("NeelNanda/pile-10k", split=f"train[:{CORPUS_DOCS}]")["text"]
            if t and t.strip()]
    ref = ngram_sets(docs)
    print(f"  reference n-gram sets: " +
          ", ".join(f"{n}-gram {len(ref[n]):,}" for n in NGRAMS), flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    stok = AutoTokenizer.from_pretrained(SCORER)
    scorer = AutoModelForCausalLM.from_pretrained(SCORER).eval().to(
        dev, torch.float16 if dev != "cpu" else torch.float32)
    print(f"  scorer {SCORER} loaded (never generates)", flush=True)

    for kind, gen, rev in GENERATORS:
        need = [f"{kind}|r{r}|T{T}" for r in RADII for T in TEMPS] + [f"{kind}|ref", f"{kind}|shuf"]
        if all(k in runs for k in need):
            print(f"  {gen}: already complete", flush=True); continue
        if kind == "ar":
            from ar_ca import ARRule, run as carun
            rule = ARRule(gen, revision=rev) if rev else ARRule(gen)
            scheme = "none"
        else:
            from mlm_ca import MLMRule, run as carun
            rule = MLMRule(gen)
            scheme = "cls_sep"
        print(f"\n  {kind} / {gen}", flush=True)

        # references, round-tripped through THIS generator's tokenizer so artifacts cancel
        if f"{kind}|ref" not in runs:
            raw = " ".join(docs[:40])[:20000]
            rt = rule.tok.decode(rule.tok(raw)["input_ids"][:N * 8])
            runs[f"{kind}|ref"] = dict(kind=kind, label="real text (round-tripped)",
                                       nll=nll(stok, scorer, rt, dev), **novelty(rt, ref))
            w = words(rt); rng = np.random.default_rng(0); rng.shuffle(w)
            sh = " ".join(w)
            runs[f"{kind}|shuf"] = dict(kind=kind, label="shuffled real text",
                                        nll=nll(stok, scorer, sh, dev), **novelty(sh, ref))
            for k in (f"{kind}|ref", f"{kind}|shuf"):
                v = runs[k]
                print(f"    {v['label']:>26}  NLL={v['nll']:.3f}  "
                      f"novel4={v['novel_4gram']}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)

        for r in RADII:
            for T in TEMPS:
                key = f"{kind}|r{r}|T{T}"
                if key in runs: continue
                s = carun(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme=scheme,
                          init="random", seed=7, order="per_replica")["final"]
                text = " ".join(rule.tok.decode(row.tolist()) for row in s)
                runs[key] = dict(kind=kind, model=gen, r=r, T=T,
                                 nll=nll(stok, scorer, text, dev),
                                 sample=text[:160], **novelty(text, ref))
                v = runs[key]
                print(f"    r={r} T={T:<4} NLL={v['nll']:.3f}  novel2={v['novel_2gram']}  "
                      f"novel4={v['novel_4gram']}  | {v['sample'][:52]!r}", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    del scorer
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = res["runs"]
    out, parts = {}, []
    for kind, gen, _ in GENERATORS:
        ref, shuf = runs.get(f"{kind}|ref"), runs.get(f"{kind}|shuf")
        cells = [(r, T, runs[f"{kind}|r{r}|T{T}"]) for r in RADII for T in TEMPS
                 if f"{kind}|r{r}|T{T}" in runs]
        if not (ref and shuf and cells): continue
        print(f"\n=== {kind} / {gen} ===")
        print(f"  {'':>12} {'NLL':>7} {'novel-2':>9} {'novel-4':>9}")
        print(f"  {'REAL TEXT':>12} {ref['nll']:>7.3f} {ref['novel_2gram']:>9.3f} "
              f"{ref['novel_4gram']:>9.3f}")
        print(f"  {'SHUFFLED':>12} {shuf['nll']:>7.3f} {shuf['novel_2gram']:>9.3f} "
              f"{shuf['novel_4gram']:>9.3f}")
        for r, T, v in cells:
            print(f"  {'r=%d T=%.1f' % (r, T):>12} {v['nll']:>7.3f} {v['novel_2gram']:>9.3f} "
                  f"{v['novel_4gram']:>9.3f}")

        # "structured novelty" = NLL no worse than real text + 1 nat, AND novelty above real text's
        band = ref["nll"] + 1.0
        hits = [(r, T, v) for r, T, v in cells
                if v["nll"] is not None and v["nll"] <= band
                and (v["novel_2gram"] or 0) > (ref["novel_2gram"] or 0)]
        out[kind] = dict(real_nll=ref["nll"], real_novel2=ref["novel_2gram"],
                         shuffled_nll=shuf["nll"], band=round(band, 3),
                         structured_novelty_cells=[[r, T] for r, T, _ in hits],
                         min_nll=min(v["nll"] for _, _, v in cells if v["nll"] is not None))
        if hits:
            best = min(hits, key=lambda x: x[2]["nll"])
            parts.append(f"{kind.upper()}: STRUCTURED NOVELTY at {len(hits)} setting(s), best "
                         f"r={best[0]} T={best[1]} with NLL={best[2]['nll']:.3f} against real "
                         f"text's {ref['nll']:.3f} and novelty {best[2]['novel_2gram']:.3f} vs "
                         f"{ref['novel_2gram']:.3f}. The construction produces sequences that are "
                         f"unseen AND predictable to a model that did not generate them. This is "
                         f"NOT evidence the model has new ideas -- novel n-grams are not ideas.")
        else:
            mn = out[kind]["min_nll"]
            parts.append(f"{kind.upper()}: NO STRUCTURED NOVELTY. Best NLL over the sweep is "
                         f"{mn:.3f} against real text's {ref['nll']:.3f} (band {band:.3f}), so the "
                         f"trajectory never reaches real-text predictability at elevated novelty. "
                         f"This construction recombines or randomises; it does not occupy the "
                         f"region where real text sits.")
    verdict = " ".join(parts) if parts else "insufficient data"
    print(f"\n  -> {verdict}")

    res["analysis"] = out
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Asks whether the CA produces structured novelty -- sequences unseen in a reference corpus "
        "AND predictable to an independent model -- or only recall and noise. Entropy alone cannot "
        "answer this because entropy is maximised by noise; the quantity that peaks at "
        "'interesting' is excess entropy, which cannot be estimated over a 50k vocabulary from a "
        "few thousand tokens. So the plane is novelty (word n-grams absent from the Pile sample) "
        "against structure (per-token NLL under gpt2-large, a third family that never generates "
        "here). r>=3 only, since F69 showed r<=2 is an out-of-distribution artifact; both "
        "constructions, since F66/F67 showed they differ; references round-tripped through each "
        "generator's tokenizer so formatting artifacts cancel. A positive result would mean the "
        "construction produces unseen, predictable sequences -- NOT that the model has new ideas. "
        "Novel n-grams are not ideas and semantic novelty is not measured here.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
