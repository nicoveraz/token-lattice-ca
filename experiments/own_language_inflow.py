"""§5.3: does Fu et al.'s inflow term behave differently on a model's OWN language?

Registered in experiments/prereg_own_language.json (frozen `b21bb918…` before any non-English inflow
value existed). PLAN.md K2: if endpoints beat their frequency-matched peers in a majority of readable
own-language cells, F171's result is an English artefact and paper 3 is about that instead.

WHY THIS RUN EXISTS AT ALL. F171 measured on English wikitext and had to throw away two of the three
models that could have tested it: `bloom-3b` because its modal endpoint `' ciudad'` occurs zero times
in English, and `polyglot-ko-1.3b` because its tokenizer was not in the local cache. Both exclusions
were statements about the CORPUS and the CACHE, never about the theory, and F171 said so. This turns
them into data.

THE ESTIMATOR IS F171'S, IMPORTED RATHER THAN RESTATED, so the two runs cannot drift apart. What
changes is the corpus and nothing else -- deliberately including the character budget, which is
20,000,000 for every cell so that corpus size cannot explain a difference (K9).

ENDPOINTS ARE KEYED ON TOKEN ID, never on the decoded string. F166 set that rule inside one model and
F173 showed why it matters across models: the glyph `'0'` names different tokens with different
corpus statistics in different tokenizers.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, hashlib, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# THIS RUN NEEDS THE NETWORK, and the import below would otherwise forbid it. inflow_funnel sets
# HF_HUB_OFFLINE / HF_DATASETS_OFFLINE via setdefault -- correct for F171, which reads only cached
# corpora -- and importing its estimator inherits that. Section 5.3 exists precisely to fetch corpora
# this machine does not have, so the flags are set EXPLICITLY here, before the import, where
# setdefault cannot win. Assignment rather than setdefault so the intent is unmistakable.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

import numpy as np

from provenance import stamp, rel
from inflow_funnel import inflow_and_freq, encode, N_CHARS, MATCH_K, MIN_COUNT  # F171's estimator

OUT = str(_ROOT / "results" / "own_language_inflow.json")
PREREG = "experiments/prereg_own_language.json"
CENSUS = _ROOT / "results" / "argmax_census_hardened.json"

ASSIGN = {                       # model -> its own language, frozen in the prereg
    "bigscience/bloom-3b":         "es",
    "llm-jp/llm-jp-3-1.8b":        "ja",
    "EleutherAI/polyglot-ko-1.3b": "ko",
}
LLMJP_ENGLISH_STORED = 36.0      # K8: F171's stored value for the one re-measured cell


def wiki(lang):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    parts, n = [], 0
    for r in ds:
        t = r["text"].strip()
        if not t:
            continue
        parts.append(t); n += len(t) + 1
        if n >= N_CHARS:
            break
    return "\n".join(parts)[:N_CHARS]


def english():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    parts, n = [], 0
    for r in ds:
        t = r["text"].strip()
        if not t:
            continue
        parts.append(t); n += len(t) + 1
        if n >= N_CHARS:
            break
    return "\n".join(parts)[:N_CHARS]


def cell(tok, text, endpoint_id):
    ids = encode(tok, text)
    V = int(max(len(tok), int(ids.max()) + 1))
    inflow, freq, _ = inflow_and_freq(ids, V)
    e = endpoint_id
    if e >= V:
        return dict(readable=False, why="endpoint id outside this tokenizer's range")
    cnt = int(freq[e])
    out = dict(endpoint_corpus_count=cnt, vocab=V, n_tokens=int(len(ids)))
    if cnt < MIN_COUNT:
        out.update(readable=False, why=f"endpoint occurs {cnt} < {MIN_COUNT} times")
        return out
    seen = freq > 0
    cand = np.flatnonzero(seen); cand = cand[cand != e]
    d = np.abs(np.log(freq[cand].astype(np.float64)) - np.log(float(cnt)))
    near = cand[np.argsort(d, kind="stable")[:MATCH_K]]
    pct = float((inflow[near] < inflow[e]).sum()) / max(len(near), 1) * 100.0
    out.update(readable=True, freq_matched_pctl=round(pct, 2), matched_n=int(len(near)),
               inflow_pctl_all_vocab=round(float((inflow < inflow[e]).sum()) / V * 100.0, 3),
               endpoint_inflow_rank=int(np.flatnonzero(np.argsort(-inflow) == e)[0]) + 1)
    return out


def main():
    res = {"_preregistration_file": PREREG,
           "_prereg_sha256": open(_ROOT / "experiments" / "prereg_own_language.sha256").read().strip(),
           "_estimator": "imported from experiments/inflow_funnel.py (F171), not restated"}
    runs = json.load(open(CENSUS))["runs"]

    endpoints = {}
    for m in ASSIGN:
        cells = [runs[k] for k in runs if k.rsplit("|", 1)[0] == m]
        tops = {int(sorted(c["endpoint_histogram"], key=lambda r: -int(r[2]))[0][0]) for c in cells}
        dec = sorted(c["modal_endpoint_token"] for c in cells)[0]
        endpoints[m] = dict(token_id=sorted(tops)[0], decoded=dec, stable=len(tops) == 1)
    res["endpoints"] = endpoints

    corpora = {}
    for lang in sorted(set(ASSIGN.values())):
        t0 = time.time(); txt = wiki(lang)
        corpora[lang] = txt
        print(f"  corpus {lang}: {len(txt):,} chars ({time.time()-t0:.0f}s)", flush=True)
    corpora["en"] = english()
    print(f"  corpus en: {len(corpora['en']):,} chars", flush=True)
    res["corpora"] = {k: dict(n_chars=len(v), sha256=hashlib.sha256(v.encode()).hexdigest()[:16],
                              size_matched=len(v) == N_CHARS) for k, v in corpora.items()}

    from transformers import AutoTokenizer
    grid = {}
    for m, own in ASSIGN.items():
        try:
            tok = AutoTokenizer.from_pretrained(m)
        except Exception as e:
            grid[m] = dict(error=type(e).__name__)
            print(f"  {m}: TOKENIZER FAILED {type(e).__name__}", flush=True)
            continue
        eid = endpoints[m]["token_id"]
        row = dict(own_language=own, endpoint=endpoints[m]["decoded"], endpoint_id=eid)
        for lang in (own, "en"):
            t0 = time.time()
            row[lang] = cell(tok, corpora[lang], eid)
            p = row[lang].get("freq_matched_pctl")
            print(f"  {m.split('/')[-1]:<22} {lang}: "
                  + (f"matched {p:6.2f}  count {row[lang]['endpoint_corpus_count']:>8}"
                     if row[lang].get("readable") else f"NOT READABLE ({row[lang].get('why')})")
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
        grid[m] = row
        del tok; gc.collect()
    res["grid"] = grid

    # ---------------- verdict ----------------
    parts = [f"OWN-LANGUAGE vs ENGLISH, {N_CHARS//10**6}M characters per corpus, estimator imported "
             f"from F171. "]

    # K8 first: if the reproduction cell drifted, nothing else here means anything.
    lj = grid.get("llm-jp/llm-jp-3-1.8b", {}).get("en", {})
    if lj.get("readable"):
        drift = abs(lj["freq_matched_pctl"] - LLMJP_ENGLISH_STORED)
        res["K8_reproduction"] = dict(stored=LLMJP_ENGLISH_STORED, remeasured=lj["freq_matched_pctl"],
                                      drift=round(drift, 2), ok=bool(drift < 1e-9))
        if drift > 1e-9:
            parts.append(f"K8 FIRES: llm-jp's English cell re-measures at {lj['freq_matched_pctl']} "
                         f"against a stored {LLMJP_ENGLISH_STORED}. The pipeline has drifted and "
                         f"every number in this run is void as evidence about language.")
            res["verdict"] = " ".join(parts)
            res["_analysis_provenance"] = stamp(__file__)
            json.dump(res, open(OUT, "w"), indent=1); print("\n" + res["verdict"]); return
        parts.append(f"K8 passes: llm-jp's English cell reproduces F171's {LLMJP_ENGLISH_STORED} "
                     f"exactly, so the pipeline is the same one. ")

    own_ok = {m: r for m, r in grid.items()
              if "error" not in r and r.get(r["own_language"], {}).get("readable")}
    res["own_readable"] = sorted(own_ok)
    parts.append(f"COVERAGE: {len(own_ok)} of {len(ASSIGN)} own-language cells clear the "
                 f"{MIN_COUNT}-occurrence floor. ")

    if len(own_ok) < 2:
        parts.append(f"K2b FIRES: NOT DECIDABLE. Fewer than two readable own-language cells cannot "
                     f"establish an artefact or refute one.")
    else:
        above = {m: r[r["own_language"]]["freq_matched_pctl"] for m, r in own_ok.items()
                 if r[r["own_language"]]["freq_matched_pctl"] > 50.0}
        parts.append(
            "OWN-LANGUAGE percentiles: "
            + "; ".join(f"{m.split('/')[-1]} ({r['own_language']}) "
                        f"{r[r['own_language']]['freq_matched_pctl']:.1f}"
                        for m, r in sorted(own_ok.items())) + ". ")
        if len(above) > len(own_ok) / 2:
            parts.append(
                f"K2 FIRES: endpoints beat their frequency-matched peers in {len(above)} of "
                f"{len(own_ok)} readable own-language cells. E2 is REVERSED on own-language "
                f"corpora, F171's result is an English artefact, and paper 3 must be re-scoped to "
                f"that. Per the registered refusals this run does NOT perform that re-scoping.")
        else:
            parts.append(
                f"K2 DOES NOT FIRE: endpoints beat matched peers in {len(above)} of {len(own_ok)} "
                f"readable own-language cells, not a majority. The direction F171 measured on "
                f"English is not reversed by measuring each model on its own language.")

    # the paired comparison, reported whatever K2 did
    paired = []
    for m, r in grid.items():
        if "error" in r:
            continue
        o, e = r.get(r["own_language"], {}), r.get("en", {})
        if o.get("readable") and e.get("readable"):
            paired.append(dict(model=m.split("/")[-1], own=r["own_language"],
                               own_pctl=o["freq_matched_pctl"], en_pctl=e["freq_matched_pctl"],
                               delta=round(o["freq_matched_pctl"] - e["freq_matched_pctl"], 2)))
    res["paired"] = paired
    if paired:
        parts.append("PAIRED (own minus English, same model, same endpoint id, same corpus size): "
                     + "; ".join(f"{p['model']} {p['delta']:+.1f}" for p in paired) + ". ")
    parts.append("REFUSALS, registered before the numbers: no p-value; no claim that one language "
                 "generalises to 'non-English'; no comparison of raw inflow VALUES across corpora, "
                 "since they are computed over different vocabularies and texts.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
