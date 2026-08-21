"""Does Fu et al.'s INFLOW predict where a model's argmax map funnels to? Zero forward passes.

F170 read arXiv:2012.14660 in full. Their Corollary 1.2 names an `inflow` term -- "the probability sum
of all words that take it as the subsequent word" -- and concludes that "high inflow words are more
likely to go back to itself and cause the repetition problem". They never measure a model's own
conditional: every transition matrix in their paper is corpus word counts. This project has the other
half, a census of 17 models' argmax maps with stored endpoint tokens. So their theory makes a
prediction about our data, and this is that prediction, run.

WHY THIS IS CHEAP AND WHY THAT MATTERS. No model is loaded and no forward pass is made -- only
tokenizers. The model side is read from results/argmax_census_hardened.json, already stamped. So the
test costs minutes, and a null result costs nothing but the truth.

THE TRAP THIS IS BUILT TO AVOID (registry R1). "Funnel endpoints are high-inflow tokens" is nearly
guaranteed if endpoints are just COMMON tokens -- '\\n', '0', ',' are high-inflow and high-frequency
in any corpus. Fu et al. draw the distinction themselves: "it is not the high-frequency words, but the
high inflow words that really lead to repetition." So H2 is judged on a FREQUENCY-MATCHED control --
e's inflow percentile among the 50 tokens nearest to it in log-frequency -- and H1 alone is treated as
supporting only the trivial claim. The control was frozen in experiments/prereg_inflow_funnel.json
before any number existed (sha256 in prereg_inflow_funnel.sha256, two hashes, both dated before this
ran).
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, hashlib, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np

from provenance import stamp, rel
from gatecheck import balance_report

OUT = str(_ROOT / "results" / "inflow_funnel.json")
PREREG = "experiments/prereg_inflow_funnel.json"
CENSUS = _ROOT / "results" / "argmax_census_hardened.json"

N_CHARS = 20_000_000
MATCH_K = 50
MIN_COUNT = 100          # K4 coverage floor
PCTL_FLOOR = 90.0        # K1
CHUNK = 200_000


def corpus_text():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    parts, n = [], 0
    for row in ds:
        t = row["text"].strip()
        if not t:
            continue
        parts.append(t)
        n += len(t) + 1
        if n >= N_CHARS:
            break
    return "\n".join(parts)[:N_CHARS]


def encode(tok, text):
    """Chunked so a 20MB string does not go through a tokenizer in one call. Chunk boundaries create
    ~100 artificial token splits over 20M characters, which cannot move any statistic reported here."""
    ids = []
    for i in range(0, len(text), CHUNK):
        ids.extend(tok(text[i:i + CHUNK], add_special_tokens=False)["input_ids"])
    return np.asarray(ids, dtype=np.int64)


def inflow_and_freq(ids, V):
    """inflow(t) = sum_a B[a,t] with B row-normalised, i.e. Fu et al. Corollary 1.2's inflow term."""
    a, b = ids[:-1], ids[1:]
    key = a * np.int64(V) + b
    uk, cnt = np.unique(key, return_counts=True)
    ua, ub = (uk // V).astype(np.int64), (uk % V).astype(np.int64)
    rowsum = np.bincount(ua, weights=cnt.astype(np.float64), minlength=V)
    w = cnt.astype(np.float64) / np.maximum(rowsum[ua], 1.0)
    inflow = np.bincount(ub, weights=w, minlength=V)
    freq = np.bincount(ids, minlength=V).astype(np.int64)
    return inflow, freq, int(len(uk))


def main():
    res = {"_preregistration_file": PREREG,
           "_prereg_sha256": open(_ROOT / "experiments" / "prereg_inflow_funnel.sha256").read().strip(),
           "_census_source": "results/argmax_census_hardened.json",
           "_forward_passes": 0}

    cen = json.load(open(CENSUS))
    runs = cen["runs"]
    models = sorted({k.rsplit("|", 1)[0] for k in runs})

    # endpoint token id per model, from the stored histogram; BOTH census seeds must agree
    endpoints, unstable = {}, []
    for m in models:
        cells = [runs[k] for k in runs if k.rsplit("|", 1)[0] == m]
        tops = []
        for c in cells:
            h = sorted(c["endpoint_histogram"], key=lambda r: -int(r[2]))
            tops.append((int(h[0][0]), h[0][1]))
        if len({t[0] for t in tops}) != 1:
            unstable.append((m, tops))
            continue
        endpoints[m] = dict(token_id=tops[0][0], decoded=tops[0][1],
                            cls=cells[0]["cls"],
                            phi=round(float(np.mean([c["fixed_point_fraction"] for c in cells])), 4))
    res["endpoints"] = endpoints
    res["excluded_unstable_endpoint"] = unstable

    text = corpus_text()
    res["corpus"] = dict(name="wikitext-103-raw-v1 train", n_chars=len(text),
                         sha256=hashlib.sha256(text.encode()).hexdigest())
    print(f"corpus {len(text):,} chars  sha {res['corpus']['sha256'][:12]}", flush=True)

    from transformers import AutoTokenizer
    rows = []
    for m in models:
        if m not in endpoints:
            continue
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(m)
        except Exception as e:
            rows.append(dict(model=m, error=type(e).__name__))
            print(f"  {m}: TOKENIZER FAILED {type(e).__name__}", flush=True)
            continue
        ids = encode(tok, text)
        V = int(max(len(tok), int(ids.max()) + 1))
        inflow, freq, n_bigrams = inflow_and_freq(ids, V)
        e = endpoints[m]["token_id"]

        seen = freq > 0
        cnt_e = int(freq[e]) if e < V else 0
        # K1 primary: percentile over the WHOLE vocabulary, as registered.
        pct_all = float((inflow < inflow[e]).sum()) / max(V, 1) * 100.0 if e < V else float("nan")
        # stricter, NOT the registered primary -- reported alongside (amendment 2)
        pct_seen = (float((inflow[seen] < inflow[e]).sum()) / max(int(seen.sum()), 1) * 100.0
                    if e < V and seen.sum() else float("nan"))

        # FREQUENCY-MATCHED CONTROL -- the load-bearing one
        pct_matched, matched_n = float("nan"), 0
        if e < V and cnt_e > 0:
            cand = np.flatnonzero(seen)
            cand = cand[cand != e]
            d = np.abs(np.log(freq[cand].astype(np.float64)) - np.log(float(cnt_e)))
            near = cand[np.argsort(d, kind="stable")[:MATCH_K]]
            matched_n = int(len(near))
            if matched_n:
                pct_matched = float((inflow[near] < inflow[e]).sum()) / matched_n * 100.0

        top_inflow = int(np.argmax(inflow))
        top_freq = int(np.argmax(freq))
        order = np.argsort(-inflow)
        rows.append(dict(
            model=m, cls=endpoints[m]["cls"], phi=endpoints[m]["phi"],
            endpoint_id=e, endpoint=endpoints[m]["decoded"], endpoint_corpus_count=cnt_e,
            inflow_pctl_all_vocab=round(pct_all, 3),
            inflow_pctl_seen_only_NOT_REGISTERED=round(pct_seen, 3),
            freq_matched_pctl=round(pct_matched, 3), matched_n=matched_n,
            endpoint_inflow_rank=int(np.flatnonzero(order == e)[0]) + 1 if e < V else None,
            trivial_top_inflow_token=repr(tok.decode([top_inflow])),
            trivial_rule_hits=bool(top_inflow == e),
            top_freq_token=repr(tok.decode([top_freq])),
            max_inflow=round(float(inflow.max()), 4),
            top10_inflow_share=round(float(np.sort(inflow)[-10:].sum() / max(inflow.sum(), 1e-9)), 5),
            vocab=V, n_tokens=int(len(ids)), n_distinct_bigrams=n_bigrams,
            secs=round(time.time() - t0, 1)))
        print(f"  {m:<42} {endpoints[m]['decoded']!r:<10} cnt {cnt_e:>8}  "
              f"pctl {pct_all:6.2f}  matched {pct_matched:6.2f}  ({time.time()-t0:.0f}s)", flush=True)
        res["per_model"] = rows
        json.dump(res, open(OUT, "w"), indent=1)
        del tok, ids, inflow, freq
        gc.collect()

    res["per_model"] = rows
    _finish(res, rows)


def _finish(res, rows):
    # THREE distinct reasons a model is absent, kept apart because they mean different things: a
    # tokenizer that would not load is an INFRASTRUCTURE failure, an endpoint the corpus never
    # produces is a WRONG-CORPUS failure, and neither is evidence about the hypothesis.
    errs = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r and r["endpoint_corpus_count"] >= MIN_COUNT]
    excl = [r for r in rows if "error" not in r and r not in ok]
    parts = [f"ZERO FORWARD PASSES. wikitext-103 ({N_CHARS//10**6}M chars), each model's own "
             f"tokenizer, inflow = Fu et al. Corollary 1.2's term. "]
    res["excluded_by_K4_coverage"] = [
        dict(model=r["model"], endpoint=r["endpoint"], count=r["endpoint_corpus_count"],
             why="endpoint token occurs <100 times in an ENGLISH corpus -- the corpus is wrong for "
                 "this model, which says nothing about the theory") for r in excl]
    res["excluded_tokenizer_load_failed"] = [
        dict(model=r["model"], error=r["error"],
             why="the tokenizer would not load from the local cache. An infrastructure failure, not "
                 "a coverage failure and not evidence about the hypothesis.") for r in errs]
    parts.append(
        f"K4 COVERAGE: {len(ok)} of {len(rows)} models readable. Excluded for coverage: "
        f"{[(r['model'].split('/')[-1], r['endpoint'], r['endpoint_corpus_count']) for r in excl]}. "
        + (f"Excluded for TOKENIZER LOAD FAILURE, a separate and non-evidential reason: "
           f"{[(r['model'].split('/')[-1], r['error']) for r in errs]}. " if errs else ""))

    if len(ok) < 3:
        parts.append(f"NOT DECIDABLE FOR INSUFFICIENCY: {len(ok)} readable models.")
        res["verdict"] = " ".join(parts)
    else:
        # K2 first: the trivial rule IS the chance baseline
        triv = sum(r["trivial_rule_hits"] for r in ok)
        parts.append(f"K2 TRIVIAL RULE: 'the endpoint is the single most-inflow token' is right for "
                     f"{triv} of {len(ok)}. That rate, not 50%, is the baseline H1 must beat. ")
        # K1
        h1 = [r for r in ok if r["inflow_pctl_all_vocab"] >= PCTL_FLOOR]
        pcts = [r["inflow_pctl_all_vocab"] for r in ok]
        parts.append(f"H1 (registered, whole-vocabulary percentile): {len(h1)} of {len(ok)} endpoints "
                     f"are at or above the {PCTL_FLOOR:.0f}th inflow percentile "
                     f"(median {np.median(pcts):.2f}). "
                     + ("H1 SURVIVES on the registered criterion. "
                        if len(h1) > len(ok) / 2 else
                        "K1 FIRES: H1 is dead on its own registered criterion. "))
        # K3 -- the load-bearing test
        mp = [r["freq_matched_pctl"] for r in ok if r["matched_n"]]
        above = [x for x in mp if x > 50.0]
        res["h2_frequency_matched"] = dict(n=len(mp), median=round(float(np.median(mp)), 2),
                                           n_above_50=len(above),
                                           values=[round(x, 1) for x in mp])
        parts.append(
            f"H2 (FREQUENCY-MATCHED, the load-bearing test): endpoint inflow beats frequency-matched "
            f"peers in {len(above)} of {len(mp)} models, median percentile "
            f"{np.median(mp):.1f} against 50 by construction. "
            + ("H2 SURVIVES: the endpoint is high-inflow BEYOND being common, which is Fu et al.'s "
               "specific claim and not the trivial one."
               if len(above) > len(mp) / 2 else
               "K3 FIRES: inflow adds nothing to frequency. The correct statement is that these maps "
               "funnel to COMMON tokens, not to HIGH-INFLOW ones. Fu et al.'s specific claim is "
               "UNSUPPORTED on this data; the trivial claim stands untouched."))

        # H3 -- a null, judged with spreads
        fun = [r for r in ok if r["cls"] == "funnel"]
        non = [r for r in ok if r["cls"] != "funnel"]
        rep = balance_report([("funnel" if r["cls"] == "funnel" else "not") for r in ok],
                            name="fixed-point class")
        res["h3_balance"] = dict(n_funnel=len(fun), n_other=len(non),
                                 gate_readable=bool(rep.readable), gate_reason=rep.reason)
        if len(fun) < 2 or len(non) < 2 or not rep.readable:
            parts.append(f"H3 NOT DECIDABLE (K5): {len(fun)} funnel, {len(non)} non-funnel readable. "
                         f"{rep.reason}")
        else:
            mf = [r["max_inflow"] for r in fun]
            mn = [r["max_inflow"] for r in non]
            overlap = not (max(mf) < min(mn) or max(mn) < min(mf))
            res["h3_concentration"] = dict(
                funnel_max_inflow=[round(x, 2) for x in mf],
                other_max_inflow=[round(x, 2) for x in mn],
                funnel_range=[round(min(mf), 2), round(max(mf), 2)],
                other_range=[round(min(mn), 2), round(max(mn), 2)], overlap=bool(overlap))
            parts.append(
                f"H3 (TIER 2, a NULL): max inflow spans [{min(mf):.1f}, {max(mf):.1f}] on funnels and "
                f"[{min(mn):.1f}, {max(mn):.1f}] on non-funnels. "
                + ("The ranges OVERLAP, so corpus inflow concentration does NOT separate the classes. "
                   "Stated as registered: this does not PROVE language-independence, it shows this "
                   "corpus statistic fails to distinguish models that differ sharply in phi -- which "
                   "is what a weights-side account predicts and a 'caused by the language itself' "
                   "account does not."
                   if overlap else
                   "The ranges are DISJOINT, so corpus inflow concentration DOES track the class. H3 "
                   "is refuted and Fu et al.'s language-side thesis gains support from this data."))
    from collections import Counter
    shared = Counter(r["endpoint"] for r in ok)
    dup = {k: v for k, v in shared.items() if v > 1}
    res["predictor_non_independence"] = dict(
        endpoint_token_counts=dict(shared),
        note="models sharing an endpoint token share its corpus inflow and frequency EXACTLY, so the "
             "13 readable models are not 13 independent tests of H1/H2. pythia-410m and -deduped "
             "additionally share a tokenizer, making their rows identical. Reported, not corrected "
             "for -- correcting would require a weighting scheme that was not pre-registered.")
    if dup:
        parts.append(f"NON-INDEPENDENCE, reported not corrected: endpoint tokens repeat across models "
                     f"({dict(dup)}), and models sharing a token share its corpus statistics exactly, "
                     f"so the readable models are fewer than {len(ok)} independent tests. ")
    parts.append("REFUSALS, recorded before the numbers: no p-value; no causal claim (corpus bigram "
                 "inflow is not the model's conditional); no claim that this validates Fu et al.'s "
                 "BOUND, which is a different object.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
