"""ARM 1: where trajectories go when they DON'T stay. Zero forward passes.

Registered in experiments/prereg_escape_rival.json (frozen `cf1e02ff...` before any destination was
decoded). The values were already stored in results/selfcont_set_*.json as `argmax_ids`, which is
exactly why the prereg had to exist first: a free number is the easiest kind to read post-hoc.

WHAT THIS IS. F179 found that GPT-Neo arrives at the newline and does not stay. This asks the
complement nobody had asked: where does it go instead? For a source token whose bit is 0, the stored
argmax IS that escape -- and is also the best rival, the same quantity split by the bit, verified on
546823 tokens before freezing.

THE F166 INVERSION GOVERNS EVERY LINE HERE. Within a model, ids are authoritative. ACROSS models they
are meaningless, so every destination is compared as a DECODED STRING and sources are keyed by the
frozen probe strings. Leading-space and byte-fallback are the hazards of that bridge and are reported
rather than assumed away.

THE NULL IS THE TEST, NOT A CONTROL ON IT (KE). Roughly 50100 of a Pythia's tokens do not
self-continue, and their argmax is whatever that model's generic high-probability continuation is.
Two models agreeing about that is Zipf, not identity, unless it survives frequency matching -- F171,
where the naive reading passed at the 99.87th percentile and died at 32.0.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import collections, itertools, json, os, re

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from provenance import stamp, rel
from gatecheck import balance_report

PREREG = "experiments/prereg_escape_rival.json"
PR = json.load(open(_ROOT / PREREG))
OUT = _ROOT / "results" / "escape_destinations.json"
FREQ = _ROOT / "results" / "corpus_token_counts.json"
PROBES = _ROOT / "experiments" / "probe_strings_selfcont.json"

DECISIVE = ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped")
FAR = [("EleutherAI/pythia-410m", "state-spaces/mamba-370m-hf"),
       ("EleutherAI/pythia-410m", "RWKV/rwkv-4-430m-pile"),
       ("EleutherAI/pythia-410m", "EleutherAI/gpt-neo-125m")]
CONTROL = ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m@bf16")

N_DOCS = PR["FREQUENCY_MATCHED_NULL_mandatory_for_every_token_content_claim"]["n_docs"] \
    if "n_docs" in PR["FREQUENCY_MATCHED_NULL_mandatory_for_every_token_content_claim"] else 2000
BAND = 50
DRAWS = PR["FREQUENCY_MATCHED_NULL_mandatory_for_every_token_content_claim"]["null_draws"]
MIN_SOURCES = 500
MODAL_GATE = 0.50
CARD_GATE = 0.7
SEED = 20260823


def load_cells():
    cells = {}
    for p in sorted((_ROOT / "results").glob("selfcont_set_*.json")):
        if p.name == "selfcont_set_failures.json":
            continue
        d = json.load(open(p))
        cells[d["cell"]] = d
    return cells


def corpus_counts(models):
    """Per-model token counts over the SAME corpus slice that defined the probe strings."""
    cache = json.load(open(FREQ)) if FREQ.exists() else {}
    todo = [m for m in models if m not in cache]
    if todo:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        texts = [ds[i]["text"] for i in range(N_DOCS)]
        for m in todo:
            tok = AutoTokenizer.from_pretrained(m)
            c = collections.Counter()
            for t in texts:
                c.update(tok(t, add_special_tokens=False)["input_ids"])
            cache[m] = {str(k): int(v) for k, v in c.items()}
            print(f"  counts {m:<34} {len(cache[m])} distinct ids", flush=True)
        json.dump(cache, open(FREQ, "w"))
    return cache


def decode_vocab(m, V):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m)
    return [tok.decode([i]) for i in range(V)]


def main():
    probe = json.load(open(PROBES))
    strings = [e["s"] for e in probe["strings"]]
    cells = load_cells()
    models = sorted(c for c in cells if "@" not in c)
    res = dict(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_escape_rival.sha256").read().split()[0],
               _probe_strings_sha256=probe["strings_sha256"],
               models=models, n_probe_candidates=len(strings))

    pid = {m: np.array(cells[m]["probe_token_ids"]) for m in cells}
    ok = np.ones(len(strings), bool)
    for m in models:
        ok &= pid[m] >= 0
    idx = np.flatnonzero(ok)
    res["intersection_size"] = int(len(idx))

    print(f"  intersection {len(idx)} of {len(strings)}", flush=True)
    counts = corpus_counts(sorted(cells))

    # decode each vocabulary once; the string bridge needs it and so does the frequency band
    vocab = {}
    for m in sorted(cells):
        V = cells[m]["vocab_measured"]
        vocab[m] = decode_vocab(cells[m]["model"], V)
        print(f"  decoded {m:<34} {V}", flush=True)

    # a global string table so agreement is an integer comparison, never a string one
    table, sidx = {}, []
    def s2i(s):
        if s not in table:
            table[s] = len(table)
        return table[s]

    esc, escv, srcmask, bands, hazard = {}, {}, {}, {}, {}
    rng = np.random.default_rng(SEED)
    for m in sorted(cells):
        d = cells[m]
        marg = np.array(d["margins_e4"], np.int64)
        amax = np.array(d["argmax_ids"], np.int64)
        sent = d.get("_unmeasured_sentinel", -2147483648)
        tid = pid[m][idx]
        measured = marg[tid] != sent
        bit0 = measured & (marg[tid] <= 0) & (amax[tid] >= 0)
        dest = np.full(len(idx), -1, np.int64)
        strs = [vocab[m][int(a)] if 0 <= a < len(vocab[m]) else None for a in amax[tid]]
        for k in np.flatnonzero(bit0):
            dest[k] = s2i(strs[int(k)])
        esc[m], srcmask[m] = dest, bit0
        escv[m] = amax[tid]
        # frequency band, by RANK over this model's own corpus counts (F171's construction)
        cm = counts[m]
        V = len(vocab[m])
        cnt = np.zeros(V, np.int64)
        for k, v in cm.items():
            k = int(k)
            if k < V:
                cnt[k] = v
        order = np.lexsort((np.arange(V), cnt))          # count asc, id asc -- deterministic
        rank = np.empty(V, np.int64); rank[order] = np.arange(V)
        bands[m] = (order, rank)
        # the string-bridge hazards, reported rather than assumed away
        hazard[m] = dict(
            destinations_with_leading_space=int(sum(1 for k in np.flatnonzero(bit0)
                                                    if strs[int(k)].startswith(" "))),
            destinations_non_ascii=int(sum(1 for k in np.flatnonzero(bit0)
                                           if not strs[int(k)].isascii())),
            destinations_empty_string=int(sum(1 for k in np.flatnonzero(bit0) if strs[int(k)] == "")))

    res["string_bridge_hazards"] = hazard
    res["n_distinct_destination_strings"] = len(table)

    # per-model destination profile, and KB's modal gate
    prof = {}
    for m in models:
        v = esc[m][srcmask[m]]
        c = collections.Counter(v.tolist())
        top, n = c.most_common(1)[0]
        inv = {i: s for s, i in table.items()}
        prof[m] = dict(n_escaping_sources=int(srcmask[m].sum()), n_distinct=len(c),
                       modal_destination=inv[top], modal_share=round(n / max(1, len(v)), 4))
    res["per_model"] = prof
    modal_max = max(p["modal_share"] for p in prof.values())
    res["KB_fires"] = bool(modal_max > MODAL_GATE)
    res["modal_share_max"] = modal_max

    def agree(a, b, exclude_modal):
        both = srcmask[a] & srcmask[b]
        if exclude_modal:
            # every cell has a profile by the time this is called, including the precision control
            keep = (esc[a] != table[prof[a]["modal_destination"]]) & \
                   (esc[b] != table[prof[b]["modal_destination"]])
            both = both & keep
        n = int(both.sum())
        if n == 0:
            return None, 0
        return float(np.mean(esc[a][both] == esc[b][both])), n

    def row(a, b):
        if a not in cells or b not in cells:
            return None
        raw, n = agree(a, b, False)
        exm, nx = agree(a, b, True)
        r = dict(a=a, b=b, n_shared_sources=n, agreement_raw=None if raw is None else round(raw, 4),
                 n_shared_excl_modal=nx,
                 agreement_excl_modal=None if exm is None else round(exm, 4))
        r["primary"] = r["agreement_excl_modal"] if res["KB_fires"] else r["agreement_raw"]
        r["KA_not_decidable"] = bool(n < MIN_SOURCES)
        return r

    # bf16 control shares pythia-410m's profile key; give it one so agree() can exclude its modal
    for m in cells:
        if m not in prof:
            v = esc[m][srcmask[m]]
            c = collections.Counter(v.tolist())
            top, n = c.most_common(1)[0]
            inv = {i: s for s, i in table.items()}
            prof[m] = dict(n_escaping_sources=int(srcmask[m].sum()), n_distinct=len(c),
                           modal_destination=inv[top], modal_share=round(n / max(1, len(v)), 4))

    res["decisive"] = row(*DECISIVE)
    res["should_be_far"] = [r for r in (row(*p) for p in FAR) if r]
    res["control"] = row(*CONTROL)

    # ---- KC: the frequency-matched null, which for this arm IS the test ----
    def null_agreement(a, b, draws):
        both = srcmask[a] & srcmask[b]
        if int(both.sum()) == 0:
            return None
        ea, eb = escv[a][both], escv[b][both]
        oa, ra = bands[a]; ob, rb = bands[b]
        Va, Vb = len(vocab[a]), len(vocab[b])
        out = []
        for _ in range(draws):
            pa = np.clip(ra[ea] + rng.integers(-BAND // 2, BAND // 2 + 1, size=len(ea)), 0, Va - 1)
            pb = np.clip(rb[eb] + rng.integers(-BAND // 2, BAND // 2 + 1, size=len(eb)), 0, Vb - 1)
            sa = [vocab[a][int(i)] for i in oa[pa]]
            sb = [vocab[b][int(i)] for i in ob[pb]]
            out.append(float(np.mean([x == y for x, y in zip(sa, sb)])))
        return out

    nulls = {}
    for lbl, (a, b) in [("decisive", DECISIVE)] + [(f"far:{p[1]}", p) for p in FAR]:
        d = null_agreement(a, b, DRAWS)
        if d is None:
            continue
        obs = row(a, b)["agreement_raw"]
        arr = np.array(d)
        nulls[lbl] = dict(a=a, b=b, observed=obs, null_mean=round(float(arr.mean()), 4),
                          null_sd=round(float(arr.std()), 4),
                          null_p95=round(float(np.percentile(arr, 95)), 4),
                          observed_percentile=round(float((arr < obs).mean() * 100), 2),
                          draws=DRAWS)
    res["frequency_null"] = nulls
    dec = nulls.get("decisive")
    res["KC_fires"] = bool(dec and dec["observed"] is not None and dec["observed"] <= dec["null_p95"])

    # ---- KD: F183's defect, checked for its sibling ----
    ags, cards = [], []
    for a, b in itertools.combinations(models, 2):
        r = row(a, b)
        if r and r["agreement_raw"] is not None:
            ags.append(r["agreement_raw"])
            cards.append(prof[a]["n_distinct"] + prof[b]["n_distinct"])
    r_card = float(np.corrcoef(np.array(ags), np.array(cards, float))[0, 1])
    res["KD_cardinality"] = dict(n_pairs=len(ags), pearson=round(r_card, 4),
                                 fires=bool(abs(r_card) > CARD_GATE))

    # ---- identification ----
    M = np.zeros((len(models), len(models)))
    for i, j in itertools.combinations(range(len(models)), 2):
        r = row(models[i], models[j])
        M[i, j] = M[j, i] = -1.0 if r["agreement_raw"] is None else r["agreement_raw"]
    fam = {m: cells[m]["family"] for m in models}
    labels = [fam[m] for m in models]
    bal = balance_report(labels, name="family label")
    n = len(models)
    hits, tie, nn = 0, 0.0, {}
    for i in range(n):
        d = M[i].copy(); d[i] = -np.inf
        j = int(np.argmax(d))
        ties = [k for k in range(n) if k != i and M[i, k] == M[i, j]]
        same = sum(1 for k in ties if fam[models[k]] == fam[models[i]])
        tie += same / len(ties)
        hits += int(fam[models[j]] == fam[models[i]])
        nn[models[i]] = dict(nearest=models[j], agreement=round(float(M[i, j]), 4),
                             same_family=bool(fam[models[j]] == fam[models[i]]),
                             n_tied=len(ties), n_tied_same_family=same)
    chance_fam = float(np.mean([(labels.count(fam[m]) - 1) / (n - 1) for m in models]))
    res["identification"] = dict(
        n=n, rank1_same_family=hits, rank1_accuracy=round(hits / n, 4),
        tie_aware_accuracy=round(tie / n, 4),
        chance_family_level=round(chance_fam, 4), chance_instance_level=round(1 / (n - 1), 4),
        majority_class_rate=round(bal.majority_rate, 4), family_counts=bal.counts,
        balance=bal.reason, nearest_neighbour=nn,
        _what_it_is_not="FAMILY ATTRIBUTION, not instance identification -- refused before the run.")
    res["agreement_matrix"] = dict(order=models, matrix=np.round(M, 4).tolist(),
                                   _status="DESCRIPTIVE; four comparisons were registered.")
    _verdict(res)


def _verdict(res):
    p = [f"ESCAPE DESTINATIONS (arm 1), registered in {PREREG} (sha256 "
         f"{res['_prereg_sha256'][:12]}..., frozen before any destination was decoded). ZERO forward "
         f"passes: the values were already stored as argmax_ids. Sources are the {res['intersection_size']} "
         f"shared probe strings; destinations are compared as DECODED STRINGS, never ids, per the "
         f"F166 inversion. "]
    d = res["decisive"]
    p.append(f"KB: the largest single-destination share in any model is {res['modal_share_max']:.1%}"
             + (f", above the registered {MODAL_GATE}, so the PRIMARY figure everywhere is the "
                f"modal-excluded one. " if res["KB_fires"] else
                f", below the registered {MODAL_GATE}, so the raw figure is primary. "))
    if d and d["KA_not_decidable"]:
        p.append(f"KA FIRES on the decisive pair: {d['n_shared_sources']} shared escaping sources is "
                 f"below the registered floor of {MIN_SOURCES}. NOT DECIDABLE. ")
    elif d:
        p.append(f"DECISIVE PAIR pythia-410m vs -deduped: agreement {d['primary']} over "
                 f"{d['n_shared_sources']} shared escaping sources (raw {d['agreement_raw']}, "
                 f"modal-excluded {d['agreement_excl_modal']} over {d['n_shared_excl_modal']}). ")
    p.append("SHOULD BE FAR: " + "; ".join(
        f"vs {r['b'].split('/')[-1]} {r['primary']}" for r in res["should_be_far"]) + ". ")
    c = res["control"]
    if c:
        p.append(f"PRECISION FLOOR (fp32 vs bf16, same weights): {c['primary']} over "
                 f"{c['n_shared_sources']} sources. ")
    n = res["frequency_null"].get("decisive")
    if n:
        p.append(f"KC, AND FOR THIS ARM THE NULL IS THE TEST: the decisive pair's observed agreement "
                 f"{n['observed']} against a frequency-matched null of {n['null_mean']} "
                 f"(sd {n['null_sd']}, p95 {n['null_p95']}), observed percentile "
                 f"{n['observed_percentile']}. ")
        p.append("KC FIRES -- the escape destination carries no identity information beyond "
                 "frequency, and the finding is that null. " if res["KC_fires"] else
                 "KC does not fire: the agreement survives frequency matching. ")
    k = res["KD_cardinality"]
    p.append(f"KD: agreement correlates with destination diversity at r = {k['pearson']} across "
             f"{k['n_pairs']} pairs"
             + (f", above the registered {CARD_GATE} -- this is substantially a diversity effect, "
                f"F183's defect in a new costume, and is reported as such. "
                if k["fires"] else f", below the registered {CARD_GATE}. "))
    i = res["identification"]
    p.append(f"IDENTIFICATION: rank-1 nearest neighbour by agreement puts {i['rank1_same_family']} of "
             f"{i['n']} models beside their own family, {i['rank1_accuracy']:.0%} against a "
             f"family-level chance of {i['chance_family_level']:.1%} and a majority-class rate of "
             f"{i['majority_class_rate']:.0%}; tie-aware {i['tie_aware_accuracy']:.0%}. FAMILY "
             f"ATTRIBUTION, never instance identification. ")
    p.append("REFUSALS, registered before the numbers: no p-value on the cohort (the null percentiles "
             "are within-comparison controls); no semantic reading of any token list; no cross-model "
             "comparison keyed on token id; no adjustment of any gate after the fact; no mechanistic "
             "claim. THE PRIOR-ART RE-CHECK OWED SINCE F183 IS STILL OWED and still blocks write-up.")
    res["verdict"] = " ".join(p)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
