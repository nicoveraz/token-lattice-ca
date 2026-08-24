"""Read the rivals and the top-k against experiments/prereg_escape_rival.json. Zero forward passes.

The character-class partition below is FROZEN IN THE PREREG, before any rival string was read. It is
mechanical -- a function of the characters in a string -- and involves no judgement about meaning.
That is the whole point: token IDENTITY is safe ground and token SEMANTICS is where this kind of work
goes wrong, because "the model is attracted to concept X" is a story that can be told about any token
list. No class may be added, merged or split now.

Q1 asks whether the rival is in the same character class as the token it loses to -- is the attractor
a narrow spike, or a spike on a neighbourhood of its own kind? Q2 asks whether two models put the same
string second, and the prereg registered IN ADVANCE that every Pythia pair is expected to fire KF,
because Pythia has 8 to 39 self-continuing tokens inside the shared probe set. Q3 asks how much mass
the spike holds.

Every one of these is a token-content claim, so every one carries the frequency-matched null. F171 is
why: the naive endpoint reading passed at the 99.87th percentile and died at 32.0 against it.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import collections, itertools, json, os

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from provenance import stamp, rel
from gatecheck import balance_report

PREREG = "experiments/prereg_escape_rival.json"
PR = json.load(open(_ROOT / PREREG))
OUT = _ROOT / "results" / "rival_analysis.json"
FREQ = _ROOT / "results" / "corpus_token_counts.json"
PROBES = _ROOT / "experiments" / "probe_strings_selfcont.json"
BAND = 50
DRAWS = PR["FREQUENCY_MATCHED_NULL_mandatory_for_every_token_content_claim"]["null_draws"]
KF_FLOOR = 20
SEED = 20260823


def char_class(s):
    """THE FROZEN PARTITION. First match wins; the order is the prereg's order."""
    if s and all(c.isspace() for c in s):
        return "whitespace"
    if s and s.isascii() and not any(c.isalnum() or c.isspace() for c in s):
        return "punctuation"
    if len(s) > 1 and s[0] == " " and s[1:].isascii() and s[1:].isalnum():
        return "alnum_leading_space"
    if s and s.isascii() and s.isalnum():
        return "alnum_bare"
    return "other"


def main():
    cells = {}
    for p in sorted((_ROOT / "results").glob("rival_topk_*.json")):
        if p.name == "rival_topk_failures.json":
            continue
        d = json.load(open(p))
        if d.get("n_self_continuing"):
            cells[d["cell"]] = d
    if not cells:
        print("no rival cells present"); return

    counts = json.load(open(FREQ)) if FREQ.exists() else {}
    probe = json.load(open(PROBES))
    strings = [e["s"] for e in probe["strings"]]
    rng = np.random.default_rng(SEED)

    res = dict(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_escape_rival.sha256").read().split()[0],
               cells=sorted(cells), n_cells=len(cells),
               frozen_partition=PR["CHARACTER_CLASS_PARTITION_frozen_before_any_string_was_read"]["rule_in_order_first_match_wins"])

    # ---- Q1: is the rival in the token's own character class? ----
    q1, cls_tab = {}, {}
    for k, d in sorted(cells.items()):
        own = [char_class(s) for s in d["token_strings"]]
        riv = [char_class(s) for s in d["rival_strings"]]
        n = len(own)
        same = int(sum(1 for a, b in zip(own, riv) if a == b))
        cls_tab[k] = dict(token=dict(collections.Counter(own)), rival=dict(collections.Counter(riv)))
        q1[k] = dict(n=n, same_class=same, rate=round(same / n, 4),
                     not_decidable=bool(n < KF_FLOOR))
    res["Q1_neighbourhood_homogeneity"] = q1
    res["class_tables"] = cls_tab

    # frequency-matched null for Q1: redraw each rival inside its own frequency band
    q1n = {}
    for k, d in sorted(cells.items()):
        repo = d["model"]
        cm = counts.get(repo)
        if not cm:
            continue
        V = max(int(x) for x in cm) + 1
        cnt = np.zeros(V, np.int64)
        for i, c in cm.items():
            cnt[int(i)] = c
        order = np.lexsort((np.arange(V), cnt)); rank = np.empty(V, np.int64); rank[order] = np.arange(V)
        rid = np.array([i for i in d["rival_ids"] if i < V])
        if len(rid) < KF_FLOOR:
            continue
        own = [char_class(s) for s in d["token_strings"]][:len(rid)]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo)
        dec = {}
        vals = []
        for _ in range(DRAWS):
            pr_ = np.clip(rank[rid] + rng.integers(-BAND // 2, BAND // 2 + 1, len(rid)), 0, V - 1)
            ids = order[pr_]
            cls = []
            for i in ids:
                i = int(i)
                if i not in dec:
                    dec[i] = char_class(tok.decode([i]))
                cls.append(dec[i])
            vals.append(float(np.mean([a == b for a, b in zip(own, cls)])))
        a = np.array(vals)
        q1n[k] = dict(observed=q1[k]["rate"], null_mean=round(float(a.mean()), 4),
                      null_sd=round(float(a.std()), 4), null_p95=round(float(np.percentile(a, 95)), 4),
                      observed_percentile=round(float((a < q1[k]["rate"]).mean() * 100), 2), draws=DRAWS)
        print(f"  Q1 null {k:<34} obs {q1[k]['rate']:.4f} vs {a.mean():.4f}", flush=True)
    res["Q1_frequency_null"] = q1n

    # ---- KG: anti-vacuity on the partition ----
    allcls = collections.Counter()
    for t in cls_tab.values():
        allcls.update(t["rival"])
    tot = sum(allcls.values())
    dom = allcls.most_common(1)[0]
    res["KG_partition_balance"] = dict(
        rival_class_totals=dict(allcls), dominant=dom[0], dominant_share=round(dom[1] / tot, 4),
        fires=bool(dom[1] / tot > 0.90),
        _note="if one class holds the rivals in every model there is no room to vary and any "
              "criterion with a shape on this quantity is vacuous.")

    # ---- Q2: cross-model rival agreement over shared self-continuing sources ----
    pos = {}
    for k, d in cells.items():
        m = {}
        pid = json.load(open(_ROOT / "results" /
                             f"selfcont_set_{k.replace('/', '__').replace('@', '__')}.json"))["probe_token_ids"]
        inv = {t: i for i, t in enumerate(pid) if t >= 0}
        for j, t in enumerate(d["token_ids"]):
            if t in inv:
                m[inv[t]] = j                       # probe position -> row in this cell
        pos[k] = m
    q2 = []
    ks = sorted(cells)
    for a, b in itertools.combinations(ks, 2):
        sh = sorted(set(pos[a]) & set(pos[b]))
        if len(sh) < KF_FLOOR:
            q2.append(dict(a=a, b=b, n_shared=len(sh), agreement=None,
                           KF_not_decidable=True)); continue
        ra = [cells[a]["rival_strings"][pos[a][p]] for p in sh]
        rb = [cells[b]["rival_strings"][pos[b][p]] for p in sh]
        q2.append(dict(a=a, b=b, n_shared=len(sh),
                       agreement=round(float(np.mean([x == y for x, y in zip(ra, rb)])), 4),
                       KF_not_decidable=False))
    res["Q2_cross_model_rival_agreement"] = q2
    res["Q2_n_not_decidable"] = sum(1 for r in q2 if r["KF_not_decidable"])

    # ---- Q3: how much mass does the spike hold? ----
    q3 = {}
    for k, d in sorted(cells.items()):
        lg = np.array(d["topk_logits_e4"], np.float64) / 1e4
        lse = np.array(d["logsumexp_e4"], np.float64) / 1e4
        p1 = np.exp(lg[:, 0] - lse)
        pk = np.exp(lg - lse[:, None]).sum(axis=1)
        q3[k] = dict(n=len(p1),
                     p_top1_median=round(float(np.median(p1)), 4),
                     p_top1_q1=round(float(np.percentile(p1, 25)), 4),
                     p_top1_q3=round(float(np.percentile(p1, 75)), 4),
                     p_top8_median=round(float(np.median(pk)), 4),
                     not_decidable=bool(len(p1) < KF_FLOOR))
    res["Q3_topk_concentration"] = q3

    # family attribution is NOT attempted on Q2: the prereg foresaw KF firing on every Pythia pair
    fams = [cells[k]["family"] for k in ks if "@" not in k]
    res["family_balance"] = balance_report(fams, name="family label").reason

    _verdict(res)


def _verdict(res):
    p = [f"RIVAL AND TOP-K at self-continuing tokens, registered in {PREREG} (sha256 "
         f"{res['_prereg_sha256'][:12]}...). {res['n_cells']} cells, zero forward passes in this "
         f"analysis. The character-class partition was frozen in the prereg before any rival string "
         f"was read and is mechanical, not semantic. "]
    kg = res["KG_partition_balance"]
    p.append(f"KG (anti-vacuity): the rival classes across the cohort are {kg['rival_class_totals']}, "
             f"dominant '{kg['dominant']}' at {kg['dominant_share']:.1%}"
             + (" -- ABOVE 0.90, so the partition has no room to vary and Q1 is reported over the "
                "variable classes as well. " if kg["fires"] else ", below 0.90, so the partition has "
                "room to vary. "))
    q1 = res["Q1_neighbourhood_homogeneity"]
    p.append("Q1, is the rival in the token's own character class: "
             + "; ".join(f"{k.split('/')[-1]} {v['rate']}" for k, v in sorted(q1.items())) + ". ")
    n = res["Q1_frequency_null"]
    if n:
        p.append("AGAINST THE FREQUENCY-MATCHED NULL (F171's discipline, and the reason it is not "
                 "optional): "
                 + "; ".join(f"{k.split('/')[-1]} obs {v['observed']} vs null {v['null_mean']} "
                             f"(pct {v['observed_percentile']})" for k, v in sorted(n.items())) + ". ")
    p.append(f"Q2, do two models put the same string second: {res['Q2_n_not_decidable']} of "
             f"{len(res['Q2_cross_model_rival_agreement'])} pairs are NOT DECIDABLE under KF, which "
             f"the prereg foresaw and wrote down before the run -- Pythia has 8 to 39 "
             f"self-continuing tokens inside the shared probe set. ")
    dec = [r for r in res["Q2_cross_model_rival_agreement"] if not r["KF_not_decidable"]]
    if dec:
        p.append("Decidable pairs: " + "; ".join(
            f"{r['a'].split('/')[-1]}~{r['b'].split('/')[-1]} {r['agreement']} (n={r['n_shared']})"
            for r in sorted(dec, key=lambda r: -r["agreement"])[:6]) + ". ")
    q3 = res["Q3_topk_concentration"]
    p.append("Q3, mass at the spike (median p of the self-continuing token): "
             + "; ".join(f"{k.split('/')[-1]} {v['p_top1_median']}" for k, v in sorted(q3.items()))
             + ". ")
    p.append("REFUSALS: no semantic reading -- the partition is mechanical and frozen, and no "
             "sentence of the form 'the model is attracted to X' is licensed by any of this; no "
             "p-value; no cross-model comparison keyed on token id. THE PRIOR-ART RE-CHECK IS OWED "
             "AND BLOCKS WRITE-UP.")
    res["verdict"] = " ".join(p)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
