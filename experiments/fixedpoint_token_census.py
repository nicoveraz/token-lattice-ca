"""What ARE the fixed points? A token-class census of terminal tokens. Zero new forward passes.

THE INVENTORY CONSTRAINT, AND WHY THIS IS NOT THE CENSUS THAT WAS ASKED FOR. The task asked for the
identities of terminal FIXED-POINT tokens. Those were never stored. `endpoint_histogram` records the
terminal token of ALL 96 trajectories with fixed-point, cyclic and wandering outcomes POOLED;
`fixed_point_fraction` and `cyclic_fraction` survive only as aggregates, and no per-trajectory outcome
label exists in any results file (verified over the union of cell keys across all sixteen
histogram-bearing files). Re-running to recover them was not authorised and is not done here.

WHAT IS RECOVERABLE, AND THE GATE THAT MAKES IT HONEST. In a cell whose fixed_point_fraction is high,
the endpoint histogram is BY CONSTRUCTION mostly fixed points: phi is exactly the fraction of the 96
endpoints that are fixed points, so phi doubles as a PURITY measure for that cell's histogram. A cell
with phi = 0.97 has a histogram that is 97% fixed-point tokens; a cell with phi = 0.10 has one that is
90% something else, and reading token classes off it would be reading cycles and wanderers.

PURITY is therefore pre-registered as a gate, not discovered afterwards: a cell enters the census only
if phi >= PURITY_MIN on the ARM being described. Contaminated cells are named, never imputed. This is
the project's own anti-vacuity rule applied to a quantity whose contamination happens to be known
exactly.

PRE-REGISTERED, BEFORE COMPUTING (the task's own prediction, restated in the terms above):
  PREDICTION  tokens GAINED under a phi-raising prefix are FORMAT-CONGRUENT with the prefix's
              document type. The F154 raising prefix is a table-of-contents fragment (Pile row 101,
              "\\n\\nGreat Britain\\n\\n# Contents\\n"), so the gained mass should be whitespace,
              punctuation, digits, list-markers or markup -- the furniture of a contents page --
              rather than ordinary word tokens.
  KILL        if gained tokens are ORDINARY-WORD-DOMINATED (ordinary-word share of gained mass > 0.5
              on a majority of readable pairs), the corpus-statistics reading dies: the prefix is not
              pulling the map toward its own document furniture.
  NOT DECIDABLE  if fewer than MIN_PAIRS readable (model, prefix) pairs survive the purity gate. The
              gate is on the RAISED arm, because that is the arm whose gained tokens the prediction
              is about.
  BOUNDARY    token CLASS is inferred from the decoded string, which is a lexical judgement, not a
              measurement. The classifier is fixed here before any output is seen and is reported in
              full so a reader can disagree with a rule rather than with a number.

EVERY NUMBER TRACES: each cell carries the results file and key it came from.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import json, re, unicodedata

import numpy as np

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "fixedpoint_token_census.json")

# (file, raw-arm key template, raised-arm key template, label) -- every pair traces to a file.
SOURCES = [
    ("domain_base.json", "{m}|s{cs}|raw", "{m}|s{cs}|bos", "bos"),
    ("text_interaction.json", None, "{m}|s{cs}|p1", "p1"),          # raw comes from domain_base
    ("structural_text.json", None, "{m}|s{cs}|t0", "struct_t0"),
    ("domain_midrange.json", "{m}|s{cs}|raw", "{m}|s{cs}|text_matched", "text_matched"),
]
RAW_FALLBACK = ("domain_base.json", "{m}|s{cs}|raw")
CENSUS_SEEDS = [20260803, 990017]
PURITY_MIN = 0.50          # the raised arm's histogram must be majority fixed-point
MIN_PAIRS = 3
WORD_SHARE_KILL = 0.50

CLASSES = ("whitespace", "punctuation", "digit", "list_marker", "markup",
           "byte_fallback", "non_latin", "word")


def classify_token(s):
    """Lexical class of a decoded token string. Fixed before any output was seen."""
    if s is None:
        return "byte_fallback"
    t = s.strip()
    if s != "" and t == "":
        return "whitespace"
    if t == "":
        return "whitespace"
    # HF byte-fallback / unknown renderings
    if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", t) or t in ("<unk>", "�") or "�" in t:
        return "byte_fallback"
    if re.fullmatch(r"[#*|=>_`~\-]{1,}", t) or t.startswith("##") or t in ("<|", "|>"):
        return "markup"
    if re.fullmatch(r"[0-9]+([.,][0-9]+)?", t):
        return "digit"
    if re.fullmatch(r"[0-9]+[.)\]]", t) or re.fullmatch(r"[ivxlcIVXLC]+[.)]", t) or t in ("•", "◦", "‣"):
        return "list_marker"
    if all(unicodedata.category(c).startswith("P") or unicodedata.category(c).startswith("S")
           for c in t):
        return "punctuation"
    letters = [c for c in t if c.isalpha()]
    if letters and not any(("LATIN" in unicodedata.name(c, "")) for c in letters):
        return "non_latin"
    return "word"


def _load(name):
    p = _ROOT / "results" / name
    if not p.exists():
        return {}
    d = json.load(open(p))
    return d.get("runs") or d.get("cells") or {}


def hist_of(runs, key):
    v = runs.get(key)
    if not isinstance(v, dict) or "endpoint_histogram" not in v:
        return None, None
    counts = {}
    for row in v["endpoint_histogram"]:
        tid, dec, n = row[0], row[1], row[2]
        counts[int(tid)] = (dec, counts.get(int(tid), (dec, 0))[1] + int(n))
    return counts, float(v["fixed_point_fraction"])


def merge_seeds(runs, tmpl, m):
    """Sum the histogram over census seeds; average phi. Returns (counts, phi, keys)."""
    tot, phis, keys = {}, [], []
    for cs in CENSUS_SEEDS:
        k = tmpl.format(m=m, cs=cs)
        c, phi = hist_of(runs, k)
        if c is None:
            continue
        keys.append(k)
        phis.append(phi)
        for tid, (dec, n) in c.items():
            d0, n0 = tot.get(tid, (dec, 0))
            tot[tid] = (dec, n0 + n)
    if not keys:
        return None, None, []
    return tot, float(np.mean(phis)), keys


def class_mass(counts):
    out = {c: 0 for c in CLASSES}
    for tid, (dec, n) in counts.items():
        out[classify_token(dec)] += n
    return out


def main():
    res = {"_preregistration": dict(
        purity_min=PURITY_MIN, min_pairs=MIN_PAIRS, word_share_kill=WORD_SHARE_KILL,
        census_seeds=CENSUS_SEEDS, classes=list(CLASSES),
        inventory_constraint="fixed-point token identities were NEVER stored; endpoint_histogram "
                             "pools fixed / cyclic / wandering outcomes and no per-trajectory label "
                             "exists in any results file. phi doubles as the histogram's PURITY.",
        gate="a (model, prefix) pair enters only if the RAISED arm has phi >= PURITY_MIN, so its "
             "histogram is majority fixed-point; contaminated pairs are named, never imputed",
        prediction="tokens GAINED under a phi-raising prefix are FORMAT-CONGRUENT with the prefix's "
                   "document type (the F154 raiser is a table-of-contents fragment)",
        kill="ordinary-word share of gained mass > 0.5 on a majority of readable pairs -> the "
             "corpus-statistics reading dies",
        not_decidable=f"fewer than {MIN_PAIRS} readable pairs",
        boundary="token CLASS is a lexical judgement from the decoded string, not a measurement; "
                 "the classifier is fixed in the script and reported in full")}

    files = {name: _load(name) for name, *_ in SOURCES}
    files[RAW_FALLBACK[0]] = _load(RAW_FALLBACK[0])
    models = sorted({k.split("|")[0] for k in files["domain_base.json"]
                     if len(k.split("|")) == 3})

    pairs, contaminated, missing = [], [], []
    for fname, raw_t, rai_t, label in SOURCES:
        runs = files[fname]
        for m in models:
            rc, rphi, rkeys = (merge_seeds(runs, raw_t, m) if raw_t
                               else merge_seeds(files[RAW_FALLBACK[0]], RAW_FALLBACK[1], m))
            ac, aphi, akeys = merge_seeds(runs, rai_t, m)
            if rc is None or ac is None:
                missing.append(dict(model=m, arm=label, file=fname))
                continue
            if aphi < PURITY_MIN:
                contaminated.append(dict(model=m, arm=label, file=fname,
                                         raised_phi=round(aphi, 4),
                                         reason=f"raised-arm phi {aphi:.3f} < purity {PURITY_MIN}"))
                continue
            gained = {}
            for tid, (dec, n) in ac.items():
                base = rc.get(tid, (dec, 0))[1]
                if n > base:
                    gained[tid] = (dec, n - base)
            gm = class_mass(gained)
            tot = sum(gm.values())
            if tot == 0:
                contaminated.append(dict(model=m, arm=label, file=fname,
                                         reason="no gained mass"))
                continue
            pairs.append(dict(
                model=m, arm=label, file=fname,
                raw_keys=rkeys, raised_keys=akeys,
                raw_phi=round(rphi, 4), raised_phi=round(aphi, 4),
                gained_total=int(tot),
                gained_class_share={c: round(gm[c] / tot, 4) for c in CLASSES},
                word_share=round(gm["word"] / tot, 4),
                format_share=round(sum(gm[c] for c in CLASSES if c != "word") / tot, 4),
                top_gained=[[int(t), d, int(n)] for t, (d, n) in
                            sorted(gained.items(), key=lambda kv: -kv[1][1])[:8]]))

    res["pairs"] = pairs
    res["contaminated"] = contaminated
    res["missing"] = missing

    parts = []
    parts.append(
        f"INVENTORY CONSTRAINT: fixed-point token identities were never stored. endpoint_histogram "
        f"pools fixed / cyclic / wandering endpoints, and phi is the fraction of them that are fixed "
        f"points -- so phi is also the histogram's PURITY. This census therefore reads only cells "
        f"whose RAISED arm has phi >= {PURITY_MIN}, where the histogram is majority fixed-point. No "
        f"model was re-run.")
    parts.append(
        f"COVERAGE: {len(pairs)} readable (model, arm) pairs; {len(contaminated)} excluded by the "
        f"purity gate or for having no gained mass; {len(missing)} absent from the files.")
    if len(pairs) < MIN_PAIRS:
        parts.append(
            f"NOT DECIDABLE: {len(pairs)} readable pairs against a floor of {MIN_PAIRS}. The purity "
            f"gate is what removes them -- most phi-raising arms in this programme raise phi to "
            f"values well below 0.5, so their endpoint histograms are mostly cycles and wanderers "
            f"and cannot speak about fixed points. Recovering this needs per-trajectory outcome "
            f"labels, i.e. a re-run, which was not authorised.")
        res["verdict"] = " ".join(parts)
    else:
        ws = [p["word_share"] for p in pairs]
        n_word = sum(1 for w in ws if w > WORD_SHARE_KILL)
        res["summary"] = dict(n_pairs=len(pairs), median_word_share=round(float(np.median(ws)), 4),
                              n_word_dominated=n_word)
        parts.append(
            "PER PAIR, share of GAINED endpoint mass by class: "
            + "; ".join("{} [{}] word {:.2f}, format {:.2f} (phi {:.2f} -> {:.2f})".format(
                p["model"].split("/")[-1], p["arm"], p["word_share"], p["format_share"],
                p["raw_phi"], p["raised_phi"]) for p in pairs) + ". ")
        parts.append(
            (f"KILL FIRED: gained mass is ordinary-word-dominated on {n_word} of {len(pairs)} pairs "
             f"(median word share {np.median(ws):.2f}). The corpus-statistics reading dies: a "
             f"phi-raising prefix does not pull the map toward its own document furniture."
             if n_word > len(pairs) / 2 else
             f"PREDICTION HELD: gained mass is format-congruent (non-word) on "
             f"{len(pairs) - n_word} of {len(pairs)} pairs, median word share "
             f"{np.median(ws):.2f}. The tokens a raising prefix adds are the furniture of its own "
             f"document type, not ordinary vocabulary."))
    parts.append(
        "BOUNDARY: token class is a lexical judgement from the decoded string, fixed in the script "
        "before any output was seen. 'Gained' is per-token-id count increase from the raw arm to the "
        "raised arm, summed over census seeds. Purity is known exactly but is not perfect: at "
        f"phi = {PURITY_MIN} a readable histogram is still half non-fixed-point endpoints.")
    res.setdefault("verdict", " ".join(parts))
    if len(pairs) >= MIN_PAIRS:
        res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
