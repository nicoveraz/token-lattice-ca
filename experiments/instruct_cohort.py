"""Choose an instruction-tuned cohort where COMPLIANCE actually varies. Gate 0 for the v3 attempt.

WHY THE MODEL SET IS THE EXPERIMENT NOW. F137 built a second compliance indicator and it could not
resolve ten base models. F138 rebuilt it to spec -- 40 constraint types, effective n 51.8 against
12.6, difficulty predictions tracking observed pass rates at +0.631 -- and it STILL could not, with
the observed across-model variance (0.00334) sitting below its own noise (0.00345). The true
between-model variance in verifiable instruction-following across those ten is consistent with
zero, so no amount of item-writing can help.

The reason is not subtle in hindsight: F117's cohort is ten BASE models, and instruction-following
on models never trained to follow instructions is a construct at its floor. The Leaderboard bears
that out -- those ten span IFEval 12.7-26.1, while instruction-tuned models at the same scale span
0.8-79.5. So the fix is the cohort, and this script picks it.

THE SELECTION PROBLEM IS PSEUDOREPLICATION, and it is why this is a script rather than a list. The
instruction-tuned end of the leaderboard is dominated by fine-tunes of ONE base model: at 1-4B,
dozens of entries are Llama-3.2-3B derivatives under different org names, with near-identical
parameter counts and benchmark rows. Ten of those would be n = 1 dressed as n = 10 -- the exact
error `gatecheck.units` exists to catch, arriving in the sampling frame instead of in the analysis.
So the cohort is one model per distinct PRETRAINING family, taken from the family's own org, which
is the closest available proxy for "a different pretraining run".

PRE-REGISTERED:
  FRAME     leaderboard rows with 1.0 <= params <= 4.0, a chat/instruct type, and all six
            benchmarks present. Rows are EMBEDDED in the results file: the leaderboard moves under
            any analysis that only points at it (the same reproducibility note band_benchmark_range
            carries).
  ONE PER FAMILY  first-party orgs only, one model each. Family is the independent unit here, as
            everywhere else in this project.
  RANGE     the cohort's IFEval span must exceed MIN_SPAN, and its correctness benchmarks must not
            be at floor -- F139 found F117's headline selectivity binding on GPQA with four of ten
            models at zero, so the comparator's range is now checked BEFORE the cohort is fixed
            rather than discovered afterwards.
  SIZE      TARGET_N models, matching F117's n = 10 so the two runs stay commensurable.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import io, json

import httpx
import numpy as np
import pandas as pd

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "instruct_cohort.json")
PARQUET_URL = ("https://huggingface.co/datasets/open-llm-leaderboard/contents/"
               "resolve/main/data/train-00000-of-00001.parquet")
BENCH = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
PARAM_MIN, PARAM_MAX = 1.0, 4.0
TARGET_N = 10
MIN_SPAN = 30.0          # IFEval points; the base cohort managed 13.4 and could not resolve
NEAR_FLOOR = 2.0         # normalized points, as in band_benchmark_range
MAX_AT_FLOOR = 2         # per correctness benchmark, out of TARGET_N -- F139's lesson

# First-party orgs: one instruction-tuned model each is one distinct pretraining run. Fine-tunes of
# someone else's base are excluded by construction, which is the whole point of the frame.
FIRST_PARTY = [
    "meta-llama", "Qwen", "google", "microsoft", "tiiuae", "HuggingFaceTB", "ibm-granite",
    "allenai", "stabilityai", "LGAI-EXAONE", "EleutherAI", "mistralai", "deepseek-ai",
    "internlm", "openbmb", "CohereForAI", "01-ai", "Zyphra", "state-spaces", "bigscience",
]

# EXCLUDED BY NAME, with reasons, because the org proxy cannot see either problem.
#
#   reasoning / distilled models -- a chain-of-thought model is not a single forward-pass
#   conditional, which is a different object on the lattice as well as a different thing on an
#   instruction-following pool (the same ground groq_share.py excluded gpt-oss and qwen3.6 on). And
#   a DISTILL inherits its teacher's pretraining family: DeepSeek-R1-Distill-Qwen-1.5B is a Qwen
#   derivative, so pairing it with Qwen2.5-3B-Instruct would be the pseudoreplication this frame
#   exists to prevent -- one org name, one family, counted twice.
#
#   language-specialised variants -- gemma-2-2b-jpn-it is the Japanese release, and F63 established
#   that corpus dominates these readouts (78.1% vs 20.4% at an identical tokenizer). Taking it over
#   gemma-2-2b-it would put a corpus confound inside a cohort assembled to vary compliance.
NAME_EXCLUDE = ("distill", "-r1", "reasoning", "thinking", "-jpn", "-ko-", "-ja-", "chinese")


def main():
    raw = httpx.get(PARQUET_URL, timeout=180, follow_redirects=True).content
    df = pd.read_parquet(io.BytesIO(raw))
    d = df[(df["#Params (B)"] >= PARAM_MIN) & (df["#Params (B)"] <= PARAM_MAX)].copy()
    d = d[d["Type"].astype(str).str.contains("chat|instruct|IFT", case=False, na=False)]
    d = d.dropna(subset=BENCH)
    d["org"] = d["fullname"].astype(str).str.split("/").str[0]
    d["available"] = d["Available on the hub"].astype(bool) if "Available on the hub" in d else True
    d = d[d["available"]]
    pool = d[d["org"].isin(FIRST_PARTY)].copy()
    low = pool["fullname"].astype(str).str.lower()
    pool = pool[~low.apply(lambda n: any(k in n for k in NAME_EXCLUDE))]

    # one per org, the highest-IFEval entry, then spread the cohort across the IFEval range so the
    # predictor is not clustered at the top -- range is the whole reason for changing cohort
    best = pool.sort_values("IFEval", ascending=False).groupby("org", as_index=False).first()
    best = best.sort_values("IFEval", ascending=False)
    if len(best) > TARGET_N:
        idx = np.linspace(0, len(best) - 1, TARGET_N).round().astype(int)
        cohort = best.iloc[sorted(set(idx))]
    else:
        cohort = best

    rows = [dict(model=r["fullname"], org=r["org"], params_b=float(r["#Params (B)"]),
                 scores={b: float(r[b]) for b in BENCH}) for _, r in cohort.iterrows()]
    span = max(r["scores"]["IFEval"] for r in rows) - min(r["scores"]["IFEval"] for r in rows)
    floors = {b: sum(1 for r in rows if r["scores"][b] <= NEAR_FLOOR) for b in BENCH}
    bad = [b for b in BENCH if b != "IFEval" and floors[b] > MAX_AT_FLOOR]
    ok = span >= MIN_SPAN and not bad and len(rows) >= 6

    res = dict(
        _preregistration=dict(
            param_min=PARAM_MIN, param_max=PARAM_MAX, target_n=TARGET_N, min_span=MIN_SPAN,
            near_floor=NEAR_FLOOR, max_at_floor=MAX_AT_FLOOR, benchmarks=BENCH,
            first_party=FIRST_PARTY, name_exclude=list(NAME_EXCLUDE), source=PARQUET_URL,
            frame="1-4B, chat/instruct type, all six benchmarks present, first-party org",
            one_per_family="family is the independent unit; the instruct end of the leaderboard is "
                           "dominated by fine-tunes of one base model, and ten of those would be "
                           "n=1 dressed as n=10",
            range_gate="IFEval span >= MIN_SPAN and no correctness benchmark with more than "
                       "MAX_AT_FLOOR models at floor -- F139's lesson, applied before the cohort "
                       "is fixed rather than after",
            why="F138 showed the ten BASE models have no measurable variance in verifiable "
                "instruction-following; the cohort is the experiment now, not the instrument"),
        cohort=rows, n=len(rows), ifeval_span=round(span, 2),
        at_floor=floors, floor_failures=bad, passes=bool(ok))
    res["verdict"] = (
        f"COHORT: {len(rows)} instruction-tuned models, one per pretraining family, "
        f"{PARAM_MIN}-{PARAM_MAX}B. IFEval spans {span:.1f} points "
        f"(gate {MIN_SPAN}); the ten BASE models of F117/F138 span 13.4 and could not be resolved. "
        + (f"Correctness benchmarks at floor (<= {NEAR_FLOOR}): "
           + ", ".join(f"{b}={floors[b]}/{len(rows)}" for b in BENCH if b != "IFEval") + ". ")
        + ("PASSES: this cohort has the range the base cohort lacked, and no correctness benchmark "
           "is floored past the limit, so F139's comparator problem is not rebuilt into it."
           if ok else
           f"FAILS: span {span:.1f} vs {MIN_SPAN}" + (f"; floored comparators {bad}" if bad else "")
           + ". Widen the frame before spending download or generation time."))
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(res["verdict"])
    print()
    print(f"  {'model':<48}{'B':>6}{'IFEval':>8}{'BBH':>7}{'GPQA':>7}{'MUSR':>7}{'MMLU-P':>8}{'MATH':>7}")
    for r in sorted(rows, key=lambda x: -x["scores"]["IFEval"]):
        s = r["scores"]
        print(f"  {r['model']:<48}{r['params_b']:>6.2f}{s['IFEval']:>8.1f}{s['BBH']:>7.1f}"
              f"{s['GPQA']:>7.1f}{s['MUSR']:>7.1f}{s['MMLU-PRO']:>8.1f}{s['MATH Lvl 5']:>7.1f}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
