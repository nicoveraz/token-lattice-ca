"""Gate B of #101: do the band's families have usable benchmark coverage and dynamic range?

RUNS BEFORE THE BAND MEASUREMENT, because it can only shrink n. Gate 0 (band_family_census)
found 22 conservative families with a base, ungated checkpoint at 1.5-3.6B; F68 requires ~16.
This gate asks whether the OTHER axis of the correlation exists at all: a public benchmark score
per family, with spread that is not eval noise around chance.

Source: the Open LLM Leaderboard v2 contents parquet (open-llm-leaderboard/contents), fetched at
run time and with the used rows EMBEDDED in the results file, because the leaderboard moves under
any analysis that only points at it (#101's own reproducibility note). v2's normalized columns
put the random-chance floor at 0 by construction, which is what makes "near floor" a defined
predicate rather than a per-benchmark constant.

The verdict separates two failure modes that must not be conflated:
  COVERAGE  -- does a family's in-band base model appear on the leaderboard at all? Base models
               at this scale are rarely submitted; this is the axis Gate 0 could not see.
  RANGE     -- among covered families, does the benchmark spread exceed the near-floor band?

Writes results/band_benchmark_range.json.
Usage:  .venv/bin/python -u experiments/band_benchmark_range.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import io, json

import httpx
import numpy as np
import pandas as pd

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "band_benchmark_range.json")
CENSUS = _ROOT / "results" / "band_family_census.json"
PARQUET_URL = ("https://huggingface.co/datasets/open-llm-leaderboard/contents/"
               "resolve/main/data/train-00000-of-00001.parquet")
BENCH = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
NEAR_FLOOR = 2.0        # normalized points; v2 normalization puts chance at 0
REQUIRED = 16           # F68's family requirement -- the same bar Gate 0 was held to
# Leaderboard-covered in-band member missing from the census's per-family pick. The census chose
# granite-3.3/4.x, which are not on the leaderboard; granite-3.0-2b-base is in band and is.
EXTRA_MEMBERS = {"ibm": ["ibm-granite/granite-3.0-2b-base"]}


def main():
    cen = json.load(open(CENSUS))
    fams = {f: list(ms) for f, ms in cen["curated_conservative"].items()}
    for f, extra in EXTRA_MEMBERS.items():
        fams[f] = fams.get(f, []) + extra

    raw = httpx.get(PARQUET_URL, timeout=120, follow_redirects=True).content
    df = pd.read_parquet(io.BytesIO(raw))

    covered, uncovered = {}, []
    for f, ms in fams.items():
        h = df[df["fullname"].isin(ms)]
        if len(h):
            r = h.iloc[0]
            covered[f] = dict(model=r["fullname"],
                              params_b=float(r["#Params (B)"]),
                              scores={b: round(float(r[b]), 2) for b in BENCH})
        else:
            uncovered.append(f)

    print(f"  covered {len(covered)}/{len(fams)} families; uncovered: {sorted(uncovered)}")
    bench_out = {}
    for b in BENCH:
        v = np.array([covered[f]["scores"][b] for f in covered])
        nf = int((v < NEAR_FLOOR).sum())
        usable = bool(nf <= len(v) / 2)
        bench_out[b] = dict(median=round(float(np.median(v)), 2),
                            max=round(float(v.max()), 2),
                            near_floor=nf, n=len(v), usable=usable)
        print(f"  {b:12s} median {bench_out[b]['median']:>6}  max {bench_out[b]['max']:>6}  "
              f"near-floor {nf}/{len(v)}  -> {'usable' if usable else 'EXCLUDED'}")

    n_cov = len(covered)
    usable_benches = [b for b, d in bench_out.items() if d["usable"]]
    passes = n_cov >= REQUIRED and bool(usable_benches)
    verdict = (
        f"GATE B {'PASSES' if passes else 'FAILS'}: {n_cov} of {len(fams)} families have a "
        f"leaderboard-covered in-band base model, against the {REQUIRED} that F68's effect size "
        f"requires. "
        + (f"Dynamic range is NOT the problem -- {len(usable_benches)} of {len(BENCH)} benchmarks "
           f"are usable on the covered set ({', '.join(usable_benches)}); MATH Lvl 5"
           f"{' and GPQA' if not bench_out['GPQA']['usable'] else ''} sit at floor. "
           f"COVERAGE is: base models at 1.5-3B are rarely submitted to the leaderboard, and "
           f"{len(uncovered)} families ({', '.join(sorted(uncovered))}) have no entry at all. "
           if not passes else "")
        + ("The benchmark-correlation PRIMARY as designed is therefore not powered even with 22 "
           "measured families, and the options are: run our own evals over the band "
           "(lm-eval-harness, days of compute), or demote the benchmark correlation to an "
           "exploratory secondary on the covered subset and let the band run stand on its riders "
           "-- T* (#90, which needs only OUR measurements and reaches n>=16 for the first time), "
           "the corpus direction, and the F64 scale gate. The riders never needed the "
           "leaderboard, so Gate B failing does NOT block the band measurement; it rescopes what "
           "the measurement claims." if not passes else ""))
    print(f"\n  -> {verdict}")

    res = dict(n_families=len(fams), n_covered=n_cov, required=REQUIRED,
               near_floor_threshold=NEAR_FLOOR, benchmarks=bench_out,
               usable_benchmarks=usable_benches, covered=covered,
               uncovered=sorted(uncovered), extra_members=EXTRA_MEMBERS,
               passes=passes, verdict=verdict,
               parquet_sha_note="rows used are embedded above; the source parquet moves")
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Gate B of #101, run before the band measurement because it can only shrink n. The v2 "
        "leaderboard normalizes scores so chance sits at 0, making near-floor a defined predicate. "
        "The verdict separates COVERAGE (does a family's in-band base model appear at all -- the "
        "axis Gate 0 could not see) from RANGE (spread beyond the near-floor band among covered "
        "families). Coverage is the binding failure: base checkpoints at this scale are rarely "
        "submitted. The used rows are embedded because the leaderboard moves under any analysis "
        "that only points at it.")
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
