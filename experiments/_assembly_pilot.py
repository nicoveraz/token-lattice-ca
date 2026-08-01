"""Assembly-theory pilot -- the four tables in assembly_theory.md, reproducible. (#20)

NOT a gated experiment and deliberately not named like one: it writes nothing to results/, carries
no provenance stamp, and is not registered in _STALENESS_PAIRS. It prints tables. Section 5.1 of
assembly_theory.md specifies the gated version (`assembly_calib.py`) that should replace it.

WHAT IT ESTABLISHES, in order:

  calib    The assembly index of a STRING is the size of the smallest binary straight-line program
           deriving it -- ASI(w) = SLP(w), proven NP-complete (Masierak, arXiv:2604.16302). RePair
           EXHIBITS a grammar, so it is a CERTIFIED UPPER BOUND, not a fitted estimate. Two string
           families have provable exact indices and the estimator is graded against them:
              a^n           -> exact = minimal addition-chain length for n
              all-distinct  -> exact = n - 1  (nothing repeats, so no reuse is possible)

  discrim  Raw assembly index vs LZ77 vs gzip vs entropy over five regimes. Under LENGTH AND
           MULTISET control -- real text against its own word-shuffle -- the raw index is NULL
           (0.4725 vs 0.4765), and so is every compression baseline. This is the Kempes et al.
           (npj Complexity 2025) permutation control, applied to text.

  weight   The exponential in A = sum_i e^{a_i} (n_i - 1) / N_T is LOAD-BEARING. Temper it to
           e^{a/2} or to linear a and the ordering INVERTS -- degenerate repetition beats real
           text. This is the difference between assembly theory and a repetition count.

  matched  Delta = log A(text) - <log A(word-shuffled)>, length-matched. Reads +0.00 on BOTH
           failure poles (degenerate repetition, random soup) and +6.87 on real text.

Usage:  .venv/bin/python experiments/_assembly_pilot.py [calib|discrim|weight|matched|all]
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, math, zlib, random, re, collections, itertools, statistics

# THE ESTIMATORS LIVE IN assembly_calib.py, NOT HERE. This file predates the gate and originally
# carried its own copies; two implementations of the same estimator can drift, and a drifted
# estimator is indistinguishable from the defect the gate exists to catch (hazard 1, F56). The
# gated module is the single source of truth and this pilot imports it, so the S3 tables below are
# produced by exactly the code the gate licenses.
from assembly_calib import (WORD, SHAKESPEARE, FLOOR, repair_assembly_index,
                            addition_chain_length, A_exp, lg, delta)
from assembly_baselines import lz77_phrases, gzip_bits, shannon_bits

NOVELTY_JSON = _ROOT / "results" / "novelty_structure.json"


# ----------------------------------------------------------------------- stages

def stage_calib():
    print("=" * 78)
    print("CALIBRATION RUNG -- strings with PROVABLE assembly indices")
    print("=" * 78)
    ok = True

    print("\n  family 1:  a^n   (exact = minimal addition-chain length)")
    print("     n   exact   repair   gap")
    for n in [2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 100, 128]:
        e, g = addition_chain_length(n), repair_assembly_index("a" * n)
        flag = "" if g == e else ("   <-- IMPOSSIBLE (below a proven bound)" if g < e
                                  else "   <-- OVERSHOOT")
        ok &= (g >= e)
        print(f"   {n:5d} {e:7d} {g:8d} {g-e:5d}{flag}")

    print("\n  family 2:  all-distinct symbols   (exact = n - 1, no reuse possible)")
    print("     n   exact   repair   gap")
    alpha = [chr(c) for c in range(33, 127)]
    for n in [2, 4, 8, 16, 32, 64, 90]:
        e, g = n - 1, repair_assembly_index(alpha[:n])
        ok &= (g == e)
        print(f"   {n:5d} {e:7d} {g:8d} {g-e:5d}{'' if g == e else '   <-- MISMATCH'}")

    print(f"\n  GATE: {'PASS' if ok else 'FAIL'} -- never below a proven bound, exact on the "
          f"no-reuse family.")
    return ok


def stage_discrim():
    random.seed(0)
    L = 2000
    corpus = SHAKESPEARE.read_text(errors="replace")[:L]
    cases = {
        "periodic  (ab)*":     "ab" * (L // 2),
        "single token  a*":    "a" * L,
        "real text (shakes.)": corpus,
        "word-shuffled text":  " ".join(random.sample(corpus.split(), len(corpus.split()))),
        "char-shuffled text":  "".join(random.sample(corpus, len(corpus))),
        "uniform random (26)": "".join(random.choice("abcdefghijklmnopqrstuvwxyz")
                                       for _ in range(L)),
    }
    print("\n" + "=" * 78)
    print("DISCRIMINATION -- raw index vs the compression baselines, matched length")
    print("=" * 78)
    print(f"\n  {'case':22s} {'a_hat':>7s} {'a/len':>7s} {'lz77_z':>7s} {'gzip_b':>8s} {'H_bits':>8s}")
    for k, v in cases.items():
        print(f"  {k:22s} {repair_assembly_index(v):7d} {repair_assembly_index(v)/len(v):7.4f} "
              f"{lz77_phrases(v):7d} {gzip_bits(v):8d} {shannon_bits(v):8.0f}")
    print("\n  Real vs word-shuffled is the LENGTH- AND MULTISET-CONTROLLED comparison, and every")
    print("  measure above is null on it. The ensemble quantity separates the same pair (see")
    print("  `matched`), which is the whole case for assembly theory here.")


def stage_weight():
    corpus = SHAKESPEARE.read_text(errors="replace")
    rng = random.Random(0)
    L = 7000
    seg = corpus[:L]
    sw = seg.split()
    cases = {
        "real text":          WORD.findall(seg.lower()),
        "word-shuffled":      WORD.findall(" ".join(rng.sample(sw, len(sw))).lower()),
        "degenerate x1":      ["the"] * 1200,
        "degenerate 2-cycle": ["the", "of"] * 600,
        "random word soup":   WORD.findall(" ".join(rng.choice(sw) for _ in range(L // 6)).lower()),
    }
    print("\n" + "=" * 88)
    print("WEIGHTING SENSITIVITY -- is the exponential decoration, or load-bearing?")
    print("=" * 88)
    print(f"\n  {'case':22s} {'logA_exp':>9s} {'logA_sqrt':>10s} {'logA_lin':>9s} {'types':>6s}")
    for k, words in cases.items():
        grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        cnt = collections.Counter(grams)
        we, ws, wl = [], [], []
        for g, c in cnt.items():
            if c < 2:
                continue
            a = repair_assembly_index(g)
            we.append(math.exp(a) * (c - 1))
            ws.append(math.exp(a / 2) * (c - 1))
            wl.append(a * (c - 1))
        m = max(len(grams), 1)
        f = lambda w: f"{math.log10(sum(w)/m):9.2f}" if w else "    -inf "
        print(f"  {k:22s} {f(we):>9s} {f(ws):>10s} {f(wl):>9s} {len(we):6d}")
    print("\n  Tempering INVERTS the ordering: under linear weighting the measure is pure copy")
    print("  number and degenerate repetition wins outright. Only the full exponential penalises")
    print("  trivially-assembled objects hard enough to keep real text on top.")


def stage_matched(budget=440, n_win=5):
    print("\n" + "=" * 92)
    print(f"LENGTH-MATCHED Delta -- {budget} words, median of {n_win} contiguous windows")
    print("=" * 92)

    def show(label, words, seed=0):
        if len(words) < budget:
            print(f"  {label:16s} {len(words):6d}   (< {budget} words -- excluded)")
            return
        starts = ([0] if len(words) == budget else
                  [round(i * (len(words) - budget) / (n_win - 1)) for i in range(n_win)])
        ds, es = [], []
        for st in starts:
            d, _, e = delta(words[st:st + budget], seed=seed)
            ds.append(d)
            es.append(e)
        print(f"  {label:16s} {len(words):6d}  {statistics.median(ds):+6.2f} "
              f"[{min(ds):+5.2f},{max(ds):+5.2f}]   eff={statistics.median(es):5.2f}")

    rng = random.Random(1)
    base = WORD.findall(SHAKESPEARE.read_text(errors="replace")[:40000].lower())
    print(f"\n  CONTROLS\n  {'case':16s} {'nWords':>6s}   {'Delta':>6s}  {'[min,max]':>13s}")
    for k, v in {"real text": base,
                 "degenerate x1": ["the"] * 2000,
                 "degenerate x2": ["the", "of"] * 1000,
                 "random soup": [rng.choice(base) for _ in range(2000)],
                 "unique tokens": [f"w{i}" for i in range(2000)]}.items():
        show(k, v)

    if not NOVELTY_JSON.exists():
        print(f"\n  ({NOVELTY_JSON.name} absent -- skipping the CA cells)")
        return
    print(f"\n  THE CA CELLS\n  {'cell':16s} {'nWords':>6s}   {'Delta':>6s}  {'[min,max]':>13s}")
    for key, v in json.loads(NOVELTY_JSON.read_text())["runs"].items():
        if v.get("full_text"):
            show(key, WORD.findall(v["full_text"].lower()))
    print("\n  NOT A FINDING: one settle run per cell, and Delta is a tail statistic (effective")
    print("  object count 1-3). Needs >= 8 seeds with the SEED as the independent unit (F57).")


STAGES = {"calib": stage_calib, "discrim": stage_discrim,
          "weight": stage_weight, "matched": stage_matched}

if __name__ == "__main__":
    which = _sys.argv[1] if len(_sys.argv) > 1 else "all"
    for name, fn in STAGES.items():
        if which in ("all", name):
            fn()
