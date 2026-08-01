"""One source of truth for the assembly-index estimator and its gate (#20; §5.1 of assembly_theory.md).

WHY THIS IS A MODULE AND NOT A COPY. The same reason `dp_calibration.py` is: F56 was a calibration
measured at one geometry and applied to another, and the moment gate code is pasted into a second
script the two can drift. A drifted gate is indistinguishable from the defect it exists to prevent.
There is exactly one implementation of the estimator and of the gate, and the scripts import it --
including `_assembly_pilot.py`, which reproduces §3's tables from these functions rather than its own.

WHAT THE GATE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT.

Computing an assembly index exactly is the smallest-grammar problem: NP-complete, APX-hard (Masierak,
arXiv:2604.16302, Lemma 1 / Theorems 1-2, which prove ASI(w) = SLP(w)). So the estimator is a greedy
RePair pass, which EXHIBITS a grammar and is therefore a **certified upper bound** rather than a
fitted quantity. Two string families have provable exact indices and pin it from opposite directions:

    a^n              exact index = minimal ADDITION-CHAIN length for n. Every joining step
                     concatenates two objects already built, so the reachable lengths form an
                     addition chain and the minimum number of steps is l(n).
    all-distinct     exact index = n - 1. Nothing repeats, so nothing can be reused, and the
                     pathway is a binary tree with n leaves.

**§5.1 of assembly_theory.md asked this gate to assert RePair is EXACT on both. It is not, and the
claim came from a 14-point sample of n that happened to contain no failure.** Swept over every n in
2..128 the estimator is exact on 75 and overshoots on 52 -- the smallest failure is n=15, where the
minimum is 5 (1,2,3,6,12,15) and greedy halving finds 6. That is the textbook smallest case where the
binary method is not an optimal addition chain, so it was always going to be there. The pilot's table
was a property of its sample, not of the estimator, which is the F64 failure mode exactly.

So the gate asserts the two properties that ARE true, and measures the third rather than claiming it:

  G1  SOUNDNESS on a^n -- repair >= exact at every n. This is the load-bearing one. A value BELOW a
      proven lower bound would mean the estimator is not an upper bound at all and every number
      downstream would be unusable. It holds at 127/127.
  G2  EXACTNESS on the no-reuse family -- repair == n-1 at every n. This is the genuine exactness
      rung, and it holds at every n tested to 256.
  G3  THE TWO FAILURE POLES -- Delta ~ 0 on degenerate repetition and on random soup, and Delta
      >> 0 on real text, at matched length. Without this the statistic could be a disorder measure
      wearing assembly theory's clothes.

  MEASURED, NOT GATED: the exactness RATE on a^n, reported with its failure list so the overshoot
  is on the record and a future refactor that changes it is visible.

WHAT "DELTA = 0 AT A POLE" ACTUALLY MEANS, since it is easy to oversell. Delta is a contrast against
a matched word-shuffle, and `lg` maps A = 0 to a FLOOR. At the noise pole BOTH the text and its
shuffle have A = 0 -- no object repeats at all -- so Delta is 0 by definition of the floor rather
than by measurement. The substantive fact there is **A(text) = 0**, which is the property that
distinguishes A from entropy, and it is reported separately as `A_is_zero`. At the degenerate pole
Delta = 0 is a genuine measurement: shuffling a string of identical tokens returns the same string.

No model, no GPU, no network. Seconds on CPU.

Writes results/assembly_calib.json.
Usage:  .venv/bin/python -u experiments/assembly_calib.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, math, random, re, collections, itertools, statistics

from provenance import stamp, rel

WORD = re.compile(r"[a-z0-9']+")
SHAKESPEARE = _ROOT / "data" / "shakespeare.txt"
OUT = str(_ROOT / "results" / "assembly_calib.json")

CHAIN_MAX = 128          # a^n sweep; every n in 2..CHAIN_MAX, not a sample of them
DISTINCT_MAX = 256       # all-distinct sweep, likewise exhaustive
NGRAM = 3                # object type; n=4 undersamples at these lengths (§3.6)
BUDGET = 440             # words per window -- A grows with length, so it must be matched
N_WIN = 5                # contiguous windows, median reported
K_SHUF = 6               # shuffles per window
POLE_TOL = 0.50          # |Delta| a failure pole may show and still count as pinned
REAL_MIN = 4.00          # Delta real text must clear
FLOOR = -3.0             # log10 stand-in for A = 0, so cells with no repeats stay comparable


# ------------------------------------------------------------------ estimators

def repair_assembly_index(seq):
    """Constructive UPPER BOUND on the assembly index, via RePair.

    Greedily replaces the most frequent adjacent pair with a new object -- exactly an assembly step
    (join two things already in the pool, add the result, reuse it for free). Total steps =
    (objects created) + (joins to concatenate what is left).

    It EXHIBITS a grammar, so the value is achievable by construction and can never be below the
    true index. It is not the minimum: see G1's docstring for where and why it overshoots.
    """
    seq = list(seq)
    n_rules = 0
    while True:
        counts = collections.Counter(zip(seq, seq[1:]))
        if not counts:
            break
        pair, c = counts.most_common(1)[0]
        if c < 2:
            break
        new, out, i = ("N", n_rules), [], 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                out.append(new)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq, n_rules = out, n_rules + 1
    return n_rules + max(len(seq) - 1, 0)


def addition_chain_length(n):
    """Exact minimal addition-chain length for n = the exact assembly index of a^n.

    Breadth-first over reachable pools, so the first time n appears the depth IS the minimum. This
    is the proven reference the estimator is checked against, not another estimate.
    """
    if n == 1:
        return 0
    frontier, seen = [(frozenset([1]), 0)], {frozenset([1])}
    while frontier:
        nxt = []
        for pool, d in frontier:
            for a, b in itertools.combinations_with_replacement(sorted(pool), 2):
                s = a + b
                if s == n:
                    return d + 1
                if s > n:
                    continue
                np_ = pool | {s}
                if np_ not in seen:
                    seen.add(np_)
                    nxt.append((np_, d + 1))
        frontier = nxt
    raise RuntimeError("unreachable")


# --------------------------------------------------------- the ensemble quantity

def A_exp(words, n=NGRAM):
    """A = sum_i e^{a_i} (n_i - 1) / N_T over distinct word-n-gram object types.

    NOT maximised by noise: a random ensemble has every object unique, so (n_i - 1) = 0 and A = 0
    however high the assembly indices are. A degenerate ensemble has huge copy number but tiny a_i,
    so e^{a_i} stays small. Returns (A, n_repeated_types, effective_object_count).

    The effective count is returned because A is a TAIL statistic -- at these lengths it is 1-3, so
    a single chance repeat moves it, and no Delta should ever be quoted without it (§6, risk 1).
    """
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if len(grams) < 2:
        return 0.0, 0, 0.0
    w = [math.exp(_cached_index(g)) * (c - 1)
         for g, c in collections.Counter(grams).items() if c >= 2]
    if not w:
        return 0.0, 0, 0.0
    s = sum(w)
    return s / len(grams), len(w), (s * s) / sum(x * x for x in w)


_INDEX_CACHE = {}


def _cached_index(gram):
    """Memoised by n-gram string. A shuffle changes the COUNTS, never the objects' own indices, so
    recomputing RePair per shuffle is pure waste -- and the z-scores in assembly_baselines.py need
    20+ shuffles per window where the pilot used 6. Pure memo on a deterministic function: the
    values are identical, only the time changes.
    """
    v = _INDEX_CACHE.get(gram)
    if v is None:
        v = _INDEX_CACHE[gram] = repair_assembly_index(gram)
    return v


def lg(a):
    return math.log10(a) if a and a > 0 else FLOOR


def delta(words, n=NGRAM, k=K_SHUF, seed=0):
    """Delta = log A(text) - <log A(word-shuffled)>.

    The shuffle preserves length, vocabulary and unigram frequencies EXACTLY, so it is the tightest
    available control -- the analogue of this project's CRN null, and the fixed-multiset permutation
    test of Kempes et al. (npj Complexity 2025) applied to text.
    """
    rng = random.Random(seed)
    a, nrep, eff = A_exp(words, n)
    nulls = []
    for _ in range(k):
        sh = words[:]
        rng.shuffle(sh)
        nulls.append(lg(A_exp(sh, n)[0]))
    return lg(a) - statistics.fmean(nulls), nrep, eff


def windowed_delta(words, budget=BUDGET, n_win=N_WIN, seed=0):
    """Median Delta over contiguous length-matched windows, with the range and effective count.

    Length matching is mandatory: A grows with word count (§3.4 measures log A at 3.87 / 4.61 / 5.20
    for 3.5k / 7k / 20k characters of the same text), so an unmatched comparison measures length.
    """
    if len(words) < budget:
        return None
    starts = ([0] if len(words) == budget else
              [round(i * (len(words) - budget) / (n_win - 1)) for i in range(n_win)])
    rows = [delta(words[s:s + budget], seed=seed) for s in starts]
    ds = [d for d, _, _ in rows]
    return dict(delta=round(statistics.median(ds), 3),
                lo=round(min(ds), 3), hi=round(max(ds), 3),
                eff_objects=round(statistics.median(e for _, _, e in rows), 2),
                A_is_zero=bool(all(A_exp(words[s:s + budget])[0] == 0.0 for s in starts)),
                n_words=len(words), windows=len(starts))


# ----------------------------------------------------------------------- the gate

def chain_soundness(nmax=CHAIN_MAX):
    """G1 + the measured exactness rate on a^n, against minimal addition-chain length."""
    rows = [(n, addition_chain_length(n), repair_assembly_index("a" * n))
            for n in range(2, nmax + 1)]
    below = [dict(n=n, exact=e, repair=g) for n, e, g in rows if g < e]
    over = [dict(n=n, exact=e, repair=g) for n, e, g in rows if g > e]
    return dict(n_tested=len(rows), n_max=nmax,
                sound=not below, below_proven_bound=below,
                n_exact=sum(1 for _, e, g in rows if e == g),
                exact_rate=round(sum(1 for _, e, g in rows if e == g) / len(rows), 3),
                max_overshoot=max((g - e for _, e, g in rows), default=0),
                smallest_overshoot=(over[0] if over else None),
                overshoots=over)


def distinct_exactness(nmax=DISTINCT_MAX):
    """G2 -- exact on the no-reuse family, where nothing repeats so nothing can be reused."""
    bad = [dict(n=n, exact=n - 1, repair=repair_assembly_index([f"s{i}" for i in range(n)]))
           for n in range(2, nmax + 1)
           if repair_assembly_index([f"s{i}" for i in range(n)]) != n - 1]
    return dict(n_tested=nmax - 1, n_max=nmax, exact=not bad, mismatches=bad)


def pole_cases(seed=1):
    """The reference regimes: one signal case and the failure poles the statistic must pin."""
    base = WORD.findall(SHAKESPEARE.read_text(errors="replace")[:40000].lower())
    rng = random.Random(seed)
    return dict(real_text=base,
                degenerate_x1=["the"] * 2000,
                degenerate_x2=["the", "of"] * 1000,
                random_soup=[rng.choice(base) for _ in range(2000)],
                # Fixed-width labels drawn at random, NOT w0..w1999 in order. Sequential labels
                # are lexicographically ordered and of growing width, so an order-sensitive
                # baseline sees strong structure that a shuffle destroys -- an artifact of how the
                # regime was written rather than a property of "all tokens distinct". It makes no
                # difference to Delta (nothing repeats either way, so A = 0) but it badly confounds
                # the compression baselines in assembly_baselines.py, which is where it surfaced.
                unique_tokens=[f"w{i:05d}" for i in rng.sample(range(90000), 2000)])


# Poles that must pin, and the one case that must separate from them. degenerate_x2 is EXCLUDED
# from the gate: shuffling a 2-cycle does not return the same string, so its residual is a real
# effect rather than a defect, and gating on it would be gating on the size of that effect.
POLES = ["degenerate_x1", "random_soup", "unique_tokens"]
SIGNAL = "real_text"


def pole_check():
    """G3 -- both failure poles pin near zero, real text clears REAL_MIN, at matched length."""
    out = {k: windowed_delta(v) for k, v in pole_cases().items()}
    out = {k: v for k, v in out.items() if v}
    pinned = {k: bool(abs(out[k]["delta"]) <= POLE_TOL) for k in POLES if k in out}
    sig = out.get(SIGNAL)
    return dict(cases=out, tolerance=POLE_TOL, real_min=REAL_MIN, poles=POLES,
                poles_pinned=pinned, all_poles_pinned=bool(pinned and all(pinned.values())),
                signal_separates=bool(sig and sig["delta"] >= REAL_MIN),
                margin=round(sig["delta"] - max(abs(out[k]["delta"]) for k in POLES if k in out), 3)
                       if sig else None)


def decides(cal):
    """Is the estimator licensed to report? Soundness, no-reuse exactness, and both poles."""
    return bool(cal["chain"]["sound"] and cal["distinct"]["exact"]
                and cal["poles"]["all_poles_pinned"] and cal["poles"]["signal_separates"])


def calibrate():
    return dict(chain=chain_soundness(), distinct=distinct_exactness(), poles=pole_check(),
                constants=dict(ngram=NGRAM, budget=BUDGET, n_windows=N_WIN, k_shuffles=K_SHUF,
                               floor=FLOOR, pole_tol=POLE_TOL, real_min=REAL_MIN),
                gate="soundness on a^n (never below a proven bound) AND exactness on the no-reuse "
                     "family AND both failure poles pinned AND real text clearing the signal floor")


def print_ladder(cal):
    """Shared rendering, so two scripts cannot disagree about what the rung means."""
    c, d, p = cal["chain"], cal["distinct"], cal["poles"]
    print("\n=== G1  a^n against minimal addition-chain length (the proven reference) ===")
    print(f"  n = 2..{c['n_max']}, every value")
    print(f"  never below a proven bound : {c['n_tested'] - len(c['below_proven_bound'])}"
          f"/{c['n_tested']}   <- REQUIRED, and what makes it an upper bound")
    print(f"  exact                      : {c['n_exact']}/{c['n_tested']} "
          f"({c['exact_rate']*100:.0f}%)   <- MEASURED, not required")
    if c["smallest_overshoot"]:
        s = c["smallest_overshoot"]
        print(f"  smallest overshoot         : n={s['n']}, exact {s['exact']}, repair {s['repair']} "
              f"(max overshoot +{c['max_overshoot']})")

    print(f"\n=== G2  all-distinct against n-1 (no reuse possible, so the index is forced) ===")
    d_msg = "EXACT throughout" if d["exact"] else f"{len(d['mismatches'])} MISMATCHES"
    print(f"  n = 2..{d['n_max']}, every value: {d_msg}")

    print(f"\n=== G3  the failure poles, at {BUDGET} words, median of {N_WIN} windows ===")
    print(f"  {'case':16s} {'Delta':>7s}  {'[min,max]':>15s} {'eff':>6s}  {'A=0':>5s}")
    for k, v in p["cases"].items():
        mark = "  <- signal" if k == SIGNAL else ("  <- pole" if k in POLES else "")
        print(f"  {k:16s} {v['delta']:+7.2f}  [{v['lo']:+6.2f},{v['hi']:+6.2f}] "
              f"{v['eff_objects']:6.2f}  {str(v['A_is_zero']):>5s}{mark}")
    print(f"\n  poles pinned within +/-{POLE_TOL} : {p['all_poles_pinned']}")
    print(f"  real text clears +{REAL_MIN}      : {p['signal_separates']} "
          f"(margin {p['margin']:+.2f})")
    print(f"\n  gate ({cal['gate']}) -> {decides(cal)}")


def main():
    cal = calibrate()
    print_ladder(cal)
    ok = decides(cal)
    c = cal["chain"]

    verdict = (
        f"CALIBRATED. The estimator is a certified upper bound -- never below a proven bound at "
        f"{c['n_tested']}/{c['n_tested']} values of n on a^n -- and EXACT on the no-reuse family at "
        f"every n to {cal['distinct']['n_max']}. Both failure poles pin within {POLE_TOL} "
        f"(degenerate repetition and random soup), while real text reads "
        f"{cal['poles']['cases'][SIGNAL]['delta']:+.2f}, a margin of {cal['poles']['margin']:+.2f}. "
        f"It is NOT exact on a^n: {c['n_exact']}/{c['n_tested']} ({c['exact_rate']*100:.0f}%), with "
        f"the smallest failure at n={c['smallest_overshoot']['n']} where greedy halving finds "
        f"{c['smallest_overshoot']['repair']} against a minimum of {c['smallest_overshoot']['exact']} "
        f"-- the textbook smallest n at which the binary method is not an optimal addition chain. "
        f"assembly_theory.md §5.1 asked this gate to assert that exactness; it was inferred from a "
        f"14-point sample that happened to contain no failure, and is corrected here. Exactness is "
        f"not needed downstream and was never the load-bearing property: an upper bound that is "
        f"sometimes loose still cannot manufacture structure that is not there, because every "
        f"overshoot INFLATES a_i and therefore e^{{a_i}}, which is the direction that would make a "
        f"degenerate ensemble look MORE structured -- and the degenerate pole still pins at "
        f"{cal['poles']['cases']['degenerate_x1']['delta']:+.2f}."
        if ok else
        f"NOT DECIDABLE -- the estimator has not earned the right to report. "
        + ("Soundness FAILED: RePair returned a value BELOW a proven lower bound, so it is not an "
           "upper bound and nothing downstream is usable. " if not c["sound"] else "")
        + ("The no-reuse family is NOT exact, so the one family where the index is forced does not "
           "reproduce. " if not cal["distinct"]["exact"] else "")
        + ("A failure pole did not pin: the statistic reads structure where there is none. "
           if not cal["poles"]["all_poles_pinned"] else "")
        + ("Real text did not separate from the poles, so the statistic has no signal to report. "
           if not cal["poles"]["signal_separates"] else "")
        + "Fix the estimator; do not reinterpret the numbers.")

    res = dict(calibration=cal, decides=ok, verdict=verdict)
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "The gated rung for assembly theory (#20, promoting §3.1/§3.4 of assembly_theory.md). Two "
        "string families have PROVABLE exact assembly indices and pin a greedy RePair estimator "
        "from opposite directions: a^n against minimal addition-chain length, and all-distinct "
        "against n-1. The exact index is the smallest-grammar problem -- NP-complete and APX-hard "
        "(Masierak, arXiv:2604.16302, proving ASI(w) = SLP(w)) -- so RePair is used because it "
        "EXHIBITS a grammar and is therefore a certified upper bound rather than a fitted value. "
        "The gate asserts soundness (never below a proven bound) and exactness on the no-reuse "
        "family, and MEASURES the exactness rate on a^n rather than claiming it: §5.1 asserted "
        "exactness there from a 14-point sample of n, and an exhaustive sweep of 2..128 finds 52 "
        "overshoots, the smallest at n=15. The third gate is behavioural: Delta must pin near zero "
        "on both failure poles and separate on real text, or the statistic is a disorder measure "
        "in assembly theory's clothes. Note that at the noise pole Delta = 0 holds because BOTH "
        "text and shuffle have A = 0, which the log floor maps to a fixed value -- the substantive "
        "fact there is A(text) = 0, reported as A_is_zero, and it is the property that "
        "distinguishes A from entropy. At the degenerate pole Delta = 0 is a genuine measurement, "
        "since shuffling identical tokens returns the same string. No model, no GPU, no network.")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {verdict}")
    print("\nwrote", rel(OUT))
    return ok


if __name__ == "__main__":
    main()
