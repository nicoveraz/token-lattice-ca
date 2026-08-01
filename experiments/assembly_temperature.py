"""Does the CA produce complexity, and at what temperature? (#20; §5.3)

THE QUESTION IS ABOUT THE CA, NOT ABOUT ASSEMBLY THEORY. Whether a language model driven as a
cellular automaton produces structure that is more than recombination is the thing worth knowing.
Assembly's Delta is ONE candidate instrument for reading that, and §5.2 measured thirteen of them
against each other. If gzip reads it, use gzip. Nothing here defends a measure; the measures are
apparatus and the CA is the subject.

WHICH INSTRUMENTS CAN BE TRUSTED -- AND HOW r=2 ANSWERS THAT FOR FREE. Every measure claiming to
track complexity should be NON-MONOTONE in temperature wherever a system is known to be degenerate
at one end and random at the other. The AR construction at r=2 is exactly that system, and both
poles were established independently by damage spreading, long before any of this:

    low-T pole    74% of the ring is a single token at T=0.02, across nineteen models and three
                  interventions (F62-F70), melting at T* ~ 0.52 for pythia-410m
    high-T pole   at T >= 0.9 the ring contains no repeated 3-gram at all (§3.5)

So r=2 is an INSTRUMENT-SELECTION RUNG for the whole suite: a measure that reads a peak there can be
believed at radii where the answer is unknown, and a measure that runs monotonically through a known
non-monotonicity cannot. This costs nothing extra -- the same settles carry every measure.

§5.3 AS DRAFTED HAD THE RADII BACKWARDS, AND A PROBE CAUGHT IT. It named r in {3,8} the headline and
r=2 "the artifact regime, never pooled" (F69). But the interior-peak prediction rests entirely on
the low-T pole, and F69 showed that pole exists only at r <= 2 -- a probe confirms it, with the r=3
ring at T=0.02 sitting at 21% its top token rather than degenerate. At r in {3,8} there is ONE pole,
and with one pole a monotone curve is what you would expect; predicting a peak there and reading
monotonicity as refutation would test a prediction nothing implies. Being an out-of-distribution
artifact is precisely what makes r=2 useful: its answer is known in advance, which is what a
calibration rung is.

    r=2      INSTRUMENT SELECTION. Both poles known -> every usable measure must be non-monotone.
             Runs FIRST and gates the rest as control flow, not as a judgement made after seeing
             numbers (the discipline that made F67's skipped M2/M3 a result rather than an excuse).
    r=3,8    THE MEASUREMENT. Only the high-T pole is known. Whatever the surviving instruments read
             here is the finding about the CA.

ORIENTATION IS DECLARED UP FRONT, NOT FITTED. A smaller compressed size or entropy means MORE
structure, a larger Delta or excess entropy means more structure. The signs are in ORIENT below,
fixed before the run, so "which measure peaks where" cannot be tuned after the fact.

EVERY MEASURE IS A CONTRAST AGAINST A WITHIN-REPLICA SHUFFLE, which does three jobs at once (F74):
it is the Kempes fixed-multiset permutation control; it cancels the finite-sample bias that makes
C_mu and excess entropy unusable at these lengths; and because it permutes WITHIN each replica, the
cross-replica redundancy of 16 similar rings sits in both terms and cancels -- so no measure can
score replica convergence as within-text structure (§3.6 confound 2). The size of that redundancy is
reported separately via Delta's within/cross copy-number split.

ONE SEEDING ARM, NOT TWO. F72 measured the basins: the prompt is ERASED (max top-1 gap 0.053 AR,
0.022 MLM; 2% and 12% of a corpus seed survives). Random seeding answers the whole question, halves
the grid, and avoids the attractor-token trap that would read zero BY CONSTRUCTION.

RING ROTATIONS (§3.6 confound 1). A period-p motif appears as p distinct linear n-grams that are
rotations of one another. Delta is reported both plain and with n-grams canonicalised to their
minimal rotation, because canonicalisation is right for a ring and wrong for linear reference text,
and the size of the difference is what nobody has measured.

PRE-REGISTERED:
  Selection (r=2).  Which measures show an INTERIOR maximum in T, judged against BETWEEN-SEED
                    spread AND against a permutation null? Those are the usable instruments. A
                    measure whose peak sits inside its own noise is recorded as monotone.

                    AMENDED AFTER THE FIRST RUN (F76). The original criterion was between-seed
                    spread ALONE, and it reported two survivors of fifteen. The permutation null
                    -- temperature labels shuffled WITHIN seed, so each seed keeps its own
                    distribution and loses only the shape -- shows that criterion fires ~10.6% of
                    the time per measure, with a 95th percentile of SIX survivors of fifteen. Two
                    is what chance produces (p = 0.32) and neither cleared BH-FDR. Selection now
                    requires clearing the null at BH-FDR 0.05, because the looser rule would have
                    licensed up to six spurious instruments for the measurement radii, where
                    nothing downstream could have caught it. Same correction shape as F59-v1 and
                    dp_calibration's mean-PLUS-spread rule.
  Measurement.      What do the surviving instruments read at r in {3,8}? Agreement across
                    independent measures is the result; disagreement is reported as disagreement.
  Null.             NO measure is non-monotone at r=2. Then none of this apparatus can read
                    complexity on a system where complexity is known to vary, and the whole
                    #20 line closes. A NULL HERE IS A GOOD RESULT. **This is what happened.**
  Kill.             If a measure reads high complexity at T=0.02, r=2 while the ring IS degenerate,
                    that measure is disqualified -- it is reading the repetition as structure.
  Power.            THE SEED IS THE INDEPENDENT UNIT (F57), not the replica. Between-seed and
                    within-seed spread are both reported; if comparable, n is effectively 1.

Writes results/assembly_temperature.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/assembly_temperature.py [--probe]
        (safe to interrupt and re-run -- it resumes, keyed per (construction, r, T, seed))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, math, time, random, collections, statistics
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from assembly_calib import WORD, NGRAM, FLOOR, lg, _cached_index, decides, calibrate
from assembly_baselines import profile
from dev_transition_phase3 import bh_fdr        # one implementation; F39 used it on a like battery

N_PERM = 2000                                   # permutations behind the selection null (F76)

GENERATORS = [("ar",  "EleutherAI/pythia-410m", "step143000", "has a low-T pole (F62-F70)"),
              ("mlm", "bert-base-uncased",      None,         "control -- pole unreachable (F67/F72)")]
TEMPS = [0.02, 0.1, 0.2, 0.3, 0.436, 0.52, 0.7, 0.9, 1.1]   # spans BOTH poles; includes T_c and T*
# r=2 IS THE INSTRUMENT TEST, NOT AN AFTERTHOUGHT. It is the only radius where BOTH poles are
# known, so the only radius where the non-monotonicity prediction is forced. It runs FIRST and gates
# the rest, in the project's own idiom: validate on a system whose answer you know, then measure the
# one you do not.
RADII_CALIB = [2]
RADII = [3, 8]                     # the measurement, interpretable only once r=2 has validated
SEEDS = list(range(11, 19))        # 8; the seed is the independent unit (F57)
N, B, SETTLE = 96, 16, 16
K_SHUF = 8                         # shuffles per Delta; memoised indices make this cheap
NGRAMS = [2, 3]                    # §3.6 point 5: n=2 AND 3 at these lengths. n=3 alone left the
                                   # median effective object count at 1.00 -- Delta resting on a
                                   # single n-gram in 58 of 72 cells, which is not an estimate.
T_STAR, T_C = 0.52, 0.436          # F68 and F58, for locating the peak against -- not for fitting
OUT = str(_ROOT / "results" / "assembly_temperature.json")

# Declared BEFORE the run, so "which measure peaks where" cannot be tuned afterwards. +1 means a
# larger contrast is more structure; -1 means a smaller one is (a shorter compressed string or a
# lower entropy is more redundancy). n_words and n_types are omitted: they are exactly invariant
# under the control, which is what stops the InChI-length confound operating (F74).
ORIENT = {"logA": +1, "excess_entropy": +1, "C_mu": +1, "mi_integrated": +1,
          "lz77_z": -1, "lzw_dict": -1, "repair_size": -1, "sequitur_size": -1,
          "gzip_bits": -1, "bz2_bits": -1, "lzma_bits": -1,
          "H0": -1, "H_block": -1, "h_rate": -1}


# ------------------------------------------------- Delta, decomposed and rotation-aware

def _canon(gram_words):
    """Lexicographically minimal rotation -- merges a ring motif's phases into one object type."""
    n = len(gram_words)
    return min(tuple(gram_words[i:] + gram_words[:i]) for i in range(n))


def _types(replicas, n=NGRAM, canon=False, ring=True):
    """Object types -> (total copies, copies attributable to within-replica repetition).

    A ring has no start, so n-grams WRAP: a linear read would silently drop the n-1 grams spanning
    the seam and under-count exactly the periodic motifs this experiment is about.
    """
    tot, within = collections.Counter(), collections.Counter()
    for rep in replicas:
        if len(rep) < n:
            continue
        idx = range(len(rep)) if ring else range(len(rep) - n + 1)
        here = collections.Counter()
        for i in idx:
            g = [rep[(i + j) % len(rep)] for j in range(n)]
            here[_canon(g) if canon else tuple(g)] += 1
        tot.update(here)
        # copies beyond the first WITHIN this replica are within-replica structure
        for g, c in here.items():
            within[g] += max(c - 1, 0)
    return tot, within


def _A(tot, within, mode):
    """A = sum_i e^{a_i} (excess_i) / N_T, with the copy excess split by provenance.

    total  : (n_i - 1), the quantity §2 defines
    within : copies repeated inside a single replica -- genuine within-text structure
    cross  : the remainder, i.e. convergence BETWEEN replicas, which is a different claim
    """
    NT = sum(tot.values())
    if NT < 2:
        return 0.0, 0, 0.0
    w = []
    for g, c in tot.items():
        exc = (c - 1) if mode == "total" else (within[g] if mode == "within"
                                              else (c - 1) - within[g])
        if exc > 0:
            w.append(math.exp(_cached_index("".join(g))) * exc)
    if not w:
        return 0.0, 0, 0.0
    s = sum(w)
    return s / NT, len(w), (s * s) / sum(x * x for x in w)


def delta_decomposed(replicas, k=K_SHUF, seed=0, canon=False, n=NGRAM):
    """Delta for total / within / cross copy excess, each against the SAME shuffle ensemble.

    The shuffle permutes words WITHIN each replica, so it preserves the per-replica multiset exactly
    and destroys only order -- the Kempes fixed-multiset control, applied per replica so that the
    cross-replica term keeps its meaning.

    DELTA IS None WHERE IT IS NOT A MEASUREMENT. `lg` maps A = 0 to a FLOOR constant, and
    differencing a real log against that constant produces a number set by the constant rather than
    by the data. Measured on the first run of this experiment: cells with one side pinned at the
    floor carried sd = 2.89, LARGER than the sd = 1.72 of cells where both sides were positive,
    while having the smaller median |Delta|. Pooling the two made a working statistic look noisy.

    So `delta` is reported only where the observed A and EVERY shuffle's A are strictly positive.
    A = 0 is not missing data -- it is the substantive fact that no object repeats at all, which is
    the property that distinguishes A from entropy (§2) -- so it is reported categorically as
    `A_obs_zero` and `n_null_zero` instead of being coerced into a number. `delta_floored` keeps the
    old floor-coded value so the size of the artifact stays visible.
    """
    rng = random.Random(seed)
    obs = {m: _A(*_types(replicas, canon=canon, n=n), mode=m) for m in ("total", "within", "cross")}
    nulls = {m: [] for m in obs}
    for _ in range(k):
        sh = []
        for rep in replicas:
            r = rep[:]; rng.shuffle(r); sh.append(r)
        t, wi = _types(sh, canon=canon, n=n)
        for m in obs:
            nulls[m].append(_A(t, wi, mode=m)[0])
    out = {}
    for m in obs:
        a = obs[m][0]
        nz = sum(1 for x in nulls[m] if x <= 0)
        defined = a > 0 and nz == 0
        floored = lg(a) - statistics.fmean([lg(x) for x in nulls[m]])
        out[m] = dict(
            delta=round(lg(a) - statistics.fmean([math.log10(x) for x in nulls[m]]), 3)
                  if defined else None,
            delta_floored=round(floored, 3),
            defined=bool(defined), A_obs_zero=bool(a <= 0), n_null_zero=nz,
            logA=round(lg(a), 3), n_types=obs[m][1], eff_objects=round(obs[m][2], 2))
    return out


def contrast_profile(replicas, k=K_SHUF, seed=0):
    """Every §5.2 measure as a contrast against a WITHIN-REPLICA shuffle, oriented so + = structure.

    Shuffling inside each replica rather than across the pool is what makes this safe for a lattice
    ensemble: 16 similar rings share a great deal of material, and a pooled shuffle would leave that
    redundancy only in the observed term, so every measure would score replica convergence as
    within-text structure (§3.6 confound 2). Permuting within replicas puts it in BOTH terms, where
    it cancels.
    """
    pooled = [w for rep in replicas for w in rep]
    if len(pooled) < 40:
        return None
    obs = profile(pooled, sep="")
    rng = random.Random(seed)
    nulls = []
    for _ in range(k):
        sh = []
        for rep in replicas:
            r = rep[:]; rng.shuffle(r); sh.extend(r)
        nulls.append(profile(sh, sep=""))
    out = {}
    for m in ORIENT:
        vs = [x[m] for x in nulls]
        mu = statistics.fmean(vs)
        out[m] = round(ORIENT[m] * (obs[m] - mu), 4)
    return out


# ------------------------------------------------------------------------- the run

def settle(kind, rule, r, T, seed, scheme):
    """Settle the ring and return it as TOKEN sequences, not word sequences.

    THE SYMBOL IS THE TOKEN, AND THIS IS LOAD-BEARING. The first version applied the pilot's word
    regex, which is right for prose and catastrophic here: at r=2, T=0.02 the ring is 56/96 newlines
    and 27/96 commas, so the regex kept 9 "words" from 96 token slots -- 84 across all 16 replicas,
    against ~1500 at mid temperature. That is an 18x LENGTH GRADIENT along the very axis the peak
    test compares, and A grows with length (§3.4). F74's guarantee that the InChI-length confound
    cannot operate held WITHIN a cell, where the control matches length exactly; extending it ACROSS
    temperatures was my error.

    On tokens every cell is exactly N*B symbols, so length is matched by construction everywhere,
    and the degenerate pole is represented faithfully as "the same symbol 56 times" instead of being
    silently deleted. The CA's state is a token ring; measuring anything else is measuring a
    projection of it.
    """
    carun = (__import__("ar_ca").run if kind == "ar" else __import__("mlm_ca").run)
    s = carun(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme=scheme,
              init="random", seed=seed, order="per_replica")["final"]
    reps = [[rule.tok.decode([int(x)]) for x in row] for row in s]
    top1 = float(np.mean([collections.Counter(row.tolist()).most_common(1)[0][1] / N for row in s]))
    return reps, top1


def cell(kind, rule, r, T, seed, scheme):
    reps, top1 = settle(kind, rule, r, T, seed, scheme)
    d = {}
    for n in NGRAMS:
        d[f"n{n}|plain"] = delta_decomposed(reps, seed=seed, n=n)
        d[f"n{n}|rotcanon"] = delta_decomposed(reps, seed=seed, canon=True, n=n)
    # THE RINGS ARE STORED. novelty_structure.py learned this the expensive way -- it kept
    # `full_text`, so when its scoring turned out to be confounded by whitespace the fix was a
    # re-analysis rather than a re-run. This experiment's first version did not carry the lesson
    # forward, so diagnosing Delta's noise cost the whole grid. 432 cells of decoded rings is ~3 MB.
    return dict(top1_share=round(top1, 4), n_words=sum(len(x) for x in reps),
                delta=d, oriented=contrast_profile(reps, seed=seed),
                rings=["".join(x) for x in reps])


def main(probe=False):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    gate = calibrate()
    if not decides(gate):
        res["verdict"] = ("NOT DECIDABLE -- assembly_calib's gate does not pass, so the estimator "
                          "has not earned the right to report on the CA.")
        json.dump(res, open(OUT, "w"), indent=1); print(res["verdict"]); return False

    res["_preregistration"] = dict(
        generators=[dict(name=g, role=ro) for _, g, _, ro in GENERATORS],
        temps=TEMPS, radii=RADII, radii_calib=RADII_CALIB, seeds=SEEDS,
        N=N, B=B, settle=SETTLE, ngram=NGRAM, k_shuffles=K_SHUF, T_star=T_STAR, T_c=T_C,
        which_radius_tests_what=(
            "§5.3 as drafted named r in {3,8} the headline and r=2 the artifact regime. That was "
            "backwards and a probe caught it: the interior-peak prediction rests entirely on the "
            "low-T single-token pole, which F69 showed exists only at r<=2, and at r=3 T=0.02 the "
            "ring is 21% its top token rather than degenerate. So r=2 is the INSTRUMENT TEST -- the "
            "one radius where both poles are known and the prediction is forced -- and it gates "
            "r in {3,8}, the measurement. Being an out-of-distribution artifact is what makes r=2 "
            "useful here: its answer is known in advance."),
        primary_ar="at r=2, Delta(T) has an INTERIOR maximum -- both poles were established "
                   "independently by damage spreading before this statistic existed, so the "
                   "prediction is forced. No interior peak is predicted at r in {3,8}, where only "
                   "the high-T pole exists and monotone decrease refutes nothing",
        primary_mlm="Delta(T) does NOT turn over at low T -- the pole exists but is unreachable "
                    "from random seeding (F67, sharpened by F72)",
        null="Delta monotone decreasing in T for BOTH constructions -- then it tracks disorder, not "
             "structure, and #20's edge-of-chaos framing is refuted. A NULL HERE IS A GOOD RESULT",
        kill="if AR's Delta at T=0.02 does not approach 0 where the ring is provably ~74% one "
             "token, fix the estimator rather than reinterpreting the numbers",
        independent_unit="the SEED (F57); between-seed and within-seed spread both reported",
        peak_test="judged against BETWEEN-SEED spread, so a peak inside its own noise is reported "
                  "as no interior peak",
        seeding="random only -- F72 showed the prompt is erased, which halves the grid and avoids "
                "the attractor-token trap that would read Delta ~ 0 by construction",
        confound_replica="copy excess decomposed into within-replica and cross-replica, which sum "
                         "to the total, so convergence between replicas is never scored as "
                         "within-text structure (§3.6 confound 2)",
        confound_rotation="reported both plain and with n-grams canonicalised to their minimal "
                          "rotation, since a ring motif appears as p rotated types (§3.6 "
                          "confound 1); reported both ways rather than chosen",
        ring_wrap="n-grams wrap the ring, so seam-spanning motifs are not silently dropped",
        baselines="every cell carries the §5.2 suite; no cell reports Delta alone (F74)",
        resumable="keyed by (construction, r, T, seed)")
    runs = res["runs"]

    radii = RADII_CALIB + RADII
    for kind, gen, rev, role in GENERATORS:
        keys = [f"{kind}|r{r}|T{T}|s{s}" for r in radii for T in TEMPS for s in SEEDS]
        if all(k in runs for k in keys):
            print(f"  {gen}: already complete", flush=True); continue
        t0 = time.time()
        if kind == "ar":
            from ar_ca import ARRule
            rule = ARRule(gen, revision=rev) if rev else ARRule(gen); scheme = "none"
        else:
            from mlm_ca import MLMRule
            rule = MLMRule(gen); scheme = "cls_sep"
        print(f"\n  {kind} / {gen} ({role}) loaded in {time.time()-t0:.0f}s", flush=True)
        for r in radii:
            for T in TEMPS:
                for s in SEEDS:
                    key = f"{kind}|r{r}|T{T}|s{s}"
                    if key in runs:
                        continue
                    t1 = time.time()
                    c = cell(kind, rule, r, T, s, scheme)
                    runs[key] = dict(kind=kind, model=gen, r=r, T=T, seed=s,
                                     secs=round(time.time() - t1, 1), **c)
                    o = c["oriented"] or {}
                    a3 = c["delta"]["n3|plain"]["total"]["delta"]
                    a2 = c["delta"]["n2|plain"]["total"]["delta"]
                    astr = f"A2={a2:+5.2f} A3={a3:+5.2f}" if (a2 is not None and a3 is not None) \
                        else f"A2={'--' if a2 is None else f'{a2:+5.2f}'} " \
                             f"A3={'--' if a3 is None else f'{a3:+5.2f}'}"
                    print(f"     r={r} T={T:<5} s={s}  {astr} "
                          f"gzip={o.get('gzip_bits', 0):+7.1f} lzma={o.get('lzma_bits', 0):+7.1f} "
                          f"top1={c['top1_share']*100:4.0f}%  "
                          f"{time.time()-t1:.0f}s", flush=True)
                    json.dump(res, open(OUT, "w"), indent=1)
                    if probe:
                        print(f"\n  PROBE: one cell took {time.time()-t1:.1f}s. Full grid = "
                              f"{len(GENERATORS)*len(radii)*len(TEMPS)*len(SEEDS)} cells "
                              f"~ {len(GENERATORS)*len(radii)*len(TEMPS)*len(SEEDS)*(time.time()-t1)/3600:.1f} h",
                              flush=True)
                        return True
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))
    return True


# ------------------------------------------------------------------------ analysis

def _val(v, m):
    """One cell's oriented contrast for measure m.

    logA_ring_n2 / _n3 are the ring-aware decomposed Deltas and may be None, which means A = 0 on
    one side and the contrast is not a measurement there (see delta_decomposed). None propagates as
    missing rather than as a value. Every other name comes from the §5.2 suite on the pooled text.
    """
    if m.startswith("logA_ring_n"):
        return v["delta"][f"n{m[-1]}|plain"]["total"]["delta"]
    return (v.get("oriented") or {}).get(m)


def _series(runs, kind, r, m):
    """Median oriented contrast per temperature, with the BETWEEN-SEED spread that judges it."""
    out = {}
    for T in TEMPS:
        vs = [_val(v, m) for k, v in runs.items()
              if v["kind"] == kind and v["r"] == r and v["T"] == T]
        vs = [x for x in vs if x is not None]
        if vs:
            out[T] = dict(value=round(statistics.median(vs), 4),
                          sd=round(statistics.pstdev(vs), 4) if len(vs) > 1 else 0.0,
                          n_seeds=len(vs))
    return out


def _shape(series, top1_at_low_T=None):
    """Interior maximum, judged against BETWEEN-SEED spread rather than the point estimate.

    A margin swamped by its own scatter is a coin flip reported as a decision -- the defect that
    killed the first DP gate (dp_calibration requires mean PLUS spread to clear) and six verdicts
    since. A peak inside its own noise is recorded here as monotone, not as a peak.
    """
    ts = sorted(series)
    if len(ts) < 3:
        # Carry the top-1 through: dropping it made the verdict print "the ring is 0% a single
        # token" on a cell that had measured 79%.
        return dict(complete=False, reason="too few temperatures", low_T_top1=top1_at_low_T)
    ds = [series[t]["value"] for t in ts]
    noise = statistics.fmean([series[t]["sd"] for t in ts])
    i = max(range(len(ds)), key=lambda j: ds[j])
    interior = 0 < i < len(ds) - 1
    margin = ds[i] - max(ds[0], ds[-1])
    clears = margin > noise
    return dict(complete=True,
                peak_T=ts[i], peak=round(ds[i], 4), endpoint_max=round(max(ds[0], ds[-1]), 4),
                margin=round(margin, 4), between_seed_noise=round(noise, 4),
                interior=bool(interior), clears_noise=bool(clears),
                interior_peak=bool(interior and clears),
                peaks_at_degenerate_end=bool(i == 0),
                low_T_top1=top1_at_low_T)


def _perm_null(runs, kind, r, measures, n_perm=N_PERM, seed=0):
    """How often does the interior-peak test fire when there is NO temperature structure? (F76)

    Permute the temperature labels WITHIN each seed. That destroys any dependence on temperature
    while preserving exactly each seed's own distribution of values, so a measure whose apparent
    peak is really between-seed scatter keeps the scatter and loses the shape. The statistic is
    recomputed by the same `_shape`, noise term included, so the comparison is self-consistent
    rather than against a fixed threshold.

    Returns per-measure permutation p (add-one, so never 0), BH-adjusted, plus the expected number
    of survivors -- which is the direct answer to "would N survivors have appeared anyway?".
    """
    per_measure = {}
    for m in measures:
        by_seed = {}
        for v in runs.values():
            if v["kind"] != kind or v["r"] != r:
                continue
            try:
                val = _val(v, m)
            except (KeyError, TypeError):
                val = None
            if val is not None:
                by_seed.setdefault(v["seed"], {})[v["T"]] = val
        by_seed = {s: d for s, d in by_seed.items() if len(d) == len(TEMPS)}
        if len(by_seed) >= 2:
            per_measure[m] = by_seed
    if not per_measure:
        return dict(p={}, p_bh={}, expected=0.0, p95=0, p_count=1.0, observed=0, n_measures=0,
                    n_perm=n_perm)

    def shape_of(by_seed):
        se = {T: dict(value=statistics.median([d[T] for d in by_seed.values()]),
                      sd=statistics.pstdev([d[T] for d in by_seed.values()]),
                      n_seeds=len(by_seed)) for T in TEMPS}
        return _shape(se)

    observed = sum(1 for d in per_measure.values() if shape_of(d).get("interior_peak"))
    rng, hits, per_perm = random.Random(seed), {m: 0 for m in per_measure}, []
    for _ in range(n_perm):
        # One shared set of permutations per draw, so the survivor count is computed on the same
        # draws the per-measure p-values are.
        orders = {s: rng.sample(list(TEMPS), len(TEMPS))
                  for s in next(iter(per_measure.values()))}
        k = 0
        for m, by_seed in per_measure.items():
            sh = {s: {TEMPS[i]: d[orders[s][i]] for i in range(len(TEMPS))}
                  for s, d in by_seed.items()}
            if shape_of(sh).get("interior_peak"):
                hits[m] += 1
                k += 1
        per_perm.append(k)

    names = list(per_measure)
    praw = [(hits[m] + 1) / (n_perm + 1) for m in names]
    padj = bh_fdr(praw)
    per_perm.sort()
    return dict(p={m: round(praw[i], 5) for i, m in enumerate(names)},
                p_bh={m: round(padj[i], 5) for i, m in enumerate(names)},
                expected=round(statistics.fmean(per_perm), 3),
                p95=per_perm[int(0.95 * len(per_perm))],
                p_count=round((sum(1 for k in per_perm if k >= observed) + 1) / (n_perm + 1), 5),
                observed=observed, n_measures=len(names), n_perm=n_perm)


def analyse(res):
    runs = res["runs"]
    if not runs:
        res["verdict"] = "no runs"; return
    measures = list(ORIENT) + [f"logA_ring_n{n}" for n in NGRAMS]
    RC = RADII_CALIB[0]
    out = {}
    for kind, _, _, _ in GENERATORS:
        for r in RADII_CALIB + RADII:
            for m in measures:
                se = _series(runs, kind, r, m)
                if se:
                    t1 = [v["top1_share"] for k, v in runs.items()
                          if v["kind"] == kind and v["r"] == r and v["T"] == TEMPS[0]]
                    out[f"{kind}|r{r}|{m}"] = dict(
                        series=se, shape=_shape(se, round(statistics.median(t1), 3) if t1 else None))

    # ---- SELECTION: which instruments read a non-monotonicity that is KNOWN to be there ----
    #
    # F76 AMENDED. The first version selected on `interior_peak` alone -- an interior maximum
    # clearing its own between-seed spread -- and reported two survivors of fifteen. The
    # permutation null below shows that criterion has a ~10.6% per-measure false-positive rate and
    # a 95th percentile of SIX survivors, so two is what chance produces (p = 0.32) and neither
    # cleared BH-FDR. Selecting without the null would have licensed up to six spurious instruments
    # for the r in {3,8} measurement, where nothing downstream could have caught it.
    #
    # So selection now requires clearing a null with NO temperature structure, not merely its own
    # scatter. This is the same correction shape as F59-v1 (a cost function that could shrink its
    # own comparison window) and dp_calibration's mean-PLUS-spread rule.
    perm = _perm_null(runs, "ar", RC, measures)
    padj = perm["p_bh"]
    print(f"\n=== INSTRUMENT SELECTION at r={RC}, AR -- both poles established independently ===")
    print(f"  a usable measure must peak at an INTERIOR temperature, by more than between-seed")
    print(f"  noise, AND clear a {N_PERM}-permutation null at BH-FDR 0.05 (F76's amendment)")
    print(f"\n  {'measure':16s} {'peak T':>7s} {'peak':>10s} {'margin':>9s} {'noise':>8s} "
          f"{'p_BH':>7s}  verdict")
    selected, disqualified = [], []
    for m in measures:
        k = f"ar|r{RC}|{m}"
        if k not in out:
            continue
        sh = out[k]["shape"]
        p = padj.get(m)
        if not sh.get("complete"):
            v = "incomplete"
        elif sh.get("interior_peak") and p is not None and p <= 0.05:
            selected.append(m); v = "USABLE"
        elif sh.get("interior_peak"):
            v = (f"interior peak, but INSIDE THE NULL"
                 + (f" (p_BH={p:.3f})" if p is not None else " (p undefined)"))
        elif sh.get("peaks_at_degenerate_end"):
            disqualified.append(m); v = "DISQUALIFIED -- peaks on the degenerate ring"
        else:
            v = "monotone / peak inside noise"
        print(f"  {m:16s} {str(sh.get('peak_T')):>7s} {sh.get('peak', 0):10.4f} "
              f"{sh.get('margin', 0):9.4f} {sh.get('between_seed_noise', 0):8.4f} "
              f"{(f'{p:.4f}' if p is not None else '   --'):>7s}  {v}")
    print(f"\n  null: {perm['observed']} interior peaks observed of {perm['n_measures']}; "
          f"{perm['expected']:.2f} expected with no temperature structure "
          f"(95th pct {perm['p95']}); P(>= observed by chance) = {perm['p_count']:.4f}")

    # ---- POWER GUARD. Absence of data is not absence of effect. Declaring the null below on a
    # partial grid would be the same defect that has killed six verdicts in this project: a
    # threshold applied to numbers that have not been measured yet.
    # Completeness is counted on the RUNS, not on a measure's series: a measure that is undefined
    # in many cells would otherwise look like a half-finished grid and stall the verdict forever.
    need_T, min_seeds = len(TEMPS), max(4, len(SEEDS) // 2)
    done = collections.Counter(v["T"] for v in runs.values()
                               if v["kind"] == "ar" and v["r"] == RC)
    have_T = sum(1 for t in TEMPS if done.get(t, 0) >= min_seeds)
    complete = have_T >= need_T
    if not complete:
        msg = (f"INCOMPLETE -- {have_T}/{need_T} temperatures at r={RC} have at least {min_seeds} "
               f"seeds, so neither the selection nor the null is decidable yet. Absence of data is "
               f"not absence of effect. Re-run to resume; this file is a checkpoint, not a result.")
        print(f"\n  -> {msg}")
        res["analysis"] = dict(curves=out, complete=False, temperatures_ready=have_T,
                               temperatures_needed=need_T, min_seeds=min_seeds)
        res["verdict"] = msg
        res["_analysis_provenance"] = stamp(__file__)
        return

    parts = []
    lowt1 = out.get(f"ar|r{RC}|gzip_bits", {}).get("shape", {}).get("low_T_top1")
    parts.append(f"At r={RC}, T={TEMPS[0]} the AR ring is {(lowt1 or 0)*100:.0f}% a single token, "
                 f"which is the degenerate pole F62-F70 established independently.")

    if not selected:
        parts.append(
            f"NULL, AND IT IS A CLEAN ONE: of {len(measures)} measures, NONE shows an interior peak "
            f"that survives a {perm['n_perm']}-permutation null at BH-FDR 0.05, on a system that is "
            f"degenerate at one end and random at the other. {perm['observed']} peak(s) clear their "
            f"own between-seed spread, but {perm['expected']:.2f} are expected with no temperature "
            f"structure at all (95th pct {perm['p95']}), so P(>= observed by chance) = "
            f"{perm['p_count']:.3f}. No instrument in this suite can read complexity where "
            f"complexity is known to vary, so none of them can be believed where it is unknown. "
            f"The #20 line closes here rather than at §5.3's measurement radii."
            + (f" {len(disqualified)} measure(s) actively FAILED, peaking on the degenerate ring: "
               f"{', '.join(disqualified)} -- they read repetition as structure." if disqualified else ""))
    else:
        parts.append(
            f"USABLE INSTRUMENTS: {', '.join(selected)} ({len(selected)}/{len(measures)}) show an "
            f"interior peak clearing between-seed noise. "
            + (f"DISQUALIFIED: {', '.join(disqualified)} peak on the degenerate ring, reading "
               f"repetition as structure. " if disqualified else "")
            + f"Peak temperatures {', '.join(str(out[f'ar|r{RC}|{m}']['shape']['peak_T']) for m in selected)} "
              f"against F68's T*={T_STAR} and F58's T_c={T_C} -- reported, not fitted.")

        # ---- MEASUREMENT: what the survivors read where the answer is not known ----
        print(f"\n=== MEASUREMENT at r={RADII}, using only the selected instruments ===")
        agree = {}
        for kind, _, _, _ in GENERATORS:
            for r in RADII:
                peaks = []
                for m in selected:
                    k = f"{kind}|r{r}|{m}"
                    if k in out:
                        sh = out[k]["shape"]
                        peaks.append((m, sh["peak_T"], sh["interior_peak"]))
                if peaks:
                    agree[f"{kind}|r{r}"] = peaks
                    ip = [x for x in peaks if x[2]]
                    print(f"  {kind} r={r}: " + ", ".join(f"{m}@T{t}{'*' if i else ''}"
                                                          for m, t, i in peaks))
                    print(f"           {len(ip)}/{len(peaks)} selected instruments show an "
                          f"interior peak (* marks one)")
        for kind, _, _, _ in GENERATORS:
            hits = [(r, [x for x in agree.get(f"{kind}|r{r}", []) if x[2]]) for r in RADII]
            got = [(r, h) for r, h in hits if h]
            if got:
                ts = sorted({t for _, h in got for _, t, _ in h})
                parts.append(
                    f"{kind.upper()} at r in {[r for r, _ in got]}: "
                    f"{'; '.join(f'r={r} -> ' + ', '.join(f'{m}@T={t}' for m, t, _ in h) for r, h in got)}. "
                    + (f"The selected instruments AGREE on where complexity peaks (T={ts[0]}), which "
                       f"is the result -- independent measures, one conclusion."
                       if len(ts) == 1 else
                       f"The selected instruments DISAGREE on the peak (T in {ts}), so the location "
                       f"is not established; what survives is that complexity is non-monotone in "
                       f"temperature here, not where the maximum sits."))
            else:
                parts.append(
                    f"{kind.upper()} at r in {RADII}: no selected instrument shows an interior peak. "
                    + ("For MLM this is the PREDICTED control behaviour -- F67 found no transition "
                       "and F72 showed why: the uniform state is self-sustaining but its basin is "
                       "negligible, so random seeding never reaches a pole to melt."
                       if kind == "mlm" else
                       "For AR this says the CA's complexity is monotone in temperature once the "
                       "two-token degeneracy is gone -- no interior optimum, so nothing that would "
                       "deserve the name edge of chaos. Note no peak was PREDICTED here: only the "
                       "high-T pole is known at these radii."))

    # ---- the confound diagnostics ----
    xs = [runs[k]["delta"]["n3|plain"]["cross"]["delta"] for k in runs
          if runs[k]["delta"]["n3|plain"]["cross"]["delta"] is not None]
    ws = [runs[k]["delta"]["n3|plain"]["within"]["delta"] for k in runs
          if runs[k]["delta"]["n3|plain"]["within"]["delta"] is not None]
    if xs and ws:
        parts.append(
            f"CONFOUND 2, SEPARATED: peak within-replica Delta {max(ws):+.2f} against "
            f"{max(xs):+.2f} cross-replica. " +
            (f"The signal is WITHIN replicas -- structure in the text rather than 16 copies of one "
             f"ring." if max(ws) >= max(xs) else
             f"The signal is CROSS-replica: what is measured is convergence BETWEEN replicas, not "
             f"structured text. Every pooled Delta in novelty_structure.py carries this."))
    pl = [runs[k]["delta"]["n3|plain"]["total"]["delta"] for k in runs
          if runs[k]["delta"]["n3|plain"]["total"]["delta"] is not None]
    rc = [runs[k]["delta"]["n3|rotcanon"]["total"]["delta"] for k in runs
          if runs[k]["delta"]["n3|rotcanon"]["total"]["delta"] is not None]
    if pl and rc:
        d = max(rc) - max(pl)
        parts.append(f"CONFOUND 1, MEASURED: canonicalising ring rotations moves peak Delta "
                     f"{max(pl):+.2f} -> {max(rc):+.2f} ({d:+.2f}), so rotation inflation is "
                     f"{'negligible' if abs(d) < 0.5 else 'material and must be disclosed'}.")
    nd = [(m, sum(1 for v in runs.values() if _val(v, m) is None))
          for m in measures if m.startswith("logA_ring")]
    for m, cnt in nd:
        parts.append(f"{m} is UNDEFINED in {cnt}/{len(runs)} cells (A = 0 on one side, so the "
                     f"contrast would be set by the log floor rather than by the data). Those cells "
                     f"are excluded rather than floor-coded -- pooling them was what made Delta look "
                     f"noisier than it is.")
    parts.append(f"r={RC} and r in {RADII} are never pooled: r={RC} asks whether an instrument reads "
                 f"a non-monotonicity known in advance, r in {RADII} asks what the CA actually does.")

    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(curves=out, selected=selected, disqualified=disqualified,
                           permutation_null=perm,
                           selection_radius=RC, measurement_radii=RADII, orientation=ORIENT)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "§5.3 of assembly_theory.md, reframed: the subject is the CA's complexity, not assembly "
        "theory's standing. Thirteen measures from §5.2 plus a ring-aware Delta are carried through "
        "the same temperature sweep, each as an oriented contrast against a WITHIN-REPLICA shuffle "
        "-- which is the Kempes fixed-multiset control, cancels the finite-sample bias that makes "
        "C_mu and excess entropy unusable at these lengths, and puts the cross-replica redundancy "
        "of 16 similar rings in both terms where it cancels (§3.6 confound 2). r=2 is the "
        "INSTRUMENT-SELECTION rung, not an afterthought: it is the only radius where both poles "
        "were established independently (F62-F70 low-T, §3.5 high-T), so any measure that tracks "
        "complexity must be non-monotone there, and one that is not cannot be believed where the "
        "answer is unknown. §5.3 as drafted had this backwards, calling r in {3,8} the headline; a "
        "probe found the r=3 ring at 21% its top token at T=0.02, so the low-T pole its prediction "
        "rested on is simply absent there. Orientations are declared before the run. The seed is "
        "the independent unit and peaks are judged against between-seed spread, so a peak inside "
        "its own noise is recorded as monotone.")


if __name__ == "__main__":
    main(probe="--probe" in _sys.argv)
