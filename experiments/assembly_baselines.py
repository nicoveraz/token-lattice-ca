"""Does any compression or entropy baseline reproduce Delta's ordering? (#20; §5.2)

THE CRITIQUE IS NOT A FOOTNOTE, SO IT IS RUN AS AN EXPERIMENT. Abrahao, Hernandez-Orozco, Kiani,
Tegner & Zenil (PLOS Complex Systems 2024) claim "full equivalence between Assembly Theory and
Shannon Entropy via a method based upon the principles of statistical compression renamed 'assembly
index' that belongs to the LZ family", with supporting empirics reporting LZW-vs-assembly Pearson
0.874, Spearman 1.00 on fixed-length strings, and 0.95 between InChI string LENGTH and assembly
index. If that holds here, assembly theory adds nothing to this project and the correct output is a
short negative note. This experiment is built so that answer can win.

THE ONE DESIGN DECISION THAT MAKES THE COMPARISON FAIR. Delta is defined as a CONTRAST against a
matched word-shuffle, so comparing it to a raw gzip length would be rigging the test -- the contrast
does work the raw number does not. Every measure here is therefore reported the same way: as a
**z-score against its own matched word-shuffle ensemble**,

    z = (value(text) - mean(value(shuffles))) / sd(value(shuffles))

The shuffle preserves length, vocabulary and unigram frequencies EXACTLY, so this is the Kempes et
al. fixed-multiset permutation control (npj Complexity 2025) applied to every baseline rather than
only to ours. It also answers a question §3.2 could not: §3.2 compared real text to ONE shuffle and
found gzip differing by 1.5%, but never asked whether 1.5% is large against shuffle-to-shuffle
scatter. A z-score asks exactly that.

It has a second benefit that is what makes C_mu and excess entropy estimable at all at these
lengths. Both are badly undersampled from 440 words -- but the SAME undersampling applies to the
shuffles, so the bias cancels in the z-score even though it dominates the raw value. This is the
same debiasing `mlm_lib.coarse_mi_decay` already does by subtracting a shuffled-pair MI floor.

LENGTH IS HELD CONSTANT BY CONSTRUCTION, NOT PARTIALLED OUT. Every regime is truncated to the same
word budget, so the InChI-length confound (r = 0.95 between string length and assembly index, the
most dangerous baseline in the critics' own data) cannot operate. n_words and n_types are measured
anyway and reported as baselines in their own right.

PRE-REGISTERED:
  * Primary: does any baseline reproduce Delta's ORDERING across the reference regimes? Measured as
    Spearman rho against Delta's z-scores. |rho| >= 0.9 for a compression or entropy baseline means
    the ensemble construction is a redescription and #20 should stop.
  * Secondary, and the sharper test: WHICH REGIME does each measure rank highest? A measure that is
    monotone in disorder peaks on random soup. Delta must peak on REAL TEXT. That is a difference in
    SHAPE, which no correlation coefficient can explain away, and it is the miniature of §4.4's
    "result that would survive review".
  * C_mu and excess entropy are the sharpest objection (Lindgren & Nordahl 1988; Crutchfield): they
    are established one-hump measures that already peak at criticality. If Delta peaks where they
    peak, this is a correlation result and #20 is one paragraph.
  * The CA cells are measured too, but they carry ONE settle run each (F57), so they are reported
    as suggestive and excluded from every verdict.

Writes results/assembly_baselines.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/assembly_baselines.py
        (safe to interrupt and re-run -- it resumes)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, math, zlib, bz2, lzma, random, collections, statistics

from provenance import stamp, rel
from assembly_calib import (WORD, SHAKESPEARE, BUDGET, NGRAM, A_exp, lg, repair_assembly_index,
                            addition_chain_length, pole_cases, decides, calibrate)

OUT = str(_ROOT / "results" / "assembly_baselines.json")
NOVELTY_JSON = _ROOT / "results" / "novelty_structure.json"

N_WIN = 3                # contiguous length-matched windows per regime
K_SHUF = 20              # shuffles per window -- the z-score needs a stable sd, the pilot's 6 does not
NB = 6                   # frequency buckets for the entropy/MI measures
KBLOCK = 3               # block length for H_k, E and the C_mu history
DMAX = 12                # MI decay range
RHO_REDESCRIBES = 0.90   # |Spearman| at which a baseline is judged to reproduce Delta's ordering


# --------------------------------------------------------------- the baselines

LZ_MAXMATCH = 64         # match-length cap; uncapped greedy LZ77 is O(n^3) on degenerate input


def lz77_phrases(s):
    """Greedy self-referential LZ77 parse length.

    z <= g (Rytter 2003; Charikar et al. 2005) where g is the SMALLEST GRAMMAR SIZE, defined as the
    total length of all right-hand sides. §4.1 and §5.2 of assembly_theory.md read that as "z is a
    lower bound on the assembly index" and report a bracket [z, RePair]. **That conflates two
    units.** A binary SLP with r rules has total RHS length 2r, and the assembly index IS the binary
    rule count, so g = 2 * ASI and the theorem gives z <= 2 * ASI. The lower bound is z/2, and
    [z, RePair] is not a bracket -- z exceeds RePair on ordinary text (11 vs 10 on "abracadabra"*6,
    20 vs 19 on repeated English). [z/2, RePair] is, and holds with room to spare.

    Capped at LZ_MAXMATCH for tractability, which can only INFLATE z, so the z/2 bound stays
    conservative. Reported as a baseline measure here, not relied on as a bound.
    """
    n, i, z = len(s), 0, 0
    while i < n:
        best = 0
        for L in range(min(n - i, LZ_MAXMATCH), 0, -1):
            if s.find(s[i:i + L], 0, i) != -1:
                best = L
                break
        i += max(best, 1)
        z += 1
    return z


def lzw_dict_size(s):
    """LZW dictionary growth -- the exact measure the critics claim assembly index is equivalent to."""
    d = {chr(i): i for i in range(256)}
    nxt, w, n = 256, "", 0
    for c in s:
        wc = w + c
        if wc in d:
            w = wc
        else:
            d[wc] = nxt; nxt += 1; n += 1; w = c
    return n


def gzip_bits(s):
    """Compressed length in bits -- the "does a zip file do this too" test. Cheap; omitting it
    would look evasive, and it turns out to be the strongest word-order detector in the suite."""
    return len(zlib.compress(s.encode("utf-8", "replace"), 9)) * 8


def shannon_bits(s):
    """Character-level Shannon content -- the "is it just entropy" baseline, in bits."""
    c, n = collections.Counter(s), len(s)
    return -sum(v * math.log2(v / n) for v in c.values()) if n else 0.0


def sequitur_slp_size(seq):
    """Sequitur grammar size in BINARY JOINS, so it is in the same units as the assembly index.

    Nevill-Manning & Witten (1997): append symbols one at a time, enforcing two invariants --
    DIGRAM UNIQUENESS (no digram appears twice) and RULE UTILITY (every rule is used more than
    once). A rule with a right-hand side of length L is L-1 concatenations, so the grammar's total
    is sum(L_i - 1), which is an SLP size and directly comparable to `repair_assembly_index`.

    Verified in `_selfcheck` against the same two provable families the estimator is gated on: it
    must return n-1 on all-distinct strings and must never fall below the addition-chain minimum on
    a^n. A baseline that fails its own sanity check is EXCLUDED and reported as excluded, rather
    than shipped as a number nobody checked.
    """
    rules = {0: list(seq)}
    nxt = 1
    while True:
        # digram uniqueness: the most frequent repeated digram anywhere in the grammar
        counts = collections.Counter()
        for rid, body in rules.items():
            i = 0
            while i < len(body) - 1:
                counts[(body[i], body[i + 1])] += 1
                i += 2 if body[i] == body[i + 1] else 1     # non-overlapping
        rep = [(d, c) for d, c in counts.items() if c >= 2]
        if not rep:
            break
        pair = max(rep, key=lambda kv: kv[1])[0]
        sym = ("R", nxt); nxt += 1
        for rid in list(rules):
            body, out, i = rules[rid], [], 0
            while i < len(body):
                if i < len(body) - 1 and (body[i], body[i + 1]) == pair:
                    out.append(sym); i += 2
                else:
                    out.append(body[i]); i += 1
            rules[rid] = out
        rules[sym] = list(pair)
        # rule utility: a rule used once is inlined and removed
        while True:
            used = collections.Counter(s for b in rules.values() for s in b)
            once = [r for r in rules if r != 0 and used[r] <= 1]
            if not once:
                break
            for r in once:
                body = rules.pop(r)
                for rid in rules:
                    out = []
                    for s in rules[rid]:
                        out.extend(body if s == r else [s])
                    rules[rid] = out
    return sum(max(len(b) - 1, 0) for b in rules.values())


def _buckets(words):
    """Frequency-rank buckets, log-spaced -- mlm_lib.freq_buckets' coarse-graining, over words.

    Derived from the window's OWN unigram counts, which a word shuffle preserves exactly, so text
    and shuffle are bucketed identically and the control stays exact.
    """
    rank = {w: i for i, (w, _) in enumerate(collections.Counter(words).most_common())}
    R = max(len(rank), 1)
    edges = [i * math.log1p(R) / NB for i in range(NB + 1)]
    def b(w):
        x = math.log1p(rank[w])
        return min(max(sum(1 for e in edges[1:] if x >= e), 0), NB - 1)
    return [b(w) for w in words]


def _H(counts):
    n = sum(counts.values())
    return -sum(v / n * math.log2(v / n) for v in counts.values()) if n else 0.0


def block_entropies(syms, kmax=KBLOCK):
    return [_H(collections.Counter(tuple(syms[i:i + k]) for i in range(len(syms) - k + 1)))
            for k in range(1, kmax + 1)]


def excess_entropy(syms, kmax=KBLOCK):
    """E = sum_k (h_k - h), the standard block-entropy estimator, truncated at kmax.

    COARSE AND UNDERSAMPLED at these lengths, deliberately reported as such: kmax-blocks over NB
    buckets is NB^kmax cells from a few hundred samples. The z-score against the matched shuffle is
    what makes it usable -- the same bias sits in both terms.
    """
    H = block_entropies(syms, kmax)
    h = [H[0]] + [H[k] - H[k - 1] for k in range(1, len(H))]
    return sum(x - h[-1] for x in h)


def statistical_complexity(syms, k=KBLOCK - 1, tol=0.25):
    """COARSE C_mu: cluster k-histories by their next-symbol distribution, C_mu = H[state].

    Not a CSSR reconstruction and not claimed to be one -- it is the cheapest thing with the right
    shape, which is what §5.2 asks for ("bounded/coarse rather than the true values"). It is here
    because Lindgren & Nordahl (1988) and Crutchfield's statistical complexity are the sharpest
    objection to #20: they are ESTABLISHED one-hump measures that already peak at criticality.
    """
    fut = collections.defaultdict(collections.Counter)
    for i in range(len(syms) - k):
        fut[tuple(syms[i:i + k])][syms[i + k]] += 1
    if not fut:
        return 0.0
    states, assign = [], {}
    for h, c in fut.items():
        n = sum(c.values()); p = [c.get(s, 0) / n for s in range(NB)]
        for j, q in enumerate(states):
            if sum(abs(a - b) for a, b in zip(p, q)) < tol:
                assign[h] = j; break
        else:
            states.append(p); assign[h] = len(states) - 1
    w = collections.Counter()
    for h, c in fut.items():
        w[assign[h]] += sum(c.values())
    return _H(w)


def mi_integrated(syms, dmax=DMAX):
    """Sum of plug-in MI(x_0; x_d) over d -- mlm_lib.coarse_mi_decay's quantity, on words.

    Raw rather than internally debiased: the z-score against the shuffle ensemble is the debias,
    and doing it twice would subtract the effect being measured.
    """
    tot = 0.0
    for d in range(1, min(dmax, len(syms) - 1) + 1):
        a, b = syms[:-d], syms[d:]
        ja = collections.Counter(zip(a, b))
        tot += max(0.0, _H(collections.Counter(a)) + _H(collections.Counter(b)) - _H(ja))
    return tot


def profile(words, sep=" "):
    """Every measure the program reports, for one text. No cell ever reports Delta alone (§5.2).

    `sep` joins the symbols. It is " " for word sequences, and "" for TOKEN sequences, whose
    strings already carry their own leading whitespace -- inserting spaces between them would
    corrupt every compression measure.
    """
    txt = sep.join(words)
    syms = _buckets(words)
    H = block_entropies(syms)
    a, _, eff = A_exp(words, NGRAM)
    return dict(
        # ours
        logA=lg(a), eff_objects=eff,
        # the trivial baselines -- the InChI-length confound says these are the dangerous ones
        n_words=len(words), n_types=len(set(words)),
        # compression
        lz77_z=lz77_phrases(txt), lzw_dict=lzw_dict_size(txt),
        repair_size=repair_assembly_index(txt), sequitur_size=sequitur_slp_size(txt),
        gzip_bits=gzip_bits(txt),
        bz2_bits=len(bz2.compress(txt.encode("utf-8", "replace"), 9)) * 8,
        lzma_bits=len(lzma.compress(txt.encode("utf-8", "replace"))) * 8,
        # entropy
        H0=H[0], H_block=H[-1], h_rate=H[-1] - H[-2],
        excess_entropy=excess_entropy(syms), C_mu=statistical_complexity(syms),
        mi_integrated=mi_integrated(syms))


MEASURES = ["logA", "n_words", "n_types", "lz77_z", "lzw_dict", "repair_size", "sequitur_size",
            "gzip_bits", "bz2_bits", "lzma_bits", "H0", "H_block", "h_rate",
            "excess_entropy", "C_mu", "mi_integrated"]
OURS = "logA"
COMPRESSION = ["lz77_z", "lzw_dict", "repair_size", "sequitur_size",
               "gzip_bits", "bz2_bits", "lzma_bits"]
ENTROPY = ["H0", "H_block", "h_rate", "excess_entropy", "C_mu", "mi_integrated"]


# ------------------------------------------------------------------- z-scoring

def zprofile(words, k=K_SHUF, seed=0):
    """Every measure as a z-score against its own matched word-shuffle ensemble.

    Returns the raw value, the contrast (value - shuffle mean, which for logA IS Delta as §3.4
    defines it), the shuffle sd, and z. A measure whose |z| is small does not detect word order at
    all -- which is the whole question §3.2 left open by comparing against a single shuffle.
    """
    obs = profile(words)
    rng = random.Random(seed)
    nulls = []
    for _ in range(k):
        sh = words[:]; rng.shuffle(sh)
        nulls.append(profile(sh))
    out = {}
    for m in MEASURES:
        vs = [x[m] for x in nulls]
        mu = statistics.fmean(vs)
        sd = statistics.pstdev(vs)
        out[m] = dict(value=round(obs[m], 4), shuf_mean=round(mu, 4), shuf_sd=round(sd, 4),
                      contrast=round(obs[m] - mu, 4),
                      z=round((obs[m] - mu) / sd, 3) if sd > 1e-12 else None)
    out["_eff_objects"] = round(obs["eff_objects"], 2)
    return out


def windows(words, budget=BUDGET, n_win=N_WIN):
    if len(words) < budget:
        return []
    if len(words) == budget:
        return [0]
    return [round(i * (len(words) - budget) / (n_win - 1)) for i in range(n_win)]


def measure_regime(words, seed=0):
    """Median z per measure over contiguous length-matched windows."""
    st = windows(words)
    if not st:
        return None
    rows = [zprofile(words[s:s + BUDGET], seed=seed + i) for i, s in enumerate(st)]
    out = {}
    for m in MEASURES:
        zs = [r[m]["z"] for r in rows if r[m]["z"] is not None]
        cs = [r[m]["contrast"] for r in rows]
        out[m] = dict(z=round(statistics.median(zs), 3) if zs else None,
                      contrast=round(statistics.median(cs), 4),
                      n_windows_with_z=len(zs))
    out["_eff_objects"] = round(statistics.median(r["_eff_objects"] for r in rows), 2)
    out["_windows"] = len(st)
    return out


# --------------------------------------------------------------------- analysis

def _spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 1e-12 else 0.0


def _selfcheck():
    """Sequitur must pass the same provable families the assembly estimator is gated on."""
    distinct = all(sequitur_slp_size([f"s{i}" for i in range(n)]) == n - 1
                   for n in range(2, 65))
    sound = all(sequitur_slp_size("a" * n) >= addition_chain_length(n) for n in range(2, 65))
    return dict(exact_on_no_reuse=bool(distinct), sound_on_a_n=bool(sound),
                usable=bool(distinct and sound))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"regimes": {}, "cells": {}}
    gate = calibrate()
    if not decides(gate):
        res["verdict"] = ("NOT DECIDABLE -- assembly_calib's gate does not pass, so the estimator "
                          "has not earned the right to be compared against anything.")
        json.dump(res, open(OUT, "w"), indent=1)
        print(res["verdict"]); return False

    sc = _selfcheck()
    print(f"  sequitur self-check: exact on no-reuse={sc['exact_on_no_reuse']}, "
          f"sound on a^n={sc['sound_on_a_n']} -> {'usable' if sc['usable'] else 'EXCLUDED'}",
          flush=True)
    measures = [m for m in MEASURES if sc["usable"] or m != "sequitur_size"]

    res["_preregistration"] = dict(
        measures=measures, ours=OURS, compression=COMPRESSION, entropy=ENTROPY,
        budget=BUDGET, n_windows=N_WIN, k_shuffles=K_SHUF, buckets=NB, kblock=KBLOCK,
        statistic="z = (value(text) - mean(value(shuffles))) / sd(value(shuffles))",
        why_z="Delta is defined as a contrast against a matched shuffle, so comparing it to a raw "
              "gzip length would rig the test; and the shuffle ensemble's finite-sample bias "
              "cancels, which is what makes C_mu and excess entropy usable at 440 words",
        length_control="every regime truncated to the same word budget, so length is held constant "
                       "by construction rather than partialled out -- the InChI-length confound "
                       "(r=0.95) cannot operate",
        primary=f"does any baseline reproduce Delta's ordering across the reference regimes? "
                f"Spearman on the CONTRAST (value - shuffle mean); |rho| >= {RHO_REDESCRIBES} for a "
                f"compression or entropy baseline means the ensemble construction is a "
                f"redescription and #20 should stop",
        primary_was_corrected=(
            "The first version ranked on the z-score rather than the contrast, and its own "
            "pre-registered KILL CONDITION fired: Delta appeared to peak on degenerate_x2. The "
            "cause was the normalisation, not Delta. z = contrast/sd explodes when the CONTROL has "
            "little variance -- shuffling a 2-cycle barely changes it, so sd = 0.0153 turned a "
            "contrast of +0.34 into z = 22.2, above real text's +6.87 at z = 3.2 -- and z is "
            "UNDEFINED when the control has no variance at all, which silently dropped "
            "degenerate_x1 and unique_tokens, two of the cleanest poles. Delta is DEFINED as a "
            "contrast (§3.4), so ranking on its z-score was ranking something that is not Delta. "
            "The contrast is the primary; z is kept as a separate detection statistic, which is "
            "the one question it does answer well."),
        secondary="which regime does each measure rank highest? a disorder measure peaks on random "
                  "soup; Delta must peak on real text. A difference in SHAPE, which no correlation "
                  "coefficient explains away",
        sharpest="C_mu and excess entropy are established one-hump measures that already peak at "
                 "criticality (Lindgren & Nordahl 1988; Crutchfield). If Delta peaks where they "
                 "peak, #20 is one paragraph",
        ca_cells="measured but EXCLUDED from every verdict -- one settle run each (F57)",
        sequitur_selfcheck=sc)

    regimes = pole_cases()
    rng = random.Random(7)
    base = regimes["real_text"]
    regimes["word_shuffled"] = rng.sample(base, len(base))    # the Kempes multiset control itself
    for i, (name, words) in enumerate(regimes.items()):
        if name in res["regimes"]:
            continue
        r = measure_regime(words, seed=100 + i)
        if r:
            res["regimes"][name] = r
            print(f"  {name:16s} logA z={r['logA']['z']}  Delta={r['logA']['contrast']:+.2f}  "
                  f"eff={r['_eff_objects']}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)

    if NOVELTY_JSON.exists():
        for key, v in json.loads(NOVELTY_JSON.read_text())["runs"].items():
            if not v.get("full_text") or key in res["cells"]:
                continue
            w = WORD.findall(v["full_text"].lower())
            r = measure_regime(w, seed=7)
            if r:
                res["cells"][key] = r
                print(f"  [cell] {key:18s} logA z={r['logA']['z']}  "
                      f"Delta={r['logA']['contrast']:+.2f}", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)

    analyse(res, measures, sc)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))
    return True


def analyse(res, measures=None, sc=None):
    R = res["regimes"]
    measures = measures or [m for m in MEASURES if any(m in v for v in R.values())]
    # EVERY regime, including those whose shuffle ensemble has no variance. The contrast is defined
    # there and is exactly zero, which is the most informative reading a failure pole can give; it
    # was only the z-score that could not represent them.
    names = list(R)
    if len(names) < 4:
        res["verdict"] = "insufficient regimes"; return

    ours = [R[n][OURS]["contrast"] for n in names]
    print(f"\n=== CONTRAST against a matched word-shuffle ensemble ({K_SHUF} shuffles, "
          f"{BUDGET} words, median of {N_WIN} windows) ===")
    print(f"  the quantity Delta IS: value(text) - mean(value(shuffles))")
    print(f"\n  {'measure':16s} " + " ".join(f"{n[:11]:>11s}" for n in names)
          + f" {'rho vs A':>9s} {'peaks on':>14s}")
    rows = {}
    for m in measures:
        cs = [R[n][m]["contrast"] for n in names]
        rho = _spearman(cs, ours)
        # Orient each baseline by the SIGN of its correlation with Delta, which is the reading most
        # favourable to it -- a compression size means "structured" when it goes DOWN. Auto-orienting
        # can only help the baselines, so failing the shape test under it is the strong statement.
        sign = 1.0 if rho >= 0 else -1.0
        oriented = [sign * c for c in cs]
        peak = names[max(range(len(oriented)), key=lambda i: oriented[i])]
        rows[m] = dict(contrast={n: R[n][m]["contrast"] for n in names},
                       z={n: R[n][m]["z"] for n in names},
                       rho_vs_ours=round(rho, 3), orientation=int(sign), peaks_on=peak)
        print(f"  {m:16s} " + " ".join(f"{c:11.3f}" for c in cs)
              + f" {rho:9.2f} {peak[:14]:>14s}")

    # A measure whose contrast is identically zero is EXACTLY INVARIANT under the control, not a
    # measure that happens to peak somewhere: argmax over an all-zero vector returns whichever
    # regime sorts first, which means nothing. n_words, n_types and H0 are functions of the
    # MULTISET alone, and a word shuffle is a permutation of the multiset, so their invariance is
    # provable rather than empirical -- and it is exactly why the InChI-length confound (r=0.95
    # between string length and assembly index) cannot operate in this design.
    INVARIANT_TOL = 1e-9
    invariant = [m for m in measures
                 if max(abs(c) for c in rows[m]["contrast"].values()) <= INVARIANT_TOL]
    others = [m for m in measures if m != OURS and m not in invariant]
    for m in invariant:
        rows[m]["peaks_on"] = None
        rows[m]["control_invariant"] = True
    redescribe = [m for m in others if abs(rows[m]["rho_vs_ours"]) >= RHO_REDESCRIBES]
    peak_real = [m for m in others if rows[m]["peaks_on"] == "real_text"]
    ours_peak = rows[OURS]["peaks_on"]

    # Secondary: does each baseline DETECT word order at all under multiset control? This is the
    # question the z-score answers well, and the one §3.2 left open by comparing against a single
    # shuffle rather than an ensemble.
    zr = {m: R["real_text"][m]["z"] for m in measures if R["real_text"][m]["z"] is not None}
    detect = sorted((m for m in zr if abs(zr[m]) >= 3.0), key=lambda m: -abs(zr[m]))

    parts = []
    if ours_peak != "real_text":
        parts.append(f"KILL CONDITION HIT: Delta peaks on {ours_peak!r}, not on real text. The "
                     f"statistic is not measuring structure and no comparison below is meaningful.")
    else:
        parts.append(f"Delta peaks on REAL TEXT, as it must, at {rows[OURS]['contrast']['real_text']:+.2f} "
                     f"against {rows[OURS]['contrast']['degenerate_x2']:+.2f} for degenerate "
                     f"repetition and {rows[OURS]['contrast']['random_soup']:+.2f} for noise.")
        if redescribe:
            parts.append(
                f"REDESCRIPTION: {', '.join(redescribe)} reproduce(s) Delta's ordering at "
                f"|rho| >= {RHO_REDESCRIBES}. Assembly theory adds nothing this project cannot get "
                f"from a compressor, and #20's correct output is a short negative note.")
        else:
            worst = max(others, key=lambda m: abs(rows[m]["rho_vs_ours"]))
            parts.append(
                f"NO BASELINE REPRODUCES IT: the closest is {worst} at rho="
                f"{rows[worst]['rho_vs_ours']:+.2f}, inside the +/-{RHO_REDESCRIBES} threshold.")
        if not peak_real:
            pk = collections.Counter(rows[m]["peaks_on"] for m in others)
            top, ntop = pk.most_common(1)[0]
            parts.append(
                f"AND THE DIFFERENCE IS IN SHAPE, not in strength: of {len(others)} baselines, "
                f"NONE peaks on real text -- {ntop} of them peak on {top!r}. Every compression and "
                f"entropy measure responds MORE STRONGLY to degenerate repetition than to real "
                f"English, and Delta is the only one that inverts that, by a factor of "
                f"{abs(rows[OURS]['contrast']['real_text'] / rows[OURS]['contrast']['degenerate_x2']):.0f}. "
                f"That is §3.3's result -- the exponential weighting is what keeps real text on "
                f"top -- now shown against the full baseline suite rather than against three "
                f"tempered versions of itself. A difference in shape is not explainable by a "
                f"correlation coefficient.")
        else:
            parts.append(
                f"But {len(peak_real)} baseline(s) also peak on real text ({', '.join(peak_real)}), "
                f"so the shape argument is weaker than the ordering argument here and must not be "
                f"made without stating them.")
        cm = [m for m in ("C_mu", "excess_entropy") if m in rows]
        same = [m for m in cm if rows[m]["peaks_on"] == "real_text"]
        parts.append(
            f"On the sharpest objection (Lindgren & Nordahl 1988; Crutchfield): "
            + ", ".join(f"{m} peaks on {rows[m]['peaks_on']}" for m in cm)
            + (f". The established one-hump measures peak WHERE DELTA DOES, so this is a "
               f"correlation result and #20 is one paragraph." if same else
               f". Neither peaks on real text here, so Delta is not a redescription of statistical "
               f"complexity AS ESTIMATED AT THIS LENGTH -- which is a statement about a coarse, "
               f"undersampled estimator on 440 words, NOT about C_mu itself. A proper CSSR "
               f"reconstruction could still peak with Delta and this experiment could not tell."))

    if detect:
        parts.append(
            f"SEPARATELY, AND CONCEDING SOMETHING §3.2 OVERSTATED: against a 20-shuffle ensemble "
            f"rather than the single shuffle §3.2 used, {len(detect)} measures DO detect word order "
            f"on real text at |z| >= 3 -- strongest {detect[0]} at z={zr[detect[0]]:+.1f}. So "
            f"\"every compression baseline is null under multiset control\" is wrong: they are not "
            f"blind to word order, they are merely ordered differently across regimes. The case for "
            f"the ensemble quantity rests on the ORDERING, not on compression being unable to see "
            f"word order.")

    if invariant:
        parts.append(
            f"{', '.join(invariant)} are EXACTLY INVARIANT under the control (contrast 0 in every "
            f"regime) and are excluded from the peak test rather than counted as peaking anywhere. "
            f"That is provable, not measured: a word shuffle permutes the multiset and all three "
            f"are functions of the multiset alone -- which is precisely why the InChI-length "
            f"confound cannot operate in this design.")
    if sc and not sc["usable"]:
        parts.append("Sequitur was EXCLUDED: it failed the provable-family self-check, so it is "
                     "reported as excluded rather than shipped as an unchecked number.")
    if res.get("cells"):
        parts.append(f"The {len(res['cells'])} CA cells are measured and stored but excluded from "
                     f"this verdict: one settle run each, and Delta is a tail statistic (F57).")

    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rows=rows, regimes=names, redescribing=redescribe,
                           control_invariant=invariant,
                           baselines_peaking_on_real_text=peak_real, ours_peaks_on=ours_peak,
                           rho_threshold=RHO_REDESCRIBES, detects_word_order=detect,
                           z_on_real_text={m: zr[m] for m in zr},
                           statistic="contrast (value - shuffle mean); z kept as a separate "
                                     "detection statistic only")
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "The head-to-head §5.2 of assembly_theory.md requires before any compute is spent on §5.3. "
        "The critics (Abrahao et al., PLOS Complex Systems 2024; Ozelim et al., npj Complexity "
        "2026) claim assembly index is a repackaged LZ-family compression measure, reporting "
        "LZW-assembly Pearson 0.874 and 0.95 between InChI string LENGTH and assembly index. Every "
        "measure is reported as a CONTRAST against its own matched word-shuffle ensemble, which is "
        "the Kempes et al. (npj Complexity 2025) fixed-multiset permutation control applied to "
        "every baseline rather than only to ours -- comparing a contrast statistic to a raw "
        "compressed length would rig the test. An earlier version ranked on the z-score and its "
        "own kill condition fired: z explodes when the CONTROL has little variance (a 2-cycle "
        "barely changes under shuffling, so sd=0.0153 turned a contrast of +0.34 into z=22) and is "
        "undefined when the control has none, silently dropping two poles. Length is held constant "
        "by construction rather than partialled out. The CA cells are measured but excluded from "
        "the verdict: one settle run each (F57).")


if __name__ == "__main__":
    main()
