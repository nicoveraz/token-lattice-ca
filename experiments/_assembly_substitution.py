"""Is it the ASSEMBLY INDEX, or just the ensemble construction? The control F74 did not run.

F74 compared Delta against FLAT compression measures -- gzip/lzma/LZ77 applied to the whole text,
contrasted against shuffles. It found 11 of 12 peak on degenerate repetition while Delta peaks on
real text, and I read that as evidence that assembly theory differs from compression.

THAT READING IS TOO STRONG, and this is the control that decides it. Delta stacks four things:

    (1) a per-object complexity  a_i
    (2) an exponential weight    e^{a_i}
    (3) copy-number coupling     (n_i - 1)
    (4) a matched-shuffle contrast

F74 varied NONE of these -- it compared the whole stack against measures that have none of them.
So it establishes that the STACK beats flat compression. It does NOT establish that (1) has to be
the assembly index.

The decisive substitution: keep (2), (3), (4) exactly, and replace a_i with a non-assembly
per-object size. If the ordering survives, the assembly index is doing no work and the honest
claim is about the assembly EQUATION's copy-number coupling, not about the index -- which is much
closer to the critics' position and must be said out loud.

    a_i           RePair assembly index (the real thing)
    len_i         character length of the object    <- the InChI-length confound, per-object
    z_i           LZ77 phrase count of the object
    words_i       constant (=n), i.e. no complexity term at all -- pure copy number
"""
import math, re, random, collections, statistics, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from assembly_calib import repair_assembly_index
from assembly_baselines import lz77_phrases

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORD = re.compile(r"[a-z0-9']+")
FLOOR = -3.0
N_GRAM, K_SHUF, BUDGET = 3, 8, 440


def weights():
    return {
        "a_i  (assembly index)": lambda g: repair_assembly_index(g),
        "len_i (char length)":   lambda g: len(g),
        "z_i  (LZ77 phrases)":   lambda g: lz77_phrases(g),
        "const (pure copy num)": lambda g: 1.0,
    }


def A(words, wfn, n=N_GRAM):
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if len(grams) < 2:
        return 0.0
    c = collections.Counter(grams)
    w = [math.exp(wfn(g)) * (v - 1) for g, v in c.items() if v >= 2]
    return (sum(w) / len(grams)) if w else 0.0


def lg(a):
    return math.log10(a) if a and a > 0 else FLOOR


def delta(words, wfn, seed=0):
    rng = random.Random(seed)
    got = lg(A(words, wfn))
    nulls = []
    for _ in range(K_SHUF):
        s = words[:]
        rng.shuffle(s)
        nulls.append(lg(A(s, wfn)))
    return got - statistics.fmean(nulls)


if __name__ == "__main__":
    corpus = (ROOT / "data" / "shakespeare.txt").read_text(errors="replace")
    rng = random.Random(1)
    base = WORD.findall(corpus[:40000].lower())[:BUDGET]

    regimes = {
        "real text":      base,
        "degenerate x1":  ["the"] * BUDGET,
        "degenerate x2":  ["the", "of"] * (BUDGET // 2),
        "random soup":    [rng.choice(base) for _ in range(BUDGET)],
        "unique tokens":  [f"w{i}" for i in range(BUDGET)],
    }

    print("=" * 84)
    print(f"SUBSTITUTION CONTROL -- same (2)(3)(4), different per-object term. {BUDGET} words, "
          f"{N_GRAM}-grams")
    print("=" * 84)
    hdr = f"  {'regime':16s}" + "".join(f"{k:>23s}" for k in weights())
    print(hdr)
    table = {}
    for name, w in regimes.items():
        row = {k: delta(w, fn) for k, fn in weights().items()}
        table[name] = row
        print(f"  {name:16s}" + "".join(f"{row[k]:>+23.2f}" for k in weights()))

    print("\n  peaks on:")
    for k in weights():
        best = max(table, key=lambda r: table[r][k])
        verdict = "REAL TEXT" if best == "real text" else best.upper()
        print(f"    {k:24s} -> {verdict}")

    print("\n  VERDICT")
    a_best = max(table, key=lambda r: table[r]["a_i  (assembly index)"])
    others = {k: max(table, key=lambda r: table[r][k]) for k in weights()
              if k != "a_i  (assembly index)"}
    same = [k for k, v in others.items() if v == a_best]
    if same:
        print(f"    The ordering SURVIVES substitution for: {same}")
        print(f"    -> the assembly INDEX is not what is doing the work; the ensemble")
        print(f"       construction (exponential x copy number x shuffle contrast) is.")
    else:
        print(f"    No substitution reproduces the assembly index's peak on {a_best!r}.")
        print(f"    -> the per-object assembly index IS load-bearing.")
