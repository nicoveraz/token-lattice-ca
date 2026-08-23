"""Freeze the self-continuation probe STRINGS, before any model is loaded. Zero forward passes.

WHY THE STRINGS AND NOT THE TOKENS. The candidate fingerprint is a bit vector indexed by token, and
two models only share an index if they share a tokenizer. Freezing token IDs would therefore freeze a
different probe for every model and make the cross-model Hamming distance meaningless. So what is
frozen here is a list of STRINGS; each model's run resolves them against its own tokenizer, and the
comparison set is the intersection -- strings that encode to exactly ONE token everywhere.

WHY IT IS A SEPARATE, HASHED FILE. The intersection is a property of the cohort, so it cannot be
known until the tokenizers are read; but the candidate list must not be. If the strings were chosen
after any model had been measured, a set that happened to separate the decisive pair could have been
selected into existence. The hash plus the commit that introduces this file are what make the
ordering checkable by someone who was not here -- the same discipline as
experiments/newline_margin_freeze.py, which froze its predictions before its census.

THE SOURCE, STATED SO IT REPRODUCES. `NeelNanda/pile-10k` at revision 127bfedcd5047750df5ccf3a12979a47bfa0bafa,
already in this repository's dataset cache and already the corpus of ten existing experiments. The
Pile is also the training corpus of every model in the registered cohort, so the frequent-word arm
probes the region of token space these models were actually fitted on. The rule is:

    N_DOCS = 2000 documents, `train` split, in stored order -- no sampling, no seed
    words   = re.findall(r'[A-Za-z]+', text), CASE-SENSITIVE ('The' and 'the' are different
              BPE tokens and both matter)
    N_WORDS = 2000 most frequent, ties broken lexicographically so the list is total-ordered
    strings = each word bare AND with a leading space, because a leading space is part of the
              token in every BPE tokenizer in the cohort
    plus     all 95 printable ASCII characters, bare and space-prefixed
    plus     a fixed whitespace list -- these are where the funnel endpoints of F172/F179 live

Nothing here is tuned. The two counts are round numbers fixed in advance; no variant was tried.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import collections, hashlib, json, os, re, string

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from provenance import stamp, rel

OUT = _ROOT / "experiments" / "probe_strings_selfcont.json"
SHA = _ROOT / "experiments" / "probe_strings_selfcont.sha256"

DATASET = "NeelNanda/pile-10k"
REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
N_DOCS = 2000
N_WORDS = 2000
WORD_RE = r"[A-Za-z]+"
WHITESPACE = ["\n", "\n\n", "\t", " ", "  ", "\r\n"]


def build():
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train")
    counts = collections.Counter()
    for i in range(N_DOCS):
        counts.update(re.findall(WORD_RE, ds[i]["text"]))
    # ties broken lexicographically: the list must be a function of the corpus, not of dict order
    top = [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:N_WORDS]]

    strings, seen = [], set()
    def add(s, arm):
        if s not in seen:
            seen.add(s)
            strings.append({"s": s, "arm": arm})

    for w in top:
        add(w, "word")
        add(" " + w, "word_sp")
    for c in string.printable[:95]:          # the printable ASCII block, ' ' through '~'
        add(c, "ascii")
        add(" " + c, "ascii_sp")
    for w in WHITESPACE:
        add(w, "whitespace")

    # every candidate is added exactly once, so a single ASCII letter that is also a frequent word
    # is stored under the `word` arm rather than `ascii`. The arm counts below therefore under-report
    # the ASCII block; this asserts the block is complete whatever arm each character landed in.
    have = {e["s"] for e in strings}
    missing = [c for c in string.printable[:95] if c not in have]
    assert not missing, f"printable ASCII characters absent from the probe set: {missing!r}"

    return dict(
        dataset=DATASET, revision=REVISION, n_docs=N_DOCS, n_words=N_WORDS,
        word_regex=WORD_RE, whitespace=WHITESPACE,
        n_distinct_words_seen=len(counts),
        min_count_of_kept_word=int(counts[top[-1]]),
        max_count=int(counts[top[0]]),
        n_strings=len(strings), strings=strings,
        _rule="frozen before any model was loaded; see this file's module docstring for the rule "
              "and experiments/prereg_selfcont.json for what it is used to decide")


def main():
    res = build()
    # the hash covers the STRINGS ONLY, so it is stable against later additions of commentary
    payload = json.dumps([e["s"] for e in res["strings"]], ensure_ascii=False).encode("utf-8")
    res["strings_sha256"] = hashlib.sha256(payload).hexdigest()
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    SHA.write_text(f"{res['strings_sha256']}  probe_strings_selfcont.json:strings  "
                   f"(frozen before any model was loaded)\n")
    arms = collections.Counter(e["arm"] for e in res["strings"])
    print(f"  {res['n_strings']} candidate strings from {res['n_distinct_words_seen']} distinct "
          f"words over {N_DOCS} Pile documents")
    print(f"  arms: {dict(sorted(arms.items()))} -- a single ASCII character that is also a "
          f"frequent word is counted under `word`, so `ascii` under-reports the block; all 95 are "
          f"present and asserted")
    print(f"  kept words range {res['max_count']} down to {res['min_count_of_kept_word']} occurrences")
    print(f"  strings sha256 {res['strings_sha256'][:32]}...")
    print("\nwrote", rel(str(OUT)), "and", rel(str(SHA)))


if __name__ == "__main__":
    main()
