"""Every number in paper 2 must trace to a results file, or be declared not a measurement.

WHY THIS EXISTS. `paper2_arxiv/main.tex` carries the same convention as paper 1 -- every number has a
`% source comment` naming its finding and results file -- and the same convention needs the same
guard. `tests/test_arxiv_paper_numbers.py` does this for paper 1; without a twin, paper 2's numbers
were governed by a comment convention and nothing that could fail.

WHICH DIRECTION THIS RUNS, and it is the harder one. Not "every derived number appears in the paper"
-- that catches a results file drifting under a static paper. This asserts the converse: every
decimal literal in the manuscript is present in some results file. That is the property making an
invented number impossible rather than merely unlikely.

MATCHING IS BOUNDARY-AWARE. A plain substring check lets `0.80` be satisfied by `0.801`, so a paper
could quote a number no run produced and still pass. The helpers are imported from paper 1's test
rather than re-implemented, so the two papers cannot drift apart in what "traces" means and a fix to
the matcher fixes both.

THE ALLOWLIST IS WHERE THIS PAPER DIFFERS FROM PAPER 1, and the reason is structural. Paper 2 argues
against a literature, so it quotes that literature's numbers -- perplexities from the attention-sink
papers, accuracies from the prompt-ordering work, a rate from the format-sensitivity study. Those are
NOT derivable from this project's runs and must not be: a number from someone else's paper appearing
in our results/ would mean we had copied it into a results file. Each is listed below with its source
named, which is the same discipline the citation ledger applies to quotes.
"""
import pathlib
import re

import pytest

from test_arxiv_paper_numbers import _numbers, _present, _results_blob

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = ROOT / "paper2_arxiv" / "main.tex"

pytestmark = pytest.mark.skipif(not TEX.exists(), reason="paper2_arxiv/main.tex not present")

# Literals that are not measurements of ours. Every entry names WHOSE number it is and where it is
# used. An entry with an empty reason fails the second test below.
ALLOWED = {
    # ---- external results, quoted while arguing with their authors ----
    "5.60": "Llama-2-13B perplexity with the first four tokens replaced by the linebreak token, from "
            "xiao2023streamingllm, quoted in the sink section. Their measurement, not ours.",
    "5.40": "Llama-2-13B perplexity with the original first four tokens (xiao2023streamingllm), the "
            "comparison point for 5.60. Their measurement, not ours.",
    "5158.07": "Llama-2-13B perplexity with those positions dropped (xiao2023streamingllm) -- the "
               "third leg of their contrast, showing the collapse. Their measurement, not ours.",
    "88.7": "Accuracy under a good permutation in lu2022ordered, quoted in the pairwise section. "
            "Their measurement.",
    "51.6": "Accuracy for the SAME permutation on a different model size in lu2022ordered -- the "
            "number that makes their point about non-transferability. Their measurement.",
    "32.4": "Rate at which graded atomic format changes give monotone accuracy triples in "
            "sclar2024format, quoted in the length section. Theirs.",
    "33.6": "The second of sclar2024format's two monotone-triple rates. Theirs.",
    "33.3": "The chance rate sclar2024format's 32.4 and 33.6 are compared against. Theirs, and a "
            "combinatorial constant rather than a measurement at all.",
    "0.238": "Kendall's W across all models in cao2024worstprompt, quoted in intro-known. Theirs.",
    "0.95": "Top-p setting of the transmission-chain work (telephone), quoted when distinguishing "
            "their regime from ours. A decoding hyperparameter of someone else's experiment.",
    "2.7": "The 2.7B model size in lu2022ordered, written as `$2.7$B`. A parameter count from "
           "another paper, not a measurement.",
}


def _body():
    """Strip comments, but NOT escaped percent signs -- and that distinction is load-bearing.

    Paper 1's `_body` uses `(?m)%.*$`, which also eats `\%`. In a line reading
    `$32.4\%$ and $33.6\%$ ... against a $33.3\%$ chance rate`, everything from the FIRST `\%`
    onward is deleted, so `33.3` is never extracted and never checked. Any number written after a
    percent sign on the same line is therefore invisible to that guard. Paper 2 quotes several such
    numbers, so the stripper here requires the `%` to be unescaped, matching what
    `test_paper2_citations.py` already does.
    """
    t = TEX.read_text()
    t = re.sub(r'(?m)(?<!\\)%.*$', '', t)              # comments carry SOURCE notes, not claims
    t = re.sub(r'\\(cite[a-z]*|ref|label|input|include|usepackage|documentclass)\{[^}]*\}', ' ', t)
    return t


def test_every_decimal_literal_in_paper2_traces_to_a_results_file():
    body, blob = _body(), _results_blob()
    lits = sorted(set(re.findall(r'(?<![\w.])\d+\.\d+(?![\w])', body)))
    assert lits, "no decimal literals found -- the extractor is broken, not the paper"
    pool = _numbers(blob)
    missing = [l for l in lits if l not in ALLOWED and not _present(l, blob, pool)]
    assert not missing, (
        f"{len(missing)} of {len(lits)} decimal literals in paper2_arxiv/main.tex do not appear in "
        f"any results/*.json under boundary-aware matching:\n  " + "\n  ".join(missing) +
        "\nEither the number was not derived from a run, or the run that produced it was never "
        "committed. If it is genuinely not a measurement of ours, add it to ALLOWED with a reason "
        "naming whose number it is.")


def test_allowed_entries_all_carry_a_reason():
    """An allowlist without reasons decays into a way to silence the test."""
    bad = [k for k, v in ALLOWED.items() if not (isinstance(v, str) and v.strip())]
    assert not bad, f"ALLOWED entries with no stated reason: {bad}"


def test_the_allowlist_does_not_excuse_numbers_the_paper_no_longer_quotes():
    """A stale allowlist silently widens what the next number may get away with."""
    body = _body()
    stale = [k for k in ALLOWED if not re.search(r'(?<![\d.])' + re.escape(k) + r'(?![\d])', body)]
    assert not stale, (
        f"ALLOWED entries no longer present in paper2_arxiv/main.tex: {stale}. Remove them; an "
        f"allowlist that outlives its literals is an allowlist nobody is reading.")


def test_every_number_carrying_paragraph_names_a_source():
    """The paper's own convention, enforced: a % source comment must exist for the findings cited.

    Not a per-number check -- a source comment covers a paragraph. This asserts the weaker, checkable
    property that the manuscript carries source comments naming results files at all, and that each
    named results file exists. A comment pointing at a file that was never committed is the failure
    this catches.
    """
    named = set(re.findall(r'%[^\n]*?(results/[A-Za-z0-9_]+\.jsonl?)', TEX.read_text()))
    assert named, "no `% ... results/*.json` source comments found -- the convention is not in force"
    absent = sorted(n for n in named if not (ROOT / n).exists())
    assert not absent, (
        f"source comments in paper2_arxiv/main.tex name results files that do not exist: {absent}")


def test_the_live_paper2_is_the_one_being_guarded():
    assert TEX.name == "main.tex" and TEX.parent.name == "paper2_arxiv"
