"""Every number in paper 3 must trace to a results file, or be declared as somebody else's.

The twin of `test_paper2_numbers.py`, sharing its matcher so the three papers cannot drift apart in
what "traces" means.

A CAVEAT THIS TEST STATES RATHER THAN HIDES. `results/` now holds well over a hundred JSON files and
many thousands of numbers, so a common two-decimal literal like `0.74` will match SOMETHING by
coincidence. Boundary-aware matching does not fix that; it only stops `0.80` matching inside `0.801`.
The test is therefore strong against invented numbers at three or more decimals, and weak against
coincidence at two. That is why every literal quoted FROM ANOTHER PAPER is allowlisted here by name
rather than left to pass on a chance collision -- an external number that happens to collide would
otherwise look verified when nothing verified it.
"""
import pathlib
import re

import pytest

from test_arxiv_paper_numbers import _numbers, _present, _results_blob

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = ROOT / "paper3_arxiv" / "main.tex"

pytestmark = pytest.mark.skipif(not TEX.exists(), reason="paper3_arxiv/main.tex not present")

# Numbers quoted from other people's papers, or derived from theirs. Each names whose it is. These
# must NOT be satisfied by a coincidental match in our own results, which is why they are explicit.
#
# Deliberately NOT here: wang2025universality's 0.74/0.76 and michaelov2025phases's 0.86-0.98. The
# manuscript reports those qualitatively ("substantial similarity", "dominates a large share")
# instead of quoting the figures, because quoting a peak R^2 as a headline is the over-reading
# CITATIONS.md exists to prevent. The staleness test below caught them sitting here unused.
ALLOWED = {
    # --- design constants: a rule fixed BEFORE the data, not a measurement from any run. These
    # would otherwise pass on a coincidental collision in results/, which is exactly the weakness
    # this file's docstring describes. Declaring them makes the pass meaningful.
    "0.30": "class threshold: FUNNEL requires phi >= 0.30 and modal share >= 0.30. Fixed in "
            "experiments/argmax_census_hardened.py before F87 and never adjusted since.",
    "0.10": "class threshold: NONE requires phi <= 0.10. Same rule, same provenance.",
    "0.20": "class threshold: FRAGMENTED requires phi >= 0.30 and modal share < 0.20. Same rule.",
    "1.45": "derived arithmetic on biderman2023pythia's stated corpus sizes (~207B deduplicated "
            "against ~300B trained), not a measurement of ours. Kept here because the extractor "
            "cannot tell a derived ratio from a measured one.",
    "2.7": "the parameter count in the model name gpt-neo-2.7B, written as $6.6\\times$ context in "
           "E3. A model identifier, not a measurement.",
    "6.6": "the size ratio between gpt-neo-2.7B and pythia-410m, stated when explaining why the "
           "earlier comparison was confounded. Derived from parameter counts, not measured.",
}


def _body():
    """Strip comments, but not escaped percent signs -- see test_paper2_numbers for why."""
    t = TEX.read_text()
    t = re.sub(r'(?m)(?<!\\)%.*$', '', t)
    t = re.sub(r'\\(cite[a-z]*|ref|label|input|include|usepackage|documentclass)\{[^}]*\}', ' ', t)
    return t


def test_every_decimal_literal_in_paper3_traces_to_a_results_file():
    body, blob = _body(), _results_blob()
    lits = sorted(set(re.findall(r'(?<![\w.])\d+\.\d+(?![\w])', body)))
    assert lits, "no decimal literals found -- the extractor is broken, not the paper"
    pool = _numbers(blob)
    missing = [l for l in lits if l not in ALLOWED and not _present(l, blob, pool)]
    assert not missing, (
        f"{len(missing)} of {len(lits)} decimal literals in paper3_arxiv/main.tex do not appear in "
        f"any results/*.json:\n  " + "\n  ".join(missing) +
        "\nEither the number was not derived from a run, or the run was never committed. If it is "
        "somebody else's number, add it to ALLOWED naming whose it is.")


def test_the_class_thresholds_in_the_paper_match_the_code():
    """The paper now PRINTS the class rule, so it can disagree with the code that applied it.

    E3's whole claim is that the class differs between models, which makes the threshold rule load
    bearing rather than descriptive. If Setup's table and classify() ever drift apart, every class
    in the paper becomes unverifiable -- so the numbers are compared directly.
    """
    import re as _re
    src = (ROOT / "experiments" / "argmax_census_hardened.py").read_text()
    fn = src[src.index("def classify("):src.index('return "borderline"')]
    code = sorted(set(_re.findall(r'0\.\d+', fn)))
    body = _body()
    setup = body[body.index("Trajectories are classified"):body.index("The funnel geometry")]
    printed = sorted(set(_re.findall(r'0\.\d+', setup)))
    assert code == printed, (
        f"the class rule printed in Setup does not match classify() in the census code.\n"
        f"  code:    {code}\n  printed: {printed}\n"
        f"E3 claims the class varies; if the rule that assigns it is misprinted, the claim is "
        f"unverifiable from the paper.")


def test_allowed_entries_all_carry_a_reason():
    bad = [k for k, v in ALLOWED.items() if not (isinstance(v, str) and v.strip())]
    assert not bad, f"ALLOWED entries with no stated reason: {bad}"


def test_the_allowlist_does_not_excuse_numbers_the_paper_no_longer_quotes():
    body = _body()
    stale = [k for k in ALLOWED if not re.search(r'(?<![\d.])' + re.escape(k) + r'(?![\d])', body)]
    assert not stale, (
        f"ALLOWED entries no longer present in paper3_arxiv/main.tex: {stale}. Remove them; an "
        f"allowlist that outlives its literals is one nobody is reading.")


def test_source_comments_name_results_files_that_exist():
    named = set(re.findall(r'%[^\n]*?(results/[A-Za-z0-9_]+\.jsonl?)', TEX.read_text()))
    assert named, "no `% ... results/*.json` source comments found -- the convention is not in force"
    absent = sorted(n for n in named if not (ROOT / n).exists())
    assert not absent, f"source comments name results files that do not exist: {absent}"


def test_the_binding_constraints_from_the_prior_art_gate_are_respected():
    """F177 left three constraints on what this paper may say. Two are checkable mechanically.

    K10: the measurement is published in arXiv:2608.10986, so Setup must cite it.
    E3:  the word "architecture" was withdrawn as a label for the split (F178) -- a transformer
         lands with the recurrent models -- so it must not reappear as the paper's own claim.
    """
    body = _body()
    # K10 is checked on the RAW source: _body() strips \citep{...}, so a citation key can never
    # survive into it. Checking the stripped text would make this assertion unfalsifiable.
    raw = TEX.read_text()
    setup = raw[raw.index("\\section{Setup}"):raw.index("\\section{Related work")]
    assert "veraz2026probes" in setup, (
        "K10: the citation to arXiv:2608.10986 must be IN Setup, beside the measurement it bounds "
        "-- the census method and the funnel/none contrast are already published there")
    bad = re.findall(r'(?i)\barchitecture (?:causes|determines|drives|explains)\b', body)
    assert not bad, (
        f"E3 may not claim architecture as the cause -- F178 withdrew that word when gpt-neo-125m, "
        f"a transformer, landed with the recurrent models. Found: {bad}")
