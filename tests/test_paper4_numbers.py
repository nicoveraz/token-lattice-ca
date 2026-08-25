"""Every number in paper 4 must trace to a results file, or be declared as somebody else's.

The twin of test_paper3_numbers.py, sharing its matcher so the papers cannot drift apart in what
"traces" means. The caveat that file states applies here unchanged: results/ holds many thousands of
numbers, so a common two-decimal literal will match SOMETHING by coincidence. The test is strong
against invented numbers at three or more decimals and weak at two, which is why every literal quoted
FROM ANOTHER PAPER is allowlisted by name rather than left to pass on a collision.
"""
import pathlib
import re

import pytest

from test_arxiv_paper_numbers import _numbers, _present, _results_blob

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = ROOT / "paper4_arxiv" / "main.tex"

pytestmark = pytest.mark.skipif(not TEX.exists(), reason="paper4_arxiv/main.tex not present")

# Numbers belonging to other people's papers, or to ours but published elsewhere. Each names whose
# it is. These must NOT be satisfied by a coincidental match in our own results.
ALLOWED = {
    "74.4": "paper 1's frozen fraction before a BOS token is prepended (arXiv:2608.10986). Quoted "
            "in the delta paragraph to state what paper 1 already established.",
    "24.1": "the same quantity after the BOS token, from paper 1. Not measured here.",
}


def _body():
    t = TEX.read_text()
    t = re.sub(r'(?m)(?<!\\)%.*$', '', t)
    t = re.sub(r'\\(cite[a-z]*|ref|label|input|include|usepackage|documentclass|newcommand)\{[^}]*\}',
               ' ', t)
    return t


def test_every_decimal_literal_in_paper4_traces_to_a_results_file():
    body, blob = _body(), _results_blob()
    lits = sorted(set(re.findall(r'(?<![\w.])\d+\.\d+(?![\w])', body)))
    assert lits, "no decimal literals found -- the extractor is broken, not the paper"
    pool = _numbers(blob)
    missing = [l for l in lits if l not in ALLOWED and not _present(l, blob, pool)]
    assert not missing, (
        f"{len(missing)} of {len(lits)} decimal literals in paper4_arxiv/main.tex do not appear in "
        f"any results/*.json:\n  " + "\n  ".join(missing) +
        "\nEither the number was not derived from a run, or the run was never committed. If it is "
        "somebody else's number, add it to ALLOWED naming whose it is.")


def test_allowed_entries_all_carry_a_reason():
    bad = [k for k, v in ALLOWED.items() if not (isinstance(v, str) and v.strip())]
    assert not bad, f"ALLOWED entries with no stated reason: {bad}"


def test_the_allowlist_does_not_excuse_numbers_the_paper_no_longer_quotes():
    body = _body()
    stale = [k for k in ALLOWED
             if not re.search(r'(?<![\d.])' + re.escape(k) + r'(?![\d])', body)]
    assert not stale, (
        f"ALLOWED entries no longer present in paper4_arxiv/main.tex: {stale}. Remove them; an "
        f"allowlist that outlives its literals is one nobody is reading.")


def test_source_comments_name_results_files_that_exist():
    named = set(re.findall(r'%[^\n]*?(results/[A-Za-z0-9_]+\.jsonl?)', TEX.read_text()))
    assert named, "no `% ... results/*.json` source comments found -- the convention is not in force"
    absent = sorted(n for n in named if not (ROOT / n).exists())
    assert not absent, f"source comments name results files that do not exist: {absent}"


def test_the_two_prohibited_pitches_never_appear():
    """F186 made these binding. They are not style preferences; both framings are taken."""
    body = _body().lower()
    for phrase, whose in (("degenerate probe", "arXiv:2410.06287 already probes with repeated tokens"),
                          ("fixed points of greedy decoding",
                           "arXiv:2410.06287 and our own paper 1 both have this framing")):
        assert phrase not in body, (
            f"paper 4 pitches itself using '{phrase}', which F186's prior-art gate prohibited: "
            f"{whose}. The registered pitch is the vocabulary-wide set-valued destination map.")


def test_the_int8_and_decisive_figures_are_on_the_same_support():
    """The paper sets 8-bit agreement beside the corpus manipulation and says "on one support".

    That claim is checkable and was FALSE when first written. The quantization cells are computed
    over each model's OWN resolvable probe set -- 3675 escaping sources for pythia-410m -- while the
    decisive-pair figure lives on the 19-model shared support of results/escape_widening.json, 3351
    escaping sources. Quoting 0.9018 beside 0.6353 compared two different index sets, which the
    paper's own re-baselining rule forbids.

    So this asserts what the manuscript must quote: the re-based 8-bit figure, not the own-probe-set
    one, and both from supports that actually match.
    """
    import json
    q = json.load(open(ROOT / "results" / "quant_robustness.json"))
    w = json.load(open(ROOT / "results" / "escape_widening.json"))
    reb = q.get("rebaselined_on_widening_support", {})
    assert reb.get("pairs"), (
        "results/quant_robustness.json carries no same-support re-baselining, so the manuscript has "
        "nothing legitimate to quote beside the decisive pair.")
    assert reb["support"] == w["intersection"], (
        f"the re-baselining used a support of {reb['support']} against the widening's "
        f"{w['intersection']}; they must be the same index set or the comparison is meaningless.")

    anchor = reb["pairs"]["EleutherAI/pythia-410m@int8"]
    dec = w["rebaselined_on_this_support"]["pairs"]["decisive"]
    assert abs(anchor["n_escaping_sources"] - dec["n_shared"]) <= 25, (
        f"the two figures rest on {anchor['n_escaping_sources']} and {dec['n_shared']} escaping "
        f"sources. They are drawn from one index set, so a large gap means one of them was computed "
        f"somewhere else.")

    body = _body()
    a, d = f"{anchor['agreement']}", f"{dec['agreement']}"
    assert a in body and d in body, (
        f"the manuscript should quote the re-based pair {a} against {d}; found "
        f"{a in body} / {d in body}.")
    own = json.load(open(ROOT / "results" / "quant_robustness.json"))["cells"] \
        ["EleutherAI/pythia-410m@int8"]["escape_agreement_vs_fp32"]
    if abs(own - anchor["agreement"]) > 1e-9:
        assert f"{own}" not in body, (
            f"the manuscript quotes {own}, the 8-bit agreement on pythia-410m's OWN probe set, "
            f"beside a decisive figure computed on the shared support. Quote {a} instead -- that is "
            f"the same-support figure, and the difference is exactly the defect this test exists for.")
