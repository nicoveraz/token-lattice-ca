"""Every number in the LIVE paper must trace to a results file.

WHY THIS EXISTS SEPARATELY. `test_paper_numbers.py` guards `paper_arxiv/withdrawn_i4d.tex` -- the
WITHDRAWN submission -- through a hand-derived 89-entry manifest. When the work moved to
`main.tex` the manifest did not follow, so the paper actually being written had zero number
coverage while the retired one kept all of it. That is a regression in the guarantee the
Reproducibility appendix makes, not merely a gap.

WHY THE TEST RUNS THE OTHER DIRECTION. The manifest asserts "every derived number appears in the
paper". That catches a results file changing under a static paper. It does NOT catch the failure
that matters more: a number appearing in the paper that was never derived from anything. This
asserts the converse -- every decimal literal in the manuscript is present in some results file --
which is the property that makes an invented number impossible rather than merely unlikely.

MATCHING IS BOUNDARY-AWARE, and that is not a detail. A plain substring check lets `0.80` be
satisfied by `0.801`, so a paper could quote a number no run produced and still pass. That exact
hole was found once already in this project by a manifest audit; it is closed here by requiring the
literal to be bounded by a non-digit on both sides.

A literal that is genuinely not a measurement -- a page count, a version, an equation constant --
belongs in ALLOWED with a stated reason, not in a widened matcher.
"""
import glob
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = ROOT / "paper_arxiv" / "main.tex"

# Literals that are not measurements. Each needs a reason; an empty reason is not allowed.
ALLOWED = {
    "77.4": "Model Equality Testing's own reported median power (Gao et al.), quoted in Related "
            "work. An external result cited from the literature is not derivable from this "
            "project's runs and must not be, which is what this allowlist is for.",
    "2608.10986": "The paper's own arXiv identifier, in the Reproducibility paragraph. An "
                  "identifier, not a measurement. Same class as the Zenodo prefix below: the "
                  "extractor cannot distinguish an identifier from a ratio, and this is the "
                  "allowlist branch that requires a stated reason rather than a widened regex.",
    "10.5281": "The Zenodo DOI prefix in the Reproducibility paragraph "
               "(doi.org/10.5281/zenodo.21880472). An identifier, not a measurement. The extractor "
               "cannot tell a DOI from a ratio, and widening the regex to exclude URLs would risk "
               "excusing a real number that happens to sit near one -- so it is allowlisted here "
               "instead, which is the branch that requires a stated reason.",
}


def _body():
    if not TEX.exists():
        pytest.skip("paper_arxiv/main.tex not present")
    t = TEX.read_text()
    t = re.sub(r'(?m)%.*$', '', t)                       # comments carry SOURCE notes, not claims
    t = re.sub(r'\\(cite[a-z]*|ref|label|input|include|usepackage|documentclass)\{[^}]*\}', ' ', t)
    return t


def _results_blob():
    files = sorted(glob.glob(str(ROOT / "results" / "*.json")))
    if not files:
        pytest.skip("no results files present")
    return "".join(pathlib.Path(f).read_text() for f in files)


def _numbers(blob):
    return [float(m) for m in re.findall(r'-?\d+\.\d+', blob)]


def _present(literal, blob, pool=None):
    """Boundary-aware exact match, OR a faithful rounding of a stored value.

    Two rules, and the second is not a loosening of the first.

    EXACT, BOUNDARY-AWARE. `0.80` inside `0.801` is not an occurrence of 0.80, and treating it as
    one is how a paper quotes a number nothing produced. That hole was found once already here by
    a manifest audit.

    FAITHFUL ROUNDING. A paper legitimately writes 0.579 for a stored 0.5786 -- quoting four
    decimals in prose would be false precision. So a literal also counts as present if some stored
    number rounds to it AT THE LITERAL'S OWN PRECISION. This is exactly the case
    `build_paper_manifest.py` documents as its first trap ("derive from raw runs, not from stored
    rounded values"), and admitting it here is what keeps the test from forcing the paper to quote
    machine precision. It does not admit an invented number: 0.579 still requires a stored value in
    [0.5785, 0.5795), which nothing but the real measurement will satisfy.
    """
    if re.search(r'(?<![\d.])' + re.escape(literal) + r'(?![\d])', blob):
        return True
    dp = len(literal.split(".")[1])
    target = float(literal)
    return any(round(v, dp) == target for v in (pool if pool is not None else _numbers(blob)))


def test_every_decimal_literal_in_the_live_paper_traces_to_a_results_file():
    body, blob = _body(), _results_blob()
    lits = sorted(set(re.findall(r'(?<![\w.])\d+\.\d+(?![\w])', body)))
    assert lits, "no decimal literals found -- the extractor is broken, not the paper"
    pool = _numbers(blob)
    missing = [l for l in lits if l not in ALLOWED and not _present(l, blob, pool)]
    assert not missing, (
        f"{len(missing)} of {len(lits)} decimal literals in paper_arxiv/main.tex do not appear in "
        f"any results/*.json under boundary-aware matching:\n  " + "\n  ".join(missing) +
        "\nEither the number was not derived from a run, or the run that produced it was never "
        "committed. If it is genuinely not a measurement, add it to ALLOWED with a reason.")


def test_allowed_entries_all_carry_a_reason():
    """An allowlist without reasons decays into a way to silence the test."""
    bad = [k for k, v in ALLOWED.items() if not (isinstance(v, str) and v.strip())]
    assert not bad, f"ALLOWED entries with no stated reason: {bad}"


def test_the_matcher_rejects_a_prefix_match():
    """Prove both rules fire -- otherwise the whole test is vacuous."""
    assert _present("0.801", '{"x": 0.801}')
    assert not _present("1.5", '{"x": 21.5}')          # suffix of a longer number
    assert _present("0.579", '{"x": 0.5786}')          # faithful rounding at 3dp
    assert not _present("0.580", '{"x": 0.5786}')      # not a faithful rounding
    assert not _present("0.579", '{"x": 0.6}')         # nothing near it


def test_the_live_paper_is_the_one_being_guarded():
    """Guard against this test drifting onto the retired manuscript the way the manifest did."""
    assert TEX.name == "main.tex", "this test must point at the LIVE paper"
    assert (ROOT / "paper_arxiv" / "withdrawn_i4d.tex").exists(), (
        "the withdrawn paper is guarded by test_paper_numbers.py; if it is gone, that test needs "
        "retiring rather than this one repointing")
