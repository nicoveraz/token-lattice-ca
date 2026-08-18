"""Paper 2's citations must be resolved, resolvable, and ledgered -- checked, not eyeballed.

WHY THIS EXISTS. `paper2_arxiv/main.tex` shipped for several commits with twelve deliberately loud
`\\citepend{...}` placeholders and a `\\paragraph{PENDING --- citations.}` in Limits reading "Do not
submit with placeholders." That warning was a note to a human, and a note to a human is the F43
shape (see `tests/test_refs_match_arxiv.py`): an audit that misses one entry is indistinguishable
from an audit that misses none. When the placeholders were resolved the warning was removed, so the
guard has to live somewhere that cannot forget -- here.

What this file asserts, all offline:

  1. No `\\citepend{}` USE survives in main.tex. The macro definition may stay -- it is kept as a
     tripwire so a future unresolved citation renders red in the PDF -- but a use of it means an
     unresolved citation is in the draft.
  2. Every `\\cite*` key resolves to an entry in `paper2_arxiv/refs.bib`. A dangling key is a "?" in
     the built bibliography, and nothing else in this suite would notice.
  3. refs.bib has no duplicate keys (BibTeX silently keeps one and drops the other).
  4. Every cited key appears in `paper2_arxiv/CITATIONS.md`, the ledger that records the verbatim
     source quote behind each claim. A citation with no ledger entry is a claim about the literature
     with nothing recording whether anyone read the work.

What this CANNOT do is confirm that a quote in the ledger is really in the source, or that the
paper's sentence is what the quote supports. Reading the paper is still reading the paper; F157's
gate refuted 13 of 74 extracted claims for overreaching their own sources, which is why the ledger
carries quotes rather than summaries.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = ROOT / "paper2_arxiv" / "main.tex"
BIB = ROOT / "paper2_arxiv" / "refs.bib"
LEDGER = ROOT / "paper2_arxiv" / "CITATIONS.md"

# Paper 2 is unpublished and local-only; skip rather than fail if it is not in this tree.
pytestmark = pytest.mark.skipif(not TEX.exists(), reason="paper2_arxiv/main.tex not present")


def _tex():
    return TEX.read_text()


def _strip_comments(text):
    """Drop LaTeX comments so a `% ...` note about placeholders is not read as a placeholder."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def _cited_keys():
    body = _strip_comments(_tex())
    keys = []
    for m in re.finditer(r"\\(cite[a-zA-Z]*)\s*(?:\[[^\]]*\])*\{([^}]*)\}", body):
        if m.group(1) == "citepend":          # a placeholder, not a key -- test 1 owns that failure
            continue
        keys += [k.strip() for k in m.group(2).split(",") if k.strip()]
    return keys


def _bib_keys():
    return re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", BIB.read_text())


def test_no_unresolved_citation_placeholders():
    """A \\citepend{} USE means the draft still names a work it has not identified."""
    body = _strip_comments(_tex())
    uses = re.findall(r"\\citepend\{([^}]*)\}", body)
    # The \newcommand definition is not a use; it is the tripwire and is allowed to stay.
    uses = [u for u in uses if u != "#1"]
    assert not uses, (
        f"{len(uses)} unresolved citation placeholder(s) in paper2_arxiv/main.tex: {uses}. "
        f"Resolve each against the source and record the quote in paper2_arxiv/CITATIONS.md.")


def test_every_cited_key_exists_in_the_bib():
    """A dangling key prints as `?` in the bibliography and fails no other check here."""
    dangling = sorted(set(_cited_keys()) - set(_bib_keys()))
    assert not dangling, (
        f"keys cited in paper2_arxiv/main.tex with no entry in paper2_arxiv/refs.bib: {dangling}")


def test_the_bib_has_no_duplicate_keys():
    """BibTeX keeps one definition and drops the other without saying which."""
    keys = _bib_keys()
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert not dups, f"duplicate keys in paper2_arxiv/refs.bib: {dups}"


def test_every_cited_work_is_in_the_ledger():
    """The ledger is where the supporting quote lives. A citation absent from it is unevidenced."""
    if not LEDGER.exists():
        pytest.fail("paper2_arxiv/CITATIONS.md is missing -- it is the evidence record for these "
                    "citations, not an optional note.")
    ledger = LEDGER.read_text()
    missing = sorted({k for k in _cited_keys() if k not in ledger})
    assert not missing, (
        f"cited in paper2_arxiv/main.tex but not recorded in paper2_arxiv/CITATIONS.md: {missing}. "
        f"Add the work with the verbatim quote that supports what the paper says about it.")
