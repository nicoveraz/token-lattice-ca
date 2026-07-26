"""Every arXiv citation in refs.bib must match what arXiv actually says (issue #71).

Why this exists. F43 (#37) found three citations carrying invented titles, and a built PDF's
bibliography literally read "Title/authors to verify". That audit was done by hand, and by hand
it was incomplete -- a FOURTH wrong title survived it and was found only when the entry happened
to be read for an unrelated reason:

    ar_tempcrit  bib:   "Critical Phase Transition in Large Language Models"
                 arXiv: "Phase transition in large language models and the criticality of
                         natural languages"

That is the same failure shape as #57: a defect fixed as an instance rather than as a class. A
hand audit that misses one entry is indistinguishable from a hand audit that misses none, so the
eyeball is replaced here by a comparison against a fetched record.

Split of responsibility. `experiments/audit_refs.py` does the network work and writes
`paper/refs_verified.json` -- the metadata AS ARXIV REPORTS IT. This test is OFFLINE: it only
checks that refs.bib still agrees with that record, so the suite never depends on arXiv being
reachable (it was not, from this machine, on the day the audit was written -- the export API
timed out and then 429'd, and the abstract pages' Highwire meta tags were used instead).

A wrong citation is not a cosmetic defect. It is an unverifiable claim about the literature
sitting in a paper whose entire argument is that its claims are checked -- and it is the one
kind of error a reviewer can confirm in ten seconds without running anything.
"""
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments")]
BIB = ROOT / "paper" / "refs.bib"
VERIFIED = ROOT / "paper" / "refs_verified.json"

from audit_refs import normalise, parse_bib          # noqa: E402  (needs the path above)


def _bib():
    if not BIB.exists():
        pytest.skip("paper/refs.bib not present")
    return parse_bib(BIB.read_text())


def _verified():
    if not VERIFIED.exists():
        pytest.skip("paper/refs_verified.json not present -- run experiments/audit_refs.py")
    return json.loads(VERIFIED.read_text())


def _arxiv_entries():
    """(key, fields, arxiv_id) for every bib entry carrying a well-formed arXiv identifier."""
    out = []
    for key, f in _bib():
        eid = (f.get("eprint") or "").strip()
        if re.fullmatch(r"\d{4}\.\d{4,5}", eid):
            out.append((key, f, eid))
    return out


def test_every_arxiv_entry_has_been_verified():
    """A new or edited citation must be re-audited, not merely added.

    This is the staleness guard. Without it, someone adds a citation, the title check below
    never runs on it because it is absent from the manifest, and the suite reports green on an
    entry nobody checked -- reintroducing exactly the F43 hole.
    """
    v = _verified()["verified"]
    missing = [k for k, _, _ in _arxiv_entries() if k not in v]
    assert not missing, (
        f"refs.bib entries with no verified record: {missing}. Run "
        f"`.venv/bin/python experiments/audit_refs.py` to refresh paper/refs_verified.json.")


def test_no_verified_record_is_orphaned():
    """The manifest must not outlive the bib entry it describes."""
    keys = {k for k, _, _ in _arxiv_entries()}
    orphans = [k for k in _verified()["verified"] if k not in keys]
    assert not orphans, (
        f"paper/refs_verified.json describes entries no longer in refs.bib: {orphans}. "
        f"Re-run experiments/audit_refs.py.")


def test_titles_match_arxiv():
    """The defect that actually shipped. Compared on content, not on LaTeX accidents."""
    v = _verified()["verified"]
    wrong = []
    for key, f, eid in _arxiv_entries():
        if key not in v:
            continue
        if normalise(f.get("title", "")) != normalise(v[key]["title"]):
            wrong.append(f"\n  {key} ({eid})\n    bib:   {f.get('title', '')}"
                         f"\n    arXiv: {v[key]['title']}")
    assert not wrong, "citation titles disagree with arXiv:" + "".join(wrong)


def test_first_authors_match_arxiv():
    v = _verified()["verified"]
    wrong = []
    for key, f, _ in _arxiv_entries():
        if key not in v:
            continue
        first = re.split(r"\s+and\s+", f.get("author", ""))[0]
        surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
        if surname and normalise(surname) not in normalise(v[key]["first_author"]):
            wrong.append(f"{key}: bib first author {first!r} vs arXiv {v[key]['first_author']!r}")
    assert not wrong, "first authors disagree with arXiv: " + "; ".join(wrong)


def test_no_citation_year_precedes_its_preprint():
    """A bib year LATER than the arXiv year is fine -- it is the peer-reviewed venue's year.

    `edgeofchaos2024` is ICLR 2025 on a 2024 preprint; `simplicitybias` is ICML 2023 on a 2022
    preprint. Both are correct as written. Only a year EARLIER than the preprint is impossible,
    and that is what this asserts -- the earlier version of this check flagged the two legitimate
    venue years and would have trained a reader to ignore it.
    """
    v = _verified()["verified"]
    bad = []
    for key, f, _ in _arxiv_entries():
        if key not in v:
            continue
        yr, ax = (f.get("year") or "").strip(), str(v[key].get("arxiv_year") or "")
        if yr.isdigit() and ax.isdigit() and int(yr) < int(ax):
            bad.append(f"{key}: bib year {yr} precedes arXiv {ax}")
    assert not bad, "impossible citation years: " + "; ".join(bad)


def test_non_arxiv_entries_are_listed_for_manual_check():
    """Entries with no arXiv id cannot be machine-verified; they must at least be enumerated.

    These are the classical references (Domany--Kinzel, Langton, Lieb--Robinson, ...). Silence
    about them would let a new un-checkable entry slip in unnoticed, which is how the class of
    defect this file guards against got started.
    """
    listed = set(_verified().get("unverifiable", []))
    keys = {k for k, _, _ in _arxiv_entries()}
    actual = {k for k, _ in _bib()} - keys
    assert actual == listed, (
        f"non-arXiv entries changed: only-in-bib={sorted(actual - listed)}, "
        f"only-in-manifest={sorted(listed - actual)}. Re-run experiments/audit_refs.py and "
        f"check the new entries by hand against their published venue.")


def test_the_paper_cites_the_prior_pythia_phase_transition_work():
    """A paper about phase transitions on Pythia checkpoints must cite the prior one.

    Not a style preference. `ar_tempcrit` (2406.05335) analyses Pythia-160m training checkpoints
    and places the emergence of critical structure at k_c ~ 10^2 steps -- adjacent to this work's
    128--256 crossing bracket for the same model, by a method (power spectra, part-of-speech
    correlation) with no overlap with damage spreading. Omitting it would read as unawareness of
    the literature, which is the cheapest available reason to be rejected.
    """
    tex = ROOT / "paper" / "paper.tex"
    if not tex.exists():
        pytest.skip("paper.tex not present")
    assert "ar_tempcrit" in tex.read_text(), (
        "paper.tex no longer cites ar_tempcrit (arXiv:2406.05335), the prior analysis of "
        "phase transitions across Pythia training checkpoints")
