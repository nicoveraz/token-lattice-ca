"""Citations that cannot be fetched must still carry evidence that someone checked them.

`tests/test_refs_match_arxiv.py` closes the arXiv half of this: every `eprint` in refs.bib is
compared against what arXiv actually says. Two things sat outside it, and both are the F43 shape
in new clothes -- a hand audit that misses one entry looks exactly like a hand audit that misses
none:

  1. OPENREVIEW CANNOT BE FETCHED. HTTP 403 `ChallengeRequiredError` on the forum page, the v1 and
     v2 APIs and the PDF endpoint. The challenge is deliberate and is not routed around.
  2. `assembly_theory.md` HAS ITS OWN BIBLIOGRAPHY AND HAD NO MACHINERY AT ALL. Its `[unverified]`
     markers were hand-typed prose, so a citation could sit behind one forever with nothing
     recording whether the work had ever been read. One of those citations -- AssemblyCA -- is the
     closest prior art to issue #20, and three claims about it are the entire basis for calling the
     assembly work novel.

Split of responsibility, as with the arXiv audit: `experiments/audit_manual.py` does the hashing
and the prose cross-check; this file is OFFLINE and asserts the record is internally honest, so
the suite never depends on anything being reachable.

What this CANNOT do is confirm that a claim about a cited work is true. It asserts the work was
obtained, that its metadata is transcribed rather than invented, and that the document's own
`[unverified]` markers agree with the record. Reading the paper is still reading the paper.
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments")]
RECORDS = ROOT / "paper" / "refs_manual.json"

from audit_manual import (SOURCES, cited_forum_ids, normalise, sha256,     # noqa: E402
                          _find, _marked_unverified_near)


def _doc():
    if not RECORDS.exists():
        pytest.skip("paper/refs_manual.json not present -- run experiments/audit_manual.py")
    return json.loads(RECORDS.read_text())


def _by_forum():
    import re
    out = {}
    for key, r in _doc()["records"].items():
        m = re.fullmatch(r"openreview:([A-Za-z0-9_-]+)", r.get("source", ""))
        if m:
            out[m.group(1)] = (key, r)
    return out


def test_every_unfetchable_citation_has_a_record():
    """The staleness guard. Without it a new OpenReview citation is simply not checked."""
    recorded = set(_by_forum())
    missing = {f: w for f, w in cited_forum_ids().items() if f not in recorded}
    assert not missing, (
        f"OpenReview citations with no record in paper/refs_manual.json: {missing}. "
        f"Add a record, then run `.venv/bin/python experiments/audit_manual.py`.")


def test_no_record_is_orphaned():
    """A record must not outlive the citation it describes."""
    cited = set(cited_forum_ids())
    orphans = [k for f, (k, _) in _by_forum().items() if f not in cited]
    assert not orphans, (
        f"paper/refs_manual.json describes works no longer cited anywhere: {orphans}")


def test_a_verified_record_carries_its_evidence():
    """`status: verified` without a date and a statement of what was read is an assertion, not a
    check -- which is the exact thing this file exists to prevent."""
    bad = []
    for key, r in _doc()["records"].items():
        if r.get("status") != "verified":
            continue
        for need in ("checked_on", "checked_against", "title", "first_author", "venue"):
            if not r.get(need):
                bad.append(f"{key} is verified but has no `{need}`")
    assert not bad, "verified records missing evidence: " + "; ".join(bad)


def test_an_unverified_record_says_why_and_what_would_close_it():
    """An unverified citation is acceptable; an unverified citation nobody can act on is not."""
    bad = []
    for key, r in _doc()["records"].items():
        if r.get("status") != "unverified":
            continue
        for need in ("reason", "action"):
            if not r.get(need):
                bad.append(f"{key} is unverified but has no `{need}`")
    assert not bad, "unverified records missing an action: " + "; ".join(bad)


def test_no_record_carries_an_unknown_status():
    ok = set(_doc()["_status_values"])
    bad = [f"{k}={r.get('status')!r}" for k, r in _doc()["records"].items()
           if r.get("status") not in ok]
    assert not bad, f"records with a status outside {sorted(ok)}: {bad}"


def test_pinned_artifacts_still_hash_to_what_was_recorded():
    """Re-verification is one shasum rather than a re-read -- but only if the hash is checked.

    Absent is not a failure: these are third-party PDFs held outside the repo and deliberately not
    redistributed. A file that is PRESENT and DIFFERENT is a failure, because it means the artifact
    behind the check changed underneath it.
    """
    bad = []
    for key, r in _doc()["records"].items():
        want = r.get("pdf_sha256")
        if not want:
            continue
        found = _find(r.get("pdf_basename", ""))
        if found is None:
            continue
        got = sha256(found)
        if got != want:
            bad.append(f"{key}: {r['pdf_basename']} hashes {got[:16]}..., "
                       f"recorded {want[:16]}...")
    assert not bad, "pinned artifacts changed: " + "; ".join(bad)


def test_prose_and_record_agree_about_what_is_verified():
    """The document must not claim more, or less, than the record supports.

    This is `test_paper_numbers.py`'s rule applied to citations: the prose must not contradict the
    file it is derived from. Both directions are failures -- a stale `[unverified]` marker
    understates, and a missing one overclaims.
    """
    bad = []
    for src in SOURCES:
        if not src.exists():
            continue
        text = src.read_text()
        for fid, (key, r) in _by_forum().items():
            if fid not in text:
                continue
            marked = _marked_unverified_near(text, fid)
            verified = r.get("status") == "verified"
            if verified and marked:
                bad.append(f"{src.name} still marks openreview:{fid} '[unverified]' "
                           f"but {key} is verified")
            if not verified and not marked:
                bad.append(f"{src.name} cites openreview:{fid} with no '[unverified]' marker "
                           f"but {key} is not verified")
    assert not bad, "prose disagrees with paper/refs_manual.json: " + "; ".join(bad)


def test_a_verified_title_is_quoted_correctly_where_it_is_cited():
    """The F43 defect itself: an invented title. Compared on content, not on markdown accidents."""
    bad = []
    for src in SOURCES:
        if not src.exists():
            continue
        body = normalise(src.read_text())
        for fid, (key, r) in _by_forum().items():
            if r.get("status") != "verified" or fid not in src.read_text():
                continue
            if normalise(r["title"]) not in body:
                bad.append(f"{src.name} cites {key} but does not carry its recorded title "
                           f"{r['title']!r}")
    assert not bad, "cited titles disagree with the record: " + "; ".join(bad)


def test_assemblyca_stays_flagged_until_its_pdf_is_read():
    """The one that matters, asserted by name rather than by class.

    Everything assembly_theory.md section 4.3 says about AssemblyCA -- 2D rather than 1D, no
    exponential assembly equation, no criticality -- comes from the project site and the GitHub
    README, NOT from the paper. Those three differentiators are the whole basis for claiming issue
    #20's work is novel against the closest prior art. If someone flips this record to `verified`
    without obtaining the PDF, this test is the thing that should have stopped them.
    """
    r = _doc()["records"].get("assemblyca_2023")
    if r is None:
        pytest.skip("assemblyca_2023 record removed")
    if r.get("status") == "verified":
        assert r.get("pdf_sha256"), (
            "assemblyca_2023 is marked verified but no PDF is pinned. The three differentiators "
            "in section 4.3 rest on the project page, not the paper -- obtain it and register it "
            "with `audit_manual.py --register` before claiming it was checked.")
