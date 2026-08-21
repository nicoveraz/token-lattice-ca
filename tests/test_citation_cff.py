"""`CITATION.cff` is machine-readable metadata, so a defect in it is silent by construction.

WHY THIS EXISTS NOW. The file is about to become load-bearing: cutting a GitHub release mints a new
Zenodo version from it, and GitHub renders "Cite this repository" from it. Until this test, nothing
in the suite read the file at all -- a malformed identifier or a half-filled template would have been
found by a reader of the Zenodo record, not by us.

THE CHECK THAT MATTERS MOST NEEDS NO DEPENDENCY, and that is deliberate. `paper2_arxiv/SUBMISSION.md`
carries a copy-paste block for adding paper 2's arXiv identifiers once it has an ID, and that block
necessarily contains a dummy ID. The failure mode is pasting it and forgetting to substitute the real
one, which would publish `arXiv:2609.01234` as though it were a real paper. That is a plain-text
check, so it runs even where PyYAML is absent. PyYAML is not a declared dependency here -- it arrives
transitively via `transformers` -- so the structural checks skip rather than fail when it is missing,
while the placeholder check never skips.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFF = ROOT / "CITATION.cff"

# arXiv IDs that must never appear as though real. The SUBMISSION.md template uses this one.
TEMPLATE_IDS = ["2609.01234"]
PLACEHOLDER_PAT = re.compile(r"XXXX|TODO|FIXME|<[a-z][a-z -]*>|PLACEHOLDER", re.IGNORECASE)

pytestmark = pytest.mark.skipif(not CFF.exists(), reason="CITATION.cff not present")


def test_no_placeholder_or_template_identifier_reaches_the_metadata():
    """The one check that must never skip: a template pasted without substitution."""
    text = CFF.read_text()
    bad = [t for t in TEMPLATE_IDS if t in text]
    assert not bad, (
        f"CITATION.cff contains the SUBMISSION.md template's dummy arXiv ID {bad}. Substitute the "
        f"real identifier from the arXiv announcement, or remove the block.")
    hit = PLACEHOLDER_PAT.search(text)
    assert not hit, (
        f"CITATION.cff contains placeholder text {hit.group(0)!r} at offset {hit.start()}. This file "
        f"is consumed by Zenodo and by GitHub's citation widget; a placeholder ships as metadata.")


def test_the_file_parses_and_its_identifiers_are_well_formed():
    yaml = pytest.importorskip("yaml", reason="PyYAML is transitive here, not declared")
    d = yaml.safe_load(CFF.read_text())

    assert d.get("cff-version"), "cff-version is required"
    assert d.get("title") and d.get("authors"), "title and authors are required"

    # The CFF 1.2.0 schema allows exactly these identifier types.
    valid = {"doi", "url", "swh", "other"}
    for i in d.get("identifiers", []):
        assert i.get("type") in valid, f"identifier type {i.get('type')!r} is not one of {valid}"
        assert str(i.get("value", "")).strip(), f"identifier {i} has no value"
        assert str(i.get("description", "")).strip(), (
            f"identifier {i.get('value')} has no description -- an undescribed DOI in a citation "
            f"record is not self-explaining to whoever reads the Zenodo page")

    dois = [str(i["value"]) for i in d.get("identifiers", []) if i["type"] == "doi"]
    for doi in dois + ([str(d["doi"])] if d.get("doi") else []):
        assert doi.startswith("10."), f"{doi!r} does not look like a DOI"


def test_every_arxiv_id_appears_in_both_forms():
    """An arXiv paper carries a bare ID and a DOI. Listing one without the other is a half-entry.

    Catches the likely slip when paper 2 is added: pasting the `arXiv:NNNN.NNNNN` line and not the
    `10.48550/arXiv.NNNN.NNNNN` one, which leaves the record without a resolvable DOI for that paper.
    """
    text = CFF.read_text()
    bare = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", text))
    viadoi = set(re.findall(r"10\.48550/arXiv\.(\d{4}\.\d{4,5})", text))
    assert bare == viadoi, (
        f"arXiv IDs listed as a bare identifier but with no matching 10.48550 DOI: "
        f"{sorted(bare - viadoi)}; and DOIs with no bare identifier: {sorted(viadoi - bare)}")
