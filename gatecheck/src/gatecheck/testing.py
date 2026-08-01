"""Test-suite helpers: staleness sweeps, retraction guards, manifest assertions.

Framework-agnostic (plain assertions), designed to be called from pytest. Typical wiring:

    # tests/test_discipline.py
    import glob, pathlib, pytest
    from gatecheck import testing

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    @pytest.mark.parametrize("f", sorted(glob.glob(str(ROOT / "results" / "*.json"))))
    def test_results_are_fresh(f):
        testing.assert_fresh(f, ROOT)

    def test_retractions_stay_retracted():
        text = (ROOT / "paper" / "paper.tex").read_text()
        testing.assert_retracted_stays_retracted(text, {
            "capacity axis": ["capacity scales with sensitivity", "p<10^{-4}"],
        })

    def test_paper_numbers():
        testing.assert_manifest(ROOT / "tests" / "manifest.json",
                                ROOT / "paper" / "paper.tex", ROOT)
"""
from __future__ import annotations

import pathlib

from .manifest import Manifest, strip_tex_comments
from .results import load_results


def assert_fresh(results_path, root):
    """A results file's provenance stamp must match the code on disk (textca #38/F45/F46).

    Files with no `_provenance` block fail: an unstamped results file is unauditable, and
    "checks only what is stamped" is how textca's guard stayed blind for a week.
    """
    _, report = load_results(results_path, root=root, verify=True)
    assert report is not None and report.ok, (
        f"{pathlib.Path(results_path).name}: {report.message()}"
    )


def assert_retracted_stays_retracted(document_text: str, forbidden: dict[str, list[str]],
                                     *, strip=strip_tex_comments):
    """No retracted claim's signature phrasing may reappear in the document.

    Origin: textca's tracker #27 — a retracted three-class ordering was still asserted
    paper-wide and plotted in two figures. `forbidden` maps a claim name to the phrases that
    would indicate its return. Keep the phrases specific enough not to match the retraction
    notice itself.
    """
    doc = strip(document_text) if strip else document_text
    offenders = {
        claim: [p for p in phrases if p in doc]
        for claim, phrases in forbidden.items()
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, (
        f"retracted claims reappeared: {offenders} — if this is a deliberate un-retraction, "
        f"update the guard in the same commit as the evidence"
    )


def assert_manifest(manifest_path, document_path, root, **check_kw):
    """The document, the manifest, and the results files must agree (textca #48/#64)."""
    m = Manifest.load(manifest_path)
    text = pathlib.Path(document_path).read_text()
    rep = m.check(text, root, **check_kw)
    assert rep.ok, rep.message()


def assert_single_implementation(pattern: str, files: list, allowed_file):
    """A shared analysis rule must have exactly one implementation (textca #63/F45).

    `pattern` is a regex matching the rule's *definition* (not its call sites); every file in
    `files` except `allowed_file` must not match. This is a grep guard, with the known limit
    textca discovered in its own repo: an import-and-wrap alias slips past it. It buys drift
    detection, not a proof — pair it with code review.
    """
    import re

    allowed = pathlib.Path(allowed_file).resolve()
    rx = re.compile(pattern)
    offenders = []
    for f in files:
        p = pathlib.Path(f).resolve()
        if p == allowed:
            continue
        if rx.search(p.read_text()):
            offenders.append(str(p))
    assert not offenders, (
        f"rule defined more than once (pattern {pattern!r}): {offenders} — import the single "
        f"definition from {allowed.name} instead of re-implementing it"
    )
