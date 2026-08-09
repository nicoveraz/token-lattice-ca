"""Every measurement quoted in findings.md must exist in a results file.

WHY THIS EXISTS. `test_arxiv_paper_numbers.py` enforces this for `paper_arxiv/main.tex`.
`findings.md` had no such guard, and that is where the numbers live longest: the paper quotes a
selected few, the findings log holds all of them and is the source everything else is written from.

WHAT IT CAUGHT ON THE FIRST RUN. F92's 8-family deflation table has no stored file behind three of
its four rows -- `fix -0.119`, `cyc +0.119`, `modal +0.595` appear in NO results file and match
neither the pre- nor post-F119 values in `tstar_second_target.json`. They came from a run whose
output was never persisted, so they cannot be re-derived and could not be screened for F119's tie
bug. Those three literals nevertheless PASS the scan, because `0.119` and `0.595` occur incidentally
in unrelated results files -- a pooled matcher cannot tell a real trace from a collision. The written
note at F92 is therefore the only record of the gap, and the last test here asserts it stays. It also
caught one genuinely stale number (F115's rwkv lambda, 0.1697 against a re-run value of 0.1698).

DIRECTION. Like the paper test, this runs the converse of a manifest: it asserts that every decimal
literal in the prose is present in some results file, which is the property that makes an INVENTED
number impossible rather than merely unlikely. A manifest ("every derived number appears in the
document") does not catch that.

MATCHING. The pool holds each stored float rendered at 2..6 decimals with trailing zeros stripped,
plus every decimal literal appearing inside verdict strings. Comparison is on those normalised
literals, so `0.80` is not satisfied by `0.801`.

A literal that is not a measurement -- an arXiv identifier, a percentage derived from counts, a
mathematical constant -- belongs in ALLOWED or in a stated structural rule, never in a widened
matcher.
"""
import glob
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FINDINGS = ROOT / "findings.md"

# arXiv identifiers are citations, not measurements. Structural rather than enumerated: new
# citations are added constantly and each would otherwise need an ALLOWED entry.
ARXIV = re.compile(r"^\d{4}\.\d{4,5}$")

# Literals that are not measurements, or are measurements this project can no longer re-derive.
# Every entry needs a reason. Grouped by why, not by which finding.
_DERIVED_PERCENT = (
    "A percentage derived from counts rather than stored as a float. The underlying counts are in "
    "the results file; the percentage is computed in prose. Storing it would be the fix, but "
    "back-filling retired runs is not worth a re-run."
)
_PRECISION = (
    "A statistic quoted at higher precision than the results file stores, or recomputed in prose "
    "from stored components. Not independently invented -- the components trace."
)
_CONSTANT = "A mathematical constant computed in prose, not a measured quantity."

ALLOWED = {}
ALLOWED.update({v: _DERIVED_PERCENT for v in (
    "99.9", "34.6", "66.1", "14.7", "40.5", "66.9", "73.7", "37.2", "19.8", "95.8", "96.1",
    "16.9", "47.05", "41.1", "60.8",
)})
ALLOWED.update({v: _PRECISION for v in (
    "0.2074", "0.4075", "1.5713", "1.6191", "1.452", "1.456", "0.00493", "0.01287", "0.02738",
    "0.3689", "0.5072", "0.0613", "185.1", "205.1", "0.0000001", "0.0000004", "0.0000063",
    "0.0000071", "0.0000377", "0.0000653", "0.0002533", "0.0002613", "0.0008106", "0.0068228",
    "0.0092886", "0.0183403", "0.02316", "0.02376", "0.09177", "0.03443", "0.6447001",
    "0.873762", "1.666",
)})
ALLOWED.update({v: _CONSTANT for v in (
    "0.9210340371976184", "0.9210340371976186",
)})

assert all(ALLOWED.values()), "every ALLOWED entry needs a stated reason"


def _pool():
    """Every number any results file holds, as normalised decimal literals."""
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            f = abs(float(o))
            for d in range(2, 7):
                out.add(f"{f:.{d}f}".rstrip("0").rstrip("."))
        elif isinstance(o, str):
            # verdict strings carry numbers that never became structured fields
            for m in re.findall(r"\d+\.\d+", o):
                out.add(m.rstrip("0").rstrip("."))

    for p in glob.glob(str(ROOT / "results" / "*.json")):
        try:
            walk(json.load(open(p)))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _sections():
    text = FINDINGS.read_text()
    parts = re.split(r"(?m)^### (F\d+ .*)$", text)
    return {parts[i].split("—")[0].strip(): parts[i + 1] for i in range(1, len(parts), 2)}


SECTIONS = _sections()
POOL = _pool()


def test_the_scan_has_something_to_scan():
    """A parser that silently matches nothing would make every test below vacuous."""
    assert len(SECTIONS) > 80, f"only {len(SECTIONS)} finding sections parsed"
    assert len(POOL) > 10_000, f"results pool holds only {len(POOL)} literals"


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_every_quoted_number_traces_to_a_results_file(name):
    body = SECTIONS[name]
    unbacked = []
    for lit in re.findall(r"[-+−]?\d+\.\d+", body):
        norm = lit.lstrip("-+−").rstrip("0").rstrip(".")
        if float(norm) == 0 or ARXIV.match(norm) or norm in ALLOWED or norm in POOL:
            continue
        unbacked.append(norm)
    assert not unbacked, (
        f"{name} quotes {sorted(set(unbacked))}, present in no results/*.json. Either re-run the "
        f"analysis that produced it, or add it to ALLOWED with a reason if it is not a measurement."
    )


def test_f92_provenance_gap_is_still_documented():
    """F92's deflation table passes only incidentally -- its literals collide with other files'.

    The three static-predictor rows are backed by NO results file (see F119). They slip through the
    scan because `0.119` and `0.595` happen to occur in unrelated results, which is precisely the
    kind of accident a pooled matcher cannot distinguish from a real trace. The written note is
    therefore the only thing recording the gap, so this asserts it stays put.
    """
    f92 = next(v for k, v in SECTIONS.items() if k.startswith("F92"))
    assert "PROVENANCE GAP" in f92, (
        "F92's unbacked deflation table is recorded only by that note. If it is removed, the table "
        "must be regenerated from a script that writes a results file instead."
    )
