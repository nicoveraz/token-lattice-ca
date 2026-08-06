"""gatecheck — a measure-and-falsification toolbox.

Extracted and generalized from the textca project's measurement discipline, in which six
confident wrong verdicts were each caught by their own check before reaching a paper. The
patterns, and the textca findings they descend from, are documented in DESIGN.md.

The one-line philosophy: an estimator must reproduce a known answer, at the geometry it will
actually be used at, before its verdict on an unknown counts — and when it cannot, the honest
return value is NOT_DECIDABLE, not a number.
"""

__version__ = "0.1.0"

from .gate import Gate, GateCheck, Verdict, gated, DECIDED, NOT_DECIDABLE
from .provenance import stamp, verify_stamp, source_sha256, rel, environment_fingerprint
from .prereg import Preregistration, verify_block, quarantine, evaluate_kills
from .units import independence_report, UnitReport
from .fits import scan_minimum, require_off_edge, EdgeRejection, slope_loglog, ScanFit
from .nulltest import (
    assert_exact_zero,
    certify_null,
    NullCertificate,
    VacuousNullError,
    BrokenCouplingError,
    effect_beyond_control,
)
from .leverage import (
    LeverageReport,
    dynamic_range,
    correlation_leverage,
    noise_gate,
    directional,
    distinct_units,
    carries_verdict,
)
from .manifest import Manifest, Entry, strip_tex_comments
from .results import save_results, load_results, check_no_absolute_paths

__all__ = [
    "Gate", "GateCheck", "Verdict", "gated", "DECIDED", "NOT_DECIDABLE",
    "stamp", "verify_stamp", "source_sha256", "rel", "environment_fingerprint",
    "Preregistration", "verify_block", "quarantine", "evaluate_kills",
    "independence_report", "UnitReport",
    "scan_minimum", "require_off_edge", "EdgeRejection", "slope_loglog", "ScanFit",
    "assert_exact_zero", "certify_null", "NullCertificate", "VacuousNullError",
    "BrokenCouplingError", "effect_beyond_control",
    "LeverageReport", "dynamic_range", "correlation_leverage", "noise_gate",
    "directional", "distinct_units", "carries_verdict",
    "Manifest", "Entry", "strip_tex_comments",
    "save_results", "load_results", "check_no_absolute_paths",
]
