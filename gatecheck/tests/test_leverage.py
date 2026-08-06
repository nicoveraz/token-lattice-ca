"""Every guard is regression-tested against the ACTUAL case that motivated it.

A gate tested only on synthetic data proves the arithmetic, not the discipline. Each test below
uses the real numbers from the finding that caught the defect by hand, so if the gate is ever
weakened the historical case fails again and says which one.
"""
import pytest

from gatecheck import (
    carries_verdict,
    correlation_leverage,
    directional,
    distinct_units,
    dynamic_range,
    noise_gate,
)


# ---------------------------------------------------------------- F94: the saturated predictor

def test_f94_saturated_predictor_is_not_interpretable():
    """lambda_MF spanned 0.050 against lambda_ca's 0.285 -- ratio 0.17, rho unquotable."""
    r = correlation_leverage([0.5130, 0.5601, 0.5157, 0.5105, 0.5258, 0.5218],
                             [-0.0926, -0.0185, 0.0679, 0.1923, 0.1558, 0.1724])
    assert not r.usable
    assert r.stats["ratio"] == pytest.approx(0.174, abs=0.01)
    assert "NOT interpretable in either direction" in r.reason


def test_f96_corrected_regime_does_have_leverage():
    """Measured on the settled ring instead of uniform noise, the same predictor clears the gate."""
    r = correlation_leverage([0.2225, 0.0492, 0.1262, 0.5384, 0.5174, 0.5145],
                             [-0.0926, -0.0185, 0.0679, 0.1923, 0.1558, 0.1724])
    assert r.usable
    assert r.stats["ratio"] == pytest.approx(1.72, abs=0.02)


def test_correlation_leverage_on_a_constant_target_is_not_usable():
    assert not correlation_leverage([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]).usable


# ---------------------------------------------------------------- F96: the collapsed input

def test_f96_settled_ring_collapses_to_ten_distinct_contexts():
    keys = [(i % 10, 0) for i in range(128)]
    r = distinct_units(keys, minimum=32)
    assert not r.usable
    assert r.stats["n"] == 128 and r.stats["n_distinct"] == 10
    assert r.stats["understatement"] == pytest.approx(3.58, abs=0.02)
    assert "understates CI width" in r.reason


def test_distinct_units_passes_once_the_ring_diversifies():
    assert distinct_units([(i, i) for i in range(104)], minimum=32).usable


# ---------------------------------------------------------------- F88: the knife edge

def test_f88_knife_edge_gap_does_not_clear_its_floor():
    """A 0.0011 gap on a 0.0247 floor was reported as a verdict before the branch existed."""
    r = noise_gate(0.0011, 0.0247)
    assert not r.usable
    assert r.stats["ratio"] == pytest.approx(0.0445, abs=1e-3)


def test_noise_gate_passes_a_real_effect():
    assert noise_gate(0.445, 0.0228).usable          # F94's deflation residual, 20x its floor


# ---------------------------------------------------------------- F93/F80: range and direction

def test_f93_dynamic_range_gate_binds_on_a_flat_series():
    r = dynamic_range([0.8352, 0.8755, 0.8374, 0.8331, 0.8459, 0.8426], floor=0.05)
    assert not r.usable


def test_dynamic_range_passes_when_the_series_actually_moves():
    assert dynamic_range([0.6246, 0.5252, 0.5673, 0.8567, 0.8388, 0.8364], floor=0.05).usable


def test_f80_wrong_sign_is_evidence_against_not_weak_evidence_for():
    r = directional(-0.31, expect="increase")
    assert not r.usable
    assert "OPPOSITE to the prediction" in r.reason
    assert directional(+0.31, expect="increase").usable


def test_directional_rejects_an_unknown_direction():
    with pytest.raises(ValueError):
        directional(1.0, expect="sideways")


# ---------------------------------------------------------------- composition

def test_carries_verdict_returns_not_decidable_and_names_the_binding_gate():
    v = carries_verdict(
        [noise_gate(0.445, 0.0228),                       # passes
         distinct_units([(0, 0)] * 128, minimum=32)],     # binds
        value=3.14)
    assert not v.decided
    assert v.value is None                                 # NOT_DECIDABLE carries no value
    assert "distinct_units" in v.reason and "noise_gate" not in v.reason


def test_carries_verdict_does_not_evaluate_the_measurement_when_a_gate_binds():
    calls = []
    v = carries_verdict([distinct_units([(0, 0)] * 8, minimum=32)],
                        lambda: calls.append(1) or 1)
    assert not v.decided and calls == []


def test_carries_verdict_passes_the_value_through_when_every_gate_clears():
    v = carries_verdict([noise_gate(0.445, 0.0228), distinct_units(range(64), minimum=32)],
                        lambda: 42)
    assert v.decided and v.value == 42


def test_reports_are_falsy_when_unusable():
    assert not distinct_units([(0, 0)] * 128, minimum=32)
    assert distinct_units(range(64), minimum=32)
