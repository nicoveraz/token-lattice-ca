"""The predictor-side balance gate, tested against the instance that created it (F163).

R1 is "a statistically-shaped criterion applied to a quantity with no room to vary". F163 showed the
same defect on the PREDICTOR side of a join: a binary predictor split 1-vs-6 returned "agrees on 5 of
7 (71%)", which reads as a screen passing and is in fact worse than the trivial rule. These tests pin
the behaviour so the gate cannot silently regress into permissiveness.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gatecheck" / "src"))

import pytest

from gatecheck import balance_report, balanced_accuracy, gate_join


def test_f163_actual_split_is_not_readable():
    """The real F163 data: 1 model carries the BOS convention, 6 do not."""
    pred = [True] + [False] * 6
    out = ["down", "up", "down", "down", "down", "down", "down"]
    rep = gate_join(pred, out, mapping={True: "up", False: "down"}, name="BOS convention")
    assert not rep.readable
    assert not rep                      # falsey, so `if not rep: return` is the whole usage
    assert rep.min_class_n == 1
    assert "imbalance" in rep.reason.lower()


def test_chance_is_the_majority_rate_not_one_half():
    """The number the screen must beat is 86%, not 50% -- that is the whole point."""
    pred = [True] + [False] * 6
    rep = balance_report(pred, name="p")
    assert rep.majority_class is False
    assert rep.majority_rate == pytest.approx(6 / 7)
    assert "MAJORITY-CLASS RATE" in rep.reason


def test_raw_agreement_can_exceed_balanced_accuracy_badly():
    """F163's numbers exactly: raw 71%, balanced 0.42. The gap is the artefact."""
    pred = [True] + [False] * 6
    out = ["down", "up", "down", "down", "down", "down", "down"]
    bal, per = balanced_accuracy(pred, out, mapping={True: "up", False: "down"})
    raw = sum(1 for p, o in zip(pred, out) if o == ({True: "up", False: "down"}[p])) / len(pred)
    assert raw == pytest.approx(5 / 7)          # 71%
    assert bal == pytest.approx((0.0 + 5 / 6) / 2)   # 0.4166...
    assert bal < 0.5 < raw                      # the screen "passes" on raw and fails on balanced


def test_single_class_predictor_is_refused():
    rep = balance_report([True] * 9, name="p")
    assert not rep.readable
    assert "ONE class" in rep.reason


def test_balanced_predictor_is_readable_and_scored():
    pred = [True, True, True, False, False, False]
    out = ["up", "up", "down", "down", "down", "up"]
    rep = gate_join(pred, out, mapping={True: "up", False: "down"}, name="p")
    assert rep.readable
    assert rep.balanced_accuracy == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert rep.raw_accuracy == pytest.approx(4 / 6)


def test_empty_predictor_is_refused_not_crashed():
    rep = balance_report([], name="p")
    assert not rep.readable
    assert rep.n == 0


def test_min_per_class_floor_is_honoured():
    """Two is the floor at which one flipped label no longer decides the answer."""
    pred = [True, True] + [False] * 5
    out = ["up", "up", "down", "down", "down", "down", "down"]
    assert gate_join(pred, out, mapping={True: "up", False: "down"}).readable
    assert not gate_join(pred, out, mapping={True: "up", False: "down"},
                         min_per_class=3).readable
