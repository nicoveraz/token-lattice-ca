"""Exact-null certification with anti-vacuity controls, and control-adjusted verdicts.

Origin: textca's CRN discipline. Three rules, each purchased with a defect:

  * The null arm of a differential measurement must be EXACTLY its known value — not "small".
    textca's coupled twin runs sharing model, init, order, and uniforms must differ in exactly
    zero cells; if that fails, every downstream number is meaningless, so the assertion has no
    tolerance parameter on purpose.
  * A null test only counts alongside a demonstration that the effect CAN appear — otherwise
    it passes vacuously. textca paired every exact-zero null with a perturbation-must-propagate
    counterpart, and its bit-exact DK identity with an off-line control that must MISmatch.
  * A treatment number is read against a control that should show nothing (F65): a radius sweep
    read as "the attractor survives to r=16" until the control acquired one there too. The
    verdict is treatment minus control, never treatment alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BrokenCouplingError(AssertionError):
    """The null arm is not exactly null: the apparatus is broken; fix it before measuring."""


class VacuousNullError(AssertionError):
    """The effect arm shows nothing either: the null test proves nothing as it stands."""


def assert_exact_zero(diff, what: str = "null arm"):
    """Assert a null-arm difference is EXACTLY zero. No tolerance, deliberately.

    Accepts scalars or arrays. If you find yourself wanting a tolerance here, the coupling is
    not the one you think you built — that discovery is this function's whole job.
    """
    a = np.asarray(diff)
    if a.size == 0:
        raise ValueError("empty difference: nothing was compared")
    if not np.all(a == 0):
        nonzero = int(np.count_nonzero(a))
        raise BrokenCouplingError(
            f"{what} differs in {nonzero}/{a.size} entries but must be exactly zero: "
            f"the coupling/apparatus is broken and every downstream number is meaningless"
        )


@dataclass
class NullCertificate:
    null_exact: bool
    effect_nonzero: bool
    effect_magnitude: float
    note: str = ("null arm exactly zero AND effect arm nonzero: the null test is certified "
                 "non-vacuous")

    def block(self) -> dict:
        return {"null_exact": self.null_exact, "effect_nonzero": self.effect_nonzero,
                "effect_magnitude": self.effect_magnitude, "note": self.note}


def certify_null(null_diff, effect_diff) -> NullCertificate:
    """Certify the pair: the null arm must vanish exactly, the effect arm must not.

    Run this in production code, not only in tests — textca asserts its null at measurement
    time (`real_generation_damage.py`), because an apparatus can break between the test suite
    and the run that matters.
    """
    assert_exact_zero(null_diff, "null arm")
    e = np.asarray(effect_diff, dtype=float)
    mag = float(np.abs(e).sum())
    if mag == 0:
        raise VacuousNullError(
            "effect arm is also exactly zero: this null test passes for any apparatus, "
            "including a disconnected one — add a perturbation that provably propagates"
        )
    return NullCertificate(null_exact=True, effect_nonzero=True, effect_magnitude=mag)


def effect_beyond_control(treatment: float, control: float, *, min_gap: float) -> dict:
    """Read a treatment effect against a control that should show nothing (textca F65).

    Returns a verdict block: the effect is claimed only when treatment - control >= min_gap.
    When the control shows the effect too, what you found is a property of the setting, not of
    the subject — that sentence, discovered late, cost textca a radius-sweep verdict.
    """
    gap = float(treatment) - float(control)
    return {
        "treatment": float(treatment),
        "control": float(control),
        "gap": gap,
        "min_gap": float(min_gap),
        "effect": bool(gap >= min_gap),
        "note": ("verdict reads treatment MINUS control; a control that acquires the effect "
                 "reclassifies it as generic"),
    }
