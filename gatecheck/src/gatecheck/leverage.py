"""Can this quantity carry a verdict at all? — the defect that recurred six times.

THE DEFECT, NAMED. A statistically-shaped criterion applied to a quantity with no room to vary.
It is not a statistics error: every instance below used a correct test, correctly computed, on a
quantity that could not have answered the question whichever way it came out. The test returns a
number, the number looks like evidence, and nothing in the pipeline objects. In the project this
package was extracted from it was caught by hand six times, each time by a different one-off guard
written after the fact:

  F80  a ratio taken before the numerator cleared its own noise floor
  F80  a directional hypothesis tested two-sided, so the sign carried no information
  F88  a knife-edge verdict on a 0.0011 gap over a 0.0247 floor -- no NOT-DECIDABLE branch existed
  F93  a target registered with no dynamic-range check; it then rejected itself
  F94  a Spearman registered against a predictor that turned out to be saturated (span ratio 0.17)
  F96  an estimator whose input collapsed to 7 distinct values, bootstrapped as if it had 128

Three of those were registered BEFORE the run and still failed, which is the important part: a
pre-registration does not protect against asking a question the data cannot answer. Only a check on
the QUANTITY does, and it has to run before the verdict is read.

WHAT THIS MODULE IS FOR, AND WHAT IT IS NOT. Every function here answers "does this quantity have
room to carry the claim being made about it?" and returns a report that is either usable or not.
None of them says whether a hypothesis is true. Composing them with `carries_verdict` gives a
Verdict of NOT_DECIDABLE with the binding reason attached, which is the honest return value when
the answer is "this measurement could not have decided it".

THE ASYMMETRY IS DELIBERATE. These gates only ever downgrade a claim to NOT_DECIDABLE. None of them
can promote one. A quantity that passes has cleared a necessary condition, never a sufficient one.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .gate import Verdict, DECIDED, NOT_DECIDABLE

__all__ = [
    "LeverageReport",
    "reduction_faithful",
    "dynamic_range",
    "correlation_leverage",
    "noise_gate",
    "directional",
    "distinct_units",
    "carries_verdict",
]


def _finite(xs: Sequence[float]) -> list[float]:
    out = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not out:
        raise ValueError("no finite values")
    return out


@dataclass
class LeverageReport:
    """Whether one quantity has room to carry one claim.

    `usable` is the only field a caller should branch on. `reason` is written to be pasted into a
    verdict string verbatim -- it states what was compared against what, with both numbers, so a
    reader can see the gate was binding rather than decorative.
    """

    kind: str
    usable: bool
    reason: str
    stats: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:                     # `if report:` reads as "is it usable"
        return self.usable

    def block(self) -> dict:
        return {"kind": self.kind, "usable": self.usable, "reason": self.reason, **self.stats}


def dynamic_range(values: Sequence[float], *, floor: float, k: float = 2.0,
                  name: str = "quantity") -> LeverageReport:
    """Does `values` span more than `k` times its own uncertainty?

    `floor` is the quantity's own noise scale in ITS OWN UNITS -- a seed standard deviation, a
    bootstrap CI width, a resampling error. Not a p-value and not a fraction. If a series moves by
    less than a couple of times the noise it is made of, "it moves" is not a finding regardless of
    what any test on it returns (F93, F94).
    """
    v = _finite(values)
    span = max(v) - min(v)
    floor = abs(float(floor))
    ok = bool(floor > 0 and span >= k * floor)
    return LeverageReport(
        "dynamic_range", ok,
        (f"{name} spans {span:.4g} against its own noise floor {floor:.4g} "
         f"({span / floor:.2f}x, gate {k}x)" if floor > 0 else
         f"{name} has a zero or undefined noise floor, so no range gate can be applied"),
        dict(span=span, floor=floor, ratio=(span / floor if floor > 0 else None), k=k))


def correlation_leverage(predictor: Sequence[float], target: Sequence[float], *,
                         min_ratio: float = 0.5, name: str = "predictor") -> LeverageReport:
    """Does the PREDICTOR vary enough, relative to the target, for a correlation to mean anything?

    F93 applied this to the target and F94 forgot to apply it to the predictor, which is the same
    omission twice. A correlation between a saturated predictor and a moving target is not weak
    evidence against the hypothesis -- it is no evidence in either direction, and it must not be
    quoted as either. Both series must be in comparable units for the ratio to be meaningful; the
    natural choice is to map the predictor through the model that relates them first.
    """
    p, t = _finite(predictor), _finite(target)
    p_span, t_span = max(p) - min(p), max(t) - min(t)
    ratio = (p_span / t_span) if t_span > 0 else 0.0
    ok = bool(t_span > 0 and ratio >= min_ratio)
    return LeverageReport(
        "correlation_leverage", ok,
        f"{name} spans {p_span:.4g} against the target's {t_span:.4g} (ratio {ratio:.2f}, gate "
        f"{min_ratio}); a correlation is "
        + ("interpretable" if ok else
           "NOT interpretable in either direction -- neither as evidence against the hypothesis "
           "nor, had it come out positive, for it"),
        dict(predictor_span=p_span, target_span=t_span, ratio=ratio, min_ratio=min_ratio))


def noise_gate(numerator: float, floor: float, *, k: float = 2.0,
               name: str = "numerator") -> LeverageReport:
    """Does a quantity clear its own noise BEFORE it is divided by something (F80)?

    A ratio whose numerator is indistinguishable from zero is unbounded noise, and it will happily
    produce a large, stable-looking number. Check the numerator, then take the ratio.
    """
    num, floor = abs(float(numerator)), abs(float(floor))
    ok = bool(floor > 0 and num >= k * floor)
    return LeverageReport(
        "noise_gate", ok,
        f"{name} is {num:.4g} against a noise floor of {floor:.4g} ({num / floor:.2f}x, gate {k}x)"
        if floor > 0 else f"{name} has no usable noise floor",
        dict(numerator=num, floor=floor, ratio=(num / floor if floor > 0 else None), k=k))


def _floor(x: float) -> float:
    return abs(float(x))


def directional(effect: float, *, expect: str, floor: float = 0.0,
                name: str = "effect") -> LeverageReport:
    """A directional hypothesis must be tested directionally (F80).

    `expect` is "increase" or "decrease". A two-sided test on a directional prediction throws away
    the sign, which is usually the only part of the prediction that was risky. An effect of the
    WRONG sign is evidence against, not weak evidence for, and this reports it as such.
    """
    if expect not in ("increase", "decrease"):
        raise ValueError("expect must be 'increase' or 'decrease'")
    e = float(effect)
    want = 1.0 if expect == "increase" else -1.0
    right_sign = (e * want) > 0
    ok = bool(right_sign and abs(e) >= _floor(floor))
    return LeverageReport(
        "directional", ok,
        f"{name} = {e:+.4g}, hypothesis predicted an {expect}; sign is "
        + ("as predicted" if right_sign else "OPPOSITE to the prediction, which is evidence "
                                             "against the hypothesis, not weak evidence for it")
        + (f" and clears the {floor:.4g} floor" if ok and floor else ""),
        dict(effect=e, expect=expect, right_sign=right_sign, floor=float(floor)))


def distinct_units(keys: Sequence[Any], *, minimum: int = 8,
                   name: str = "contexts") -> LeverageReport:
    """Is the estimator's INPUT made of enough distinct units to be an across-unit measurement?

    F96: a settled lattice collapsed onto 7 distinct tokens, so 128 sampled "contexts" were 10
    distinct ones repeated. Every across-context statistic computed there -- and its bootstrap CI
    -- described one context, not a population. The effective n is the number of DISTINCT units,
    and a row bootstrap over n draws from d distinct values understates the width by ~sqrt(n/d).

    Note this gates the input, not the output. It is the only function here that does, and it is
    the one that generalizes furthest: any statistic over a population needs a population.
    """
    n = len(list(keys))
    d = len(set(keys))
    ok = bool(d >= minimum)
    return LeverageReport(
        "distinct_units", ok,
        f"{n} sampled {name} collapse to {d} distinct (gate {minimum})"
        + ("" if ok else f"; a row bootstrap here understates CI width by ~{math.sqrt(n / max(d, 1)):.1f}x "
                         f"-- resample distinct units, and treat any across-{name.rstrip('s')} "
                         f"statistic as undefined"),
        dict(n=n, n_distinct=d, minimum=minimum,
             understatement=(math.sqrt(n / d) if d else None)))


def reduction_faithful(values, *, axis_name: str = "component",
                      condition_name: str = "condition", k: float = 1.0,
                      name: str = "reduction") -> LeverageReport:
    """Is a mean over an axis faithful, or is that axis where the structure lives?

    THE DEFECT, WITH FIVE INSTANCES. A statistic is reported as a scalar -- a mean over positions,
    over contexts, over replicas, over an ensemble -- and the reduction is quoted while the axis it
    reduced is not. When the reduced axis varies MORE than the quantity varies across the conditions
    under study, the mean is not a summary of the thing, it is a summary of something else:

      F94   s averaged over window position AND context: "flat at 0.85", predicted growth everywhere
      F96   s measured on random windows, not the ensemble the ring occupies
      F99   the same, resolved by transplanting model onto ensemble
      F109  s averaged over position on a restricted support: hid an 8x far/near asymmetry
      F110  s averaged over position on the FULL vocabulary: the branching ratio is the SUM of the
            per-position values, not r times their mean, and computing it correctly moved the
            predictor from flat-and-uninterpretable (span ratio 0.17) to tracking (2.09)

    THE TEST. `values` is (n_conditions, n_components). Compare the spread WITHIN the reduced axis
    against the movement of the reduction ACROSS conditions:

        within  = mean over conditions of (max - min across components)
        across  = (max - min) over conditions of the per-condition mean

    If `within > k * across`, the axis you averaged over varies more than the thing you are
    studying, and the mean must not be quoted alone. This does not say the mean is wrong -- it says
    it is not the whole quantity, and the decomposition has to be reported with it.

    It cannot tell you WHICH decomposition matters, only that one does. That judgement stays with
    the author; what this removes is the option of not noticing.
    """
    import numpy as np
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("values must be 2-D: (n_conditions, n_components) with >= 2 components")
    m = a.mean(axis=1)
    within = float((a.max(axis=1) - a.min(axis=1)).mean())
    across = float(m.max() - m.min())
    ratio = within / across if across > 0 else float("inf")
    ok = bool(across > 0 and within <= k * across)
    return LeverageReport(
        "reduction_faithful", ok,
        f"{name}: spread WITHIN the {axis_name} axis is {within:.4g}; movement of the mean ACROSS "
        f"{condition_name}s is {across:.4g} (ratio {ratio:.2f}, gate {k}). "
        + ("The reduction is faithful -- the axis it averages over varies less than the quantity "
           "being studied." if ok else
           f"THE AXIS DOMINATES: averaging over {axis_name} discards more variation than the "
           f"{condition_name} effect it is being used to describe, so the mean must not be quoted "
           f"alone and the per-{axis_name} decomposition has to be reported with it."),
        dict(within=within, across=across, ratio=ratio, k=k,
             n_conditions=int(a.shape[0]), n_components=int(a.shape[1])))


def carries_verdict(reports: Sequence[LeverageReport], measure: Callable[[], Any] | None = None,
                    *, value: Any = None) -> Verdict:
    """Compose leverage reports into a Verdict; NOT_DECIDABLE if any is unusable, or if there are none.

    Mirrors `gate.gated`: the reports must already exist before the value is read, and the binding
    reason is carried on the verdict so the write-up cannot quietly omit which gate bound. Pass
    either a `measure` callable (evaluated only if every gate passes) or a precomputed `value`.

    AN EMPTY REPORT LIST IS NOT A PASS. The requirement this module was built for is that *every*
    registered criterion ships with these gates, so the one case that must never return DECIDED is
    the case where none were applied. Returning DECIDED there makes the discipline opt-in, and the
    caller whose gate list was emptied by a filter -- or never populated -- receives a clean verdict
    carrying the words "no leverage gates applied", which reads like the gates were considered and
    found irrelevant. That is precisely the defect class this module exists to catch, reproduced
    inside the catcher. `measure` is not evaluated in that branch, for the same reason it is not
    evaluated when a gate binds.
    """
    reports = list(reports)
    if not reports:
        return Verdict(
            status=NOT_DECIDABLE,
            reason=("no leverage gates were applied, so nothing established that this quantity has "
                    "room to carry a verdict -- supply at least one gate, or state in the "
                    "registration why none applies"))
    failed = [r for r in reports if not r.usable]
    if failed:
        return Verdict(
            status=NOT_DECIDABLE,
            reason="; ".join(f"{r.kind}: {r.reason}" for r in failed))
    v = measure() if measure is not None else value
    return Verdict(status=DECIDED, value=v,
                   reason="; ".join(f"{r.kind}: {r.reason}" for r in reports))
