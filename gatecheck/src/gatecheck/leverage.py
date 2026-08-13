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
    "resolves_units",
    "expected_range",
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


_E_RANGE = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847,
            9: 2.970, 10: 3.078, 12: 3.258, 15: 3.472, 20: 3.735, 30: 4.086, 50: 4.498}


def expected_range(k: int) -> float:
    """E[max - min] of k standard normals. The reference a SPAN must be judged against."""
    if k < 2:
        return 0.0
    ks = sorted(_E_RANGE)
    if k in _E_RANGE:
        return _E_RANGE[k]
    if k > ks[-1]:                                  # slow growth; extrapolate on log k
        a, b = ks[-2], ks[-1]
        return _E_RANGE[b] + (_E_RANGE[b] - _E_RANGE[a]) * (math.log(k / b) / math.log(b / a))
    lo = max(x for x in ks if x < k)
    hi = min(x for x in ks if x > k)
    t = (k - lo) / (hi - lo)
    return _E_RANGE[lo] + t * (_E_RANGE[hi] - _E_RANGE[lo])


def resolves_units(values: Sequence[float], *, noise_sd: float | Sequence[float],
                   min_reliability: float = 0.5,
                   name: str = "measure") -> LeverageReport:
    """Does this measure separate the things it is measuring, above its own noise?

    THE FOOTGUN THIS CLOSES, and it bit the author of this package. `dynamic_range` asks whether a
    SPAN exceeds k times a noise floor, and the natural floor to reach for is one observation's
    standard error. That comparison is close to meaningless: the span of k draws from PURE NOISE is
    already ~2.3 SD at k=5 and ~3.1 SD at k=10, so a `span >= 2 * SD` gate is passed BY NOISE, by
    construction. It can only fail on something MORE degenerate than noise. In the incident that
    produced this function it reported "2.27x, gate 2.0x -- passes" for a measure whose ten values
    had less variance than the noise they were made of.

    Two corrections, both here:
      * the span is compared against `expected_range(k) * noise_sd`, what noise alone would give;
      * the decisive quantity is RELIABILITY, 1 - var_noise / var_observed -- the share of observed
        spread that is not noise. Reliability at or below zero means the units are not resolved at
        all and every correlation computed from the measure is attenuated to zero. A rank
        correlation from such a measure is not weak evidence; it is no evidence, and its failure
        must not be reported as a finding about the thing being measured.

    `noise_sd` is per-unit and is the CALLER'S to get right -- pass a scalar or one value per unit.
    Compute it at the level of the independent unit (see `gatecheck.units`): with clustered
    observations a per-observation standard error understates the noise by the square root of the
    design effect, which was the second half of the same incident.
    """
    v = _finite(values)
    k = len(v)
    sds = [float(noise_sd)] if isinstance(noise_sd, (int, float)) else _finite(noise_sd)
    var_noise = sum(s * s for s in sds) / len(sds)
    if k > 1:
        mean = sum(v) / k
        var_obs = sum((x - mean) ** 2 for x in v) / (k - 1)
    else:
        var_obs = 0.0
    span = max(v) - min(v)
    noise_span = expected_range(k) * math.sqrt(var_noise) if var_noise > 0 else float("inf")
    reliability = (1.0 - var_noise / var_obs) if var_obs > 0 else float("-inf")
    ok = bool(k > 1 and var_noise > 0 and reliability >= min_reliability)
    return LeverageReport(
        "resolves_units", ok,
        f"{name} spans {span:.4g} across {k} units against the {noise_span:.4g} that pure noise "
        f"would produce (E[range] at k={k} is {expected_range(k):.2f} SD); reliability "
        f"{reliability:+.3f} against a floor of {min_reliability}"
        + ("" if ok else
           " -- the units are NOT resolved, so every correlation computed from this measure is "
           "attenuated toward zero and a failed correlation says nothing about what was measured"),
        dict(k=k, span=span, expected_noise_span=noise_span, var_obs=var_obs,
             var_noise=var_noise, reliability=reliability, min_reliability=min_reliability))


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
