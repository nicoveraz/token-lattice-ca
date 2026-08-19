"""Balance gates for CATEGORICAL PREDICTORS -- the vacuity defect on the predictor side.

WHY THIS MODULE EXISTS, AND THE INSTANCE THAT CREATED IT.

`leverage` gates the MEASURED quantity: does it have room to vary? This gates the PREDICTOR: does the
thing doing the predicting have room to vary? They are the same defect on opposite sides of a join,
and the second is easier to miss because a lopsided predictor still yields a high, confident-looking
number.

F163 is the instance. A binary predictor split 1-vs-6 was joined against a two-class outcome and
returned "agrees on 5 of 7 (71%)", which reads as a screen passing. It is a base-rate artefact: a
rule that IGNORES the predictor entirely and always answers the majority class scores 6 of 7. The
quantity that can actually fail is balanced accuracy, which was 0.42 -- below chance. The imbalance
gate was added after seeing the split, which is exactly why it now lives in code: the honest report
of that near-miss says vigilance caught it once, and vigilance is not a control.

THE RULE. For a binary (or k-class) predictor, chance is the MAJORITY-CLASS RATE, never 1/k, and
never the raw agreement. A predictor whose minority class has fewer than `min_per_class` members
cannot discriminate at all, and the verdict is NOT DECIDABLE for predictor imbalance -- returned
BEFORE any join is read.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BalanceReport:
    """The result of gating a categorical predictor before a join."""
    n: int
    counts: dict
    majority_class: object
    majority_rate: float
    min_class_n: int
    readable: bool
    reason: str
    balanced_accuracy: float | None = None
    raw_accuracy: float | None = None
    per_class_rate: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.readable


def balance_report(predictor, *, min_per_class: int = 2, name: str = "predictor") -> BalanceReport:
    """Can this categorical predictor discriminate at all? Call BEFORE joining to an outcome.

    `min_per_class` is the smallest minority class that can carry a two-class call. Two is the floor
    at which a single flipped label no longer decides the answer.
    """
    vals = list(predictor)
    counts = dict(collections.Counter(vals))
    n = len(vals)
    if n == 0:
        return BalanceReport(0, {}, None, float("nan"), 0, False,
                             f"{name} is empty")
    maj, maj_n = max(counts.items(), key=lambda kv: kv[1])
    lo = min(counts.values())
    if len(counts) < 2:
        return BalanceReport(
            n, counts, maj, maj_n / n, lo, False,
            f"{name} has ONE class ({maj!r}) over {n} units: it cannot discriminate, and any "
            f"agreement it appears to show is the base rate. NOT DECIDABLE for predictor imbalance.")
    if lo < min_per_class:
        return BalanceReport(
            n, counts, maj, maj_n / n, lo, False,
            f"{name} is imbalanced {counts}: the minority class has {lo} member(s), below the floor "
            f"of {min_per_class}. Chance here is the MAJORITY-CLASS RATE {maj_n / n:.0%}, not 50%, "
            f"and raw agreement inherits it. NOT DECIDABLE for predictor imbalance.")
    return BalanceReport(
        n, counts, maj, maj_n / n, lo, True,
        f"{name} is usable: {counts}, minority class {lo} >= {min_per_class}. Chance is the "
        f"majority-class rate {maj_n / n:.0%}; report balanced accuracy against it, not raw "
        f"agreement.")


def balanced_accuracy(predictor, outcome, *, mapping=None):
    """Mean of the per-class hit rates. The quantity that can fail when classes are lopsided.

    `mapping` maps a predictor value to the outcome value it predicts; defaults to identity.
    """
    pred, out = list(predictor), list(outcome)
    if len(pred) != len(out) or not pred:
        raise ValueError("predictor and outcome must be the same non-zero length")
    mp = (lambda v: v) if mapping is None else (lambda v: mapping[v])
    per = {}
    for cls in set(pred):
        idx = [i for i, p in enumerate(pred) if p == cls]
        hits = sum(1 for i in idx if out[i] == mp(cls))
        per[cls] = hits / len(idx)
    return sum(per.values()) / len(per), per


def gate_join(predictor, outcome, *, mapping=None, min_per_class: int = 2,
              name: str = "predictor") -> BalanceReport:
    """Balance-gate a categorical predictor, then score the join. NOT DECIDABLE short-circuits.

    Returns a BalanceReport that is falsey when the predictor cannot discriminate. When it is
    readable, `balanced_accuracy` and `raw_accuracy` are filled and the report names the
    majority-class rate as the chance level to beat.
    """
    rep = balance_report(predictor, min_per_class=min_per_class, name=name)
    if not rep.readable:
        return rep
    mp = (lambda v: v) if mapping is None else (lambda v: mapping[v])
    raw = sum(1 for p, o in zip(predictor, outcome) if o == mp(p)) / rep.n
    bal, per = balanced_accuracy(predictor, outcome, mapping=mapping)
    return BalanceReport(
        rep.n, rep.counts, rep.majority_class, rep.majority_rate, rep.min_class_n, True,
        rep.reason + f" Balanced accuracy {bal:.2f} against a majority-class rate of "
                     f"{rep.majority_rate:.2f}; raw agreement {raw:.2f} is reported only for "
                     f"comparison and is not the verdict.",
        balanced_accuracy=bal, raw_accuracy=raw, per_class_rate=per)
