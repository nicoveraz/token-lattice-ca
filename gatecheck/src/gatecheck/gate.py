"""Calibration gates: no verdict without a passing gate at the measurement's own geometry.

Origin: textca F56. A tolerance calibrated on a reference system at N=512/200 sweeps was applied
to a measurement at N=96/40 — where the same estimator misses the known answer by ~20% on data
that provably IS the reference class. The resulting "not in class X" verdict rejected X using a
threshold that rejects the reference itself. The repair, generalized here:

  * The gate runs the IDENTICAL estimator on a reference system with a known answer, at the SAME
    geometry (lattice size, series length, sample size, whatever "geometry" means for you).
  * The gate is evaluated on the reference ALONE, blind to the target numbers — enforced by API
    shape: `Gate.check` never sees target data.
  * Passing requires mean deviation PLUS its own seed-to-seed spread inside tolerance. A margin
    swamped by its own noise is a coin flip reported as a decision (textca's first gate failed
    by 0.7 points against a 9.6-point spread).
  * When the literature disagrees about the reference's true value, the KINDEST candidate is
    used: if even the most favourable truth fails the gate, the failure is robust to the dispute.
  * A failed gate does not soften the number — it replaces it. `gated()` returns NOT_DECIDABLE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

DECIDED = "DECIDED"
NOT_DECIDABLE = "NOT_DECIDABLE"


def _as_mapping(x: float | Mapping[str, float]) -> dict[str, float]:
    if isinstance(x, Mapping):
        return {str(k): float(v) for k, v in x.items()}
    return {"value": float(x)}


@dataclass
class GateCheck:
    """Result of running the gate at one geometry. Embed `.block()` in your results file."""

    name: str
    geometry: Any
    tolerance_pct: float
    n_seeds: int
    truth_label: str                       # which truth candidate was kept (kindest)
    quantities: dict[str, dict]            # per quantity: mean, truth, dev_pct, spread_pct
    passes: bool

    def block(self) -> dict:
        return {
            "gate": self.name,
            "geometry": repr(self.geometry),
            "tolerance_pct": self.tolerance_pct,
            "n_seeds": self.n_seeds,
            "truth_label": self.truth_label,
            "quantities": self.quantities,
            "passes": self.passes,
            "rule": "mean deviation PLUS its seed-to-seed spread must clear the tolerance",
        }

    def worst(self) -> float:
        """The binding margin: max over quantities of dev_pct + spread_pct."""
        return max(q["dev_pct"] + q["spread_pct"] for q in self.quantities.values())


@dataclass
class Verdict:
    """What a gated measurement returns. NOT_DECIDABLE carries no value, deliberately."""

    status: str                            # DECIDED | NOT_DECIDABLE
    value: Any = None
    gate: GateCheck | None = None
    reason: str = ""

    @property
    def decided(self) -> bool:
        return self.status == DECIDED

    def block(self) -> dict:
        return {
            "status": self.status,
            "value": self.value if self.decided else None,
            "reason": self.reason,
            "gate": self.gate.block() if self.gate is not None else None,
        }


class Gate:
    """A calibration gate around one estimator.

    Parameters
    ----------
    run_reference:
        callable(geometry, seed) -> float | Mapping[str, float]. Must generate data from the
        known-answer reference system AT the given geometry and run the *same* estimator on it
        that the target measurement uses. Sharing the estimator is the caller's obligation and
        the entire point: a gate around a different implementation gates nothing.
    truth:
        the reference's known answer(s). A float, a mapping of named quantities, or a sequence
        of such candidates when the literature disagrees (the kindest candidate is kept).
    tolerance_pct:
        the gate tolerance, in percent of the truth value, applied per quantity.
    """

    def __init__(
        self,
        run_reference: Callable[[Any, int], float | Mapping[str, float]],
        truth: float | Mapping[str, float] | Sequence[float | Mapping[str, float]],
        tolerance_pct: float,
        name: str = "gate",
    ):
        self.run_reference = run_reference
        if isinstance(truth, Sequence) and not isinstance(truth, (str, Mapping)):
            self.truths = [_as_mapping(t) for t in truth]
        else:
            self.truths = [_as_mapping(truth)]
        self.tolerance_pct = float(tolerance_pct)
        self.name = name

    def check(self, geometry: Any, seeds: Sequence[int]) -> GateCheck:
        """Run the estimator on the reference at `geometry` over `seeds`; score vs each truth.

        Few seeds gate nothing: textca's gate flipped verdicts between 5 and 20 seeds while the
        reference was pure numpy and free. Be generous here; this is the cheap part.
        """
        seeds = list(seeds)
        if len(seeds) < 2:
            raise ValueError("a gate needs >= 2 seeds to measure its own spread")
        ests = [_as_mapping(self.run_reference(geometry, s)) for s in seeds]
        keys = sorted(ests[0])
        for e in ests:
            if sorted(e) != keys:
                raise ValueError("run_reference returned inconsistent quantity names")

        candidates = []
        for i, truth in enumerate(self.truths):
            if sorted(truth) != keys:
                raise ValueError(
                    f"truth candidate {i} names {sorted(truth)} but estimator returns {keys}"
                )
            qs = {}
            for k in keys:
                vals = np.array([e[k] for e in ests], dtype=float)
                t = truth[k]
                scale = abs(t) if t != 0 else 1.0
                qs[k] = {
                    "mean": float(vals.mean()),
                    "truth": t,
                    "dev_pct": float(abs(vals.mean() - t) / scale * 100),
                    "spread_pct": float(vals.std() / scale * 100),
                }
            candidates.append((f"truth[{i}]", qs))

        label, qs = min(
            candidates, key=lambda c: max(q["dev_pct"] + q["spread_pct"] for q in c[1].values())
        )
        passes = all(
            q["dev_pct"] + q["spread_pct"] <= self.tolerance_pct for q in qs.values()
        )
        return GateCheck(
            name=self.name, geometry=geometry, tolerance_pct=self.tolerance_pct,
            n_seeds=len(seeds), truth_label=label, quantities=qs, passes=passes,
        )

    def ladder(
        self,
        geometries: Sequence[Any],
        seeds: Sequence[int],
        cost_hours: Callable[[Any], float] | None = None,
    ) -> dict:
        """Which geometry WOULD decide, and what it costs — so "get more compute" is a number.

        Origin: textca's `dp_calibration.calibrate`, whose ladder turned an impasse into a
        purchasable line item ("cheapest that decides: N=192, 23.7 h").
        """
        rungs = []
        for g in geometries:
            c = self.check(g, seeds)
            rungs.append({
                "geometry": g, "passes": c.passes, "worst_pct": c.worst(),
                "hours": (float(cost_hours(g)) if cost_hours else None), "check": c,
            })
        passing = [r for r in rungs if r["passes"]]
        cheapest = None
        if passing:
            key = (lambda r: r["hours"]) if cost_hours else (lambda r: r["worst_pct"])
            cheapest = min(passing, key=key)
        return {"rungs": rungs, "cheapest_passing": cheapest, "tolerance_pct": self.tolerance_pct}


def gated(check: GateCheck, measure: Callable[[], Any]) -> Verdict:
    """Run `measure` only behind a passing gate; otherwise return NOT_DECIDABLE.

    The order is deliberate: the gate result must already exist (computed blind, on the
    reference alone) before the target measurement runs. A gate consulted after seeing the
    target number is a rationalization, not a gate.
    """
    if not check.passes:
        return Verdict(
            status=NOT_DECIDABLE, gate=check,
            reason=(f"gate '{check.name}' fails at geometry {check.geometry!r}: worst "
                    f"deviation+spread {check.worst():.1f}% > tolerance {check.tolerance_pct}%"),
        )
    return Verdict(status=DECIDED, value=measure(), gate=check,
                   reason="gate passed at the measurement's own geometry")
