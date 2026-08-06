"""Was the verdict computed over the cohort it was registered against?

THE DEFECT. A study declares the units it will measure -- models, subjects, conditions -- and then
some of them fail to load, time out, or go missing. The run records the failures honestly, prints
"3 loads failed", and computes the headline over whatever survived. Nothing is hidden and nothing
objects, but the number now answers a different question than the one registered, over a
denominator nobody chose.

It is the same shape as everything in `leverage`: a statistic applied to a basis that was never
checked. `distinct_units` guards the estimator's INPUT; this guards the verdict's COHORT.

WHY IT NEEDS ITS OWN CHECK RATHER THAN CARE FROM THE AUTHOR. The failure is silent by construction
-- a smaller cohort produces a perfectly well-formed number. In the project this was extracted
from, five gated model repositories became unfetchable between one run and the next; the affected
scripts dropped them, recomputed, and emitted new headline correlations over 18 families where 22
had been registered. The printed "loads failed" line was there the whole time and was read by
nobody, because a result that looks finished does not invite an audit of its denominator.

WHAT IT CANNOT DO. It compares a declared set against a realised one. It cannot tell you whether
the declared set was the right one, and it cannot detect a unit that loaded successfully but
returned garbage. Like everything in this package it only ever downgrades a verdict.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .gate import Verdict, DECIDED, NOT_DECIDABLE

__all__ = ["CohortReport", "cohort_complete", "require_cohort"]


@dataclass
class CohortReport:
    """Whether the units actually measured are the units that were registered."""

    complete: bool
    declared: list[Any]
    realised: list[Any]
    missing: list[Any]
    extra: list[Any]
    reason: str
    tolerated: list[Any] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.complete

    def block(self) -> dict:
        return {
            "complete": self.complete,
            "n_declared": len(self.declared),
            "n_realised": len(self.realised),
            "missing": sorted(map(str, self.missing)),
            "extra": sorted(map(str, self.extra)),
            "tolerated": sorted(map(str, self.tolerated)),
            "reason": self.reason,
        }


def cohort_complete(declared: Iterable[Any], realised: Iterable[Any], *,
                    tolerate: Iterable[Any] = (), unit: str = "unit") -> CohortReport:
    """Compare the registered cohort against the one actually measured.

    `tolerate` names units whose absence was itself registered -- a model known to be unavailable
    and declared as excluded BEFORE the run. Anything absent and not tolerated makes the cohort
    incomplete. Extra units are reported too: measuring something that was never declared is a
    different problem, but it is also not the study that was registered.
    """
    d, r, t = list(declared), list(realised), set(map(str, tolerate))
    ds, rs = {str(x) for x in d}, {str(x) for x in r}
    missing = sorted(ds - rs - t)
    tolerated = sorted((ds - rs) & t)
    extra = sorted(rs - ds)
    complete = not missing and not extra
    if complete:
        reason = (f"cohort intact: all {len(ds)} declared {unit}s measured"
                  + (f" ({len(tolerated)} pre-declared exclusion(s))" if tolerated else ""))
    else:
        bits = []
        if missing:
            bits.append(f"{len(missing)} declared {unit}(s) MISSING from the results "
                        f"({', '.join(missing[:6])}{'...' if len(missing) > 6 else ''})")
        if extra:
            bits.append(f"{len(extra)} {unit}(s) measured that were never declared "
                        f"({', '.join(extra[:6])}{'...' if len(extra) > 6 else ''})")
        reason = ("; ".join(bits)
                  + f". The verdict would be computed over {len(rs)} {unit}s where {len(ds)} were "
                    f"registered, which answers a different question over a denominator nobody "
                    f"chose.")
    return CohortReport(complete=complete, declared=d, realised=r, missing=missing,
                        extra=extra, tolerated=tolerated, reason=reason)


def require_cohort(report: CohortReport, value: Any = None) -> Verdict:
    """Turn a cohort report into a Verdict; NOT_DECIDABLE when the cohort is not what was declared.

    Deliberately not an exception. A run that has already spent hours of compute should WRITE its
    partial results and refuse to emit a headline, not lose the data to a traceback -- the numbers
    it did collect are still worth keeping, and the next run resumes from them.
    """
    if not report.complete:
        return Verdict(status=NOT_DECIDABLE, reason=report.reason)
    return Verdict(status=DECIDED, value=value, reason=report.reason)
