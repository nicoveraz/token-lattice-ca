"""Independent-unit accounting: pseudoreplication detection and effective sample size.

Origin: textca F57 and audit item W1/A1. Two separate incidents, one defect class:

  * A "p<1e-4" flagship result was a signed-rank test over 15 grid cells that all came from the
    SAME two seeds — effective n was 2, not 15. Retracted.
  * A shared per-sweep visit permutation made 512 "replicas" carry the statistical weight of a
    single draw; error bars were ~8x too small. ("The real independent unit was the seed.")

The defense is boring and works: declare what the independent unit is, then CHECK that
observations within a unit are not so correlated that your n is a fiction. This module
implements the check via the one-way random-effects intraclass correlation and the resulting
design effect / effective sample size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class UnitReport:
    n_obs: int
    n_units: int
    mean_obs_per_unit: float
    icc: float                 # intraclass correlation, clipped to [0, 1]
    design_effect: float       # 1 + (m - 1) * icc
    effective_n: float         # n_obs / design_effect
    warn: bool                 # effective_n much smaller than n_obs
    unit_name: str = ""

    def message(self) -> str:
        head = (f"{self.n_obs} observations in {self.n_units} "
                f"{self.unit_name or 'unit'}(s): ICC={self.icc:.3f}, "
                f"design effect {self.design_effect:.2f}, effective n ~ {self.effective_n:.1f}")
        if self.warn:
            head += (" — PSEUDOREPLICATION HAZARD: analyze at the unit level "
                     "(one summary statistic per unit), not the observation level")
        return head


def independence_report(
    values: Sequence[float],
    units: Sequence,
    *,
    unit_name: str = "",
    warn_ratio: float = 0.5,
) -> UnitReport:
    """How many independent observations do you actually have?

    `values` are the observations, `units` the label of the independent unit each came from
    (seed, subject, batch, rule, ...). ICC(1) is estimated from the one-way ANOVA mean squares
    with the unbalanced-design group-size correction; the design effect 1+(m-1)*ICC converts it
    into an effective n. `warn` fires when effective_n < warn_ratio * n_obs.

    Interpretation guide: ICC ~ 0 means observation-level analysis is defensible; ICC near 1
    means your n is the number of UNITS and any test run at the observation level is
    overconfident by roughly the design effect.
    """
    v = np.asarray(values, dtype=float)
    u = np.asarray(units)
    if v.shape != u.shape or v.ndim != 1:
        raise ValueError("values and units must be 1-D and the same length")
    n = v.size
    labels, inverse = np.unique(u, return_inverse=True)
    k = labels.size
    if k < 1 or n < 2:
        raise ValueError("need at least 2 observations")
    sizes = np.bincount(inverse).astype(float)
    means = np.array([v[inverse == i].mean() for i in range(k)])
    grand = v.mean()

    if k == n:                       # one observation per unit: independent by construction
        icc = 0.0
    elif k == 1:                     # a single unit: nothing is independent
        icc = 1.0
    else:
        ss_between = float((sizes * (means - grand) ** 2).sum())
        ss_within = float(sum(((v[inverse == i] - means[i]) ** 2).sum() for i in range(k)))
        ms_between = ss_between / (k - 1)
        ms_within = ss_within / (n - k) if n > k else 0.0
        # unbalanced-design average group size (Searle's n0)
        n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
        if ms_within <= 0:
            icc = 1.0 if ms_between > 0 else 0.0
        else:
            icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
        icc = float(np.clip(icc, 0.0, 1.0))

    m_bar = n / k
    deff = 1.0 + (m_bar - 1.0) * icc
    eff_n = n / deff
    return UnitReport(
        n_obs=n, n_units=k, mean_obs_per_unit=float(m_bar), icc=icc,
        design_effect=float(deff), effective_n=float(eff_n),
        warn=bool(eff_n < warn_ratio * n), unit_name=unit_name,
    )


def assert_effective_n(report: UnitReport, required: float):
    """Fail loudly when the effective sample size cannot support the claim."""
    if report.effective_n < required:
        raise AssertionError(
            f"effective n ~ {report.effective_n:.1f} < required {required}: "
            f"{report.message()}"
        )


def unit_level(values: Sequence[float], units: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Collapse observations to one mean per unit — the analysis textca's retractions point to.

    Returns (unit_labels, unit_means). Run your test on these when the report warns: a rank
    test over unit means cannot be pseudoreplicated by within-unit correlation, whatever the
    ICC is.
    """
    v = np.asarray(values, dtype=float)
    u = np.asarray(units)
    labels, inverse = np.unique(u, return_inverse=True)
    means = np.array([v[inverse == i].mean() for i in range(labels.size)])
    return labels, means
