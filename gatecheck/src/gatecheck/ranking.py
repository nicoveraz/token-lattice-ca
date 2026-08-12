"""Tie-aware ranking, because `np.argsort(np.argsort(x))` is not one.

THE BUG THIS REPLACES. Fifteen scripts in the project this package was extracted from ranked with

    rk = lambda x: np.argsort(np.argsort(x))

which is correct only when every value is distinct. `argsort` breaks ties by INPUT POSITION, so a
vector with repeats is assigned strictly increasing ranks encoding the order the values happened to
be listed in. The degenerate case is the loud one: on a CONSTANT vector the idiom returns
[0, 1, ..., n-1] -- a perfectly monotone rank sequence for a quantity that never moves. A
correlation on those ranks then reports a confident number. One shape scalar measured at exactly
0.000 across 24 checkpoints came back at rho = +0.829 against the growth rate, i.e. against the
order the checkpoints were listed in.

WHY IT LIVES IN THE PACKAGE. This is the defect class `leverage` exists for -- a statistically
shaped criterion applied to a quantity with no room to vary -- reached through the CORRELATION
FUNCTION rather than through the data. Every data-inspecting gate in this package was structurally
unable to see it: the data was honest, and the estimator manufactured the order. A toolbox that
ships range gates but not a ranking primitive leaves its users exposed to the one instance its own
gates cannot catch.

WHAT THIS DOES.
  * ties get AVERAGED ranks, which is the Spearman definition
  * a zero-variance or non-finite input returns all-nan, so any correlation computed from it is nan
    rather than a number. Callers must branch on nan. Returning a plausible float for an
    unmeasurable quantity is exactly the failure being removed, so this propagates instead of
    defaulting.

Ranks are 1-based floats. Every intended call site feeds them to a correlation, where an affine
shift is irrelevant; do not use these as indices.

numpy only, deliberately: `scipy.stats.rankdata` has identical semantics and this package's
dependency list is one line long. `tests/test_ranking.py` checks the agreement where scipy is
importable, and skips rather than failing where it is not.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["rank", "spearman"]


def rank(x: Sequence[float]) -> np.ndarray:
    """Averaged ranks; all-nan when `x` has no variance to rank."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return a
    if not np.isfinite(a).all():
        return np.full(a.shape, np.nan)
    if a.std() == 0:
        return np.full(a.shape, np.nan)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    out = np.empty(a.size, dtype=float)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and s[j + 1] == s[i]:
            j += 1
        out[order[i:j + 1]] = 0.5 * (i + j) + 1.0      # mean of the 1-based ranks i+1 .. j+1
        i = j + 1
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho, nan when either side is degenerate. No p-value: callers own their nulls."""
    ra, rb = rank(a), rank(b)
    if ra.size != rb.size or ra.size < 2:
        return float("nan")
    if not (np.isfinite(ra).all() and np.isfinite(rb).all()):
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
