"""Tie-aware ranking, because `np.argsort(np.argsort(x))` is not one.

THE BUG THIS REPLACES. Fifteen scripts in this project ranked with the idiom

    rk = lambda x: np.argsort(np.argsort(x))

which is correct only when every value is distinct. `argsort` breaks ties by INPUT POSITION, so a
vector with repeats is assigned strictly increasing ranks that encode the order the values happened
to be listed in. The degenerate case is the loud one: on a CONSTANT vector the idiom returns
[0, 1, ..., n-1], i.e. a perfectly monotone rank sequence for a quantity that never moves. Pearson
on those ranks then reports a confident correlation. `damage_geometry` produced rho = +0.829,
p = 0.058 against lambda for `front_width` when all 24 measured values were exactly 0.000; scipy
returns nan for the same input.

This is the project's recurring defect class -- a statistically-shaped criterion applied to a
quantity with no room to vary -- reached for the first time through the CORRELATION FUNCTION rather
than through the data. It is worse than the earlier instances in one specific way: `gatecheck`'s
leverage primitives inspect the data, and the data here was honest. Nothing outside the ranking
could have seen it.

WHAT THIS DOES.
  * ties get AVERAGED ranks (`scipy.stats.rankdata`), which is the Spearman definition
  * a zero-variance input returns all-nan, so any correlation computed from it is nan rather than a
    number. Callers must gate on nan. Returning a plausible float for an unmeasurable quantity is
    exactly the failure being removed, so this deliberately propagates instead of defaulting.

Ranks are 1-based floats rather than 0-based ints. Every call site feeds them to a correlation,
where an affine shift is irrelevant; do not use these as indices.
"""
import numpy as np
from scipy.stats import rankdata


def rank(x):
    """Averaged ranks; all-nan when `x` has no variance to rank."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return a
    if not np.isfinite(a).all():
        return np.full(a.shape, np.nan)
    if a.std() == 0:
        return np.full(a.shape, np.nan)
    return rankdata(a).astype(float)


def spearman(a, b):
    """Spearman rho, nan when either side is degenerate. No p-value: callers own their nulls."""
    ra, rb = rank(a), rank(b)
    if not (np.isfinite(ra).all() and np.isfinite(rb).all()):
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
