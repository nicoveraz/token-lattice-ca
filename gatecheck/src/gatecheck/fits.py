"""Fitted-scan hygiene: reject minima on the scan edge; shared log-log slope estimator.

Origin: textca F59. A finite-size-scaling fit reported z = 1.325 "excluding" the reference
class — and the reported minimum was sitting on the edge of the scan grid, i.e. it was the
grid, not a measurement. The repaired rule, generalized here: scan far wider than any
plausible value, and REJECT a fit whose optimum lands within `edge_margin` cells of either
end. An optimum on the edge means the true optimum may be outside the scan, and the honest
statement is "widen the scan", not the edge value.

`slope_loglog` is textca's shared power-law estimator (`dp_calibration.slope`): one
implementation, imported everywhere, because a pasted copy of a fitting convention is a drift
waiting to invert a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class EdgeRejection(ValueError):
    """A fitted optimum landed on the scan edge; the scan must be widened, not trusted."""


@dataclass
class ScanFit:
    value: float               # grid value at the optimum
    index: int
    cost: float                # cost at the optimum
    on_edge: bool
    edge_margin: int
    grid_lo: float
    grid_hi: float

    def block(self) -> dict:
        return {
            "value": self.value, "index": self.index, "cost": self.cost,
            "on_edge": self.on_edge, "edge_margin": self.edge_margin,
            "grid": [self.grid_lo, self.grid_hi],
        }


def scan_minimum(grid: Sequence[float], costs: Sequence[float], *, edge_margin: int = 1) -> ScanFit:
    """Locate the minimum of a scanned cost and record whether it touches the scan edge."""
    g = np.asarray(grid, dtype=float)
    c = np.asarray(costs, dtype=float)
    if g.shape != c.shape or g.ndim != 1 or g.size < 3:
        raise ValueError("grid and costs must be 1-D, equal length, and size >= 3")
    if edge_margin < 1:
        raise ValueError("edge_margin must be >= 1")
    i = int(np.nanargmin(c))
    on_edge = i < edge_margin or i >= g.size - edge_margin
    return ScanFit(
        value=float(g[i]), index=i, cost=float(c[i]), on_edge=on_edge,
        edge_margin=edge_margin, grid_lo=float(g[0]), grid_hi=float(g[-1]),
    )


def require_off_edge(fit: ScanFit) -> ScanFit:
    """Raise EdgeRejection when the optimum is on (or within margin of) the scan edge."""
    if fit.on_edge:
        raise EdgeRejection(
            f"fitted optimum {fit.value} at index {fit.index} lies within "
            f"{fit.edge_margin} cell(s) of the scan edge [{fit.grid_lo}, {fit.grid_hi}]: "
            f"the value is the grid, not a measurement — widen the scan"
        )
    return fit


def slope_loglog(t: Sequence[float], y: Sequence[float], *, min_points: int = 4):
    """Log-log slope and R^2, skipping non-positive y. Returns (None, None) when underpowered.

    The shared estimator convention matters more than the arithmetic: every power-law fit in a
    project should go through one function, so that fit windows and dropped points cannot
    silently differ between the calibration and the measurement (that difference is exactly
    what textca's F56 gate exists to catch).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = y > 0
    if int(ok.sum()) < min_points:
        return None, None
    lt, ly = np.log(t[ok]), np.log(y[ok])
    coef = np.polyfit(lt, ly, 1)
    resid = ly - np.polyval(coef, lt)
    denom = max(float(((ly - ly.mean()) ** 2).sum()), 1e-12)
    r2 = 1.0 - float((resid ** 2).sum()) / denom
    return float(coef[0]), r2
