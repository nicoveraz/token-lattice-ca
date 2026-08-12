"""The ranking primitive must fail where the old idiom silently succeeded.

Each test below is a case `np.argsort(np.argsort(x))` gets wrong. The constant-vector case is the
one that actually fired in production (damage_geometry's front_width, rho = +0.829 on 24 identical
zeros), so it is asserted against the exact old behaviour rather than only against the new.
"""
import sys, pathlib
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / "experiments")]

import numpy as np
import pytest
from scipy.stats import spearmanr

from ranking import rank, spearman

LAM = [-0.0826, -0.0303, 0.0332, 0.1983, 0.1440, 0.1783]


def test_constant_input_is_nan_not_a_correlation():
    """The regression. The old idiom returned +0.829 here; scipy returns nan."""
    old = np.argsort(np.argsort([0.0] * 6))
    assert list(old) == [0, 1, 2, 3, 4, 5], "the old idiom ranks a constant as strictly increasing"
    assert abs(float(np.corrcoef(old, np.argsort(np.argsort(LAM)))[0, 1]) - 0.8286) < 1e-3

    assert np.isnan(rank([0.0] * 6)).all()
    assert np.isnan(spearman([0.0] * 6, LAM))


def test_ties_get_averaged_ranks_not_input_order():
    x = [5.0, 1.0, 5.0, 3.0]
    assert list(rank(x)) == [3.5, 1.0, 3.5, 2.0]
    # the old idiom would have split the tie by position, giving distinct ranks
    assert len(set(np.argsort(np.argsort(x)).tolist())) == 4


@pytest.mark.parametrize("a,b", [
    ([1.0, 2, 3, 4, 5, 6], LAM),
    ([3.0, 1, 4, 1, 5, 9], LAM),          # contains a tie
    ([2.0, 2, 2, 1, 1, 3], LAM),          # several ties
])
def test_matches_scipy_where_scipy_is_defined(a, b):
    assert abs(spearman(a, b) - float(spearmanr(a, b).statistic)) < 1e-9


def test_nonfinite_input_is_nan():
    assert np.isnan(rank([1.0, 2.0, np.nan])).all()
    assert np.isnan(spearman([1.0, 2.0, np.inf], [1.0, 2.0, 3.0]))


def test_agrees_with_the_packaged_copy_in_gatecheck():
    """`gatecheck.ranking` is a SECOND implementation, and the duplication is deliberate.

    gatecheck is laid out as a separable package and is not installed into this venv; its own
    conftest puts `gatecheck/src` on sys.path for its tests only, so that `import gatecheck` from
    the main suite keeps resolving to the namespace directory rather than quietly undoing the
    separation (see gatecheck/conftest.py). Making this module import the package would break that.

    So there are two copies, and the copies must not drift. This test is what makes that safe: it
    adds the path the way an experiment script does, and asserts the two agree on every case the
    rest of this file cares about -- including the degenerate ones, which is where an accidental
    "improvement" to either copy would show up first.
    """
    gc_src = pathlib.Path(__file__).resolve().parents[1] / "gatecheck" / "src"
    sys.path.insert(0, str(gc_src))
    try:
        for mod in ("gatecheck", "gatecheck.ranking"):
            sys.modules.pop(mod, None)
        from gatecheck.ranking import rank as grank, spearman as gspearman
    finally:
        sys.path.remove(str(gc_src))
        for mod in ("gatecheck", "gatecheck.ranking"):
            sys.modules.pop(mod, None)

    cases = [[0.0] * 6, LAM, [5.0, 1.0, 5.0, 3.0], [2.0, 2, 2, 1, 1, 3], [1.0, 2.0, np.nan], []]
    for x in cases:
        a, b = rank(x), grank(x)
        assert a.shape == b.shape
        assert np.array_equal(a, b, equal_nan=True), f"ranking copies disagree on {x}"
    for x in ([1.0, 2, 3, 4, 5, 6], [3.0, 1, 4, 1, 5, 9], [0.0] * 6):
        p, q = spearman(x, LAM), gspearman(x, LAM)
        assert (np.isnan(p) and np.isnan(q)) or abs(p - q) < 1e-12


def test_no_experiment_script_still_uses_the_untied_idiom():
    """Guard against the idiom coming back by copy-paste."""
    root = pathlib.Path(__file__).resolve().parents[1] / "experiments"
    bad = []
    for f in sorted(root.glob("*.py")):
        if f.name == "ranking.py":
            continue                                  # documents the idiom it replaces
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "argsort(np.argsort" in line and not line.lstrip().startswith(("#", "*")):
                if "`np.argsort" in line or "the old idiom" in line:
                    continue                      # prose inside a docstring
                bad.append(f"{f.name}:{i}")
    assert not bad, f"untied ranking idiom present in: {bad}"
