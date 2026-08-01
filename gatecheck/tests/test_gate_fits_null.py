import numpy as np
import pytest

from gatecheck.gate import Gate, gated, DECIDED, NOT_DECIDABLE
from gatecheck.fits import scan_minimum, require_off_edge, EdgeRejection, slope_loglog
from gatecheck.nulltest import (
    assert_exact_zero, certify_null, VacuousNullError, BrokenCouplingError,
    effect_beyond_control,
)

TRUTH = 2.0


def mean_estimator_on_reference(geometry, seed):
    """Known-answer reference: N(TRUTH, 1) sampled `geometry` times; estimator = sample mean."""
    rng = np.random.default_rng(seed)
    return float(rng.normal(TRUTH, 1.0, size=geometry).mean())


class TestGate:
    def test_generous_geometry_passes(self):
        g = Gate(mean_estimator_on_reference, TRUTH, tolerance_pct=5.0)
        c = g.check(geometry=4096, seeds=range(20))
        assert c.passes and c.n_seeds == 20
        q = c.quantities["value"]
        assert q["dev_pct"] + q["spread_pct"] <= 5.0

    def test_starved_geometry_fails_on_spread_not_just_bias(self):
        # the estimator is unbiased at any n; the gate must still fail when the
        # seed-to-seed spread swamps the tolerance (textca: "a coin flip reported
        # as a decision")
        g = Gate(mean_estimator_on_reference, TRUTH, tolerance_pct=5.0)
        c = g.check(geometry=4, seeds=range(20))
        assert not c.passes
        assert c.quantities["value"]["spread_pct"] > 5.0

    def test_disputed_truth_keeps_kindest(self):
        g = Gate(mean_estimator_on_reference, [10.0, TRUTH], tolerance_pct=5.0)
        c = g.check(geometry=4096, seeds=range(10))
        assert c.truth_label == "truth[1]" and c.passes

    def test_named_quantities_and_mismatch_detection(self):
        def est(geometry, seed):
            rng = np.random.default_rng(seed)
            x = rng.normal(TRUTH, 1.0, size=geometry)
            return {"mean": float(x.mean()), "sd": float(x.std())}

        g = Gate(est, {"mean": TRUTH, "sd": 1.0}, tolerance_pct=8.0)
        assert g.check(4096, seeds=range(12)).passes
        bad = Gate(est, {"mean": TRUTH, "wrong_name": 1.0}, tolerance_pct=8.0)
        with pytest.raises(ValueError):
            bad.check(64, seeds=range(3))

    def test_needs_two_seeds(self):
        g = Gate(mean_estimator_on_reference, TRUTH, tolerance_pct=5.0)
        with pytest.raises(ValueError):
            g.check(64, seeds=[1])

    def test_gated_verdicts(self):
        g = Gate(mean_estimator_on_reference, TRUTH, tolerance_pct=5.0)
        ok, starved = g.check(4096, range(12)), g.check(4, range(12))

        v = gated(ok, measure=lambda: 42.0)
        assert v.decided and v.status == DECIDED and v.value == 42.0

        ran = []
        v2 = gated(starved, measure=lambda: ran.append(1) or 42.0)
        assert v2.status == NOT_DECIDABLE and v2.value is None
        assert ran == [], "measurement must not run behind a failing gate"
        assert "tolerance" in v2.reason and v2.block()["value"] is None

    def test_ladder_finds_cheapest_passing(self):
        g = Gate(mean_estimator_on_reference, TRUTH, tolerance_pct=5.0)
        lad = g.ladder([4, 256, 4096], seeds=range(12), cost_hours=lambda n: n / 100.0)
        by_geom = {r["geometry"]: r for r in lad["rungs"]}
        assert not by_geom[4]["passes"]
        assert lad["cheapest_passing"] is not None
        # cheapest passing rung by cost, not the most precise one
        passing = [r for r in lad["rungs"] if r["passes"]]
        assert lad["cheapest_passing"]["hours"] == min(r["hours"] for r in passing)


class TestFits:
    def test_interior_minimum_ok(self):
        grid = np.linspace(0, 4, 41)
        costs = (grid - 2.0) ** 2
        fit = require_off_edge(scan_minimum(grid, costs))
        assert fit.value == pytest.approx(2.0) and not fit.on_edge

    def test_edge_minimum_rejected(self):
        grid = np.linspace(0, 4, 41)
        costs = grid.copy()          # minimum at the left edge
        fit = scan_minimum(grid, costs, edge_margin=2)
        assert fit.on_edge
        with pytest.raises(EdgeRejection):
            require_off_edge(fit)

    def test_slope_loglog_recovers_power_law(self):
        t = np.arange(1, 40, dtype=float)
        slope, r2 = slope_loglog(t, t ** 2.5)
        assert slope == pytest.approx(2.5, abs=1e-9) and r2 > 0.999999

    def test_slope_loglog_underpowered_returns_none(self):
        s, r2 = slope_loglog([1, 2, 3], [1, 0, 0])
        assert s is None and r2 is None


class TestNull:
    def test_exact_zero_passes_and_fails(self):
        assert_exact_zero(np.zeros(100))
        with pytest.raises(BrokenCouplingError):
            assert_exact_zero(np.array([0.0, 1e-12, 0.0]))

    def test_certified_pair(self):
        cert = certify_null(np.zeros(10), np.array([0, 1, 1, 0]))
        assert cert.null_exact and cert.effect_nonzero and cert.effect_magnitude == 2.0
        assert cert.block()["null_exact"] is True

    def test_vacuous_null_detected(self):
        with pytest.raises(VacuousNullError):
            certify_null(np.zeros(10), np.zeros(10))

    def test_effect_beyond_control(self):
        # textca F65: the control acquired the effect too -> generic, not a finding
        generic = effect_beyond_control(0.55, 0.52, min_gap=0.10)
        real = effect_beyond_control(0.74, 0.14, min_gap=0.10)
        assert not generic["effect"] and real["effect"]
        assert real["gap"] == pytest.approx(0.60)
