"""A range gate that pure noise passes is not a gate.

`resolves_units` exists because `dynamic_range` has a footgun that bit this package's own author:
the natural floor to reach for is one observation's standard error, and comparing a SPAN to one SD
is close to meaningless, because the span of k draws from pure noise is already ~3.1 SD at k=10.

So the load-bearing test here is `test_pure_noise_does_not_pass`: synthetic data with NO true
between-unit signal must fail, and must fail at every k. If that ever passes, the gate is decorative
again.
"""
import math
import random

import pytest

from gatecheck import dynamic_range, expected_range, resolves_units


def noise_only(k, sd, seed):
    """k units drawn from ONE distribution: any spread between them is noise by construction."""
    rng = random.Random(seed)
    return [rng.gauss(0.2, sd) for _ in range(k)]


@pytest.mark.parametrize("k", [5, 10, 20])
@pytest.mark.parametrize("seed", range(8))
def test_pure_noise_does_not_pass(k, seed):
    sd = 0.1
    vals = noise_only(k, sd, seed)
    assert not resolves_units(vals, noise_sd=sd).usable


# Measured over 3000 noise-only draws per k, sd = 0.1. Both columns are the point: the old
# comparison is not merely imperfect, it is ~93% vacuous exactly where the incident sat (k = 10),
# and fully vacuous by k = 50. The new gate's own false-positive rate is reported rather than
# claimed to be zero -- it is 8% at k = 5, which is an honest statement that five units is hard.
#
#    k     old: span >= 2*SD passes noise     new: resolves_units passes noise
#    5                  60.9%                            8.3%
#   10                  92.5%                            3.3%
#   20                  99.7%                            0.5%
#   50                 100.0%                            0.0%
@pytest.mark.parametrize("k,old_min,new_max", [(5, 0.55, 0.15), (10, 0.88, 0.08), (20, 0.97, 0.03)])
def test_old_gate_passes_noise_and_the_new_one_rejects_it(k, old_min, new_max):
    """The defect measured as a RATE, because it is k-dependent and that detail matters.

    `span >= 2 * SD` passes whenever the realised range exceeds 2 SD, and E[range] grows with k:
    2.33 SD at k=5, 3.08 at k=10, 3.74 at k=20. So the old comparison is only *usually* vacuous at
    k=5 and essentially always vacuous from k=10 up.

    TWO OVER-STRONG ASSERTIONS WERE WRITTEN HERE BEFORE THIS ONE, both claiming a rate of 1.0 --
    first that the old gate passes noise at every k and seed (it fails 2 of 8 at k=5), then that the
    new gate catches noise every time (it lets 1 of 200 through). A gate's error rate is a number to
    measure, not a perfection to assert, and asserting perfection is how a test becomes flaky
    instead of informative.
    """
    sd, n = 0.1, 1000
    old = sum(dynamic_range(noise_only(k, sd, s), floor=sd, k=2.0).usable for s in range(n)) / n
    new = sum(resolves_units(noise_only(k, sd, s), noise_sd=sd).usable for s in range(n)) / n
    assert old >= old_min, (
        f"at k={k} the span-vs-one-SD gate passed pure noise only {old:.0%} of the time; if that is "
        f"genuinely lower now the illustration is stale, not the gate")
    assert new <= new_max, f"at k={k} resolves_units passed pure noise {new:.0%} of the time"
    assert new < old


def test_real_between_unit_signal_passes():
    """Units genuinely spread over 0.1-0.9 with small noise must be resolved."""
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.55]
    assert resolves_units(vals, noise_sd=0.02).usable


def test_the_incident_fails_the_new_gate_and_passed_the_old_one():
    """The measured case that produced this function: ten base models, cluster SD 0.107."""
    vals = [0.250, 0.167, 0.233, 0.175, 0.183, 0.225, 0.208, 0.208, 0.217, 0.167]
    assert dynamic_range(vals, floor=0.0367, k=2.0).usable          # what was reported at the time
    rep = resolves_units(vals, noise_sd=0.107)
    assert not rep.usable
    assert rep.stats["reliability"] < 0
    assert "attenuated" in rep.reason


def test_per_unit_noise_is_accepted():
    vals = [0.1, 0.5, 0.9, 0.3, 0.7]
    a = resolves_units(vals, noise_sd=0.02)
    b = resolves_units(vals, noise_sd=[0.02] * 5)
    assert a.usable and b.usable
    assert a.stats["var_noise"] == pytest.approx(b.stats["var_noise"])


def test_reliability_is_reported_even_when_it_passes():
    rep = resolves_units([0.1, 0.5, 0.9, 0.3, 0.7], noise_sd=0.02)
    assert 0.9 < rep.stats["reliability"] <= 1.0
    assert rep.stats["expected_noise_span"] > 0


def test_expected_range_is_monotone_and_matches_known_values():
    assert expected_range(2) == pytest.approx(1.128)
    assert expected_range(10) == pytest.approx(3.078)
    vals = [expected_range(k) for k in range(2, 60)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), "E[range] must not decrease in k"
    assert expected_range(1) == 0.0
    assert 3.9 < expected_range(25) < 4.1                    # interpolated
    assert 4.5 < expected_range(80) < 5.5                    # extrapolated, still sane


def test_zero_noise_is_refused_rather_than_dividing_by_zero():
    rep = resolves_units([0.1, 0.2, 0.3], noise_sd=0.0)
    assert not rep.usable
    assert math.isinf(rep.stats["expected_noise_span"])
