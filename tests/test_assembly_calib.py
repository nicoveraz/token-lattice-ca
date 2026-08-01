"""The assembly estimator's provable properties, asserted directly (#20, §5.1).

WHY THESE EXIST SEPARATELY FROM THE EXPERIMENT. `assembly_calib.py` gates itself at run time, but a
refactor of the estimator would be caught only the next time somebody ran it -- and the results file
would still be sitting there looking calibrated. These assert the properties against their PROVEN
references, so degrading the estimator fails the suite immediately.

The properties are not all of the same kind, and the distinction is the point:

  SOUNDNESS is guaranteed by construction and is load-bearing. RePair exhibits a grammar, so its
  value is achievable and can never be below the true index. Everything downstream assumes the
  number is an upper bound; if that breaks, nothing is usable.

  EXACTNESS holds on the no-reuse family only, where nothing repeats so the pathway is forced.

  NON-exactness on a^n is asserted too, at the specific n where it starts. That looks odd until you
  notice the alternative: assembly_theory.md §5.1 claimed exactness there from a 14-point sample,
  and pinning the counterexample is what stops that claim coming back.
"""
import sys
import pathlib
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]

from assembly_calib import (repair_assembly_index, addition_chain_length, A_exp, delta,
                            distinct_exactness, pole_check, decides, calibrate, POLES, SIGNAL)

# OEIS A003313, l(n) for n = 1..20 -- an INDEPENDENT reference for the reference itself. The BFS in
# addition_chain_length is code like any other, and checking the estimator against it would be
# circular if the chain lengths were never checked against anything.
PUBLISHED_CHAIN_LENGTHS = [0, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 4, 5, 5, 5, 4, 5, 5, 6, 5]


def test_addition_chain_lengths_match_the_published_values():
    """The proven reference must itself reproduce a published table, or it proves nothing."""
    got = [addition_chain_length(n) for n in range(1, 21)]
    assert got == PUBLISHED_CHAIN_LENGTHS, (
        f"addition_chain_length disagrees with OEIS A003313: got {got}, "
        f"expected {PUBLISHED_CHAIN_LENGTHS}. The reference the estimator is gated against is "
        f"wrong, so the gate means nothing.")


@pytest.mark.parametrize("n", list(range(2, 49)) + [63, 100, 128])
def test_repair_never_returns_below_a_proven_lower_bound(n):
    """SOUNDNESS -- the load-bearing property, and the one that makes it a certified upper bound.

    A value below the minimal addition-chain length would mean RePair reported a pathway shorter
    than the shortest pathway that exists, which is impossible -- so it would mean the estimator or
    the step accounting is wrong, and every assembly number in the project would be unusable.
    """
    exact, got = addition_chain_length(n), repair_assembly_index("a" * n)
    assert got >= exact, (
        f"repair_assembly_index('a'*{n}) = {got} is BELOW the proven minimum {exact}. RePair "
        f"exhibits a grammar, so its value is achievable by construction and cannot be below the "
        f"true index -- this means the step accounting is wrong, not that a shorter pathway exists.")


@pytest.mark.parametrize("n", [2, 3, 5, 8, 16, 24, 32, 64, 100, 128])
def test_repair_is_exact_where_the_pilot_sampled(n):
    """The 14-point sample §3.1 reported. It is genuinely exact here -- that was never the error."""
    assert repair_assembly_index("a" * n) == addition_chain_length(n)


@pytest.mark.parametrize("n,exact,repair", [(15, 5, 6), (23, 6, 7), (63, 8, 10)])
def test_repair_is_NOT_exact_on_a_n_at_the_known_counterexamples(n, exact, repair):
    """Pins the overshoot, so "RePair is exact on a^n" cannot quietly return.

    n=15 is the textbook smallest case where the binary method is not an optimal addition chain:
    the minimum is 5 (1,2,3,6,12,15) and greedy halving finds 6. §5.1 asserted exactness for n up
    to 128 on the strength of a sample that skipped every failure; an exhaustive sweep finds 52.
    """
    assert addition_chain_length(n) == exact
    assert repair_assembly_index("a" * n) == repair, (
        f"the overshoot at n={n} changed. If the estimator genuinely improved, update this test and "
        f"assembly_calib's measured rate together -- do not delete the case.")


def test_repair_is_exact_on_the_no_reuse_family():
    """EXACTNESS -- the genuine rung. Nothing repeats, so nothing can be reused and the index is n-1."""
    d = distinct_exactness(nmax=128)
    assert d["exact"], f"no-reuse family is no longer exact: {d['mismatches'][:5]}"


def test_A_is_zero_when_no_object_repeats():
    """The property that separates A from entropy, and the reason it is not maximised by noise.

    Every object unique means (n_i - 1) = 0 for every type, so A = 0 no matter how high the
    individual assembly indices are. An entropy would be MAXIMAL on this input.
    """
    a, n_rep, _ = A_exp([f"w{i}" for i in range(600)])
    assert a == 0.0 and n_rep == 0


def test_A_stays_small_when_copy_number_is_huge_but_assembly_index_is_tiny():
    """The other pole: enormous repetition of a trivially-assembled object must not score high."""
    degenerate, _, _ = A_exp(["the"] * 600)
    real, _, _ = A_exp((_ROOT / "data" / "shakespeare.txt")
                       .read_text(errors="replace")[:12000].lower().split())
    assert degenerate < real, (
        f"degenerate repetition ({degenerate:.3g}) scored at or above real text ({real:.3g}). "
        f"That is the failure the exponential weighting exists to prevent -- under linear "
        f"weighting the measure is pure copy number and this inverts.")


def test_delta_pins_at_the_failure_poles_and_separates_on_real_text():
    """G3 end to end, at the experiment's own settings, so the suite gates what the run reports."""
    p = pole_check()
    for k in POLES:
        assert p["poles_pinned"][k], (
            f"failure pole {k!r} read Delta = {p['cases'][k]['delta']:+.3f}, outside "
            f"+/-{p['tolerance']}. The statistic is reporting structure where there is none.")
    assert p["signal_separates"], (
        f"real text read Delta = {p['cases'][SIGNAL]['delta']:+.3f}, below the floor "
        f"{p['real_min']}. The statistic has no signal to report.")


def test_the_gate_passes_as_a_whole():
    assert decides(calibrate()), "the calibration rung no longer passes its own gate"


def test_the_pilot_uses_the_gated_estimator_rather_than_a_copy_of_it():
    """Anti-drift (hazard 1). Two implementations that can disagree are the F56 defect in miniature."""
    import _assembly_pilot as pilot
    import assembly_calib as calib
    for fn in ("repair_assembly_index", "addition_chain_length", "A_exp", "delta", "lg"):
        assert getattr(pilot, fn) is getattr(calib, fn), (
            f"_assembly_pilot.{fn} is not assembly_calib.{fn} -- the pilot has its own copy again, "
            f"so the §3 tables can drift from the code the gate licenses.")
