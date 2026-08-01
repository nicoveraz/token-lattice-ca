"""The baseline suite's own invariants (#20, §5.2).

The head-to-head answers "is assembly theory a repackaged compressor?", so a broken baseline biases
the answer toward yes-it-is-different, which is the direction that flatters the project. These
assert the properties that keep the comparison honest.
"""
import sys
import pathlib
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]

from assembly_calib import addition_chain_length, A_exp, lg, pole_cases, BUDGET
from assembly_baselines import (sequitur_slp_size, lz77_phrases, repair_assembly_index,
                                profile, zprofile, _selfcheck, MEASURES, OURS)

MULTISET_INVARIANTS = ["n_words", "n_types", "H0"]


def test_sequitur_passes_the_same_provable_families_as_the_gated_estimator():
    """A baseline gets no free pass. It is checked against the references the estimator is gated on."""
    sc = _selfcheck()
    assert sc["exact_on_no_reuse"], "Sequitur is not exact on all-distinct strings, where the index is forced"
    assert sc["sound_on_a_n"], "Sequitur returned a value below a proven addition-chain minimum"


@pytest.mark.parametrize("n", [2, 4, 8, 15, 16, 32, 64])
def test_sequitur_and_repair_agree_with_each_other_on_the_no_reuse_family(n):
    """Two independent grammar heuristics must both hit n-1 where no reuse is possible."""
    seq = [f"s{i}" for i in range(n)]
    assert sequitur_slp_size(seq) == n - 1 == repair_assembly_index(seq)


@pytest.mark.parametrize("s", ["abracadabra" * 6, "the cat sat on the mat " * 9, "a" * 100,
                               "".join(chr(97 + i % 26) for i in range(300))])
def test_lz77_bounds_TWICE_the_assembly_index_not_the_assembly_index(s):
    """z/2 <= ASI, and asserting the stronger z <= ASI would be asserting something false.

    z <= g (Rytter 2003; Charikar et al. 2005) is stated for g = SUM |RHS| over the grammar's
    rules. A binary SLP with r rules has total RHS length 2r, and the assembly index IS the binary
    rule count, so g = 2 * ASI. §4.1 and §5.2 of assembly_theory.md read the theorem as making z a
    lower bound on the assembly index and report a bracket [z, RePair]; that is a unit error, and
    z exceeds RePair on ordinary text. Both halves are pinned here.
    """
    z, g = lz77_phrases(s), repair_assembly_index(s)
    assert z / 2 <= g, (
        f"z/2 = {z/2} exceeded the assembly index {g} on {s[:24]!r} -- the corrected bound fails, "
        f"so the bracket [z/2, RePair] is not a bracket either.")


@pytest.mark.parametrize("measure", MULTISET_INVARIANTS)
def test_the_multiset_invariants_really_are_invariant_under_the_control(measure):
    """Provable, so asserted: a word shuffle permutes the multiset and these depend on it alone.

    This is what makes the InChI-length confound (r = 0.95 between string length and assembly
    index, the critics' most dangerous baseline) unable to operate in this design -- and the
    analysis excludes these from the peak test on the strength of it.
    """
    words = pole_cases()["real_text"][:BUDGET]
    z = zprofile(words, k=3, seed=0)[measure]
    assert abs(z["contrast"]) < 1e-9, (
        f"{measure} moved by {z['contrast']} under a word shuffle, so it is NOT a function of the "
        f"multiset alone and excluding it from the peak test is unjustified.")


def test_z_is_NOT_a_lower_bound_on_the_assembly_index_as_the_doc_claimed():
    """The counterexample, pinned so the unit error cannot return as a "bracket [z, RePair]"."""
    s = "abracadabra" * 6
    assert lz77_phrases(s) > repair_assembly_index(s), (
        "z no longer exceeds RePair here. If the LZ77 implementation changed, re-derive the bound "
        "rather than restoring the [z, RePair] wording -- the units argument stands regardless.")


def test_A_ranks_real_text_above_degenerate_repetition_where_compression_does_the_opposite():
    """The headline shape result, asserted on the two regimes that produce it.

    Every compression baseline responds more strongly to a 2-cycle than to real English; A inverts
    that, and the exponential weighting is why (§3.3). If this ever flips, the §5.2 verdict is void.
    """
    c = pole_cases()
    real, degen = c["real_text"][:BUDGET], c["degenerate_x2"][:BUDGET]
    assert lg(A_exp(real)[0]) > lg(A_exp(degen)[0]), (
        "A ranks degenerate repetition at or above real text -- the failure the exponential "
        "weighting exists to prevent.")
    pr, pd = profile(real), profile(degen)
    assert pd["gzip_bits"] < pr["gzip_bits"], (
        "gzip no longer compresses a 2-cycle harder than real text, so the contrast that makes "
        "the shape argument is gone.")
