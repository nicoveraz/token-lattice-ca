"""The sub-alphabet sampler, tested against hand-computable cases before any model runs on it."""
import numpy as np
import pytest
from subalphabet import make_sampler, sub_init, pick_tokens


def test_sampler_returns_real_token_ids_from_the_sub_alphabet_only():
    sub = np.array([100, 200, 300])
    s = make_sampler(sub)
    probs = np.tile(np.eye(1, 500, 200)[0] * 0 + 1.0 / 500, (4, 1))
    out = s(probs, np.array([0.1, 0.4, 0.7, 0.99]))
    assert set(out.tolist()) <= set(sub.tolist())
    assert out.dtype == np.int64


def test_inverse_cdf_boundaries_are_exact_on_a_two_token_alphabet():
    """p = [0.3, 0.7] over the sub-alphabet: u<0.3 -> first, u>=0.3 -> second. No sampling."""
    sub = np.array([10, 20])
    s = make_sampler(sub)
    probs = np.zeros((4, 50)); probs[:, 10] = 0.3; probs[:, 20] = 0.7
    out = s(probs, np.array([0.0, 0.2999, 0.3001, 0.999]))
    assert out.tolist() == [10, 10, 20, 20]


def test_order_permutation_changes_which_token_a_given_uniform_selects():
    """This is the entire premise of #105: the coupling depends on the layout, the model does not.

    u must be chosen where the two layouts actually disagree. With p = [0.3, 0.7]:
        layout [10, 20]: cells [0,0.3) -> 10, [0.3,1.0) -> 20
        layout [20, 10]: cells [0,0.7) -> 20, [0.7,1.0) -> 10
    so u = 0.5 lands on 20 under BOTH and discriminates nothing, while u = 0.8 separates them.
    The first version of this test used 0.5 and failed for that reason -- the sampler was right.
    """
    sub = np.array([10, 20])
    probs = np.zeros((1, 50)); probs[:, 10] = 0.3; probs[:, 20] = 0.7
    u = np.array([0.8])
    assert make_sampler(sub, order=[0, 1])(probs, u).tolist() == [20]
    assert make_sampler(sub, order=[1, 0])(probs, u).tolist() == [10]


def test_renormalisation_ignores_mass_outside_the_sub_alphabet():
    """Only relative weights within the support may matter; everything else is projected away."""
    sub = np.array([10, 20])
    a = np.zeros((1, 50)); a[:, 10] = 0.3; a[:, 20] = 0.7
    b = np.zeros((1, 50)); b[:, 10] = 0.03; b[:, 20] = 0.07; b[:, 30] = 0.90
    u = np.array([0.5])
    assert make_sampler(sub)(a, u) == make_sampler(sub)(b, u)


def test_a_row_with_no_sub_alphabet_mass_falls_back_to_uniform_not_nan():
    sub = np.array([10, 20])
    probs = np.zeros((1, 50)); probs[:, 30] = 1.0
    out = make_sampler(sub)(probs, np.array([0.75]))
    assert out.tolist() == [20] and np.isfinite(out).all()


def test_order_must_be_a_permutation():
    with pytest.raises(ValueError):
        make_sampler(np.array([1, 2, 3]), order=[0, 0, 1])


def test_sub_init_stays_inside_the_support():
    sub = np.array([7, 8, 9])
    lat = sub_init(sub, 5, 12, np.random.default_rng(0))
    assert lat.shape == (5, 12) and set(np.unique(lat).tolist()) <= set(sub.tolist())


def test_binary_alphabet_has_one_distinct_ordering_up_to_symmetry():
    """#105's known-answer control: at |V|=2 the two orderings are mirror images, so under a
    shared uniform stream the ORDERING SPREAD of any symmetric statistic must be exactly zero.
    Verified here as the sampler-level identity it rests on."""
    sub = np.array([10, 20])
    probs = np.zeros((3, 50)); probs[:, 10] = 0.4; probs[:, 20] = 0.6
    u = np.array([0.1, 0.5, 0.9])
    # Exact statement of the symmetry: reversing the layout AND complementing the uniform is the
    # identity. layout [a(p), b(1-p)] sends u<p to a; layout [b(1-p), a(p)] sends 1-u<1-p, i.e.
    # u>p, to b -- the same partition. So at |V|=2 ordering carries no information.
    fwd = make_sampler(sub, order=[0, 1])(probs, u)
    rev = make_sampler(sub, order=[1, 0])(probs, 1.0 - u)
    assert fwd.tolist() == rev.tolist()
