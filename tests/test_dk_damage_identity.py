"""Phase 2.2 -- the Domany-Kinzel rung as a BIT-EXACT test of the CRN damage machinery.

Every other check in this project on damage spreading is statistical: a critical point with
an error bar, a bootstrap CI, a p-value. This one is not. Kohring & Schreckenberg (J. Phys.
I France 2, 2033 (1992)) showed that on the p2 = 0 line of the DK automaton the damage
field between two commonly-driven replicas is *itself* a DK automaton at the same p1:

    d'_i = (d_{i-1} XOR d_{i+1}) . theta(p1 - z_i)

So we can predict the damage trajectory exactly, from an independent simulator, and demand
bit-identity. If the shared-uniform consumption order, the window indexing, the inverse-CDF
sampling or the synchronous update were wrong in `lattice.run`, this fails immediately --
and `lattice.run` is the loop that produces every language-model number in the project.

The identity is special to p2 = 0. `test_identity_fails_off_the_line` checks that it does
fail elsewhere, so the exact test cannot be passing vacuously.

Do NOT relax any assertion here to `allclose`: these are integer lattices and the whole
value of the rung is that agreement is exact.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import numpy as np
import pytest

from lattice import run as lattice_run
from dk import DKRule, dk_run, seed_state

B, N, SW = 4, 32, 12


def _stream(seed, sweeps=SW, n=N, b=B):
    return np.random.default_rng(seed).random(sweeps * n * b)


def _init(seed, n=N, b=B):
    return np.random.default_rng(seed).integers(0, 2, size=(b, n)).astype(np.int64)


# --------------------------------------------------------------- adapter fidelity
@pytest.mark.parametrize("p1,p2", [(0.0, 0.0), (0.3, 0.0), (0.8087, 0.0),
                                   (1.0, 1.0), (0.6, 0.5), (0.705489, 0.705489)])
def test_rule_path_matches_vectorised_reference(p1, p2):
    """`DKRule` through the shared loop == the vectorised reference, bit for bit.

    This is what licenses using the fast reference for the sweeps in `dk_calib.py` while
    claiming the results describe the same dynamics the `Rule` path implements.
    """
    init, u = _init(1), _stream(2)
    got = lattice_run(DKRule(p1, p2), B=B, N=N, r=1, T=1.0, sweeps=SW, mode="sync",
                      seed=71, init_state=init, u_stream=u)["snaps"]
    want = dk_run(init, u, p1, p2, SW)
    assert got.shape == want.shape
    assert np.array_equal(got, want), (
        f"Rule path and reference disagree at p1={p1}, p2={p2}: "
        f"{(got != want).sum()} differing cells")


# --------------------------------------------------------------- the exact anchor
@pytest.mark.parametrize("p1", [0.2, 0.5, 0.8087, 0.95, 1.0])
@pytest.mark.parametrize("seed", [0, 17])
def test_damage_field_is_a_dk_automaton_on_the_p2_zero_line(p1, seed):
    """Kohring-Schreckenberg: at p2=0 the damage field obeys the DK rule at the same p1.

    Twins run through `lattice.run` sharing one uniform stream (CRN). Their XOR must equal,
    at every site and every sweep, an independent DK run started from the initial damage.
    """
    init, u = _init(seed), _stream(seed + 100)
    perturbed = init.copy()
    perturbed[:, N // 2] ^= 1
    kw = dict(B=B, N=N, r=1, T=1.0, sweeps=SW, mode="sync", seed=71, u_stream=u)
    a = lattice_run(DKRule(p1, 0.0), init_state=init, **kw)["snaps"]
    b = lattice_run(DKRule(p1, 0.0), init_state=perturbed, **kw)["snaps"]

    observed = a ^ b                                   # damage field, from the real loop
    predicted = dk_run(init ^ perturbed, u, p1, 0.0, SW)   # damage field, predicted exactly

    assert np.array_equal(observed, predicted), (
        f"damage field is not a DK automaton at p1={p1}, seed={seed}: "
        f"{(observed != predicted).sum()} differing cells -- the CRN coupling, the window "
        f"indexing or the uniform consumption order is wrong")


def test_identity_is_not_vacuous_damage_actually_survives():
    """The exact test would be trivial if damage always died. Above the transition it doesn't."""
    init, u = _init(3), _stream(4)
    perturbed = init.copy()
    perturbed[:, N // 2] ^= 1
    kw = dict(B=B, N=N, r=1, T=1.0, sweeps=SW, mode="sync", seed=71, u_stream=u)
    a = lattice_run(DKRule(0.95, 0.0), init_state=init, **kw)["snaps"]
    b = lattice_run(DKRule(0.95, 0.0), init_state=perturbed, **kw)["snaps"]
    assert (a[-1] != b[-1]).any(), "damage vanished everywhere; the exact test is vacuous"


@pytest.mark.parametrize("p1,p2", [(0.6, 0.5), (0.705489, 0.705489), (0.8, 0.9)])
def test_identity_fails_off_the_p2_zero_line(p1, p2):
    """Control: the mapping is special to p2=0 and MUST break elsewhere.

    Without this, a bug that made both sides trivially equal (e.g. damage always zero)
    would look like a passing exact test.
    """
    init, u = _init(5), _stream(6)
    perturbed = init.copy()
    perturbed[:, N // 2] ^= 1
    kw = dict(B=B, N=N, r=1, T=1.0, sweeps=SW, mode="sync", seed=71, u_stream=u)
    a = lattice_run(DKRule(p1, p2), init_state=init, **kw)["snaps"]
    b = lattice_run(DKRule(p1, p2), init_state=perturbed, **kw)["snaps"]
    predicted = dk_run(init ^ perturbed, u, p1, p2, SW)
    assert not np.array_equal(a ^ b, predicted), (
        f"the damage identity held at p2={p2} != 0, where it has no right to")


# --------------------------------------------------------------- null + guards
@pytest.mark.parametrize("p1,p2", [(0.3, 0.0), (0.705489, 0.705489), (0.8087, 0.0)])
def test_null_dk_exact_zero(p1, p2):
    """The project-wide guarantee, extended to the DK backend: identical inputs, zero drift."""
    init, u = _init(7), _stream(8)
    kw = dict(B=B, N=N, r=1, T=1.0, sweeps=SW, mode="sync", seed=71,
              init_state=init, u_stream=u)
    a = lattice_run(DKRule(p1, p2), **kw)["snaps"]
    b = lattice_run(DKRule(p1, p2), **kw)["snaps"]
    assert np.array_equal(a, b), "DK null arm diverged: CRN coupling broken"


def test_odd_ring_is_rejected():
    """An odd ring joins DK's two sublattices through the seam; that must not pass silently."""
    with pytest.raises(ValueError, match="sublattices"):
        seed_state(2, 31)
    with pytest.raises(ValueError, match="sublattices"):
        DKRule(0.5, 0.5).random_lattice(np.random.default_rng(0), 2, 31)


def test_radius_other_than_one_is_rejected():
    """DK is a two-input rule; silently running it at r=2 would not be the DK model."""
    with pytest.raises(ValueError, match="two-input"):
        DKRule(0.5, 0.5).window(0, 2, N)


def test_absorbing_state_is_absorbing():
    """P[1|0,0]=0 exactly: the all-zero configuration can never revive, at any p1, p2."""
    zeros = np.zeros((B, N), dtype=np.int64)
    u = _stream(9)
    for p1, p2 in [(1.0, 1.0), (0.9, 0.4)]:
        snaps = dk_run(zeros, u, p1, p2, SW)
        assert snaps.sum() == 0, f"absorbing state revived at p1={p1}, p2={p2}"
