"""Harness regression tests for the token-lattice CA.

Run from the repo root:  ``.venv/bin/pytest -q``

The null test is the critical one: coupled twin runs that share model, init,
update order, and the uniform stream must diverge by *exactly* zero. If it ever
fails the common-random-number coupling is broken and every damage / differential
result is meaningless — fix the harness before trusting anything else.
"""
import numpy as np
import pytest

import ca
from model import CFG, load
from ca import run
from differential import coupled

CKPT = "ckpt/final.npz"


@pytest.fixture(scope="module")
def params():
    return load(CKPT)


# ---------------------------------------------------------------- null test
def test_null_coupling_is_exactly_zero(params):
    """Identical model + init + order + uniforms => zero divergence, all sweeps."""
    for T in (0.3, 0.7, 1.5):
        d = coupled(params, params, T, sample_b=None, B=4, N=48, sweeps=8, seed=71)
        assert d.max() == 0.0, f"null coupling diverged at T={T}: max={d.max()}"


def test_null_identical_stream_zero_divergence(params):
    """Lower-level check: two runs with the SAME init_state and u_stream are identical."""
    B, N, sweeps = 4, 48, 8
    rng = np.random.default_rng(123)
    init = rng.integers(2, CFG["vocab"], size=(B, N)).astype(np.int32)
    u = np.random.default_rng(9).random(sweeps * N * B)
    a = run(params, B=B, N=N, r=2, T=0.7, sweeps=sweeps, mode="async",
            init_state=init, seed=5, u_stream=u)
    b = run(params, B=B, N=N, r=2, T=0.7, sweeps=sweeps, mode="async",
            init_state=init, seed=5, u_stream=u)
    assert np.array_equal(a["snaps"], b["snaps"])


# ------------------------------------------------------------- determinism
def test_determinism_same_seed_bit_identical(params):
    """Same seed twice (init, order, and u_stream all derived from it) => identical."""
    kw = dict(B=6, N=48, r=2, T=1.0, sweeps=10, mode="async", init="random", seed=42)
    a = run(params, **kw)
    b = run(params, **kw)
    assert np.array_equal(a["snaps"], b["snaps"])
    assert np.array_equal(a["final"], b["final"])


def test_different_seed_differs(params):
    """Sanity converse: different seeds should NOT be identical (else seeding is dead)."""
    a = run(params, B=6, N=48, r=2, T=1.0, sweeps=10, mode="async", init="random", seed=1)
    b = run(params, B=6, N=48, r=2, T=1.0, sweeps=10, mode="async", init="random", seed=2)
    assert not np.array_equal(a["snaps"], b["snaps"])


# --------------------------------------------------------- block-Gibbs sanity
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_shapes_and_mask_never_emitted(params, mode):
    B, N, sweeps, r = 5, 48, 12, 2
    out = run(params, B=B, N=N, r=r, T=1.0, sweeps=sweeps, mode=mode,
              init="random", seed=7)
    assert out["snaps"].shape == (sweeps + 1, B, N)
    assert out["final"].shape == (B, N)
    assert out["activity"].shape == (sweeps, B)
    # MASK id (0) must never appear: the rule sets its logit to -1e9.
    assert (out["snaps"] == 0).sum() == 0, f"{mode}: MASK token emitted"
    # all tokens are valid vocab ids
    assert out["snaps"].min() >= 0 and out["snaps"].max() < CFG["vocab"]


def test_activity_bounds(params):
    out = run(params, B=4, N=48, r=2, T=2.5, sweeps=10, mode="async",
              init="random", seed=3)
    assert (out["activity"] >= 0).all() and (out["activity"] <= 1).all()
