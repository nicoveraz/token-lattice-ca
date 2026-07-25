"""Phase 1.4 -- the exact-zero CRN null, parametrized over EVERY backend and mode.

This is the point of the unification. The null test is the single guarantee that makes
every damage and differential number in the project meaningful: twin runs sharing model,
init, update order and uniform stream must diverge by EXACTLY zero. Before this file it
covered only the toy JAX path, while `mlm_ca.run` and `ar_ca.run` -- which produce every
headline result in the paper -- had no regression coverage at all.

Because all three backends now go through `lattice.run`, a `StubRule` (no model load, runs
in milliseconds) exercises the same loop the real backends use, so the coupling guarantee is
covered BY CONSTRUCTION. The real backends are additionally tested directly, marked slow.
"""
import sys, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pytest

from lattice import run as lattice_run, symmetric_window, causal_window

B, N, R, SW = 4, 24, 2, 6
V = 32


class StubRule:
    """Deterministic toy distribution -- no model, no device, exercises the real loop.

    Probabilities depend on the window contents (so the dynamics are non-trivial and a
    perturbation can actually propagate), but contain no randomness of their own: every
    stochastic choice comes from the external uniform stream, which is what CRN requires.
    """

    def __init__(self, causal=False, vocab=V):
        self.causal, self.V = causal, vocab

    def window(self, i, r, N):
        return causal_window(i, r, N) if self.causal else symmetric_window(i, r, N)

    def probs(self, win, T):
        w = np.asarray(win, dtype=np.int64)
        # a smooth, deterministic function of the window: peak near the window mean
        centre = (w.sum(axis=1) % self.V).astype(np.float64)          # (B,)
        grid = np.arange(self.V)[None, :]                             # (1,V)
        d = np.minimum(np.abs(grid - centre[:, None]),
                       self.V - np.abs(grid - centre[:, None]))
        logits = -d / max(T, 1e-6)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def sample(self, probs, u):
        cdf = np.cumsum(probs, axis=-1)
        cdf /= cdf[:, -1:]
        return np.array([np.searchsorted(cdf[b], u[b]) for b in range(len(u))],
                        dtype=np.int64)

    def random_lattice(self, rng, B, N):
        return rng.integers(0, self.V, size=(B, N)).astype(np.int64)


def _fixed(seed=5):
    rng = np.random.default_rng(seed)
    init = rng.integers(0, V, size=(B, N)).astype(np.int64)
    u = np.random.default_rng(seed + 1).random(SW * N * B)
    return init, u


# ------------------------------------------------------------------ stub: loop-level
@pytest.mark.parametrize("causal", [False, True], ids=["symmetric", "causal"])
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("T", [0.3, 0.7, 1.5])
def test_null_identical_stream_zero_divergence_stub(causal, mode, T):
    """Same rule + init + order + uniforms => bit-identical trajectories, all sweeps."""
    rule = StubRule(causal=causal)
    init, u = _fixed()
    kw = dict(B=B, N=N, r=R, T=T, sweeps=SW, mode=mode, seed=71, u_stream=u)
    a = lattice_run(rule, init_state=init, **kw)
    b = lattice_run(rule, init_state=init, **kw)
    assert np.array_equal(a["snaps"], b["snaps"]), "null arm diverged: CRN coupling broken"
    assert (a["snaps"] != b["snaps"]).sum() == 0


@pytest.mark.parametrize("causal", [False, True], ids=["symmetric", "causal"])
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_perturbation_does_propagate_stub(causal, mode):
    """Sanity counterpart: with a flip, the SAME stream must NOT stay identical.

    Without this the null test could pass trivially (e.g. if the rule ignored its window),
    which would make the exact-zero guarantee meaningless.

    Uses a BLOCK flip, matching the real damage protocol (`block_damage(..., block=3)`),
    not a single site. Reason, verified while writing this test: under a CAUSAL window a
    site's new value does not depend on its own old value, so in async mode a single-site
    flip is erased outright whenever that site is visited before its dependants (with
    seed 71: site 12 is visited at step 8, its dependants 13 and 14 at steps 17 and 10).
    That is a genuine property of the dynamics, not a harness bug -- and it is exactly why
    the AR/MLM experiments perturb a block.
    """
    rule = StubRule(causal=causal)
    init, u = _fixed()
    c = N // 2
    flipped = init.copy()
    for j in (c - 1, c, c + 1):
        flipped[:, j] = (flipped[:, j] + 7) % V
    kw = dict(B=B, N=N, r=R, T=0.3, sweeps=SW, mode=mode, seed=71, u_stream=u)
    a = lattice_run(rule, init_state=init, **kw)
    b = lattice_run(rule, init_state=flipped, **kw)
    assert (a["snaps"][-1] != b["snaps"][-1]).any(), "perturbation vanished; null test is vacuous"


# ------------------------------------------------------------------ real backends
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_null_mlm(mode):
    from mlm_ca import MLMRule, run
    rule = MLMRule("prajjwal1/bert-tiny")
    rng = np.random.default_rng(3)
    init = rule.init_pool[rng.integers(0, len(rule.init_pool), size=(B, N))]
    u = np.random.default_rng(4).random(SW * N * B)
    kw = dict(B=B, N=N, r=R, T=0.7, sweeps=SW, mode=mode, scheme="cls_sep",
              seed=71, init_state=init, u_stream=u)
    a, b = run(rule, **kw), run(rule, **kw)
    assert np.array_equal(a["snaps"], b["snaps"]), "MLM null arm diverged"


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_null_ar(mode):
    from ar_ca import ARRule, run
    rule = ARRule("EleutherAI/pythia-14m")
    rng = np.random.default_rng(3)
    init = rule.init_pool[rng.integers(0, len(rule.init_pool), size=(B, N))]
    u = np.random.default_rng(4).random(SW * N * B)
    kw = dict(B=B, N=N, r=R, T=0.7, sweeps=SW, mode=mode, scheme="none",
              seed=71, init_state=init, u_stream=u)
    a, b = run(rule, **kw), run(rule, **kw)
    assert np.array_equal(a["snaps"], b["snaps"]), "AR null arm diverged"
