"""Harness regression test for the real-MLM path (Phase 3).

The same critical invariant as the toy null test: coupled twin runs sharing model,
init, update order, and uniforms must diverge by exactly zero. Skipped when torch /
transformers are missing or bert-tiny is not cached (so `pytest tests/` stays green
on a CPU-only checkout without the Phase-3 deps or network).
"""
import os
import numpy as np
import pytest

from _backends import load_or_skip

os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def rule():
    from mlm_ca import MLMRule
    return load_or_skip(lambda: MLMRule("prajjwal1/bert-tiny", fp16=False),
                        "prajjwal1/bert-tiny")


def test_mlm_null_coupling_is_exactly_zero(rule):
    from mlm_ca import run
    B, N, sweeps = 3, 24, 6
    rng = np.random.default_rng(7)
    init = rule.random_lattice(rng, B, N)
    u = np.random.default_rng(9).random(sweeps * N * B)
    a = run(rule, B=B, N=N, r=2, T=0.8, sweeps=sweeps, init_state=init, seed=5, u_stream=u)
    b = run(rule, B=B, N=N, r=2, T=0.8, sweeps=sweeps, init_state=init, seed=5, u_stream=u)
    assert np.array_equal(a["snaps"], b["snaps"]), "MLM null coupling diverged"


def test_mlm_mask_never_emitted(rule):
    from mlm_ca import run
    out = run(rule, B=3, N=24, r=2, T=1.0, sweeps=6, init="random", seed=1)
    assert out["snaps"].shape == (7, 3, 24)
    # no forbidden token (mask/special/unused) is ever emitted
    forb = set(rule.forbidden.tolist())
    assert not (set(np.unique(out["snaps"]).tolist()) & forb), "forbidden token emitted"
