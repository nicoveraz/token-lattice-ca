"""Golden-file regression: the simulation core must stay BIT-IDENTICAL across refactors.

The reference files in `tests/golden/` were generated from the pre-refactor code by
`tests/make_golden.py`. Any change to the simulation loop that alters a single token is a
behaviour change and must fail here.

DO NOT relax these assertions to `np.allclose` to make a refactor pass. If a backend cannot
be made bit-identical, stop and report why.
"""
import sys, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pytest

from make_golden import CFG_RUN, TOY_CKPT, MLM_NAME, AR_NAME, _streams

GOLD = ROOT / "tests" / "golden"
pytestmark = pytest.mark.skipif(not (GOLD / "ca.npz").exists(),
                                reason="golden files absent; run tests/make_golden.py")


def _assert_identical(tag, out):
    ref = np.load(GOLD / f"{tag}.npz")
    for key in ("snaps", "activity", "final"):
        got = np.asarray(out[key])
        exp = ref[key]
        assert got.shape == exp.shape, f"{tag}.{key}: shape {got.shape} != golden {exp.shape}"
        assert np.array_equal(got, exp), (
            f"{tag}.{key}: NOT bit-identical to golden "
            f"({int((got != exp).sum())} differing elements). "
            "Do not relax to allclose -- report the behaviour change.")


def test_golden_ca():
    from model import CFG, load
    import ca
    params = load(TOY_CKPT)
    init, u = _streams(CFG_RUN["B"], CFG_RUN["N"], CFG_RUN["sweeps"], 2, CFG["vocab"])
    out = ca.run(params, B=CFG_RUN["B"], N=CFG_RUN["N"], r=CFG_RUN["r"], T=CFG_RUN["T"],
                 sweeps=CFG_RUN["sweeps"], mode="async", seed=CFG_RUN["seed"],
                 init_state=init.astype(np.int32), u_stream=u)
    _assert_identical("ca", out)


def test_golden_mlm():
    from mlm_ca import MLMRule, run
    try:
        rule = MLMRule(MLM_NAME)
    except Exception as e:
        pytest.skip(f"{MLM_NAME} unavailable: {e}")
    init, u = _streams(CFG_RUN["B"], CFG_RUN["N"], CFG_RUN["sweeps"], 0, len(rule.init_pool))
    init = rule.init_pool[init]
    out = run(rule, B=CFG_RUN["B"], N=CFG_RUN["N"], r=CFG_RUN["r"], T=CFG_RUN["T"],
              sweeps=CFG_RUN["sweeps"], mode="async", scheme="cls_sep",
              seed=CFG_RUN["seed"], init_state=init, u_stream=u)
    _assert_identical("mlm", out)


def test_golden_ar():
    from ar_ca import ARRule, run
    try:
        rule = ARRule(AR_NAME)
    except Exception as e:
        pytest.skip(f"{AR_NAME} unavailable: {e}")
    init, u = _streams(CFG_RUN["B"], CFG_RUN["N"], CFG_RUN["sweeps"], 0, len(rule.init_pool))
    init = rule.init_pool[init]
    out = run(rule, B=CFG_RUN["B"], N=CFG_RUN["N"], r=CFG_RUN["r"], T=CFG_RUN["T"],
              sweeps=CFG_RUN["sweeps"], scheme="none", seed=CFG_RUN["seed"],
              init_state=init, u_stream=u)
    _assert_identical("ar", out)
