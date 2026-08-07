"""Rung 1 of the coupling ladder: the paired loop must reproduce the production twin protocol.

This looks like a tautology and is not. `lattice.run` advances one lattice; the CRN protocol runs
the twins as two separate calls sharing a `u_stream`. The paired loop advances both together so a
coupling can be chosen, and it re-implements the sweep, the permutation draw and the window slice.
Any divergence in how it consumes randomness makes every later maximal-vs-monotone comparison a
measurement of that divergence.

It caught one on the first run: forcing float64 where the production sampler cumsums the backend's
native float32 moved `searchsorted` across bin boundaries in 163 of 1296 cells.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src"), str(ROOT / "gatecheck" / "src")]


def test_paired_loop_reproduces_the_production_twin_protocol():
    pr = pytest.importorskip("paired_runner", reason="needs the jax toy backend")
    if not (ROOT / "ckpt" / "final.npz").exists():
        pytest.skip("toy checkpoint not present")
    r = pr.rung1_reproduces_production()
    assert r["production_damage_cells"] > 0, (
        "the comparison is vacuous: production produced no damage at all, so agreeing with it "
        "proves nothing")
    assert r["mismatch_a"] == 0 and r["mismatch_b"] == 0, (
        f"paired loop diverges from production: {r['mismatch_a']} cells in twin A, "
        f"{r['mismatch_b']} in twin B, of {r['cells']}")
    assert r["mismatch_damage_field"] == 0
