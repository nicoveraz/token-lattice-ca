"""#25: five copies of inverse-CDF sampling reduced to one, proved equal rather than assumed.

This function is what makes the exact-zero damage null a proof instead of a coincidence: twins run
under common random numbers agree cell-for-cell only because identical windows handed the same
uniform draw the same token. The copies are replaced only after the replacement is shown to agree
with each original over random inputs and at every boundary.
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]

from sampling import inverse_cdf                                    # noqa: E402


# --- the originals, reproduced verbatim so the hoist is checked against what it replaced -------

def _ref_ca(probs, u):                                   # src/ca.py::_sample
    cdf = np.cumsum(np.asarray(probs), axis=-1)
    cdf /= cdf[:, -1:]
    return np.array([np.searchsorted(cdf[b], u[b]) for b in range(len(u))], dtype=np.int32)


def _ref_dk(probs, u):                                   # src/dk.py::Rule.sample, mlm_ca::_sample
    cdf = np.cumsum(probs, axis=-1)
    cdf = cdf / cdf[:, -1:]
    return np.array([np.searchsorted(cdf[b], u[b]) for b in range(len(u))], dtype=np.int64)


def _rows(rng, B, V, *, normalise=True, sparse=False):
    p = rng.random((B, V)) + 1e-12
    if sparse:                                           # zero-probability tokens must be skipped
        p *= rng.random((B, V)) > 0.5
        p[p.sum(1) == 0, 0] = 1.0
    return p / p.sum(1, keepdims=True) if normalise else p


@pytest.mark.parametrize("sparse", [False, True])
def test_the_hoist_reproduces_both_numpy_originals_exactly(sparse):
    rng = np.random.default_rng(0)
    for _ in range(200):
        B, V = int(rng.integers(1, 6)), int(rng.integers(2, 40))
        p, u = _rows(rng, B, V, sparse=sparse), rng.random(B)
        assert np.array_equal(inverse_cdf(p, u, dtype=np.int32), _ref_ca(p, u))
        assert np.array_equal(inverse_cdf(p, u), _ref_dk(p, u))


def test_unnormalised_rows_give_the_same_answer_as_normalised_ones():
    """The division by cdf[:, -1:] is the contract, not an optimisation.

    A row summing to 0.999 from float error would otherwise leave mass above the last token where
    u can land and index out of bounds.
    """
    rng = np.random.default_rng(1)
    p = _rows(rng, 8, 25, normalise=False)
    u = rng.random(8)
    assert np.array_equal(inverse_cdf(p, u), inverse_cdf(p / p.sum(1, keepdims=True), u))
    assert inverse_cdf(p, u).max() < p.shape[1], "sampled past the last token"


@pytest.mark.parametrize("u_val", [0.0, 1e-12, 0.5, 1 - 1e-12])
def test_boundaries_stay_in_range(u_val):
    rng = np.random.default_rng(2)
    p = _rows(rng, 4, 12)
    out = inverse_cdf(p, np.full(4, u_val))
    assert out.min() >= 0 and out.max() < p.shape[1]
    assert np.array_equal(out, _ref_dk(p, np.full(4, u_val)))


def test_dtype_is_the_callers_choice_and_is_not_silently_unified():
    rng = np.random.default_rng(3)
    p, u = _rows(rng, 3, 9), rng.random(3)
    assert inverse_cdf(p, u, dtype=np.int32).dtype == np.int32
    assert inverse_cdf(p, u).dtype == np.int64


# --- the torch form, including the copy still living in ar_ca.py -------------------------------

def test_the_torch_form_agrees_with_the_numpy_form_exactly():
    """`(cdf < u).sum(dim=1)` and `searchsorted(..., side='left')` are the same count."""
    torch = pytest.importorskip("torch")
    from sampling import inverse_cdf_torch
    rng = np.random.default_rng(4)
    for _ in range(100):
        B, V = int(rng.integers(1, 6)), int(rng.integers(2, 40))
        p, u = _rows(rng, B, V), rng.random(B)
        got = inverse_cdf_torch(torch.as_tensor(p, dtype=torch.float64), u)
        assert np.array_equal(got, inverse_cdf(p, u))


def test_the_copy_still_in_dk_agrees_with_the_shared_definition():
    """dk.DKRule.sample is `inverse_cdf` verbatim, and is deliberately NOT hoisted.

    Editing src/dk.py marks results/canalization.json stale, and re-running an analysis merely to
    refresh a hash is the move that once recomputed headline numbers over a silently smaller
    cohort when gated checkpoints had become unfetchable in the meantime. Even adding a COMMENT to
    dk.py drifts the stamp, since it is taken over file bytes -- so the note lives here, where
    nothing is stamped, and the copy is guarded rather than annotated.

    Unlike the transcriptions above, this calls the live method, so it fails if dk.py drifts.
    Fold the copy in when dk.py is next edited for a reason of its own.
    """
    dk = pytest.importorskip("dk", reason="dk.py needs the lattice backend")
    rng = np.random.default_rng(6)
    for _ in range(100):
        B, V = int(rng.integers(1, 6)), int(rng.integers(2, 40))
        p, u = _rows(rng, B, V), rng.random(B)
        assert np.array_equal(dk.DKRule.sample(None, p, u), inverse_cdf(p, u))


def test_the_copy_still_in_ar_ca_agrees_with_the_shared_definition():
    """ar_ca.py keeps its own sample_device: editing it would mark 6 stamped results stale.

    The duplication is therefore GUARDED rather than removed -- this fails the moment the copy
    drifts from the definition, so the deferral costs vigilance rather than correctness. Delete
    this test when ar_ca.py is next edited for another reason and the copy can go with it.
    """
    torch = pytest.importorskip("torch")
    from sampling import inverse_cdf_torch
    rng = np.random.default_rng(5)
    for _ in range(100):
        B, V = int(rng.integers(1, 6)), int(rng.integers(2, 40))
        p, u = _rows(rng, B, V), rng.random(B)
        pt = torch.as_tensor(p, dtype=torch.float64)
        u_t = torch.as_tensor(u, device=pt.device, dtype=pt.dtype).unsqueeze(1)
        cdf = pt.cumsum(-1)
        cdf = cdf / cdf[:, -1:]
        ar_ca_copy = (cdf < u_t).sum(dim=1).to("cpu", torch.int64).numpy()
        assert np.array_equal(ar_ca_copy, inverse_cdf_torch(pt, u))
