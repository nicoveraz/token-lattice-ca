"""One definition of inverse-CDF sampling against an external uniform.

WHY THIS EXISTS. The rule was written five times -- three NumPy copies (`ca._sample`,
`mlm_ca._sample`, `dk.Rule.sample`) and two Torch copies (`mlm_ca.*.sample_device`,
`ar_ca.*.sample_device`) -- and it is the single most load-bearing line in the repository. Damage
spreading is measured by running twins under COMMON RANDOM NUMBERS: the exact-zero null holds only
because two lattices with identical windows, handed the same uniform, draw the same token. That is
a property of this function and of nothing else. Five copies means five chances for one of them to
drift and turn the null from a proof into a coincidence.

THE TWO FORMS ARE THE SAME FUNCTION, not merely similar. `np.searchsorted(cdf, u, side="left")`
returns the number of entries strictly less than `u`; `(cdf < u).sum(dim=1)` computes that count
directly. They agree exactly, including on ties and on the boundaries u=0 and u=1, and
`test_sampling.py` pins that over random inputs rather than asserting it in prose.

NORMALISATION IS PART OF THE CONTRACT. Every copy divided the CDF by its last entry before
searching, which makes the draw invariant to unnormalised probabilities -- a row that sums to 0.999
from float error must not leave a sliver of mass above the last token where `u` can land and index
out of bounds. Kept here rather than left to callers.

DTYPE IS A CALLER DECISION AND IS NOT UNIFIED. `ca` returned int32 and the others int64. Those feed
different downstream buffers and golden files, so the shared function takes the dtype rather than
picking one and silently changing what a caller writes.
"""
import numpy as np

__all__ = ["inverse_cdf", "inverse_cdf_torch"]


def inverse_cdf(probs, u, dtype=np.int64):
    """probs (B,V), external uniforms u (B,) -> tokens (B,).

    `np.asarray` because callers pass JAX arrays as well as NumPy ones.
    """
    cdf = np.cumsum(np.asarray(probs), axis=-1)
    cdf = cdf / cdf[:, -1:]
    return np.array([np.searchsorted(cdf[b], u[b]) for b in range(len(u))], dtype=dtype)


def inverse_cdf_torch(probs_t, u, *, device=None):
    """On-device form: probs_t (B,V) torch, u (B,) numpy -> tokens (B,) numpy.

    Deterministic given (probs_t, u), which is what keeps the null test exact. The comparison runs
    on device and only the result is moved, so a large vocabulary does not cross the bus.
    """
    import torch
    dev = device if device is not None else probs_t.device
    u_t = torch.as_tensor(u, device=dev, dtype=probs_t.dtype).unsqueeze(1)
    cdf = probs_t.cumsum(-1)
    cdf = cdf / cdf[:, -1:]
    return (cdf < u_t).sum(dim=1).to("cpu", torch.int64).numpy()
