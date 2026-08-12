"""Keep the object the measurement was reduced FROM, not only the number it was reduced TO.

THE DEFECT, WITH THREE INSTANCES. A run computes a rich state -- a lattice, a damage field, a
trajectory -- reduces it to a scalar, writes the scalar, and drops the state. Two things follow, and
the second is the dangerous one:

  1. every later question about that state costs a full re-run. Expensive, but visible.
  2. a DEGENERACY in the scalar cannot be detected at all, because the evidence that would reveal
     it no longer exists. Invisible, and it does not announce itself.

The instances, from the project this package was extracted from: a damage-cone study whose results
files stored no cone, so the cone-shape question needed a re-run; a remote campaign that stored only
an attractor share, which is `1/period` on a lattice that crystallised into a periodic orbit, so a
period-3 crystal and a diffuse ring were indistinguishable -- discovered only when the rings were
finally kept; and the local version of the same readout, where 120 stored cells could not answer
whether the same degeneracy was present and a fresh grid had to be run to find out.

`leverage` and `gate` inspect numbers. This module is what lets `discriminator.nuisance_identity`
inspect the thing the numbers came from, which is the only check that catches instance 2.

THE DOWNSAMPLE RULE IS THE WHOLE DESIGN. A cap without a rule is a silent truncation, and a rule
that strides the WRONG axis destroys exactly what the state was kept for -- subsampling a ring's
sites is what makes a period-3 orbit look like noise. So `pack` strides the axis you name, refuses
to touch the last axis unless you say so in as many words, and records what it did inside the
returned block. A reader of the results file can always see whether they are looking at all of it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["pack", "unpack", "has_state", "STATE_KEY"]

STATE_KEY = "_state"

# 8192 elements is ~64 kB of JSON per cell at 5-digit token ids: large enough to hold a 16x48
# lattice or a 48x96 damage field whole, small enough that a 120-cell grid stays a file a human
# can open. Callers with bigger states pass their own cap and name the axis to stride.
DEFAULT_CAP = 8192


def pack(array: Any, *, cap: int = DEFAULT_CAP, stride_axis: int = 0,
         allow_last_axis: bool = False, note: str = "") -> dict:
    """JSON-safe encoding of a state array, downsampled along ONE named axis if it exceeds `cap`.

    `stride_axis` is the axis that carries repeats -- replicas, seeds, sweeps -- i.e. the one whose
    entries are interchangeable samples of the same thing. Striding it costs resolution in the
    number of samples, which is recoverable by running more.

    The LAST axis is refused by default because in every use this module was written for it is the
    axis the structure lives on: ring sites, sequence positions, lattice columns. Striding it
    destroys periodicity, locality and front shape -- the properties the state was kept in order to
    be able to check. Pass `allow_last_axis=True` if your last axis really is the interchangeable
    one, and the block will record that you did.
    """
    a = np.asarray(array)
    if a.ndim == 0:
        raise ValueError("pack() is for states, not scalars; a 0-d array has nothing to keep")
    if stride_axis == a.ndim - 1 and not allow_last_axis:
        raise ValueError(
            f"stride_axis={stride_axis} is the last axis, which is refused by default: in a "
            f"lattice, a sequence or a field the last axis carries the structure (periodicity, "
            f"locality, front shape) that keeping the state was supposed to make checkable. Stride "
            f"an axis of interchangeable samples instead, or pass allow_last_axis=True and say why "
            f"in `note`")
    step, kept = 1, a
    if a.size > cap:
        per = max(1, a.size // max(1, a.shape[stride_axis]))
        want = max(1, cap // max(1, per))
        step = max(1, int(np.ceil(a.shape[stride_axis] / want)))
        kept = np.take(a, np.arange(0, a.shape[stride_axis], step), axis=stride_axis)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "stride_axis": int(stride_axis),
        "step": int(step),
        "complete": bool(step == 1),
        "kept_shape": list(kept.shape),
        "rule": (f"every {step}th entry along axis {stride_axis}" if step > 1
                 else "complete, no downsampling applied"),
        "note": note,
        "data": kept.tolist(),
    }


def unpack(block: dict) -> np.ndarray:
    """The stored array. Shape is `kept_shape`, which equals `shape` when `complete` is true."""
    return np.asarray(block["data"], dtype=block.get("dtype", None))


def has_state(cell: dict, key: str = STATE_KEY) -> bool:
    """Does this results cell carry the object its numbers were reduced from?"""
    b = cell.get(key)
    return isinstance(b, dict) and "data" in b and bool(b["data"])
