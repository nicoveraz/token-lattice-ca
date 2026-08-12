"""The downsample rule is the design, so it is what gets tested.

A cap without a rule is a silent truncation; a rule that strides the wrong axis destroys exactly
what the state was kept for. Both are asserted here, and the second is asserted as a REFUSAL rather
than as a warning, because a warning in a batch run is a line nobody reads.
"""
import numpy as np
import pytest

from gatecheck import has_state, pack_state, unpack_state, STATE_KEY
from gatecheck.state import DEFAULT_CAP


def test_small_state_is_stored_whole():
    a = np.arange(16 * 48).reshape(16, 48)
    b = pack_state(a)
    assert b["complete"] and b["step"] == 1
    assert b["kept_shape"] == [16, 48]
    assert np.array_equal(unpack_state(b), a)


def test_large_state_strides_the_named_axis_only():
    """The ring axis must survive intact at every cap, or periodicity becomes unaskable."""
    a = np.arange(400 * 96).reshape(400, 96)
    b = pack_state(a, cap=4096, stride_axis=0)
    assert not b["complete"] and b["step"] > 1
    kept = unpack_state(b)
    assert kept.shape[1] == 96, "the site axis was subsampled; a strided ring has no period"
    assert kept.shape[0] < 400
    assert np.array_equal(kept[0], a[0])
    assert np.array_equal(kept[1], a[b["step"]])


def test_striding_the_last_axis_is_refused_by_default():
    a = np.arange(400 * 96).reshape(400, 96)
    with pytest.raises(ValueError, match="last axis"):
        pack_state(a, cap=4096, stride_axis=1)


def test_last_axis_can_be_strided_when_the_caller_says_so():
    a = np.arange(400 * 96).reshape(400, 96)
    b = pack_state(a, cap=4096, stride_axis=1, allow_last_axis=True,
                   note="last axis is the replica axis in this layout")
    assert b["stride_axis"] == 1 and b["note"]


def test_rule_is_recorded_so_a_reader_can_see_the_truncation():
    a = np.arange(400 * 96).reshape(400, 96)
    whole = pack_state(np.arange(20).reshape(4, 5))
    part = pack_state(a, cap=4096)
    assert whole["rule"] == "complete, no downsampling applied"
    assert "every" in part["rule"] and str(part["step"]) in part["rule"]
    assert part["shape"] == [400, 96] and part["kept_shape"] != part["shape"]


def test_scalars_are_refused():
    with pytest.raises(ValueError, match="0-d"):
        pack_state(np.float64(0.5))


def test_has_state_rejects_a_cell_with_only_scalars():
    assert not has_state({"top1": 0.9, "rep2": 0.8})
    assert not has_state({STATE_KEY: {}})
    assert not has_state({STATE_KEY: {"data": []}})
    assert has_state({STATE_KEY: pack_state(np.zeros((2, 3)))})


def test_default_cap_holds_a_typical_lattice_whole():
    """The cap must not bind on the geometry it was chosen for, or every cell ships truncated."""
    assert pack_state(np.zeros((16, 48)))["complete"]
    assert 16 * 48 < DEFAULT_CAP
