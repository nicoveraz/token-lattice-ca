"""The unit gate must catch a units error and must not catch a badly-trained model.

The registered interval (0.4, 2.5) conflated those two. These pin the distinction, because the
replacement is only worth having if it is stronger where the old one was aimed and weaker only
where the old one was wrong.
"""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "gatecheck" / "src")]


def _cell(nats, tokens, nbytes, *, bpb=None, family="f", revision="r"):
    """A cell whose bpb is the identity's answer unless deliberately corrupted."""
    true = nats * tokens / (math.log(2) * nbytes)
    return dict(family=family, revision=revision, nats_per_token=nats, n_tokens=tokens,
                n_bytes=nbytes, bpb=true if bpb is None else bpb)


def _gate(cells):
    # The decider imports FAMILIES and _spread_at_matched from the measuring script rather than
    # duplicating them, so it is unimportable until that script is committed. Skipping keeps this
    # file honest about which half is missing instead of failing as though the gate were broken.
    pytest.importorskip("loss_collapse_families",
                        reason="measuring script not present; decider cannot import its grid")
    from loss_collapse_decide import _unit_gate
    return _unit_gate(cells)


def test_a_random_init_checkpoint_passes_though_the_old_interval_rejected_it():
    """3.95 bpb is a correct number for a model that has learned nothing.

    The grid includes random init deliberately, as its chaotic-init control, so a gate that
    rejects it rejects the design rather than the data.
    """
    assert _gate([_cell(11.9007, 61724, 257163)]) == []


def test_a_dip_region_checkpoint_passes_though_the_old_interval_rejected_it():
    """pythia-410m|step128 read 2.6275, just over the registered 2.5 ceiling."""
    assert _gate([_cell(7.58787, 61724, 257163)]) == []


def test_nats_per_token_recorded_as_bits_per_byte_is_caught():
    """The error the gate was actually written for: the two differ by a factor of ln2 and more."""
    c = _cell(7.58787, 61724, 257163)
    c["bpb"] = c["nats_per_token"]                      # the confusion, verbatim
    bad = _gate([c])
    assert bad and "not the same quantity" in bad[0]


def test_a_units_error_inside_the_old_interval_is_still_caught():
    """The decisive case: the old range gate could not see this at all.

    A value that is wrong but plausible-looking sits inside (0.4, 2.5) and passes any interval,
    distribution-derived or otherwise. Only the identity catches it.
    """
    c = _cell(7.58787, 61724, 257163, bpb=1.9)
    assert 0.4 <= c["bpb"] <= 2.5, "this case must be inside the old interval to be meaningful"
    assert _gate([c]), "a wrong value inside the plausible band must still fail the identity"


def test_the_ceiling_rejects_what_cannot_be_a_bits_per_byte_loss():
    c = _cell(11.9007, 61724, 257163)
    c["bpb"] = 9.0
    c["n_bytes"] = None                                  # identity unavailable; ceiling must hold
    assert _gate([c])


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 0.0, -1.0])
def test_non_numbers_and_impossible_values_are_rejected(bad):
    c = _cell(11.9007, 61724, 257163)
    c["bpb"] = bad
    assert _gate([c])


def test_the_gate_accepts_the_measured_grid_as_a_whole():
    """Every cell the real run has produced satisfies the identity; skipped if absent."""
    import json
    p = ROOT / "results" / "loss_collapse_families.json"
    if not p.exists():
        pytest.skip("no measured grid present")
    cells = [c for c in json.loads(p.read_text())["cells"].values() if "bpb" in c]
    if not cells:
        pytest.skip("grid has no measured cells yet")
    assert _gate(cells) == []
