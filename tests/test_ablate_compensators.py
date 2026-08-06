"""#103: the verdict layer, exercised on synthetic inputs before the run spends five hours.

F80's meta-defect was a declared statistic correctly frozen mid-run while the verdict logic that
CONSUMED it was not -- it applied an absolute-value threshold to a directional hypothesis and
computed a ratio with no noise gate. The lesson is that the branch structure has to be tested on
inputs whose right answer is known, not inspected. These construct each outcome deliberately.
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src"), str(ROOT / "gatecheck" / "src")]

ac = pytest.importorskip("ablate_compensators", reason="needs the torch/ar backend")

SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]


def _runs(levels, *, sd=0.004, compound_sd=None, ign=None):
    """{arm: lambda} -> a synthetic runs dict with a little seed noise, all ignited.

    `compound_sd` gives the compound arms their own spread, so a run can be underpowered on the
    comparison while the calibration rung still reproduces -- which is the realistic shape, since
    the rung arms are the ones already known to behave.
    """
    rng = np.random.default_rng(0)
    out = {}
    for arm, lam in levels.items():
        sd_a = compound_sd if (compound_sd is not None and "+" in arm) else sd
        for s in SEEDS:
            ip = 1.0 if ign is None else ign.get(arm, ign.get("*", 1.0))
            out[f"{arm}|s{s}"] = dict(arm=arm, seed=s, lambda_ca=float(lam + rng.normal(0, sd_a)),
                                      D_norm=0.5, mean_damage=0.5, ignition_prob=ip)
    return out


def _levels(delta_by_layer, *, l_none=0.3566, l_early=0.0115, singles=None):
    """Build arm levels that realise a chosen delta(L) for each downstream layer.

    delta(L) = [l_early - l_compound(L)] - [l_none - singles(L)], so
    l_compound(L) = l_early - delta(L) - l_none + singles(L).
    """
    singles = singles or {L: 0.35 for L in delta_by_layer}
    lv = {"none": l_none, ac.EARLY: l_early}
    for L, d in delta_by_layer.items():
        lv[f"{ac.EARLY}+attn_L{L:02d}"] = l_early - d - l_none + singles[L]
    return lv, singles


def _decide(delta_by_layer, **kw):
    lv, singles = _levels(delta_by_layer, **kw)
    res = {"runs": _runs(lv)}
    return ac.analyse(res, singles), res


def test_the_rung_blocks_everything_when_the_harness_does_not_reproduce():
    """If `none` no longer reproduces its recorded level, the borrowed singles are not comparable."""
    lv, singles = _levels({L: 0.0 for L in ac.DOWNSTREAM})
    lv["none"] = 0.3566 + 5 * ac.REF_TOL                      # harness has drifted
    res = {"runs": _runs(lv)}
    v = ac.analyse(res, singles)
    assert "NOT DECIDABLE" in v and "DOES NOT REPRODUCE" in v
    assert res["analysis"]["decided"] is False
    assert "rows" not in res["analysis"], "primary was computed despite a failed rung"


def test_no_compensation_is_a_kill_not_an_undecided():
    """All deltas at zero: static redundancy accounts for F80 and route 5 closes.

    The registration names this a decidable outcome, so it must not come back NOT DECIDABLE --
    that was the exact defect corrected in this issue's own v1 registration.
    """
    v, res = _decide({L: 0.0 for L in ac.DOWNSTREAM})
    assert "KILL" in v and "A NULL IS A GOOD RESULT" in v
    assert res["analysis"]["decided"] is True


def test_a_negative_delta_is_evidence_against_not_an_inability_to_decide():
    v, res = _decide({L: -0.08 for L in ac.DOWNSTREAM})
    assert "KILL" in v
    assert res["analysis"]["decided"] is True
    assert res["analysis"]["directional"]["usable"] is False


def test_one_clearly_compensating_layer_is_identified():
    d = {L: 0.0 for L in ac.DOWNSTREAM}
    d[17] = 0.30                                              # far above any plausible floor
    v, res = _decide(d)
    assert "COMPENSATION" in v and "L17" in v
    assert res["analysis"]["decided"] is True
    assert res["analysis"]["directional"]["usable"] is True


def test_a_uniform_tiny_positive_offset_is_a_kill_not_an_identification():
    """A constant bias in the compound arms must not read as 'every layer compensates'."""
    v, res = _decide({L: 0.0008 for L in ac.DOWNSTREAM})
    assert "KILL" in v
    assert res["analysis"]["decided"] is True


def test_an_underpowered_run_can_decide_neither_way():
    """The power gate: if the floor swamps MIN_DETECTABLE, the kill is not readable either."""
    v, res = _decide({L: 0.0 for L in ac.DOWNSTREAM}, )
    assert "KILL" in v                                   # sanity: tight seeds -> decidable
    lv, singles = _levels({L: 0.0 for L in ac.DOWNSTREAM})
    res2 = {"runs": _runs(lv, compound_sd=0.5)}          # comparison arms all over the place
    v2 = ac.analyse(res2, singles)
    assert "NOT DECIDABLE" in v2 and "underpowered" in v2
    assert res2["analysis"]["decided"] is False


def test_unignited_downstream_arms_do_not_silently_shrink_the_comparison():
    """F42: lambda is undefined on an unignited run, and dropping arms must be visible."""
    lv, singles = _levels({L: 0.0 for L in ac.DOWNSTREAM})
    runs = _runs(lv)
    for k, r in runs.items():
        if r["arm"].endswith("attn_L08"):
            r["D_norm"] = 0.0
            r["mean_damage"] = 0.0            # is_unignited prefers mean_damage
    res = {"runs": runs}
    ac.analyse(res, singles)
    layers = [r["layer"] for r in res["analysis"]["rows"]]
    assert 8 not in layers, "an unignited arm contributed a lambda anyway"
    assert len(layers) == len(ac.DOWNSTREAM) - 1


# ---------------------------------------------------------------- the comparability gate

def _with_ignition(delta_by_layer, ign):
    lv, singles = _levels(delta_by_layer)
    res = {"runs": _runs(lv, ign=ign)}
    return ac.analyse(res, singles), res


def test_arms_igniting_unlike_the_reference_are_dropped_and_reported():
    """lambda lives on ignited replicas, so an arm at a different rate is a different selection."""
    d = {L: 0.0 for L in ac.DOWNSTREAM}
    ign = {"*": 0.15, "none": 1.0}
    for L in list(ac.DOWNSTREAM)[:3]:
        ign[f"{ac.EARLY}+attn_L{L:02d}"] = 0.9              # far from the reference's 0.15
    v, res = _with_ignition(d, ign)
    dropped = [r["layer"] for r in res["analysis"]["dropped"]]
    assert dropped == list(ac.DOWNSTREAM)[:3]
    assert "DROPPED as not comparable" in v
    for L in dropped:
        assert f"L{L}(" in v, "a dropped arm was not named in the verdict"


def test_too_few_comparable_arms_is_not_decidable_and_says_why():
    """The failure this gate exists for: the reference is so dead that nothing matches it."""
    d = {L: 0.0 for L in ac.DOWNSTREAM}
    ign = {"*": 0.9, ac.EARLY: 0.15, "none": 1.0}           # every compound arm unlike the ref
    v, res = _with_ignition(d, ign)
    assert "NOT DECIDABLE" in v
    assert "differently-selected replica subsets" in v
    assert res["analysis"]["decided"] is False


def test_a_selection_artifact_cannot_be_read_as_compensation():
    """The whole point: a big delta on a non-comparable arm must not become the headline."""
    d = {L: 0.0 for L in ac.DOWNSTREAM}
    d[19] = 0.40                                             # huge, but on an arm that ignites unlike the ref
    ign = {"*": 0.15, "none": 1.0, f"{ac.EARLY}+attn_L19": 0.95}
    v, res = _with_ignition(d, ign)
    assert "COMPENSATION" not in v, "a non-comparable arm supplied the positive"
    assert 19 in [r["layer"] for r in res["analysis"]["dropped"]]
