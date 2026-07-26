"""Rule 8: a results file must not contradict its own declared design.

This test exists because of two defects that a prose grep could not have found -- neither
document in the repo contains the string `sel(N, {256}, m)`:

  * F39: `dev_transition_shape.json` declared `"pre": [256, 512]` and computed the headline
    over `{256}` alone. `n_pre` was 8 where the design gives 16, while the `variance` block
    in the SAME file used all 16. One assertion would have failed the run.
  * F42: `lambda_ca` was averaged over runs where damage never ignited, for which it is
    undefined, and the emitted value spans an order of magnitude for the same physical
    outcome (-0.1649 vs -1.7130).

The checks are about the DESIGN, not the values: they compare what a file says it did
against what it must have done had its stated design been followed. A future subset can only
pass by also changing the declared design, which is visible in review.
"""
import json, pathlib, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
RESULTS = ROOT / "results"


def _load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    return json.load(open(p))


def _unignited(v):
    from lyapunov import is_unignited
    return (is_unignited(mean_damage=v["mean_damage"]) if "mean_damage" in v
            else is_unignited(D_norm=v["D_norm"]))


# ------------------------------------------------------- F39: declared design vs computed n
def test_shape_groups_match_declared_definitions():
    """The exact check that would have failed the first version of the shape script."""
    d = _load("dev_transition_shape.json")
    defs = d["_definitions"]
    exp_pre = len(defs["pre"]) * defs["n_seeds"]
    exp_plateau = len(defs["plateau"]) * defs["n_seeds"]
    assert defs["expected_n_pre"] == exp_pre
    assert defs["expected_n_plateau"] == exp_plateau
    for N in defs["sizes"]:
        for m in ("lambda_ca", "D_norm"):
            h = d["headline"][f"N{N}_{m}"]
            assert h["n_pre"] == exp_pre, (
                f"N{N}_{m}: headline used n_pre={h['n_pre']} but the declared pre set "
                f"{defs['pre']} x {defs['n_seeds']} seeds = {exp_pre}. This is F39's defect.")
            assert h["n_plateau"] == exp_plateau


def test_shape_retains_the_unregistered_variants_for_audit():
    """The inflated numbers stay visible, under names that say what they are."""
    d = _load("dev_transition_shape.json")
    for k, v in d["headline"].items():
        assert "cohens_d_from_step256_only_UNREGISTERED" in v, k
        assert "cohens_d_vs_peak_INFLATED" in v, k
        assert v["cohens_d"] != v["cohens_d_from_step256_only_UNREGISTERED"], (
            f"{k}: pre-registered and step256-only effect sizes are identical -- the "
            f"correction was reverted or the selection is wrong again")


def test_phase3_runs_have_the_designed_count():
    d = _load("dev_transition_phase3.json")
    runs = [v for v in d["runs"].values() if "lambda_ca" in v]
    steps = {v["step"] for v in runs}; sizes = {v["N"] for v in runs}
    seeds = {v["seed"] for v in runs}
    assert len(runs) == len(steps) * len(sizes) * len(seeds)


# ------------------------------------------------------------------ F42: the ignition rule
def test_unignited_predicate_catches_what_the_floor_predicate_misses():
    from lyapunov import is_unignited, is_dead_damage_floor
    for lam in (-0.1649, -1.7130):                 # both observed with zero final damage
        assert not is_dead_damage_floor(lam), (
            f"lambda={lam} must NOT match the floor sentinel; if it does, F42's premise "
            f"is wrong and is_unignited is unnecessary")
    assert is_unignited(mean_damage=0.0)
    assert is_unignited(D_norm=0.0)
    # a negative-but-IGNITED run is a real measurement and must be kept
    assert not is_unignited(mean_damage=0.0250)
    assert not is_unignited(D_norm=0.0250)
    with pytest.raises(ValueError):
        is_unignited()


def test_estimator_is_N_independent_for_a_fixed_cone():
    """Guards the corrected mechanism statement in `is_unignited`'s docstring.

    It is tempting to explain the -0.1649 vs -1.7130 spread as the estimator scaling with
    lattice size. It does not: N appears only in the second return value. If this test ever
    fails, the docstring's mechanism paragraph must be rewritten, not the test relaxed.
    """
    from lyapunov import lyap_from_cone
    lams = []
    for N in (48, 96, 192):
        cone = np.zeros((23, N))
        cone[0, N // 2 - 1:N // 2 + 2] = 1.0       # 3-site seed that dies immediately
        lams.append(lyap_from_cone(cone, N)[0])
    assert len({round(x, 9) for x in lams}) == 1, (
        f"lambda varies with N for an identical cone: {lams}. The estimator IS "
        f"size-dependent and is_unignited's docstring is wrong.")


def test_D_norm_zero_fallback_is_sound_at_the_configurations_used():
    """`is_unignited(D_norm=0)` is only valid if no nonzero damage can round to 0.00000.

    Smallest nonzero mean_damage is 1/(tail*N*B) with tail=8; the stored D_norm is
    round(mean_damage/max(D0,1e-3), 5). Checked against the largest plausible floor.
    """
    tail = 8
    for N, B in ((48, 16), (96, 8), (192, 4)):
        smallest = 1.0 / (tail * N) / B
        for D0 in (0.1, 0.3, 0.6, 1.0):
            assert round(smallest / max(D0, 1e-3), 5) > 0.0, (
                f"at N={N} B={B} D0={D0} a single damaged site rounds D_norm to 0.00000 -- "
                f"the D_norm fallback in is_unignited is unsound here")


def test_phase3_unignited_count_is_what_F42_documents():
    """Phase 3 contains exactly one unignited run. If this changes, update F42, not the test."""
    d = _load("dev_transition_phase3.json")
    runs = [v for v in d["runs"].values() if "lambda_ca" in v]
    dead = [v for v in runs if _unignited(v)]
    assert len(dead) == 1, f"expected 1 unignited run, found {len(dead)}: {dead}"
    only = dead[0]
    assert (only["N"], only["step"], only["seed"]) == (96, 256, 22)
