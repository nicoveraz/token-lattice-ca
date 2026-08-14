"""The discriminator must reach NOT_DECIDABLE for the RIGHT reason, not just often enough.

Every test here builds a grid whose correct verdict is known by construction, because the module's
whole job is to refuse questions the data cannot answer -- and a refusal for the wrong reason is a
different failure from a refusal for the right one. Two of these (`test_pinned_across_constructions`
and `test_nuisance_identity_catches_period_readout`) encode defects that reached results files in
the project this package came from.
"""
import math

import pytest

from gatecheck import (
    CONSTRUCTION_DETERMINED,
    MODEL_DETERMINED,
    Loopness,
    discriminate,
    nuisance_identity,
    rank,
    spearman,
)
from gatecheck.gate import NOT_DECIDABLE

MODELS = ["a", "b", "c", "d", "e"]
CONS = ["r2.T0.2", "r2.T0.7", "r3.T0.2"]
SEEDS = [1, 2]


def grid(fn):
    return {(m, c, s): fn(m, c, s) for m in MODELS for c in CONS for s in SEEDS}


def test_model_determined_when_ranking_survives_construction():
    """A per-model level plus a small per-construction offset: same ranking everywhere."""
    lvl = {m: 0.1 * i for i, m in enumerate(MODELS)}
    off = {c: 0.01 * i for i, c in enumerate(CONS)}
    rep = discriminate(grid(lambda m, c, s: lvl[m] + off[c] + 1e-4 * s))
    assert rep.verdict.decided
    assert rep.verdict.value == MODEL_DETERMINED
    assert rep.seed_stability == pytest.approx(1.0)
    assert rep.invariance == pytest.approx(1.0)


def test_construction_determined_when_each_construction_reorders():
    """Signal and seed stability, but every construction permutes the models differently."""
    perms = {CONS[0]: [0, 1, 2, 3, 4], CONS[1]: [4, 3, 2, 1, 0], CONS[2]: [2, 0, 4, 1, 3]}
    rep = discriminate(grid(lambda m, c, s: 0.1 * perms[c][MODELS.index(m)] + 1e-5 * s))
    assert rep.verdict.decided
    assert rep.verdict.value == CONSTRUCTION_DETERMINED
    assert rep.invariance < 0.6


def test_spread_without_a_rankable_ordering_is_not_decidable():
    """THE BRANCH THAT MATTERS (F129): a real spread carrying no reproducible ordering.

    THE SHAPE MATTERS, and getting it wrong is instructive: a ranking that simply REVERSES between
    seeds has, by arithmetic, a seed noise as large as the spread, so it fails the signal step
    instead and never reaches the branch under test. The combination F129 actually saw is one
    OUTLIER setting the spread while the remaining models sit clustered and re-order freely inside
    the cluster -- a spread/noise ratio of 30x carrying a rank correlation of 0.2.

    Two other fixture errors this test has already caught: no per-construction offset (the
    anti-vacuity gate binds first) and too few models (the min_models floor binds first). Each
    produced a passing `NOT_DECIDABLE` for the wrong reason.
    """
    a = [0.0, 9.0, 9.2, 9.4, 9.6]
    b = [0.0, 9.4, 9.6, 9.0, 9.2]                      # same cluster, scrambled inside it

    def f(m, c, s):
        i = MODELS.index(m)
        return 3.0 * CONS.index(c) + (a[i] if s == SEEDS[0] else b[i])
    rep = discriminate(grid(f))
    assert not rep.verdict.decided
    assert rep.verdict.status == NOT_DECIDABLE
    assert "cannot rank" in rep.verdict.reason
    assert rep.seed_stability == pytest.approx(0.2)     # computed, reported, and below the floor
    assert min(rep.model_spread[c] / rep.seed_noise[c] for c in rep.signal) > 10
    assert rep.invariance is None                      # never computed, which is the discipline


def test_pinned_across_constructions_is_vacuous_not_invariant():
    """A readout the construction cannot move ranks models identically everywhere, trivially.

    This is the anti-vacuity gate's whole reason to exist: without it the grid below returns a
    confident MODEL_DETERMINED with invariance +1.000, which is true and empty -- the ranking
    survived construction change because the construction changed nothing.
    """
    lvl = {m: 0.1 * i for i, m in enumerate(MODELS)}
    rep = discriminate(grid(lambda m, c, s: lvl[m] + 1e-9 * s))
    assert not rep.verdict.decided
    assert "across constructions" in rep.verdict.reason


def test_single_seed_refused_before_any_gate():
    obs = {(m, c, 1): 0.1 * MODELS.index(m) for m in MODELS for c in CONS}
    rep = discriminate(obs)
    assert rep.verdict.status == NOT_DECIDABLE
    assert "noise floor" in rep.verdict.reason


def test_too_few_models_refused():
    """A rank correlation on three points takes a handful of values and cannot fail informatively."""
    few = MODELS[:3]
    obs = {(m, c, s): 0.1 * few.index(m) + 0.01 * CONS.index(c) + 1e-5 * s
           for m in few for c in CONS for s in SEEDS}
    rep = discriminate(obs)
    assert rep.verdict.status == NOT_DECIDABLE
    assert "fewer points" in rep.verdict.reason or "min_models" in str(rep.thresholds)


def test_nuisance_identity_catches_period_readout():
    """F136: a share readout that is 1/period on a crystallised lattice.

    The values are well separated and perfectly reproducible, so no range or stability gate can
    object. Only comparing against the nuisance prediction does.
    """
    values = [1 / 3, 1 / 4, 1 / 3, 0.87]
    predicted = [1 / 3, 1 / 4, 1 / 3, 1 / 2]
    g = nuisance_identity(values, predicted, name="top1", nuisance="1/period")
    assert not g.usable
    assert "3 of 4" in g.reason


def test_nuisance_identity_passes_when_readout_is_not_the_nuisance():
    g = nuisance_identity([0.9, 0.8, 0.7], [0.5, 0.5, 0.33], name="top1", nuisance="1/period")
    assert g.usable


def test_nuisance_gate_blocks_an_otherwise_model_determined_grid():
    """The gate must be able to overturn a verdict, or it is decorative."""
    lvl = {m: 0.1 * i for i, m in enumerate(MODELS)}
    off = {c: 0.01 * i for i, c in enumerate(CONS)}
    obs = grid(lambda m, c, s: lvl[m] + off[c] + 1e-4 * s)
    clean = discriminate(obs)
    assert clean.verdict.value == MODEL_DETERMINED
    gated = discriminate(obs, nuisance_prediction=dict(obs))     # every cell equals its prediction
    assert not gated.verdict.decided
    assert "nuisance_identity" in gated.verdict.reason


def test_loopness_rejects_an_unlisted_commitment():
    Loopness(commitment="free_ar")                                # listed, fine
    with pytest.raises(ValueError):
        Loopness(commitment="whatever_i_did_in_my_script")


def test_loopness_orders_the_commitment_axis():
    ring = Loopness(radius=2, temperature=0.2, commitment="in_place")
    ar = Loopness(radius=None, temperature=0.2, commitment="free_ar")
    assert ring.commitment_rank < ar.commitment_rank
    assert ring.name == "r2.T0.2.async.in_place"


def test_loopness_keys_are_accepted_directly():
    lvl = {m: 0.1 * i for i, m in enumerate(MODELS)}
    cons = [Loopness(radius=2, temperature=t, label=f"T{t}") for t in (0.2, 0.7, 1.0)]
    obs = {(m, c, s): lvl[m] + 0.01 * i + 1e-4 * s
           for m in MODELS for i, c in enumerate(cons) for s in SEEDS}
    rep = discriminate(obs)
    assert rep.verdict.value == MODEL_DETERMINED
    assert set(rep.constructions) == {"T0.2", "T0.7", "T1.0"}


def test_rank_returns_nan_on_a_constant_vector():
    """R2: `argsort(argsort(x))` returns [0..n-1] here and manufactures a correlation."""
    r = rank([3.0, 3.0, 3.0, 3.0])
    assert all(math.isnan(v) for v in r)
    assert math.isnan(spearman([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]))


def test_rank_averages_ties():
    assert list(rank([10.0, 20.0, 20.0, 40.0])) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_matches_scipy_where_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    a = [1.0, 5.0, 5.0, 2.0, 9.0, 3.0]
    b = [2.0, 2.0, 7.0, 1.0, 8.0, 4.0]
    assert spearman(a, b) == pytest.approx(float(scipy_stats.spearmanr(a, b).statistic))


# --- the domain axis -------------------------------------------------------------------------

def test_domain_defaults_to_raw_and_stays_silent_in_the_name():
    lp = Loopness(radius=2, temperature=0.2)
    assert lp.domain == "raw" and lp.domain_tokens == 0
    assert "raw" not in lp.name            # the default does not clutter every report


def test_domain_appears_in_the_name_and_block_when_set():
    lp = Loopness(radius=2, temperature=0.2, domain="chat_template", domain_tokens=30)
    assert lp.name.endswith("chat_template30")
    b = lp.block()
    assert b["domain"] == "chat_template" and b["domain_tokens"] == 30


def test_two_domains_are_two_constructions():
    """The whole point: a readout measured raw and behind a template is measured twice, not once."""
    a = Loopness(radius=2, temperature=0.2, label="")
    b = Loopness(radius=2, temperature=0.2, domain="chat_template", domain_tokens=30)
    assert a.name != b.name and a != b


def test_unknown_domain_kind_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        Loopness(domain="template")        # near-miss for chat_template


def test_a_prefix_of_unknown_size_is_refused():
    """One BOS token and a thirty-token template are both 'a prefix' and are not the same thing."""
    with pytest.raises(ValueError, match="unknown size"):
        Loopness(domain="bos")
    Loopness(domain="bos", domain_tokens=1)          # stating the size is enough


def test_raw_with_a_prefix_is_refused():
    with pytest.raises(ValueError, match="domain_tokens must be 0"):
        Loopness(domain="raw", domain_tokens=5)
