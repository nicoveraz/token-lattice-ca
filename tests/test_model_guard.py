"""The pre-flight model guard: it must refuse a run rather than shrink its cohort silently."""
import pytest
from model_guard import ModelsUnavailable, audit_models, require_models


def test_offline_uncached_model_is_unavailable_not_assumed_present():
    r = audit_models(["definitely/not-a-real-model-xyz"], offline=True)
    ok, why = r["definitely/not-a-real-model-xyz"]
    assert not ok and "offline" in why


def test_require_models_raises_and_names_every_offender():
    with pytest.raises(ModelsUnavailable) as e:
        require_models(["fake/one", "fake/two"], offline=True)
    msg = str(e.value)
    assert "fake/one" in msg and "fake/two" in msg
    assert "SMALLER COHORT" in msg and "denominator nobody chose" in msg


def test_tolerated_models_are_not_checked_at_all():
    # a registered exclusion must not make the run fail, and must not hit the network either
    require_models(["fake/one"], offline=True, tolerate=["fake/one"])


def test_a_cached_model_passes_offline():
    from model_guard import _cached
    cached = _cached()
    if not cached:
        pytest.skip("no local HF cache to test against")
    name = sorted(cached)[0]
    ok, why = audit_models([name], offline=True)[name]
    assert ok and why == "cached"
