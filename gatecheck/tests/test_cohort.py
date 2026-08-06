"""The cohort guard, tested against the incident that produced it."""
import pytest
from gatecheck.cohort import cohort_complete, require_cohort

DECLARED = ["EleutherAI/pythia-410m", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B",
            "sapienzanlp/Minerva-3B-base-v1.0", "pfnet/plamo-3-nict-2b-base", "gpt2"]


def test_the_actual_incident_five_gated_models_vanish():
    r = cohort_complete(DECLARED, ["EleutherAI/pythia-410m", "gpt2"], unit="model")
    assert not r.complete and len(r.missing) == 4
    assert "denominator nobody chose" in r.reason


def test_intact_cohort_passes():
    r = cohort_complete(DECLARED, list(DECLARED), unit="model")
    assert r.complete and not r.missing and not r.extra


def test_a_pre_declared_exclusion_is_tolerated_and_still_reported():
    r = cohort_complete(DECLARED, [d for d in DECLARED if d != "gpt2"],
                        tolerate=["gpt2"], unit="model")
    assert r.complete and r.tolerated == ["gpt2"] and not r.missing
    assert "pre-declared exclusion" in r.reason


def test_measuring_something_never_declared_is_also_incomplete():
    r = cohort_complete(["a", "b"], ["a", "b", "c"], unit="model")
    assert not r.complete and r.extra == ["c"]


def test_require_cohort_returns_not_decidable_and_carries_no_value():
    v = require_cohort(cohort_complete(["a", "b"], ["a"]), value=0.833)
    assert not v.decided and v.value is None
    assert require_cohort(cohort_complete(["a"], ["a"]), value=0.833).value == 0.833


def test_report_is_falsy_when_incomplete_and_serialises():
    r = cohort_complete(["a", "b"], ["a"])
    assert not r
    b = r.block()
    assert b["n_declared"] == 2 and b["n_realised"] == 1 and b["missing"] == ["b"]
