"""The scorer gets a rung before the models do.

`compliance_second_measure` exists to test whether F117's compliance signal survives on an
INDEPENDENT indicator. That question is only worth asking if the new indicator is correct, and a
verifiable-instruction benchmark is exactly the kind of thing that can be subtly wrong in a
direction nobody notices: a predicate that accepts everything makes a model look compliant, a
predicate that accepts nothing pins the type at zero and quietly removes it from the measure.

So every constraint predicate is checked against a hand-written compliant answer and a hand-written
non-compliant one. This is the estimator-must-reproduce-a-known-answer discipline applied to a
scoring function rather than to a measurement.

The `loosen` cases matter for a second reason: loose scoring exists because base models wrap a
compliant answer in boilerplate, and a loosener that is too aggressive would start rescuing answers
that genuinely ignored the constraint -- turning the measure's own noise floor into signal.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src"), str(ROOT / "gatecheck" / "src")]

import importlib.util                                                        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "csm", ROOT / "experiments" / "compliance_second_measure.py")
csm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csm)

# (compliant, non-compliant) for every constraint. Written by hand, not generated, so they are an
# independent statement of what each instruction means.
CASES = {
    "all_caps": ("THE WATER CYCLE MOVES WATER", "the water cycle moves water"),
    "lowercase": ("the water cycle moves water", "The Water Cycle Moves Water"),
    "no_commas": ("water rises then falls", "water rises, then falls"),
    "include_word": ("a lantern glows softly", "a candle glows softly"),
    "forbid_word": ("water rises and falls", "the water rises"),
    "min_words": (" ".join(["word"] * 55), "a short answer"),
    "end_phrase": ("Rain falls from clouds. That is all.", "Rain falls from clouds."),
    "three_bullets": ("* first\n* second\n* third", "* first\n* second"),
    "wrap_quotes": ('"rain falls from clouds"', "rain falls from clouds"),
    "title_brackets": ("<<Water>> rain falls from clouds", "Water: rain falls from clouds"),
}


def test_every_constraint_has_a_case():
    """A predicate with no known-answer check is a predicate nobody has verified."""
    assert set(CASES) == set(csm.CONSTRAINTS)


@pytest.mark.parametrize("name", sorted(CASES))
def test_predicate_accepts_compliant_and_rejects_noncompliant(name):
    good, bad = CASES[name]
    pred = csm.CONSTRAINTS[name][1]
    assert pred(good), f"{name} rejected a compliant answer"
    assert not pred(bad), f"{name} accepted a non-compliant answer"


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_predicate_accepts_empty_output(name):
    """An empty generation must never score as compliant -- it is the cheapest way to fake a pass."""
    pred = csm.CONSTRAINTS[name][1]
    assert not pred(""), f"{name} accepted an empty response"
    assert not pred("   \n  "), f"{name} accepted whitespace"


def test_loosen_strips_wrappers_but_does_not_rescue_a_violation():
    """Loose scoring may remove boilerplate AROUND a compliant answer. It may not invent one."""
    wrapped = '```\nResponse: "rain falls"\n```'
    assert csm.CONSTRAINTS["wrap_quotes"][1](csm.loosen(wrapped))
    # a genuine violation stays a violation after loosening
    for name in ("all_caps", "no_commas", "include_word", "three_bullets"):
        _, bad = CASES[name]
        assert not csm.CONSTRAINTS[name][1](csm.loosen(bad)), \
            f"loosen() rescued a genuine {name} violation"


def test_loose_score_is_never_below_strict():
    """By construction loose = strict OR loosened; an inversion would mean the scorer is wrong."""
    responses = {(c, 0): CASES[c][0] for c in csm.CONSTRAINTS}
    responses.update({(c, 1): CASES[c][1] for c in csm.CONSTRAINTS})
    strict, loose = csm.score(responses)
    for c in csm.CONSTRAINTS:
        assert sum(loose[c]) >= sum(strict[c]), f"{c}: loose scored below strict"


def test_prompt_format_carries_both_task_and_constraint():
    p = csm.prompt_for("Describe the water cycle.", csm.CONSTRAINTS["all_caps"][0])
    assert "Describe the water cycle." in p and "capital letters" in p
    assert p.rstrip().endswith("Response:")


def test_item_pool_is_the_registered_size_and_shares_nothing_with_ifeval():
    """The independence claim is structural, so it is asserted rather than described.

    IFEval's pool is not vendored here to check against, so what is checkable is that the pool is
    OURS: every prompt and every constraint string is a literal in the script's source, and no item
    data is loaded from anywhere.

    The first version of this test grepped the source for the string "ifeval" and failed, because
    the script must NAME IFEval -- the convergence rung correlates against it. Independence is about
    where the ITEMS come from, not about whether the benchmark is mentioned; a test that conflates
    the two forbids the script from documenting its own rung.
    """
    assert len(csm.PROMPTS) * len(csm.CONSTRAINTS) == 120
    assert all(isinstance(p, str) and p for p in csm.PROMPTS)
    src = (ROOT / "experiments" / "compliance_second_measure.py").read_text()
    for p in csm.PROMPTS:
        assert p in src, "prompts must be literals in the script, not loaded"
    for bad in ("load_dataset", "hf_hub_download", "google/IFEval", "HuggingFaceH4/ifeval"):
        assert bad not in src, f"the item pool must not be sourced from {bad}"
