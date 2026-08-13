"""Forty predicates, forty known-answer checks. The pool gets a rung before any model runs.

A verifiable-instruction benchmark can be wrong in a way nobody notices: a predicate that accepts
everything makes every model look compliant, one that accepts nothing pins its type at zero and
silently removes it from the measure. F137 lost 24 of 120 items to two pinned types. So each
constraint is checked against a hand-written compliant answer AND a hand-written non-compliant one,
both written independently of the predicate's implementation.

The structural tests matter as much as the per-predicate ones: the pool's whole justification is
that resolution comes from the NUMBER OF INDEPENDENT TYPES, so anything that quietly makes two
types the same type undoes it.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src"), str(ROOT / "gatecheck" / "src")]

from verifiable_constraints import (                                          # noqa: E402
    CATEGORIES, CONSTRAINTS, PROMPTS, PROMPTS_PER_TYPE, prompts_for,
)

# Realistic base-model output, deliberately VARIED. A single sample cannot establish "satisfied by
# default": whether prose has commas, newlines or digits is a property of that sample, not of
# prose. The first version of this file used one clean paragraph and six constraints "failed" for
# no better reason than that the paragraph happened to have no commas and fit in 120 characters.
SAMPLES = [
    "The water cycle moves water between the sea and the sky. Rain falls onto the land.",
    "Water rises, condenses, and falls again, which is why rivers never run dry.",
    "Question: what is a volcano?\nAnswer: a mountain.\nQuestion: why does it erupt?",
    " ".join(["The water cycle carries water from the sea to the sky and back again."] * 6),
    "In 1823 (roughly) about 3 volcanoes erupted, and the ash spread for 400 km.",
]
PROSE = SAMPLES[0]

# (compliant, non-compliant) per constraint, written by hand as an independent statement of what
# each instruction means.
CASES = {
    "include_lantern": ("A lantern glows beside the river.", "A candle glows beside the river."),
    "include_zebra": ("A zebra drinks at the pool.", "A horse drinks at the pool."),
    "include_purple": ("The sky turned purple at dusk.", "The sky turned dark at dusk."),
    "include_phrase_in_fact": ("Rain falls; in fact it falls often.", "Rain falls often indeed."),
    "include_river_twice": ("The river swelled and the river fell.", "The river swelled."),

    "no_commas": ("Water rises then falls", "Water rises, then falls"),
    "no_periods": ("Water rises then falls", "Water rises. Then it falls."),
    "no_word_and": ("Water rises then falls", "Water rises and falls"),
    "no_word_is": ("Water rose quickly", "Water is rising"),
    "no_digits": ("Water rises then falls", "Water rises 3 times"),

    "title_brackets": ("<<Water>> Rain falls from clouds.", "Water: rain falls from clouds."),
    "start_certainly": ("Certainly, rain falls from clouds.", "Rain falls from clouds."),
    "start_numbered": ("1. Rain falls from clouds.", "Rain falls from clouds."),
    "start_hash": ("# Water\nRain falls.", "Water\nRain falls."),
    "start_answer_label": ("Answer: rain falls from clouds.", "Rain falls from clouds."),
    "start_caps_word": ("WATER rises then falls.", "Water rises then falls."),

    "end_that_is_all": ("Rain falls from clouds. That is all.", "Rain falls from clouds."),
    "end_question_mark": ("Why does rain fall?", "Rain falls from clouds."),
    "end_ellipsis": ("Rain falls from clouds...", "Rain falls from clouds."),
    "end_word_done": ("Rain falls from clouds, done", "Rain falls from clouds."),
    "end_exclamation": ("Rain falls from clouds!", "Rain falls from clouds."),

    "three_bullets": ("* rain\n* rivers\n* sea", "* rain\n* rivers"),
    "five_bullets": ("* a\n* b\n* c\n* d\n* e", "* a\n* b\n* c"),
    "four_dashes": ("- a\n- b\n- c\n- d", "- a\n- b\n- c"),
    "numbered_three": ("1. rain\n2. rivers\n3. sea", "1. rain\n2. rivers"),
    "two_paragraphs": ("Rain falls.\n\nRivers carry it.", "Rain falls. Rivers carry it."),
    "single_line": ("Rain falls from clouds.", "Rain falls.\nRivers carry it."),
    "every_line_period": ("Rain falls.\nRivers carry it.", "Rain falls\nRivers carry it."),

    "min_words_50": (" ".join(["water"] * 55), "Water rises then falls."),
    "max_words_25": ("Water rises then falls.", " ".join(["water"] * 40)),
    "one_sentence": ("Water rises then falls.", "Water rises. Then it falls."),
    "min_sentences_3": ("Water rises. It condenses. It falls.", "Water rises then falls."),
    "max_chars_120": ("Water rises then falls.", "x" * 200),
    "min_chars_300": ("y" * 350, "Water rises then falls."),

    "wrap_brackets": ("[Rain falls from clouds]", "Rain falls from clouds"),
    "wrap_quotes": ('"Rain falls from clouds"', "Rain falls from clouds"),
    "markdown_bold": ("Rain **falls** from clouds.", "Rain falls from clouds."),
    "parenthetical": ("Rain falls (usually) from clouds.", "Rain falls from clouds."),
    "json_answer": ('{"answer": "rain falls"}', '{"result": "rain falls"}'),

    "two_colours": ("The red roof under a blue sky.", "The red roof under a clear sky."),
}


def test_every_constraint_has_a_case():
    """A predicate with no known-answer check is a predicate nobody has verified."""
    assert set(CASES) == set(CONSTRAINTS)


def test_there_are_forty_types():
    assert len(CONSTRAINTS) == 40


@pytest.mark.parametrize("name", sorted(CASES))
def test_predicate_accepts_compliant_and_rejects_noncompliant(name):
    good, bad = CASES[name]
    pred = CONSTRAINTS[name][1]
    assert pred(good), f"{name} rejected a compliant answer: {good!r}"
    assert not pred(bad), f"{name} accepted a non-compliant answer: {bad!r}"


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_predicate_accepts_empty_output(name):
    """An empty generation must never score as compliant -- the cheapest way to fake a pass."""
    pred = CONSTRAINTS[name][1]
    assert not pred(""), f"{name} accepted an empty response"
    assert not pred("   \n\t "), f"{name} accepted whitespace"


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_constraint_is_satisfied_by_all_realistic_output(name):
    """The pinned-at-ONE failure, the mirror of pinned-at-zero and exactly as useless.

    A constraint that ordinary continuations satisfy every time carries no model information. What
    is checkable without a run is the weaker, honest version: on VARIED realistic output there is at
    least one sample the predicate rejects. Whether a type is actually pinned in practice is a
    question only the per-type range report can answer, and it is gated there.
    """
    pred = CONSTRAINTS[name][1]
    assert not all(pred(s) for s in SAMPLES), (
        f"{name} is satisfied by every realistic sample, so it cannot distinguish models")


def test_instructions_are_distinct_and_nonempty():
    texts = [v[0] for v in CONSTRAINTS.values()]
    assert len(set(texts)) == len(texts), "two types share an instruction string"
    # instructions legitimately END on a quoted literal -- 'Begin your response with "1."' -- so
    # requiring a trailing full stop was a wrong assertion about the data, not a finding about it
    assert all(t and t[0].isupper() and t.rstrip()[-1] in '."' for t in texts)


def test_no_two_types_agree_on_every_case():
    """Two predicates that never disagree are ONE type wearing two names.

    That matters more here than anywhere else: the pool's justification is that resolution scales
    with the number of INDEPENDENT types, so a duplicate is not merely redundant -- it inflates the
    apparent effective n while contributing nothing.
    """
    corpus = ([g for g, _ in CASES.values()] + [b for _, b in CASES.values()]
              + SAMPLES + [""])
    sigs = {}
    for name, (_i, pred, _c, _d) in CONSTRAINTS.items():
        sig = tuple(bool(pred(x)) for x in corpus)
        assert sig not in sigs, f"{name} and {sigs[sig]} agree on every test string"
        sigs[sig] = name


def test_difficulty_labels_are_from_the_registered_set():
    assert {v[3] for v in CONSTRAINTS.values()} <= {"easy", "medium", "hard"}


def test_categories_are_populated_and_none_dominates():
    """Range comes from diverse types; one category holding half the pool would undercut that."""
    counts = {c: sum(1 for v in CONSTRAINTS.values() if v[2] == c) for c in CATEGORIES}
    assert len(CATEGORIES) >= 7
    assert max(counts.values()) <= len(CONSTRAINTS) // 4, counts


def test_globally_reformatting_constraints_are_excluded_on_evidence():
    """F137 measured all_caps and lowercase at span 0.00 across ten models. They stay out."""
    assert "all_caps" not in CONSTRAINTS
    assert "lowercase" not in CONSTRAINTS
    for name, (instr, _p, _c, _d) in CONSTRAINTS.items():
        low = instr.lower()
        assert not ("entire response in capital" in low or "entirely in lowercase" in low), name


def test_prompt_rotation_uses_every_prompt_and_is_deterministic():
    seen = set()
    for name in CONSTRAINTS:
        ps = prompts_for(name)
        assert len(ps) == PROMPTS_PER_TYPE
        assert ps == prompts_for(name), "prompt assignment must be deterministic"
        seen.update(ps)
    assert seen == set(PROMPTS), "the rotation must exercise every prompt across the pool"


def test_item_count_and_effective_n_beat_the_design_that_failed():
    """The pool exists to clear F137's resolution failure, so the arithmetic is asserted.

    With ICC 0.774 measured in F137, effective n = n_items / (1 + (m-1) * ICC) where m is the
    number of prompts per type. The old design (10 types x 12) gave 12.6; this must beat it by a
    wide margin or the pool does not solve the problem it was built for.
    """
    icc, m = 0.774, PROMPTS_PER_TYPE
    n_items = len(CONSTRAINTS) * m
    eff = n_items / (1 + (m - 1) * icc)
    assert n_items == 240
    assert eff > 40, f"effective n {eff:.0f} is not a clear improvement on F137's 12.6"


def test_pool_is_local_not_loaded():
    src = (ROOT / "experiments" / "verifiable_constraints.py").read_text()
    for p in PROMPTS:
        assert p in src
    for bad in ("load_dataset", "hf_hub_download", "google/IFEval", "HuggingFaceH4/ifeval"):
        assert bad not in src
