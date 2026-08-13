"""Forty programmatically verifiable output constraints — the item pool F137 said was needed.

WHY FORTY, AND WHY TYPES RATHER THAN ITEMS. F137 built a compliance measure from 10 constraint
types x 12 prompts and it did not resolve ten models: reliability -12.4, an across-model span four
times SMALLER than pure noise would produce. The diagnosis was not "too few items" but "too few
TYPES": the intra-class correlation by constraint type came in at **ICC = 0.774**, so a model that
fails one prompt of a type fails nearly all of them, and 120 items were worth an effective n of
12.6. Adding prompts inside a type buys almost nothing; the design effect at 12 prompts is 9.5x.

So the arithmetic runs the other way from the obvious fix:

    40 types x 6 prompts  = 240 items, deff = 1 + 5(0.774) = 4.87, effective n ~ 49
    10 types x 48 prompts = 480 items, deff = 1 + 47(0.774) = 37.4, effective n ~ 13

Twice the items and four times the resolution, versus four times the items and none. Resolution
lives in the number of independent types.

WHAT THE POOL IS DESIGNED AGAINST, learned from F137's per-type spans rather than guessed:

  no_commas      0.92    include_word 0.67   title_brackets 0.67   three_bullets 0.58
  min_words      0.33    end_phrase   0.25   forbid "the" 0.08     wrap_quotes    0.08
  all_caps       0.00    lowercase    0.00

The pattern is sharp and it is the design rule here: **constraints satisfiable by a LOCAL property
of the text carry range; constraints demanding GLOBAL reformatting are pinned at zero.** A base
model continues text, it does not restructure its whole output, so "write everything in capitals"
is not a hard instruction to it -- it is an impossible one, and 24 of F137's 120 items were dead
weight. `all_caps` and `lowercase` are therefore excluded on evidence, not on taste. Their local
cousins (first word in capitals) are kept, because a local casing constraint can be satisfied by a
continuation.

A pinned-at-ONE constraint is exactly as useless as a pinned-at-zero one, so constraints a
continuation satisfies by accident are also avoided (no "avoid semicolons").

EVERY PREDICATE IS A TOTAL FUNCTION of the response text. No judge model, no rubric, no tolerance
chosen after seeing an answer. `difficulty` is a PREDICTION recorded before the run so the pool can
be scored against it afterwards: if predicted difficulty does not track observed pass rates, the
author does not understand the measure, and that is worth finding out.

INDEPENDENCE, RESTATED PRECISELY. The constraint FAMILY here is the same as IFEval's -- verifiable
instruction following is what "compliance" means, and a second measure of a different construct
would answer a different question. What is independent is the POOL: every prompt, keyword, phrase
and count below is written here, and nothing is loaded from any dataset.
"""
import json as _json
import re

__all__ = ["CONSTRAINTS", "PROMPTS", "prompts_for", "CATEGORIES"]

PROMPTS = [
    "Describe the water cycle.",
    "Explain why the sky appears blue.",
    "Write a short paragraph about the history of paper.",
    "Summarise what a volcano is.",
    "Explain what causes the seasons on Earth.",
    "Describe how bread is made.",
    "Explain the purpose of a public library.",
    "Describe what happens during a thunderstorm.",
    "Explain how a bicycle works.",
    "Describe the life cycle of a butterfly.",
    "Explain what a desert is.",
    "Describe how honey is produced.",
]

PROMPTS_PER_TYPE = 6

_COLOURS = ("red", "green", "blue", "yellow", "black", "white", "purple", "orange",
            "brown", "grey", "gray", "pink")


def _words(t):
    return re.findall(r"[A-Za-z']+", t)


def _lines(t):
    return [l for l in t.splitlines() if l.strip()]


def _sentence_marks(t):
    return re.findall(r"[.!?]", t)


def _nonempty(t):
    return bool(t) and bool(t.strip())


def _is_json_answer(t):
    try:
        v = _json.loads(t.strip())
    except Exception:
        return False
    return isinstance(v, dict) and "answer" in v


# name -> (instruction, predicate, category, predicted difficulty)
# Difficulty is the AUTHOR'S PREDICTION, recorded before any run: easy = most base models comply,
# hard = few do. A pool whose observed pass rates ignore these predictions is a pool its author
# did not understand.
CONSTRAINTS = {
    # ---- lexical inclusion: local, cheap to satisfy, F137's best-performing family --------------
    "include_lantern": ('Include the word "lantern" somewhere in your response.',
                        lambda t: "lantern" in t.lower(), "lexical_include", "medium"),
    "include_zebra": ('Include the word "zebra" somewhere in your response.',
                      lambda t: "zebra" in t.lower(), "lexical_include", "medium"),
    "include_purple": ('Include the word "purple" somewhere in your response.',
                       lambda t: "purple" in t.lower(), "lexical_include", "medium"),
    "include_phrase_in_fact": ('Include the exact phrase "in fact" in your response.',
                               lambda t: "in fact" in t.lower(), "lexical_include", "medium"),
    "include_river_twice": ('Use the word "river" at least twice in your response.',
                            lambda t: [w.lower() for w in _words(t)].count("river") >= 2,
                            "lexical_include", "hard"),

    # ---- lexical exclusion: no_commas had the widest span of all ten in F137 --------------------
    "no_commas": ("Do not use any commas in your response.",
                  lambda t: _nonempty(t) and "," not in t, "lexical_exclude", "medium"),
    "no_periods": ("Do not use any full stops in your response.",
                   lambda t: _nonempty(t) and "." not in t, "lexical_exclude", "hard"),
    "no_word_and": ('Do not use the word "and" anywhere in your response.',
                    lambda t: _nonempty(t) and "and" not in [w.lower() for w in _words(t)],
                    "lexical_exclude", "medium"),
    "no_word_is": ('Do not use the word "is" anywhere in your response.',
                   lambda t: _nonempty(t) and "is" not in [w.lower() for w in _words(t)],
                   "lexical_exclude", "medium"),
    "no_digits": ("Do not use any digits in your response.",
                  lambda t: _nonempty(t) and not re.search(r"\d", t),
                  "lexical_exclude", "easy"),

    # ---- opening: a continuation CAN start a particular way, so these carry range ---------------
    "title_brackets": ("Begin your response with a title enclosed in double angular brackets, "
                       "like <<Title>>.",
                       lambda t: bool(re.match(r"\s*<<[^<>]+>>", t)), "opening", "medium"),
    "start_certainly": ('Begin your response with the word "Certainly".',
                        lambda t: t.strip().lower().startswith("certainly"), "opening", "medium"),
    "start_numbered": ('Begin your response with "1."',
                       lambda t: t.strip().startswith("1."), "opening", "medium"),
    "start_hash": ('Begin your response with a hash and a space, like "# ".',
                   lambda t: t.strip().startswith("# "), "opening", "medium"),
    # NOT "begin with a double quote": that is satisfied by every response `wrap_quotes` satisfies,
    # so the two would be one type wearing two names -- and the pool's whole justification is that
    # resolution scales with the number of INDEPENDENT types. Caught by the duplicate-signature test.
    "start_answer_label": ('Begin your response with "Answer:".',
                           lambda t: t.strip().startswith("Answer:"), "opening", "medium"),
    "start_caps_word": ("Begin your response with a single word written in capital letters.",
                        lambda t: bool(_words(t)) and len(_words(t)[0]) >= 2
                        and _words(t)[0].isupper(), "opening", "hard"),

    # ---- closing: harder than opening -- the model must stop deliberately -----------------------
    "end_that_is_all": ('Finish your response with the exact phrase "That is all."',
                        lambda t: t.strip().rstrip('"').rstrip().endswith("That is all."),
                        "closing", "hard"),
    "end_question_mark": ("End your response with a question mark.",
                          lambda t: t.strip().endswith("?"), "closing", "medium"),
    "end_ellipsis": ("End your response with three dots.",
                     lambda t: t.strip().endswith("..."), "closing", "hard"),
    "end_word_done": ('End your response with the word "done".',
                      lambda t: bool(_words(t)) and _words(t)[-1].lower() == "done",
                      "closing", "hard"),
    "end_exclamation": ("End your response with an exclamation mark.",
                        lambda t: t.strip().endswith("!"), "closing", "medium"),

    # ---- structure: three_bullets scored 0.58 in F137, the best structural signal ---------------
    "three_bullets": ('Answer using exactly three bullet points, each line starting with "* ".',
                      lambda t: len([l for l in _lines(t) if l.strip().startswith("* ")]) == 3,
                      "structure", "medium"),
    "five_bullets": ('Answer using exactly five bullet points, each line starting with "* ".',
                     lambda t: len([l for l in _lines(t) if l.strip().startswith("* ")]) == 5,
                     "structure", "hard"),
    "four_dashes": ('Answer using exactly four bullet points, each line starting with "- ".',
                    lambda t: len([l for l in _lines(t) if l.strip().startswith("- ")]) == 4,
                    "structure", "hard"),
    "numbered_three": ('Answer with exactly three numbered lines, starting "1.", "2." and "3.".',
                       lambda t: [l.strip()[:2] for l in _lines(t)[:3]] == ["1.", "2.", "3."]
                       and len(_lines(t)) == 3, "structure", "hard"),
    "two_paragraphs": ("Write exactly two paragraphs separated by a blank line.",
                       lambda t: len([b for b in re.split(r"\n\s*\n", t.strip()) if b.strip()]) == 2,
                       "structure", "medium"),
    "single_line": ("Write your entire response on a single line with no line breaks.",
                    lambda t: _nonempty(t) and "\n" not in t.strip(), "structure", "medium"),
    "every_line_period": ("End every line of your response with a full stop.",
                          lambda t: bool(_lines(t))
                          and all(l.rstrip().endswith(".") for l in _lines(t)),
                          "structure", "medium"),

    # ---- length: min_words scored 0.33; both directions are included ----------------------------
    "min_words_50": ("Write at least 50 words.", lambda t: len(_words(t)) >= 50,
                     "length", "easy"),
    "max_words_25": ("Write no more than 25 words.",
                     lambda t: _nonempty(t) and len(_words(t)) <= 25, "length", "hard"),
    "one_sentence": ("Write exactly one sentence.",
                     lambda t: len(_sentence_marks(t)) == 1, "length", "hard"),
    "min_sentences_3": ("Write at least three sentences.",
                        lambda t: len(_sentence_marks(t)) >= 3, "length", "easy"),
    "max_chars_120": ("Write no more than 120 characters.",
                      lambda t: _nonempty(t) and len(t.strip()) <= 120, "length", "hard"),
    "min_chars_300": ("Write at least 300 characters.",
                      lambda t: len(t.strip()) >= 300, "length", "medium"),

    # ---- markup: wrap_quotes scored only 0.08, so its siblings are varied rather than copied ----
    "wrap_brackets": ("Wrap your entire response in square brackets.",
                      lambda t: t.strip().startswith("[") and t.strip().endswith("]")
                      and len(t.strip()) > 2, "markup", "hard"),
    "wrap_quotes": ("Wrap your entire response in double quotation marks.",
                    lambda t: t.strip().startswith('"') and t.strip().endswith('"')
                    and len(t.strip()) > 2, "markup", "hard"),
    "markdown_bold": ("Include at least one word in bold, using double asterisks.",
                      lambda t: bool(re.search(r"\*\*[^*\s][^*]*\*\*", t)), "markup", "medium"),
    "parenthetical": ("Include a remark in round brackets.",
                      lambda t: bool(re.search(r"\([^()]+\)", t)), "markup", "medium"),
    "json_answer": ('Reply with a JSON object with a single key "answer".',
                    _is_json_answer, "markup", "hard"),

    # ---- content-shaped but still mechanically checkable ----------------------------------------
    "two_colours": ("Mention at least two different colours.",
                    lambda t: len({w.lower() for w in _words(t)} & set(_COLOURS)) >= 2,
                    "content", "medium"),
}

CATEGORIES = sorted({c for _, _, c, _ in CONSTRAINTS.values()})


def prompts_for(name):
    """Six prompts per type, rotated so the 40 types between them use all twelve prompts.

    Rotation rather than "the first six": if every type saw the same six, prompt-specific quirks
    would be perfectly confounded with the type effect, and the pool would be narrower than its
    item count suggests. Deterministic, so every model sees an identical assignment.
    """
    i = sorted(CONSTRAINTS).index(name)
    return [PROMPTS[(i * PROMPTS_PER_TYPE + j) % len(PROMPTS)] for j in range(PROMPTS_PER_TYPE)]
