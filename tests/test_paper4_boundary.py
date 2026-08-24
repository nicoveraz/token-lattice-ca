"""The boundary paragraph must survive into paper 4, and this fails the day it does not.

WHY A TEST AND NOT A NOTE. F185's boundary is not decoration; each clause exists because a specific
reading of the numbers would otherwise be wrong:

  * both leave-one-out attribution misses are Mamba landing on RWKV -- recurrent on recurrent -- so
    what the fingerprint recovers may be ARCHITECTURE CLASS rather than family, and 10/12 may not be
    quoted without it;
  * family is confounded with tokenizer (Pythia/RWKV/Mamba use GPT-NeoX vocabularies, GPT-Neo uses
    GPT-2's), so a family result is not a family result alone;
  * every model is Pile-trained, so nothing here is a corpus effect AND nothing here says anything
    about corpus;
  * quantization robustness is OWED -- bfloat16 does not discharge it, because 4- and 8-bit are
    categorically larger perturbations and are the deployment-relevant ones;
  * the tau=0.5 rung is not the registered primary and may not be promoted without its own
    pre-registered replication.

A caveat that lives only in a findings entry is a caveat that gets dropped in drafting. This test
skips while paper4_arxiv/main.tex does not exist and fires the moment it does, so the boundary
cannot fossilise out of the paper silently.

It checks for the CLAIMS, not for exact sentences: each rule below accepts any phrasing that carries
the substance, so the paper can be written in its own voice and still be held to the constraint.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = ROOT / "paper4_arxiv" / "main.tex"
DELTA = ROOT / "paper4_arxiv" / "DELTA.md"

# Applied per-test rather than module-wide, because the LAST test in this file guards DELTA.md and
# its prohibitions, which exist NOW and can drift NOW -- gating it on a manuscript that does not yet
# exist would leave the only currently-checkable thing unchecked.
needs_draft = pytest.mark.skipif(
    not TEX.exists(),
    reason="paper4_arxiv/main.tex not present -- the boundary cannot be checked until a draft exists")


def _body():
    t = TEX.read_text()
    return re.sub(r'(?m)(?<!\\)%.*$', '', t)


# (name, alternative patterns -- ANY match satisfies it, so the paper keeps its own voice)
REQUIRED = [
    ("architecture-class caveat on the attribution",
     [r"(?is)architecture class", r"(?is)both\s+miss\w*\s+are\s+\\?texttt\{?mamba",
      r"(?is)recurrent[^.]{0,80}recurrent"]),
    ("family/tokenizer confound",
     [r"(?is)confound\w*[^.]{0,120}tokeni[sz]er", r"(?is)tokeni[sz]er[^.]{0,120}confound"]),
    ("one-corpus scope",
     [r"(?is)(all|every|each)[^.]{0,60}(the\s+)?pile", r"(?is)one\s+corpus",
      r"(?is)single\s+corpus"]),
    ("quantization owed or discharged, explicitly",
     [r"(?is)quanti[sz]ation", r"(?is)\b4-?bit\b", r"(?is)\b8-?bit\b"]),
]


@needs_draft
def test_the_boundary_claims_survive_into_the_manuscript():
    body = _body()
    missing = [name for name, pats in REQUIRED
               if not any(re.search(p, body) for p in pats)]
    assert not missing, (
        f"paper4_arxiv/main.tex does not carry these boundary claims: {missing}. Each one exists "
        f"because a specific reading of the numbers is wrong without it -- see F185's boundary "
        f"paragraph and this file's docstring. Add the claim, do not delete the check.")


@needs_draft
def test_the_attribution_number_is_never_quoted_bare():
    """10/12 is the number a reader will lift. It may not appear without its caveat nearby."""
    body = _body()
    # Match the ATTRIBUTION figure specifically. An earlier version also matched a bare "83%",
    # which caught E1's cardinality variance -- an unrelated percentage that happens to share two
    # digits with the accuracy. A guard that fires on the wrong sentence trains its reader to
    # ignore it, so the pattern names the figure as the paper actually writes it: 0.8333, or a
    # spelled-out ratio.
    hits = [m.start() for m in re.finditer(r"(?i)\b10\s*(of|/)\s*12\b|0\.8333", body)]
    if not hits:
        pytest.skip("the attribution figure is not quoted in the manuscript")
    caveat = re.compile(r"(?is)architecture class|recurrent|tokeni[sz]er|chance")
    naked = [h for h in hits if not caveat.search(body[max(0, h - 700):h + 700])]
    assert not naked, (
        f"the attribution figure is quoted {len(naked)} time(s) with no caveat within 700 "
        f"characters. Both misses are Mamba on RWKV, and family is confounded with tokenizer; the "
        f"number is not reportable on its own.")


@needs_draft
def test_the_tau_half_rung_is_not_promoted():
    """tau=1.0 was the registered primary. Promoting the best rung after the fact is the defect."""
    body = _body()
    # Trigger on the RESOLUTION FIGURE alone. An earlier version also required the string "0.5" to
    # appear, which let a draft quoting "a resolution of 11.76 at the best threshold" pass silently
    # -- exactly the sentence this test exists to catch. Proved by writing that sentence.
    if not re.search(r"(?i)11\.7\d?", body):
        pytest.skip("the tau=0.5 resolution figure is not quoted")
    ok = re.search(r"(?is)(registered|primary|pre-?registered|replicat|not\s+promot|observation)", body)
    assert ok, (
        "the tau=0.5 resolution is quoted without any of: the registered primary, the word "
        "replication, or a statement that it is an observation rather than a claim. The ladder was "
        "registered in full precisely so no rung would be chosen afterwards.")


@needs_draft
def test_the_delta_paragraphs_are_carried_into_the_manuscript():
    """F186 made a delta against BOTH arXiv:2410.06287 and paper 1 mandatory before any write-up."""
    raw = TEX.read_text()
    for key, why in (("hammouri2025nonhalting",
                      "arXiv:2410.06287 partially anticipates the self-continuation set"),
                     ("veraz2026probes",
                      "our own paper 1 already banks the argmax map and the fixed-point contrast")):
        assert key in raw, (
            f"paper4_arxiv/main.tex does not cite `{key}`: {why}. F186 registered a delta against "
            f"it as mandatory; the paragraphs are drafted in paper4_arxiv/DELTA.md.")


def test_the_delta_source_file_still_exists_and_carries_its_prohibitions():
    """The drafted paragraphs and their binding constraints must not drift apart from the paper."""
    assert DELTA.exists(), "paper4_arxiv/DELTA.md is gone; F186's mandated delta has no source"
    d = DELTA.read_text()
    for phrase in ("degenerate probes", "fixed points of greedy decoding"):
        assert phrase in d, (
            f"DELTA.md no longer records the prohibition on pitching the work as '{phrase}'. "
            f"F186 made it binding.")
