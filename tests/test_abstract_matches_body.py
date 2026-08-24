"""The abstract must not say something its own body denies. R15's mechanical half.

WHY THIS EXISTS. An abstract is the body with the qualifications removed, and removing a
qualification sometimes changes the truth value rather than merely losing detail. Five documented
instances across two of this project's papers, all the same shape and all invisible to every other
check we run:

  * paper 3 compressed "concentrates more strongly than SOME Pythias" into "concentrating most
    strongly", which admits a superlative reading that the ladder refutes;
  * paper 4 asserted a novelty contrast that its own related-work section withdraws two pages later;
  * paper 4 wrote "0.8333 over 19 models" when the scored set is twelve and nineteen is the
    distractor pool;
  * paper 4 quoted balanced accuracy 1.0 while the body says "The honest figure is 0.90";
  * paper 4 wrote "4-bit destroys it" without the registered bound that round-to-nearest is the
    weakest standard quantizer.

Number-tracing passes on all five, because the figures are real. Prior-art gates pass, because the
concession exists -- in the body. The defect lives in the GAP between the two halves, and nothing
else looks at the gap.

WHAT THIS CAN AND CANNOT DO, stated so the pass is not over-read. Two of the five shapes are
mechanical and are enforced here. The other three are prose -- a withdrawn contrast, a misleading
juxtaposition, a dropped scope -- and need the human pass. A green run on this file means two
specific failure modes are absent, not that the abstract agrees with the body.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPERS = [p / "main.tex" for p in sorted(ROOT.glob("paper*_arxiv"))]
PRESENT = [p for p in PAPERS if p.exists()]

pytestmark = pytest.mark.skipif(not PRESENT, reason="no manuscripts present")

# Phrases by which a body designates one figure as the one to believe. Deliberately narrow: each is
# an explicit authorial ruling, not a turn of phrase.
AUTHORITATIVE = [
    r"honest figure is[^.\d]{0,30}\$?(\d+\.\d+)",
    r"figure we stand behind[^.\d]{0,40}\$?(\d+\.\d+)",
    r"\$?(\d+\.\d+)\$?[^.]{0,40}\bis the (?:figure|number) we stand behind",
    r"primary figure is[^.\d]{0,30}\$?(\d+\.\d+)",
    r"the primary is[^.\d]{0,30}\$?(\d+\.\d+)",
]


def _split(tex_path):
    t = tex_path.read_text()
    t = re.sub(r'(?m)(?<!\\)%.*$', '', t)                       # comments are not the paper
    i, j = t.index(r"\begin{abstract}"), t.index(r"\end{abstract}")
    abstract = t[i + len(r"\begin{abstract}"):j]
    body = t[:i] + t[j + len(r"\end{abstract}"):]
    return abstract, body


def _lits(s):
    return set(re.findall(r'(?<![\w.])\d+\.\d+(?![\w])', s))


def _covered(a, body_lits):
    """Is abstract figure `a` present in the body, allowing the abstract to have ROUNDED it?

    An abstract legitimately rounds for readability -- paper 2 writes $0.21$ for a body figure of
    $0.214$ -- and a check that calls that a defect would be turned off within a week. So a match is
    exact, or some body figure rounds to it at the abstract's own precision. What this deliberately
    does NOT allow is an abstract figure with no body figure anywhere near it, which is the case
    worth catching.
    """
    if a in body_lits:
        return True
    d = len(a.split(".")[1])
    return any(round(float(b), d) == float(a) for b in body_lits)


@pytest.mark.parametrize("tex", PRESENT, ids=lambda p: p.parent.name)
def test_no_number_appears_only_in_the_abstract(tex):
    """An abstract-only figure is either unsupported or computed where the paper never shows it."""
    abstract, body = _split(tex)
    body_lits = _lits(body)
    orphan = sorted(l for l in _lits(abstract) if not _covered(l, body_lits))
    assert not orphan, (
        f"{tex.parent.name}: {len(orphan)} figure(s) appear in the abstract and nowhere in the "
        f"body: {orphan}. Either the body should state them, or the abstract should not. An "
        f"abstract is a compression of the paper, not an additional source of numbers. Rounding is "
        f"allowed: a body figure that rounds to the abstract's value at the abstract's own "
        f"precision counts as present.")


@pytest.mark.parametrize("tex", PRESENT, ids=lambda p: p.parent.name)
def test_a_figure_the_body_calls_authoritative_reaches_the_abstract(tex):
    """R15 instance 4: the abstract quoted 1.0 while the body said the honest figure is 0.90."""
    abstract, body = _split(tex)
    ruled = set()
    for pat in AUTHORITATIVE:
        ruled |= set(re.findall(pat, body))
    if not ruled:
        pytest.skip("the body designates no figure as authoritative")
    abstract_lits = _lits(abstract)
    missing = sorted(v for v in ruled if v not in abstract_lits)
    assert not missing, (
        f"{tex.parent.name}: the body names {missing} as the figure to believe, and the abstract "
        f"does not carry it. If a paper has to tell the reader which of two numbers is honest, the "
        f"abstract is where that number belongs.")
