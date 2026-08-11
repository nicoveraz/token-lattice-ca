# Contributing

This is a research repository, not a library, and the thing it is trying to protect is the
credibility of its own numbers. Contributions are welcome; the conventions below are not style
preferences but the mechanisms that have caught eleven confident wrong verdicts before they reached
a paper. Most of them exist because something got past the previous version.

## The one rule everything else follows from

**A check that cannot fail is not a check.** The recurring defect in this project — eleven instances
and counting — is *a statistically-shaped criterion applied to a quantity with no room to vary*: a
correlation whose predictor is saturated, a ratio whose denominator is noise, a null whose p-value
is structurally floored, a control that recomputes the thing it is controlling. Each returns a
confident number from a comparison that could not have come out otherwise.

Before adding a check, ask what result would make it fail. If you cannot name one, it is decoration.

## What a new experiment needs

**A pre-registration block**, written before the run, stating the primary statistic, the reading
that would kill the hypothesis, and the boundary. `experiments/` is full of examples. The block goes
in the results file, not only in the docstring, so the claim and the data travel together.

**A rung — a known-answer check that gates the read.** The pattern throughout: measure something
whose answer is established independently, and if it does not reproduce, *stop and read nothing
below it*. Domany–Kinzel is the strictest host (the damage field is provably the automaton itself,
so the prediction has no error bar); elementary CA rules and previously-committed results are looser
ones. Scripts here return `NOT DECIDABLE` and refuse rather than degrade.

**Thresholds derived from measured noise, not from reasoning.** Three rungs failed in one session
because a tolerance was set by argument. Measure the estimator's seed noise, then set the tolerance
from it, and say in the code that this is what you did. A tolerance loosened *after* a failure is
only legitimate when the justification is a prior result rather than the wish for a passing test —
and the code should say which.

**A provenance stamp.** `provenance.stamp(__file__)` records the SHA-256 of the analysis and of
every repo-local module it imports. `tests/test_results_self_consistency.py` fails if a results file
disagrees with the code that claims to have written it. This is not bureaucracy: it is the check
that stops a figure outliving the code that produced it.

## Working with the findings ledger

`findings.md` is a dated record, numbered `F1`…, and it is append-mostly. **Retractions and
amendments are stated in place, not by deletion.** If a finding turns out to be wrong, the entry
says so where the claim was made and names what supersedes it. Several entries in this repository
are corrections of other entries; that is the design.

If you contribute a finding:

- state the boundary as specifically as the result — "n = 10, base models, benchmark scores
  downloaded rather than measured" is the useful kind
- record the registered reading *before* the number, especially when the expected outcome is the
  deflationary one
- if your summariser could print a misleading sentence on the data you got, fix the summariser and
  say so in the entry. Several findings here document exactly that

Results files whose *design* was superseded are kept as `*_superseded.json` and are exempt from the
drift check. They must not be quoted; the finding that supersedes them carries the numbers.

## Code

- Python, `numpy`-first. `.venv/bin/python -m pytest tests/ -q` must pass before a commit — the
  suite is ~470 tests and takes a few minutes.
- Rank correlations go through `experiments/ranking.py`. Do **not** reintroduce
  `np.argsort(np.argsort(x))`: it breaks ties by input position, so a constant vector is ranked as
  strictly increasing. It was in fifteen scripts for months and reported ρ = +0.829 for a quantity
  whose every measured value was 0.000. `tests/test_ranking.py` greps for it.
- Long runs get wrapped in `caffeinate -i` and write results incrementally, so a crash or a sleeping
  machine loses nothing.
- Never commit a results file a job is still writing. A pre-commit guard blocks this; it exists
  because a partial file that looks finished is worse than an absent one.

## Papers

Every decimal literal in `paper_arxiv/main.tex` must trace to a file in `results/`, asserted by
`tests/test_arxiv_paper_numbers.py`. Note what that test does *not* do: it checks numbers, not
claims. Three claim-level errors were found in one read-through — prose that sounded right and
contradicted the repository's own findings. If you edit the paper, check that each assertion's
supporting finding has not since been retracted, amended, or scoped.

## Scope

Issues and pull requests that extend the instrument, add rungs, or find defects in existing results
are all welcome — the last category most of all. If you think a number here is wrong, the fastest
path is a script that demonstrates it, with a rung showing the demonstration itself works.

## Licence

Code is MIT (`LICENSE`); prose, the findings ledger, and `results/` as accompanying data are
CC BY 4.0 (`LICENSE-docs`). By contributing you agree your contribution is released under whichever
of the two covers the files you touched.
