# Merge manifest — `paper4` → `main`

> **EXECUTED 25 Aug 2026, on the author's instruction.** `paper4` fast-forwarded into `main` at
> `52336d3`, the suite ran green on `main` (833 passed, 9 skipped), and `main` was pushed — 61
> commits, `e8062ab..52336d3`. Two things this document did not anticipate, recorded here rather
> than rewritten above. The privacy sweep below **asserted a check it had not run**: four files
> carried `/Users/<name>/…`, none printed by our own code, and they were scrubbed in `52336d3`
> before the push. And the release was **not** cut with the merge — paper 3 is on hold at arXiv, so
> `v1.3.0` and paper 4's submission are both parked; the notes are written in `RELEASE_NOTES.md`
> with the two lines the identifier will fill.
>
> **Merging was deliberately decoupled from submitting.** Publishing the evidence is the
> precondition for submitting, not a decision to submit, and submitting a fourth paper while the
> third sits under moderation review is the action most likely to lengthen that review.

**Prepared, not executed** *(as written; see the banner above)*. The merge is the author's act. Nothing here has been run against `main`
and no branch has been pushed. This exists so the decision can be made from facts rather than from a
diff scrolled past at midnight.

**Why it is needed at all.** Paper 4's results files exist only on this branch. Its comments line
points readers at `https://github.com/nicoveraz/token-lattice-ca` for "code, per-run results,
pre-registrations and the findings ledger", and until `paper4` lands on `main` that link resolves to
a repository containing none of them. Submitting first would publish a citation to something not yet
public — the same reason `v1.1.0` and `v1.2.0` were cut before their papers were submitted.

---

## Headline

**56 commits, 104 files: 101 added, 3 modified, 0 deleted. `main` has no commits `paper4` lacks, so
this is a fast-forward and no merge conflict is possible.**

```
git rev-list --count main..paper4   →  56
git rev-list --count paper4..main   →   0
```

| group | files |
|---|---|
| `results/` | 47 |
| `experiments/` | 32 |
| `logs/` | 13 |
| `paper4_arxiv/` | 5 |
| `tests/` | 4 |
| `fingerprint/` | 1 |
| root (`findings.md`, `.gitignore`) | 2 |

## What lands, and why each group is there

**Eight frozen pre-registrations with their `.sha256` sidecars**, each committed before the run it
governs: `selfcont`, `escape_rival`, `escape_confidence`, `escape_widening`, `quant_robustness`,
`quant_grouped`, `floor_survey`, plus the probe-string freeze. The commit graph is the ordering
evidence — `probe_strings_selfcont` precedes `prereg_selfcont` precedes the first cell, and
`matched_entropy_pairing` was committed in a tree containing no Δφ at all.

**Findings F183–F190**, including three that record defects in this work rather than in the world:
R13's cardinality confound found in a registered primary estimand, R15's abstract-versus-body
compression found in a one-day-old draft, and F190's censoring, where an anti-vacuity gate tested the
wrong quantity and the first reading said the opposite of the truth.

**Four test files.** `test_selfcont_prereg.py` (freeze integrity and base rates),
`test_paper4_numbers.py` (number tracing plus the two prohibited pitches),
`test_paper4_boundary.py` (the boundary claims must survive into the manuscript), and
`test_abstract_matches_body.py` (R15's mechanical half, which runs against **all four** manuscripts,
not just this one).

**One `fingerprint/PROGRAM.md` change**: §1's verdict (b) is narrowed, because F186 found it false
for these two features. That file described a feature set nobody was using any more.

## A size decision the author should make consciously

**The results payload is 35.8 MB**, dominated by per-token arrays:

```
6.5 MB  results/corpus_token_counts.json
4.2 MB  results/rival_topk_EleutherAI__gpt-neo-1.3B.json
2.6 MB  results/rival_topk_state-spaces__mamba-370m-hf.json
2.2 MB  results/rival_topk_state-spaces__mamba-130m-hf.json
1.3 MB  results/rival_topk_RWKV__rwkv-4-430m-pile.json
```

These are the raw per-token measurements — margins, argmaxes, top-8 logits — and they are what makes
the paper's numbers checkable rather than merely stated. They are stored as **scaled integers, not
floats**, specifically so they do not swamp the pool `tests/test_findings_numbers.py` traces the
whole repository against; that decision is recorded in `selfcont_set.py`. If 35.8 MB in git is
unwanted, the alternative is a release asset rather than deletion — but note that deleting them
would leave `results/*.json` source comments in the manuscript pointing at files that do not exist,
which is the defect `test_paper4_numbers.py` fails on.

## Files that must NOT merge — confirmed they cannot

```
git diff --name-only main..paper4 | grep -E 'PLAN|future_work|failure_registry'   →  (empty)
```

- `future_work.md`, `failure_registry.md` — gitignored, and the registry now carries R13, R14 and
  R15, which are the most transferable output of this work and are deliberately local.
- `paper3_arxiv/PLAN.md` — untracked on both branches since `2759ce2`; present on disk, absent from
  the tree.
- Build artefacts (`main.pdf`, `main.aux`, `arxiv-submission.tar.gz`) — gitignored for
  `paper4_arxiv/` as they are for papers 1–3.

## Privacy sweep of the files that WOULD merge

- No absolute paths or usernames in any added file — **after a correction.** This claim was written
  before it was checked and was false: a pre-push sweep found `/Users/<name>/…` in four files, none
  of them printed by our own code. Two logs carried HuggingFace's dataset-cache path, one carried a
  Python `multiprocessing` warning, and `results/prior_art_selfcont_gate.json` recorded local file
  paths in the deep-research gate's `sources` field. The repository's own test passed throughout,
  because it is scoped to paths *we* print. All four are scrubbed to repo-relative form; the sweep
  now returns nothing. The username is public anyway via the repository name and `CITATION.cff`, so
  the exposure was directory layout rather than identity — but `.gitignore` already excludes one
  file specifically for carrying "the author's username and absolute paths", and consistency with
  that policy is the reason this was fixed rather than waved through.
- No API keys: the GROQ key lives in a gitignored `.env` and no run in this branch touches a remote
  API — every measurement is local, CPU, offline (`HF_HUB_OFFLINE=1`).
- `logs/` carries per-cell wall times and oracle checks, no paths.

## Suggested sequence — for the author to run, not for an agent

```bash
git checkout main
git merge --ff-only paper4          # fast-forward; no conflict is possible
python -m pytest -q                 # expect 832 passed, 9 skipped
git push origin main
```

Then cut a release, as `v1.1.0` and `v1.2.0` did, so the archived DOI in paper 4's comments line
resolves to a snapshot containing the results it names. Against `v1.2.0` this branch adds every
`results/selfcont_*`, `escape_*`, `rival_*`, `quant_*`, `floor_survey` and `matched_entropy` file —
which is to say, all of paper 4's evidence.

**Merging does not commit you to submitting.** Paper 4's status is "draft complete and reviewed, not
submitted", and the merge only makes its evidence public, which is the precondition for submitting
rather than a decision to.
