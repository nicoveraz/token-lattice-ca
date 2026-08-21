# Merge manifest — `paper2` → `main`

> **EXECUTED 21 Aug 2026, on the author's instruction.** `paper2` fast-forwarded into `main` at
> `a7e3c8a`, the suite ran green on `main` (741 passed, 5 skipped), and `main` was pushed. Two things
> this document did not anticipate and which are recorded in the commit history rather than rewritten
> here: `main` was locked by a **prunable worktree** at `/private/tmp/tlca-main` whose directory had
> outlived its `.git` link, and a `paper3` branch existed carrying a next-paper plan, so the merge was
> `paper2` only and one stranded commit was cherry-picked across. The plan below is kept as written,
> as the record of what was decided before it was done.

**Prepared, not executed** *(as written; see the banner above)*. The merge is the author's act.
Nothing in this file has been run against `main`, and no branch was pushed. This exists so the
decision can be made from facts rather than from a diff scrolled past at midnight.

**Why it is needed at all.** Paper 2's results files exist only on this branch. The submission
comments line points readers at `https://github.com/nicoveraz/token-lattice-ca`, and until `paper2`
lands on `main` that link does not resolve to the evidence the paper cites — `results/text_interaction_fill.json`,
`results/token_partition_rank.json`, and the F162–F171 entries in `findings.md` among them.
Submitting first publishes a citation to something not yet public.

---

## Headline

**81 commits, 160 files: 140 added, 20 modified, 0 deleted. `main` has no commits `paper2` lacks, so
this is a fast-forward — no merge conflicts are possible.**

```
git rev-list --count main..paper2   →  81
git rev-list --count paper2..main   →   0
```

## What changes on `main`, grouped

### Added (140)

| group | count | what it is |
|---|---|---|
| `experiments/` | 44 | probe scripts and their frozen pre-registration JSONs |
| `results/` | 42 | per-run outputs, each carrying `_analysis_provenance` |
| `logs/` | 33 | run logs (see the privacy note below) |
| `gatecheck/` | 8 | the balance-gate package added after F163 |
| `tests/` | 7 | new guards, including the paper-2 number and citation tests |
| `paper2_arxiv/` | 6 | `main.tex`, `refs.bib`, `CITATIONS.md`, `SUBMISSION.md`, this manifest, packaging script |

### Modified (20) — the ones a reviewer should actually look at

**Content of the record**
- `findings.md` — **5 975 → 8 207 lines.** The bulk of the diff. Append-mostly: new entries are
  prepended and existing entries gain qualification banners rather than edits. Worth a skim of the
  banners specifically, since those alter what earlier findings claim.

**Repo metadata**
- `README.md` (+3/−1), `CITATION.cff` (+10), `.github/workflows/tests.yml` (+7), `.gitignore` (+32/−1),
  `conftest.py` (+64)

**Existing code touched**
- `experiments/share_invariance.py`, `experiments/topk_ablation.py`
- `gatecheck/README.md`, `gatecheck/src/gatecheck/__init__.py`, `gatecheck/src/gatecheck/leverage.py`

**Existing results re-written** — check these deliberately; a changed results file can invalidate a
number already quoted in paper 1:
- `results/revival_replication.json`, `results/share_invariance.json`, `results/topk_ablation.json`

**Existing tests touched**
- `tests/test_arxiv_paper_numbers.py` — paper 1's number guard. Its comment stripper was fixed to
  stop eating escaped `\%`; two literals (`0.714`, `24.1`) were invisible to it and both trace, so no
  verdict changed.
- `tests/test_findings_numbers.py`, `tests/test_golden.py`, `tests/test_mlm_harness.py`,
  `tests/test_null_all_backends.py`, `tests/test_ranking.py`

---

## Files that must NOT merge — confirmed they cannot

All four are untracked **and absent from both trees**, so a merge cannot carry them. Verified with
`git ls-tree -r <branch> --name-only`, not by looking at `.gitignore`:

| file | tracked? | in `paper2` tree | in `main` tree | on disk |
|---|---|---|---|---|
| `future_work.md` | no | **0** | **0** | yes |
| `failure_registry.md` | no | **0** | **0** | yes |
| `.env` (GROQ key) | no | **0** | **0** | yes |
| `quantum_reading.md` | no | **0** | **0** | yes |

These stay local by standing instruction. Nothing in this manifest changes that, and the merge does
not touch them.

---

## Privacy sweep of the files that WOULD merge

Scanned all 158 files present at sweep time (this manifest and `SUBMISSION.md` were
written afterwards and contain no run data) for credential and personal-data patterns.

| pattern | result |
|---|---|
| Groq / OpenAI-style keys (`gsk_`, `sk-`) | **clean** |
| Hugging Face tokens (`hf_`) | **clean** |
| `api_key = "..."` assignments | **clean** |
| `Bearer` tokens | **clean** |

### Two flags, both examined and neither blocking

**1. Absolute home paths in 11 files — and NOT all of them are logs.** Corrected after a recount:
an earlier pass of this sweep said "11 log files", which was wrong in a way that would have
understated it. The actual set is **8 logs and 3 results files**:

```
logs/compliance_v3.log          results/compliance_v3.json
logs/copy_precision_k256.log    results/midrange_screen.json
logs/instruct_raisable.log      results/prior_art_domain_journal.jsonl
logs/midrange_screen.log
logs/scaffold_effect.log
logs/share_instruct.log
logs/structural_text.log
logs/text_interaction.log
```

(This manifest is a twelfth match, because the paragraph you are reading quotes the path.)

What it discloses is the local account name, which already matches the public GitHub handle and the
git author on every commit. **It is pre-existing policy rather than a new exposure: `main` already
tracks 19 files containing `/Users/` paths — 11 logs, 4 results, and one each in `experiments/`,
`tests/`, `paper_arxiv/` and `build_mirror.py`.** Raised so the choice is made rather than inherited;
scrubbing is a one-line `sed` across `logs/` and `results/` if preferred, though rewriting stored
results to change a path would invalidate their provenance stamps and is not a free edit.

**2. An apparent email address in two results files — FALSE POSITIVE, confirmed by reading it.**
`results/compliance_second_measure.json` and `results/compliance_v2.json` matched an email regex. The
match is `@client.event`, a Python decorator inside **model-generated text** (a discord.py snippet the
model produced), not an address. No personal data. Recorded rather than silently dismissed, because
"I checked and it was fine" is not a check anyone else can audit.

The only real email addresses in the merge set are `nicovera@quetru.cl` in `CITATION.cff` and
`paper2_arxiv/main.tex` — the author's own published contact, intentional.

---

## Suggested sequence — for the author to run, not for an agent

```bash
git checkout main
git merge --ff-only paper2          # fast-forward; refuses if history diverged since this was written
python -m pytest tests/ -q          # expect green before pushing anything
git push origin main
```

`--ff-only` is deliberate: if it refuses, `main` has moved since this manifest was written and the
whole document should be regenerated rather than forced through.

**Do not push before the tests pass on `main` itself.** The suite is green on `paper2`; it has never
been run on the merged result, because the merge has not happened.
