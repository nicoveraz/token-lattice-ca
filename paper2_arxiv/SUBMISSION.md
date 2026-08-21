# arXiv submission metadata — paper 2

> **SUBMITTED 21 Aug 2026** — arXiv `submit/7978448`, status *processing*. That is a temporary
> submission handle, **not** the paper's identifier: the real `arXiv:NNNN.NNNNN` appears only on
> announcement. Do not put `submit/7978448` in `CITATION.cff`, the README, or anywhere else — the
> block below waits for the announced ID.
>
> **Known cosmetic issue, fixable while the status is still *processing*.** The title was pasted with
> `--` where the LaTeX source has an em dash (`---`). arXiv's title field is plain text, so it renders
> as two literal hyphens. Use **Update** to replace it with `—` or a single `-`. The abstract's `--`
> are correct and conventional and should be left alone.

Everything below is ready to paste into the arXiv submission form. Numbers in the comments line come
from the verified build (`./make_arxiv_package.sh`), not from an estimate: 11 pages, 4 tables.

**Source package:** `paper2_arxiv/arxiv-submission.tar.gz` — rebuild with
`./paper2_arxiv/make_arxiv_package.sh`, which verifies the tarball by unpacking it into a clean
directory, building from its own contents, and checking the resulting PDF. The tarball is gitignored
as a build artefact. It ships `main.tex`, `main.bbl` and `refs.bib`; the drafting-notes header is
stripped from the shipped `.tex` and the strip is verified by grep, while the repo copy keeps it.

---

## Title

```
Prompt-Model Interaction Reaches the Fixed Points: A deterministic, task-free structural readout -- and the factorizations of it that failed
```

The LaTeX title is split across two lines with `\\` and a `{\large ...}` subtitle. arXiv's title
field is plain text, so the two halves are joined with a colon above. If the form is fussy about
length, the subtitle after the colon can be dropped without loss — the paper is referred to by the
first clause throughout.

## Authors

```
Nicolás Vera Zúñiga
```

Affiliation as it appears in the paper: Independent Researcher, Chile. Contact: nicovera@quetru.cl

## Primary category

```
cs.CL
```

No cross-list is proposed here. `cs.LG` would be defensible, and is the author's call rather than a
mechanical one.

## Comments

```
11 pages, 4 tables. Companion to arXiv:2608.10986. Code, per-run results, and the findings ledger: https://github.com/nicoveraz/token-lattice-ca (archived: https://doi.org/10.5281/zenodo.21880472)
```

## Abstract (plain text, ready to paste)

**arXiv caps this field at 1,920 characters.** The manuscript's abstract is 2,269, so what follows is
a trimmed version at **1,885 characters** — 35 to spare, deliberately, because a field that fits by
one character fails the moment the form counts a trailing newline differently.

```
That a prompt's effect is not a property of the prompt is established: prompts optimised for one model degrade on another, and rankings reorder under neutral reformatting. That evidence is about task accuracy, which cannot say whether the interaction is a fact about task machinery or about the conditional distribution itself. We ask on a readout with no task in it: the fixed-point structure of the short-window argmax map x_{t+1} = argmax_x p(x | x_{t-1}, x_t), censused from 96 starts. It is deterministic, so nothing can be helped or hurt, and it exists only at short windows -- four of six models lose it entirely by window 16 -- so everything here concerns how a model reads a fragment. Two results. First, the interaction reaches this readout at full magnitude: nine tokens of conditioning move the fixed-point fraction across most of its range, change a four-way structural class, and reorder models, while instruction tuning worth 60.5 IFEval points moves the class by zero. Second, nothing we proposed carries it. Prefix length fails: the effect is not monotone. Four phenomenological factors -- prose-versus-markup, a universal direction, bidirectionality, instruct-resistance -- were each withdrawn within one run of being proposed, dissolved by widening the sample. And the nearest mechanistic account, attention-sink dominance of early tokens, predicts the sign of the shift on 2 of 5 models -- chance -- while a length-by-content cross shows it holds on real text and fails on our probe's uniformly random input, so we are outside its regime, not against it. One fixed nine-token prefix drives four models toward 0 and two toward 1; the bidirectionality survives in-distribution starts. On this readout the unit of explanation is the prompt-model pair. The recurring error it caught in us has a name: a criterion with a shape applied to a quantity with no room to vary.
```

**Nothing was dropped that the paper claims.** Every result, both headline findings and every number
that survived — 96 starts, window 16, 60.5 IFEval points, 2 of 5 models, four models to 0 and two to
1 — is still here. The 384 characters came out of phrasing, plus one illustration: the manuscript
also cites a single BOS token taking one model from 0.21 to 0.91, and that example is the only
content the arXiv field lacks. It is an *additional* instance of bidirectionality, which the retained
"four models toward 0 and two toward 1" already states.

The manuscript's own abstract is unchanged. A shorter arXiv field is routine and expected at this
limit; do not edit `main.tex` to match, or the paper and the record diverge for no gain.

If the abstract in `main.tex` ever changes, regenerate both from it rather than editing this file by
hand.

---

## Before uploading — the checks that are already green

| check | state |
|---|---|
| `\citepend` uses in the shipped source | 0 (the macro definition stays as a tripwire) |
| undefined citations / references | 0 |
| literal `[?]` markers in the PDF | 0 |
| DRAFTING NOTES block in the shipped `.tex` | 0 (19 inline `% F1xx` source comments preserved) |
| bibliography | all keys resolve; BibTeX log clean |
| every decimal literal traces to `results/` | enforced by `tests/test_paper2_numbers.py` |
| every cited key is ledgered with a quote | enforced by `tests/test_paper2_citations.py` |

## The one blocker — CLEARED 21 Aug 2026

The comments line points readers at the repository and the archived DOI, and for most of this
paper's life both resolved to a snapshot without its evidence. That is now fixed, in three steps,
each verified:

1. **`paper2` merged into `main`** (fast-forward, `a7e3c8a`) and pushed. `results/text_interaction.json`,
   `results/text_interaction_fill.json`, `results/token_partition_rank.json` and the F162–F171 entries
   in `findings.md` are public.
2. **Release [`v1.1.0`](https://github.com/nicoveraz/token-lattice-ca/releases/tag/v1.1.0)** cut at
   `5d2ae1c`, the exact commit CI passed on, with the verified `arxiv-submission.tar.gz` and
   `main.pdf` attached.
3. **Zenodo archived it**, confirmed by the author. The concept DOI
   [10.5281/zenodo.21880472](https://doi.org/10.5281/zenodo.21880472) resolves to *latest*, so it now
   reaches a snapshot containing everything the paper cites — including
   `results/text_interaction.json`, which the converse-exhibit sentence names directly. No link in
   the manuscript changed, because the concept DOI is stable across versions.

**Nothing now blocks submission.** What remains is the submission itself, and then the
`CITATION.cff` step below once an ID exists.

---

## After the ID exists — add paper 2 to `CITATION.cff`

Do this once arXiv announces and the identifier is final, **not** at submission time: an ID that
does not resolve yet is worse in a citation record than an absent one.

Paper 2 goes in `identifiers`, alongside paper 1. It does **not** replace `preferred-citation`,
which stays pointed at paper 1 — that field names the citation for *the software*, and paper 1 is
the instrument paper.

Both lines are required. An arXiv paper carries a bare ID and a resolvable DOI, and listing one
without the other leaves the record with a paper nobody can follow:

```yaml
  - type: other
    value: "arXiv:NNNN.NNNNN"
    description: "Paper 2 preprint, cs.CL, DD Mon 2026"
  - type: doi
    value: 10.48550/arXiv.NNNN.NNNNN
    description: "Paper 2 arXiv DOI"
```

Substitute `NNNN.NNNNN` in **both** lines and set the real announcement date. Then:

```bash
python -m pytest tests/test_citation_cff.py -q
```

`tests/test_citation_cff.py` guards exactly the two ways this goes wrong. It fails if a template or
placeholder ID reaches the file — that check runs without PyYAML, so it cannot be skipped — and it
fails if an arXiv ID appears in one form but not the other. Both were verified to fire on the
mistake and pass on the correct entry.

While you are in the file, the `abstract` describes the repository rather than either paper and
needs no change; if paper 2's subject belongs in `keywords`, `prompt sensitivity` and
`fixed points` are the two that are missing.
