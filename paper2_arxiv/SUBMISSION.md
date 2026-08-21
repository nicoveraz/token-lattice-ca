# arXiv submission metadata — paper 2

Everything below is ready to paste into the arXiv submission form. Numbers in the comments line come
from the verified build (`./make_arxiv_package.sh`), not from an estimate: 10 pages, 4 tables.

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
10 pages, 4 tables. Companion to arXiv:2608.10986. Code, per-run results, and the findings ledger: https://github.com/nicoveraz/token-lattice-ca (archived: https://doi.org/10.5281/zenodo.21880472)
```

## Abstract (plain text, ready to paste)

```
That a prompt's effect on a language model is not a property of the prompt is established: prompts optimised for one model degrade on another, formats have no model-independent valence, and benchmark rankings reorder under semantically neutral reformatting. All of that evidence is about task accuracy, and a task readout cannot say where the interaction lives -- in the machinery of performing tasks, or in the conditional distribution itself. We ask on a readout with no task in it: the fixed-point structure of the short-window argmax map x_{t+1} = argmax_x p(x | x_{t-1}, x_t), censused from 96 starts. It is deterministic, so nothing can be helped or hurt; and it exists only at short windows -- four of six models lose it entirely by window 16 -- so everything here is a statement about how a model reads a fragment. Two results. First, the interaction reaches this readout at full magnitude: nine tokens of conditioning move the fixed-point fraction across most of its range, change a four-way structural class, and reorder models, while an instruction-tuning intervention that moves IFEval by 60.5 points moves the class by zero. Second, nothing we proposed carries the effect. Prefix length fails: the effect is not monotone. Four phenomenological factors -- prose-versus-markup, a universal direction, bidirectionality as a model property, instruct-resistance -- were each withdrawn within one run of being proposed, dissolved by widening the sample. And the nearest mechanistic account, attention-sink dominance of early tokens, predicts the sign of the shift on 2 of 5 models -- chance -- with a length-by-content cross showing why: the account holds on real text and fails on the uniformly random input our probe feeds it, so we operate outside its regime rather than against it. One fixed nine-token prefix drives four models toward fixed-point fraction 0 and two toward 1; a single beginning-of-sequence token takes one model from 0.21 to 0.91 while collapsing others; the bidirectionality survives in-distribution starts. On this readout the unit of explanation is the prompt-model pair. We close with the discipline that caught our own four factors, whose recurring error has a name: a criterion with a shape applied to a quantity with no room to vary.```

The abstract is extracted mechanically from `\begin{abstract}` in `main.tex` and de-TeXed: `\Nstarts`
resolved to 96, `\fpf` to "phi", the display of the map written inline as
`x_{t+1} = argmax_x p(x | x_{t-1}, x_t)`, em-dashes as ` -- `, and compound en-dashes as single
hyphens. No wording was changed. If the abstract in `main.tex` changes, regenerate rather than edit
this file by hand.

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

## One thing that is NOT ready, and is the author's decision

**The repository link in the comments line points at `main`, and paper 2's results files exist only
on the local `paper2` branch.** Until `paper2` is merged, a reader following that URL will not find
`results/text_interaction_fill.json`, `results/token_partition_rank.json`, or the F162–F171 entries in
`findings.md` — the evidence for the Note added, among others. `paper2_arxiv/MERGE_MANIFEST.md`
prepares that merge and does not perform it.

Submitting before the merge would publish a link that does not yet resolve to the cited evidence.
