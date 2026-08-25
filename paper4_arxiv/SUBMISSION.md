# arXiv submission metadata — paper 4

Everything the arXiv form asks for, in the order it asks. Adapted from `paper3_arxiv/SUBMISSION.md`.

**Nothing here is uploaded automatically.** The tarball is built and verified by
`./make_arxiv_package.sh`; the fields below are pasted by hand.

## Status

**PARKED 25 Aug 2026, by author decision, pending paper 3's arXiv identifier.** The draft is complete,
reviewed and green; the release is prepared but not cut. The prior-art gate that blocked write-up
(F186) has run, the delta paragraphs it mandated are §3, and three arms run *after* the draft (F189,
F190) have been folded in.

**What the wait is actually for, and it is short.** Nothing in paper 4's *argument* depends on paper
3 — the manuscript cites paper 1 and no other companion, because paper 1 is the only one its delta
runs against. What the identifier buys is coherence in the metadata: paper 4's comments line names
its companions, and naming two of three while the third sits announced-but-unnamed is the kind of
small wrongness that outlives the reason for it. Cutting one release that carries both papers'
identifiers is cheaper than cutting two.

**Resume list, when the identifier exists:**
1. Add paper 3 to the **Comments** line below, making it three companions.
2. Add paper 3 to `CITATION.cff` under `identifiers` in **both** forms, then
   `python -m pytest tests/test_citation_cff.py -q`.
3. Update the README — badge row, papers section, and the Citation section, replacing the
   placeholder written at `7e102bb`.
4. Merge `paper4` into `main` per `MERGE_MANIFEST.md`, push, and cut the release from
   `paper4_arxiv/RELEASE_NOTES.md`.

Steps 1–3 are metadata. Step 4 is the release, and its notes are already written.

## Title

```
Where a Model Sends Its Own Repeated Token
```

The LaTeX title carries a `{\large ...}` subtitle — *A vocabulary-wide destination map, its measured
robustness envelope, and the estimand it replaced* — on a second line. arXiv's title field is plain
text. Paper 2 learned that a dash in that field renders as two literal hyphens, so the subtitle is
dropped here rather than joined with punctuation; the paper is referred to by the first clause
throughout.

## Authors

```
Nicolás Vera Zúñiga
```

Independent Researcher, Chile. `nicovera@quetru.cl`.

## Primary category

```
cs.CL
```

Same as papers 1–3. `cs.LG` is defensible as a cross-list and is the author's call — this paper has a
stronger claim to it than its companions, since its object is model identification rather than a
language-model readout, but consistency across a four-paper series is worth more than one paper's
reach.

## Comments

```
8 pages, 3 tables. Companion to arXiv:2608.10986 and arXiv:2608.21315. Code, per-run results, pre-registrations and the findings ledger: https://github.com/nicoveraz/token-lattice-ca (archived: https://doi.org/10.5281/zenodo.21880472)
```

Paper 3 is not named because it has no identifier yet. Add it when it announces.

## Abstract (plain text, ready to paste)

**arXiv caps this field at 1,920 characters.** The manuscript's abstract is 2,360, so what follows is
a trimmed version at **1,888 characters** — 32 to spare, for the reason paper 2 recorded: a field
that fits by one character fails the moment the form counts a trailing newline differently.

```
Black-box model identification works by scoring a model's response to natural-language prompts. One line of work feeds models a degenerate input -- their own token, repeated -- to find a failure mode rather than an identity. We take that input and ask where the model goes when it does not. For each token t, read argmax p(. | t, t) in one forward pass; the result is a map on the whole vocabulary, with two halves. The first -- which tokens are fixed points -- is partially anticipated, and we report it as a failed estimand: the natural distance on it is 83% cardinality, separates a corpus manipulation by two bits in 3471 against a precision floor of zero, and attributes families at 0.5833. The second half, where the map sends tokens that are not fixed points, is unrecorded; the one paper holding those tokens logged them as a zero. Pairing on the source token removes the cardinality confound by construction (r from 0.9128 to -0.0932) and attributes families at 0.8333 -- twelve models scored against a pool of nineteen -- with chance 0.1389, across seven tokenizer groups and several corpora. Two nulls clear it: frequency-matched destinations agree at 0.1429, independent marginals at 0.0798. Family predicts agreement better than tokenizer (0.2031 against 0.1205), and recurrent architectures cluster at balanced accuracy 1.0 against a 0.7895 majority rate, or 0.90 once each model's dominant destination is excluded -- the figure we stand behind. We measure the robustness envelope: 8-bit weight rounding moves the map less than deduplicating the training corpus does (0.9018 against 0.6353), 4-bit destroys it (0.0098; 0.1812 at deployment granularity, so not a coarseness artefact), and the precision floor varies by model from 0.201 to 0.9778. All estimands and kill conditions were registered before the data, and the failed one is reported as fully as the surviving one.
```

**Nothing was dropped that the paper claims.** Every figure survives — 83% cardinality, two bits in
3471, 0.5833, r 0.9128 to −0.0932, 0.8333 over twelve scored against nineteen, chance 0.1389, the
two nulls, 0.2031 against 0.1205, 1.0 and 0.90, 0.9018 against 0.6353, 0.0098 and 0.1812, and the
0.201–0.9778 floor spread. The characters came out of phrasing.

## Before uploading — the checks that are already green

`./make_arxiv_package.sh` builds the tarball, unpacks it into a clean directory, builds **from the
tarball's own contents**, and inspects the result. It exits non-zero and prints `FAIL -- do not
upload` if any check trips.

| check | state |
|---|---|
| undefined citations/references | 0 |
| literal `[?]` markers in the PDF | 0 |
| `DRAFTING NOTES` in the shipped `.tex` | 0 |
| `\citepend` **uses** | 0 |
| shipped `.tex` starts with `\documentclass` | pass |
| self-citation lines survive the strip | 2 |
| LaTeX warnings | 0 |

Repository-side, three test files gate this manuscript specifically.
`tests/test_paper4_numbers.py` requires every decimal literal to trace to a results file and **fails
if the two prohibited pitches appear** — F186 barred "degenerate probes" and "fixed points of greedy
decoding", and those are prohibitions rather than preferences.
`tests/test_paper4_boundary.py` requires the boundary claims to survive into the text, forbids the
attribution figure being quoted without a caveat within 700 characters, and requires both mandated
delta citations. `tests/test_abstract_matches_body.py` enforces R15: no figure may appear in the
abstract and nowhere in the body, and any figure the body calls authoritative must reach the
abstract.

## Citation health

**5 works cited, 5 verified, 0 dangling, 0 unledgered, 0 orphans.** Every entry is `LOCAL FULL TEXT`
or `SELF` — no entry rests on the F186 gate's summary, and fetching the three that did corrected the
record twice. `veraz2026domain` is deliberately absent from `refs.bib` until a sentence cites it.

## The one field this document cannot fill

**The arXiv licence selector.** It is irrevocable once submitted and is not recorded for papers 1–3,
so it cannot be matched from this repository. The repo licenses its prose and research record as
**CC BY 4.0** (`LICENSE-docs`), which makes CC BY 4.0 the consistent choice — but confirm it against
a companion's abstract page before selecting, and then record it here so paper 5 does not rediscover
the question. This is the second paper to reach submission with this gap.

## After upload

1. Do **not** put `submit/NNNNNNN` anywhere. Wait for the announced ID.
2. Add paper 4 to `CITATION.cff` under `identifiers` in **both** forms, then
   `python -m pytest tests/test_citation_cff.py -q`.
3. `preferred-citation` stays pointed at paper 1 — it names the citation for the *software*.
4. Update the README: badge row, the papers section, and the Citation section.
5. Cut a release, as `v1.1.0` and `v1.2.0` did, so the archived DOI in the comments line resolves to
   a snapshot containing the results this paper names.
