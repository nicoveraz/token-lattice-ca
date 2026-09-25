# arXiv submission metadata — paper 4

Everything the arXiv form asks for, in the order it asks. Adapted from `paper3_arxiv/SUBMISSION.md`.

**Nothing here is uploaded automatically.** The tarball is built and verified by
`./make_arxiv_package.sh`; the fields below are pasted by hand.

## Status

**UNPARKED 25 Sep 2026: paper 3 announced as arXiv:2609.29507.** Resume steps 1–3 below are done;
step 4's merge was already executed on 25 Aug (`MERGE_MANIFEST.md`), so what is left is to push
these commits and cut `v1.3.0`. What follows records the park as it stood.

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

**The cost of not waiting was checked, 10 Sep 2026, and it is larger than the paragraph above
assumed.** arXiv will not edit the Comments field of an announced paper as metadata: *"To allow
authors to make changes to the comments and other metadata fields of publicly announced articles
without generating a new version … is open to abuse and thus not allowed"*
([replace](https://info.arxiv.org/help/replace.html)). The documented exceptions are journal
reference, DOI and report number only — *"no new article version will be generated when journal
reference, DOI or report number information is added"*
([jref](https://info.arxiv.org/help/jref.html)). So uploading before paper 3 announces does not
defer naming it; it converts naming it into **paper 4 v2, whose only change is one line of
comments**, permanently in the version history. That is the argument for waiting, and it is stronger
than metadata tidiness.

**A second reason to wait, weaker and separate.** `MERGE_MANIFEST.md` states that submitting a
fourth paper while the third is under moderation review "is the action most likely to lengthen that
review". That is the author's judgement, not arXiv policy — the moderation page says nothing about
holds or concurrent submissions, and its only rate guidance is *"no more than three papers per
day"*. What it does say is adjacent and does apply: *"Submitters who have had works previously
delayed or declined by arXiv should anticipate closer scrutiny on future submissions"*
([moderation](https://info.arxiv.org/help/moderation/index.html)). Paper 3 is currently delayed, so
paper 4 submitted now should expect a hold of its own — which is a cost, but not evidence that it
lengthens paper 3's.

**Resume list, when the identifier exists:**
1. Add paper 3 to the **Comments** line below, making it three companions.
2. Add paper 3 to `CITATION.cff` under `identifiers` in **both** forms, then
   `python -m pytest tests/test_citation_cff.py -q`.
3. Update the README — badge row, papers section, and the Citation section, replacing the
   placeholder written at `7e102bb`.
4. Merge `paper4` into `main` per `MERGE_MANIFEST.md`, push, and cut the release from
   `paper4_arxiv/RELEASE_NOTES.md`.

Steps 1–3 are metadata. Step 4 is the release, and its notes are already written. **Steps 1–3 were
done 25 Sep 2026.**

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
8 pages, 3 tables. Companion to arXiv:2608.10986, arXiv:2608.21315 and arXiv:2609.29507. Code, per-run results, pre-registrations and the findings ledger: https://github.com/nicoveraz/token-lattice-ca (archived: https://doi.org/10.5281/zenodo.21880472)
```

All three companions are named, in announcement order. Paper 3 was added on 25 Sep 2026 when it
announced; this is the one-line change the park was waiting for.

## Abstract (plain text, ready to paste)

**arXiv caps this field at 1,920 characters.** The manuscript's abstract is 2,360, so what follows is
a trimmed version at **1,904 characters** — 16 to spare, for the reason paper 2 recorded: a field
that fits by one character fails the moment the form counts a trailing newline differently.

```
Black-box model identification works by scoring a model's response to natural-language prompts. One line of work feeds models a degenerate input -- their own token, repeated -- to find a failure mode rather than an identity. We take that input and ask where the model goes when it does not. For each token t, read argmax p(. | t, t) in one forward pass; the result is a map on the whole vocabulary, with two halves. The first -- which tokens are fixed points -- is partially anticipated, and we report it as a failed estimand: the natural distance on it is 83% cardinality, separates a corpus manipulation by two bits in 3471 against a precision floor of zero, and attributes families at 0.5833. The second half, where the map sends tokens that are not fixed points, is unrecorded; the one paper holding those tokens logged them as a zero. Pairing on the source token removes the cardinality confound by construction (r from 0.9128 to -0.0932) and attributes families at 0.8333 -- twelve models scored against a pool of nineteen -- with chance 0.1389, across seven tokenizer groups and several corpora. Two nulls clear it: frequency-matched destinations agree at 0.1429, independent marginals at 0.0798. Family predicts agreement better than tokenizer (0.2031 against 0.1205), and recurrent architectures cluster at balanced accuracy 1.0 against a 0.7895 majority rate, or 0.90 once each model's dominant destination is excluded -- the figure we stand behind. We measure the robustness envelope: 8-bit weight rounding moves the map less than deduplicating the training corpus does (0.9004 against 0.6353, on one support), 4-bit destroys it (0.0098; 0.1812 at deployment granularity, so not a coarseness artefact), and the precision floor varies by model from 0.201 to 0.9778. All estimands and kill conditions were registered before the data, and the failed one is reported as fully as the surviving one.
```

**Nothing was dropped that the paper claims.** Every figure survives — 83% cardinality, two bits in
3471, 0.5833, r 0.9128 to −0.0932, 0.8333 over twelve scored against nineteen, chance 0.1389, the
two nulls, 0.2031 against 0.1205, 1.0 and 0.90, 0.9004 against 0.6353, 0.0098 and 0.1812, and the
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

## Licence — resolved, no longer an open field

```
Creative Commons Attribution 4.0 (CC BY 4.0)
```

**Checked against the companions' own abstract pages, 10 Sep 2026, rather than inferred.** Both
announced papers carry CC BY 4.0: [arXiv:2608.10986](https://arxiv.org/abs/2608.10986) (v1, 11 Aug
2026) and [arXiv:2608.21315](https://arxiv.org/abs/2608.21315) (v1, 21 Aug 2026). That matches the
repository's own `LICENSE-docs`, so consistency and the repo's stated policy point the same way and
the selector needs no judgement call at the form.

The selection is **irrevocable once submitted**, which is why it is verified here rather than at the
form. It is recorded so paper 5 does not rediscover the question — paper 4 was the second paper to
reach submission with this gap, and `paper3_arxiv/SUBMISSION.md` has no licence section at all.

**Paper 3's selection is CC BY 4.0**, read off [its abstract page](https://arxiv.org/abs/2609.29507)
on 25 Sep 2026 and now recorded in `paper3_arxiv/SUBMISSION.md`. All four papers are consistent.

## After upload

1. Do **not** put `submit/NNNNNNN` anywhere. Wait for the announced ID.
2. Add paper 4 to `CITATION.cff` under `identifiers` in **both** forms, then
   `python -m pytest tests/test_citation_cff.py -q`.
3. `preferred-citation` stays pointed at paper 1 — it names the citation for the *software*.
4. Update the README: badge row, the papers section, and the Citation section.
5. Cut a release, as `v1.1.0` and `v1.2.0` did, so the archived DOI in the comments line resolves to
   a snapshot containing the results this paper names.
