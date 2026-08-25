# arXiv submission metadata — paper 3

Everything the arXiv form asks for, in the order it asks. Adapted from `paper2_arxiv/SUBMISSION.md`,
which was written for the same form and whose lessons are carried rather than rediscovered.

**Nothing here is uploaded automatically.** The tarball is built and verified by
`./make_arxiv_package.sh`; the fields below are pasted by hand.

## Status

**SUBMITTED 25 Aug 2026, and ON HOLD.** arXiv has the submission and its moderators are reviewing
it; it has no identifier and may not be cited, linked or recorded anywhere until it announces. The
submission number is deliberately **not written down here** — it does not resolve, and an identifier
that does not resolve is worse in a citation record than an absent one.

**On hold is a normal state, not a rejection.** arXiv holds a fraction of submissions for moderator
review and contacts the author only if something is needed; most clear within days. Nothing is
owed from this end unless they write. If they do ask about placement, the relevant facts are that
this paper was submitted to `cs.CL`, matching papers 1 and 2, while the account's default category
is `cs.LG` — and that consistency across the series is the reason, recorded below under Primary
category before the submission was made.

**Downstream:** paper 4 is parked on this identifier (`paper4_arxiv/SUBMISSION.md`), so a long hold
delays that release rather than anything about paper 4 itself.

The one thing paper 3 had been waiting for — paper 2's arXiv identifier — exists:
**arXiv:2608.21315**, announced 21 Aug 2026. `refs.bib` now carries `veraz2026domain`,
`CITATIONS.md` entry 11 records it, and Setup and §E3 cite it.

## Title

```
What a Cross-Model Fixed-Point Census Can and Cannot Arbitrate About Repetition
```

One line, no subtitle, no dash — so the plain-text rendering that made paper 2's title carry two
literal hyphens cannot happen here.

## Authors

```
Nicolás Vera Zúñiga
```

Independent Researcher, Chile. `nicovera@quetru.cl`.

## Primary category

```
cs.CL
```

Same as papers 1 and 2. No cross-list is proposed; `cs.LG` would be defensible and is the author's
call rather than a mechanical one.

## Comments

```
9 pages, 2 tables. Companion to arXiv:2608.21315 and arXiv:2608.10986. Code, per-run results, and the findings ledger: https://github.com/nicoveraz/token-lattice-ca (archived: https://doi.org/10.5281/zenodo.21880472)
```

Two numbered tables; the class rule in Setup renders as a third but is unnumbered and is not counted. Both companions are named. Paper 2 measures the *same readout* and bounds this paper's scope in two
directions — the window and the domain — so a reader who follows only one link should get the one
that carries the caveats.

## Abstract (plain text, ready to paste)

**arXiv caps this field at 1,920 characters.** The manuscript's abstract is 2,184, so what follows is
a trimmed version at **1,880 characters** — 40 to spare, deliberately, for the reason paper 2
recorded: a field that fits by one character fails the moment the form counts a trailing newline
differently.

```
Two accounts of neural text degeneration coexist. One locates the cause in the training data -- repetition in the corpus produces repetition in the output, established by training on repetition-sorted data -- the other in the trained network, in copying circuits and repetition features. Neither has been arbitrated across a broad cohort of pretrained models: the causal work trains its own. We report an observational measurement in a different currency: the fixed-point structure of a model's own short-window argmax map, censused from 96 random two-token starts over 17 off-the-shelf models, always unprompted -- a companion paper shows nine tokens of conditioning move this readout across most of its range. The four-way class is stable across census seeds on 17 of 17. Three exhibits. At fixed corpus (The Pile), fixed scale and that fixed domain, the class is not determined: across two size-matched tiers, pythia is a funnel while RWKV, Mamba and a second transformer family are not, and both hold their class across an order of magnitude of scale. Six of seven models in that ladder reach the same endpoint token, and those concentrating on it most strongly are among those that never stay there -- what varies is not where trajectories go but whether the destination self-continues. The deduplicated Pythia suite does not change the class. And the corpus-side inflow term proposed for this phenomenon does not select our endpoints once frequency is controlled, in English and three other languages. This is observational and cannot refute a training intervention. Funnels are common: eight of seventeen models, seven families, five corpora -- so the limit is not that the phenomenon is one model's peculiarity, but that within the one corpus where training data can be held fixed only one available family funnels; that subset cannot show the split is corpus-independent.
```

**Nothing was dropped that the paper claims.** Every result and every number survives — 96 starts,
17 models, 17 of 17 seed stability, two size-matched tiers, six of seven models on one endpoint
token, an order of magnitude of scale, eight of seventeen funnels across seven families and five
corpora, and the narrow limit stated in full. The characters came out of phrasing: four clauses were
compressed and two framings ("a scope condition, not a detail", "in our cohort") were dropped as
commentary on facts the text already states.

The manuscript's own abstract is unchanged and remains the longer one. A shorter arXiv field is
routine.

## Before uploading — the checks that are already green

Run `./make_arxiv_package.sh`. It builds the tarball, unpacks it into a clean directory, builds
**from the tarball's own contents**, and inspects the result. It is a gate, not a report: it exits
non-zero and prints `FAIL -- do not upload` if any check trips.

| check | why it exists | state |
|---|---|---|
| undefined citations/references | a paper that builds with `[?]` in the text looks fine to the submitter | 0 |
| literal `[?]` markers in the PDF | the .bbl-without-refs.bib trap, observed on paper 1 | 0 |
| `DRAFTING NOTES` in the shipped `.tex` | arXiv distributes source, and the header names kill conditions and a withdrawn framing | 0 |
| `\citepend` **uses** | the red tripwire; the definition is not a use | 0 |
| shipped `.tex` starts with `\documentclass` | the strip is the only transformation, and one that ate too much has no other symptom | pass |
| self-citation lines survive the strip | same reason, from the other side | 4 |

The last two are additions beyond paper 2's script, and both were proved to fire: disabling the
strip makes the run report `DRAFTING NOTES: 1`, the wrong first line, and exit 1.

Repository-side, `tests/test_paper3_numbers.py` enforces what the prose cannot check itself — every
decimal literal traces to a results file, the class-rule table matches `classify()` in the census
code, every `% ... results/*.json` source comment names a file that exists, K10's citation is in
Setup, and the word "architecture" never returns as a causal claim.

## Citation health at the time of writing

**11 works cited, 11 verified, 0 dangling, 0 unledgered, 0 ledger orphans.** `CITATIONS.md` records
the verification basis per entry and distinguishes `LOCAL FULL TEXT` from `GATE FULL TEXT`, because
they are not equivalent bases and a future reader may need to escalate one.

## After upload

1. Do **not** put the submission identifier (`submit/NNNNNNN`) in `CITATION.cff`, the README, or
   anywhere else. Wait for the announced ID, exactly as paper 2 did.
2. Once announced, add paper 3 to `CITATION.cff` under `identifiers`, in **both** forms:

```yaml
  - type: other
    value: "arXiv:NNNN.NNNNN"
    description: "Paper 3 preprint, cs.CL, DD Mon 2026"
  - type: doi
    value: 10.48550/arXiv.NNNN.NNNNN
    description: "Paper 3 arXiv DOI"
```

   Substitute in **both** lines, then run `python -m pytest tests/test_citation_cff.py -q`. That test
   fails if a placeholder reaches the file and if an arXiv ID appears in one form but not the other;
   both directions were verified to fire when paper 2 was added.

3. `preferred-citation` stays pointed at paper 1. It names the citation for the *software*, not for
   the newest result.
4. Update the README: the badge row, the "Two papers" section — which will need to become three —
   and the Citation section. Paper 2's announcement is the worked example; three statements there had
   to be corrected because they were written before it.
