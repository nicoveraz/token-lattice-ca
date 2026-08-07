# arXiv paper — drafting state

Target: **arXiv only**. Portable preamble (`article` + standard packages), no venue style file, no
second variant maintained. A journal route stays open later if wanted, since TMLR's reuse exemption
names preprint servers explicitly — but nothing here is shaped around that.

## Status

| Section | State |
|---|---|
| Abstract, Introduction | **drafted** |
| §2 The construction | **drafted + figure** — `instrument.png` (existing) |
| §3 Validation by reproduction | **drafted + figure** — `validation_ladder.png` (existing) |
| §4 A phase transition that belongs to the probe | **drafted + figure** — `fig/manufactured.png`, new |
| §5 The discriminator | **drafted + figure** — Table 1, `developmental.png` (existing) |
| §5.3 Ablation response | **drafted** — partial regression, slope +0.568 / +0.724 (F107) |
| §6 Gating estimators | **drafted** — five retractions, one defect class, the SE case study |
| §7 Limits | **drafted** |
| §8 Related work | **drafted** — taken vs open, per F95 |
| §9 Conclusion | **drafted** |

## No blockers left

`§5.3` is written. `ignition_level.py` returned neither registered hypothesis: the compound arm
tracks its reference at slope +0.568 (L8) and +0.724 (L22), both intervals excluding 0 and 1 —
**partial regression**, not revival and not a common level. Row 4 is unaffected; F104's
anti-monotone headline is amended by F107.

**Nothing is outstanding.** All nine citations resolve and every one is arXiv-verified through
`experiments/audit_refs.py` (0 mismatched of 29). The paper compiles to 12 pages.

## Figures and tables

| | | Source |
|---|---|---|
| Fig. 1 | The construction | `fig/instrument.png` (existing) |
| Fig. 2 | Validation ladder | `fig/validation_ladder.png` (existing) |
| Fig. 3 | The manufactured transition | `fig/manufactured.png` — **new**, `experiments/fig_manufactured.py` |
| Fig. 4 | The developmental transition | `fig/developmental.png` (existing) |
| Table 1 | The discriminator | four manipulations; row 4 pending |
| Table 2 | The retractions | four retractions + one same-class defect + one post-guard |

Table 2 is the methods contribution in scannable form: five paragraphs of prose is its least usable
shape. The final row matters most — it is the one that arrived *after* every guard existed, and it
survived a pre-registration, a power calculation and a fixed stopping rule.

## Bibliography

`refs.bib` is copied from `paper/refs.bib` (35 entries), which carries an audit trail:
`paper/refs_verified.json` records 24 entries verified against sources and 11 explicitly marked
unverifiable. **Do not add entries by hand without verifying them** — `experiments/audit_manual.py`
and `tests/test_refs_manual.py` exist because a citation once sat behind an `[unverified]` marker
with nothing recording whether the work had ever been read.

Every key currently cited resolves. Three citations the paper still *wants* and does not have, all
of which must be verified rather than invented:

All four have since been added and verified against arXiv by `audit_refs.py`: `selfconsistency`
(2203.11171), `cot` (2201.11903), `selfrefine` (2303.17651), `modelequality` (2410.20247) and
`iris` (2607.20860). Nothing is cited that has not been fetched and checked.

### A correction the bibliography forced

§8 originally claimed *"no prior method feeds a model its own output back in."* That is false, and
the counterexamples were already in this bibliography and cited by paper 1: `telephone` (multi-turn
cultural attractors) and `paraphrase2cycle` (attractor cycles under successive paraphrasing). F95's
claim was scoped to *model-identification feature sets*, and the draft over-generalised it. §8 now
cites both, concedes the observation that iteration has attractors, and locates the contribution in
the discriminator instead — which is a narrower and defensible claim.

## Conventions

- **Every number carries a source comment** naming the finding and the results file it came from.
  Do not edit a number without editing its comment.
- Comments marked `VERIFY ... before submit` are numbers taken from `plan_paper2.md` (the
  author's own claim set) that have not been re-read from the results files in this draft.
  **§4's headline numbers are now verified**: building `fig/manufactured.png` printed them straight
  from `results/attractor_construction.json` and they match the plan exactly (0.7435 / 0.2409 /
  0.1471 / 0.0990 / 0.1133). The DK cell counts and the ECA effect size are still unchecked.
- `TODO(author)` marks decisions that are not mine: affiliation line, repository URL, Zenodo DOI.

## Building

**It compiles.** `tectonic -X compile main.tex` — 12 pages, no undefined references or citations,
bibliography resolved through natbib/plainnat. Three underfull-hbox warnings at Table 2's narrow
columns; cosmetic.

Tectonic is self-contained and fetches what it needs, so no TeX distribution is required. Build
artifacts (`*.aux`, `*.bbl`, `*.xdv`, `*.log`, `*.pdf`) are gitignored.

## Not yet decided

Whether §6 includes the standard-error case study: one line — the floor for a difference of four
measured centres — written three ways, each changing the verdict, with the middle version producing
a positive that was recorded as F103 and withdrawn the same day. It is the strongest evidence for
Claim B and the most embarrassing. It survived a pre-registration, a power calculation and a fixed
stopping rule, which none of the four original retractions did.
