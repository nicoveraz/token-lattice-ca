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

Remaining before submission: two citations (IRIS, and self-consistency/CoT/self-refine for §1),
and a first compile.

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

| Needed for | Work | Status |
|---|---|---|
| §1 framing | self-consistency, chain-of-thought, self-refine | **absent** — §1 describes these without citing them |
| §8 taken | Model Equality Testing (ICLR 2025) | **absent** — named in F95's prior-art gate, not in refs.bib |
| §8 taken | IRIS (black-box model ID, ~0.99 AUROC) | **absent** — same |

The §8 entries matter more than the §1 ones: the section concedes those results are taken, and
conceding to an uncited work reads as vagueness rather than candour.

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

No LaTeX toolchain on the machine this was drafted on, so **the draft has never been compiled**.
Before trusting it: install a TeX distribution or let arXiv's build check it (arXiv compiles the
source and shows errors before anything is announced, so submitting unverified source is recoverable
rather than fatal). `refs.bib` does not exist yet — either create it or drop `\bibliography`.

## Not yet decided

Whether §6 includes the standard-error case study: one line — the floor for a difference of four
measured centres — written three ways, each changing the verdict, with the middle version producing
a positive that was recorded as F103 and withdrawn the same day. It is the strongest evidence for
Claim B and the most embarrassing. It survived a pre-registration, a power calculation and a fixed
stopping rule, which none of the four original retractions did.
