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
| §5.3 Ablation response | **PENDING** — blocked on `experiments/ignition_level.py` |
| §6 Gating estimators | **drafted** — five retractions, one defect class, the SE case study |
| §7 Limits | **drafted** |
| §8 Related work | **drafted** — taken vs open, per F95 |
| §9 Conclusion | **drafted** |

## The one blocker

`§5.3` is row 4 of the discriminator table. `ignition_level.py` decides whether F104 reads as
**revival** (adding an ablation raises ignition) or as **regression to a common level** (adding an
ablation drives ignition toward a characteristic value regardless of where the reference sits).

Row 4 survives either way — both are the instrument responding to a model change with the
construction held fixed — but the prose is completely different and must not be written twice.
See `findings.md` F104, and the withdrawal notice at the top of F103.

## Conventions

- **Every number carries a source comment** naming the finding and the results file it came from.
  Do not edit a number without editing its comment.
- Comments marked `VERIFY ... before submit` are numbers taken from `paper/plan_paper2.md` (the
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
