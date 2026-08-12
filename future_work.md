# Future work — gated plans

**Status: private.** This file lives on the `paper2` branch, which is unpublished. Items 2 and 3
telegraph unpublished research directions; they do not go to the public tracker or `main` until
executed, or until priority is protected another way (sha256 of this file committed publicly, or a
closed Zenodo deposit — the prereg self-hash discipline applied to planning). Item 1 is
public-safe at any time: it describes packaging an already-published artifact, and every fact in
it is already in the public README.

**Provenance.** Distilled 12 Aug 2026 from an external LLM roadmap ("deep future development"),
after gating it against the ledger. Three items survived. What was discarded, and why, is recorded
at the end — the roadmap was written from the arXiv paper and was blind to F128–F130, so its
strongest-sounding proposals aimed at objects the ledger had already deflated.

---

## 1. Package the discriminator protocol: loopness as an explicit parameter

**Public-safe. No new runs. Do first.**

`gatecheck` ships the estimator guards, but the discriminator itself — the two-axis test (vary
construction / fix model; vary model / fix construction) — exists only as prose in the paper.
Package it as a protocol module plus a `PROTOCOL.md` template, with the construction parameterised
as an explicit **loopness vector**: radius, temperature schedule, visit scheme (sync / async /
ordered), masking policy, and commitment (in-place vs append-only vs free autoregression).

The gradient version is the point: sweep loopness from the ring toward free AR generation and
record **where each observable's model-attribution dies**. Observables that collapse immediately
are kinematic; observables that survive loosening are model-attributable. Output per observable:
construction-range vs model-range (the F128/F129 statistic) with verdict
`construction-determined | model-determined | NOT_DECIDABLE`. This is the commitment-schedule
axis made operational.

Gates, inherited not optional: anti-vacuity (a pinned observable reads as "construction-invariant"
vacuously — dynamic-range check on both axes before any verdict); seed-stability floor before any
ranking (the 0.030 lesson); family as the independent unit wherever models vary.

Why now: the arXiv Comments field points readers at this repo, and the protocol is the first thing
a reader with their own self-feeding loop will want. v1 needs no new runs — F128/F129/F130 data
exercises every branch. Deliverable: code + doc + one worked example.

## 2. Share-ranking stabilization time across public developmental ladders

**Private until run. After the paper2 share program lands.**

F77 established the developmental crossing on one family. F130 established the share as the
construction-robust cross-model instrument. The open measurement joining them: **how early in
training does the share ranking stabilise?** If the ranking is informative at low compute, the
share becomes a cheap monitoring signal during pre-training — a use nothing else in the
instrument offers.

Public intermediate checkpoints make this runnable without lobbying anyone: Pythia (154
checkpoints, one family done), OLMo-2 (intermediate checkpoints public), LLM360 Amber/Crystal
(~360 checkpoints by design). Two to three families of developmental resolution from public
artifacts. Verify availability at execution time.

Pre-registration, before any run:

- **Prior-art gate first, mandatory** (F90/F91): "early-checkpoint signals predict final-model
  properties" is an active literature. The specific question — a *dynamical* ranking from
  settled-state statistics stabilising early — must be checked against it before a token is spent.
- **Stabilization defined before looking**: earliest checkpoint whose ranking correlates with the
  final ranking above a pre-registered threshold *and remains above it for all later checkpoints*
  — the "remains" clause written down first.
- **Dynamic-range gate on the share across each ladder**: early checkpoints may sit at degenerate
  share; a stabilization time read off a pinned quantity is the defect class again.
  NOT_DECIDABLE branch required.
- **Family as unit**: with 2–3 families this is a resolution study, not a hypothesis test. Say so.

Kill condition: if stabilization coincides with the F77 crossing window in every family, the
result is the crossing restated, not a new monitor. Record and stop.

Cost: inference-only on public checkpoints; the expensive part is downloads and storage.

## 3. Survival table for unanchored self-feeding readings (paper3 candidate)

**Private until run. Strictly after item 1 and after paper2.**

Apply the packaged discriminator to readings the self-feeding literature already quotes, and
publish the survival/failure table. **Scope is the contribution**: only *unanchored dynamical
readings* are in jurisdiction — self-consistency agreement rates and dispersion,
iterated-refinement convergence claims, multi-turn agreement drift, transmission-chain attractor
statistics. Externally anchored gains (task-accuracy improvements) are explicitly OUT: external
anchoring is precisely what the ring removes by design, so the discriminator has nothing to say
about them. Recording this boundary is what keeps the paper from overclaiming.

Pre-registration, per target, before any run:

- **Prior-art check per target** — each literature may already contain its own
  construction-vs-model analysis.
- **Saturation gate on the target observable first**: agreement rates on easy tasks sit near 1.0
  — restriction of range, the paper's own defect class. Audit only where the observable has room
  to vary, or scan task difficulty until it does (scan-then-gate), with the range gate
  pre-registered.
- **Construction axis defined per target**: mapping the loopness vector onto self-consistency /
  refinement / agentic loops is itself research content, not an implementation detail. Each
  mapping written and frozen before its runs.
- Family as unit; seed-stability floor on any reported ranking.

Kill / honesty condition: a target whose observable cannot be given dynamic range at feasible
cost enters the table as NOT_DECIDABLE — published, not dropped. The table's honesty is the
contribution.

Cost: largest of the three — open models plus sampling budgets. This was the external roadmap's
step 2; it is deliberately third here: the protocol must exist before it is applied at scale, and
paper2 establishes the instrument's credibility this paper spends.

---

## Discarded from the external roadmap, and why

- **Coupling comparison on λ_ca model orderings**: F128/F129 put the λ_ca cross-model ordering at
  seed stability 0.030 — rank-correlating noise across couplings is a wasted run. The coupling
  axis is worth having, but the object that can carry it is the share ordering (0.848). Deferred,
  reformulated, not adopted as written.
- **"Are self-consistency / Self-Refine gains construction artefacts?" at headline scope**:
  jurisdiction error — the gains are externally anchored (task accuracy), and anchoring is what
  the ring removes by design. The salvageable core is item 3's scoped version.
- **"Publish the package"**: done (gatecheck, MIT, cited in the paper). **"Expand the suite to
  cover every statistic the community invents"**: inverts the project's epistemology — guards are
  born from post-mortems, not anticipation. **"Require the community to pass the calibration"**:
  wrong verb for a solo researcher; the real verb is make it trivial and demonstrate it.
- **Training bespoke checkpoint ladders**: compute-unrealistic here; public ladders (item 2)
  first. **Internal mechanism hunts**: sequenced last by the roadmap and by this file, for the
  same reason — the state-level map is not yet dense.
