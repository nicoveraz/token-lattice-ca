# Adversarial review synthesis (pre-submission, NeurIPS)

Two independent adversarial reviews were run against the current draft
(`paper/paper.md`), the ground-truth log (`findings.md`), and the raw result
JSON in `results/`. One reviewer took an **originality** lens, the other an
**experimental-rigor / statistics / validity** lens. Both recommend **Reject**
(4/10 originality, 3/10 rigor). Every specific rigor objection below was
re-verified against the raw JSON and **confirmed** (verification script output
in the commit that adds this file).

The honest conclusion: the engineering and the honesty culture are strong, but
**the one novel scientific claim — the capacity → sensitivity axis (§3.3) — is
not supported by its own data**, and two of the paper's marquee decompositions
do not survive contact with the raw JSON. This is a statistical-repair +
reframe job, not a polish job.

---

## Reviewer 1 — Originality (Reject, 4/10)

**Core objection: no external significance.** The instrument is a well-built
thermometer never shown to track a disease. Everything the paper can defend
rigorously it explicitly labels as *imported physics* (damage-front velocity
∝ r = classic CA light cone; Lieb–Robinson; Bagnoli–Rechtman–Ruffo) or
*apparatus-determined*. The one capability-linked claim (capacity → sensitivity)
fails its own robustness checks. Method is novel at the *protocol* level (CRN
certification, ground-truth census calibration) but the *substrate* (iterated
token dynamics / edge-of-chaos in trained LMs) is not.

**Reviewer 1's top-ranked fix (verbatim intent):** make the instrument predict
something the field cares about — correlate a measured token-space quantity
(damping length / edge-of-chaos position / census recovery) with an external
model property across a real ladder; and do the cross-level validation the paper
only promises (§6-i): on the same open model, test whether black-box token-space
damping length agrees with white-box activation-space criticality (SPARC ρ(F_T);
the Jacobian–Lyapunov of arXiv:2505.19458).

---

## Reviewer 2 — Rigor / statistics / validity (Reject, 3/10)

Ranked by severity. **All numeric claims re-verified against the JSON.**

### W1 (most damaging) — The central novel claim rests on n=2 seeds with pseudoreplicated significance.
`repair_{tiny,mini,base}.json` all record `seeds: [21, 22]` — **two seeds**.
The "decisive" capacity result is a Wilcoxon over the 15 (r,T) cells
("mini≫tiny, 15/15 cells, p<10⁻⁴"). But a signed-rank test on 15 same-sign
pairs returns p ≈ 2/2¹⁵ ≈ 6×10⁻⁵ **by construction** — it merely restates "all
15 cells shared a sign." Those cells are a smooth 5r×3T grid from the *same 2
seeds*: strongly correlated, effective n = 2, not 15. Textbook
pseudoreplication. And the honest residue is thin: **base vs mini is null
(8/15, p=0.21)**, so the "3-point ladder" collapses to "tiny ≪ {mini, base}" —
one gap between a 4M and an 11M toy model.
**VERIFIED:** seeds = [21,22] in all three files.

### W2 — D_norm normalization can manufacture the monotone rise and the ">1 chaos signature."
1. **Coupling mismatch.** D (perturbed) uses CRN (common uniforms → the flip is
   the only divergence source); D0 (floor) uses **independent** noise with **no**
   flip. CRN actively suppresses divergence; independent noise does not. The
   ratio divides a matched-noise perturbation response by an unmatched-noise
   decorrelation rate. The natural CRN floor is the null itself (= 0), not an
   independent-noise run.
2. **"Damping shrinks with r" is denominator-driven at large r.** tiny, T=0.7:
   raw D **peaks at r=4 (0.537) and falls to 0.391 at r=16**, yet D_norm keeps
   **rising** (0.748 → 0.803 → 0.810) because D0 falls faster (0.719 → 0.621 →
   0.481). The metric reports damping still shrinking exactly where the raw
   perturbation is *decaying* — the collapse-into-repetition the metric claims to
   remove, reappearing in the denominator. **VERIFIED.**
3. **">1" is within ~1σ of the saturation value.** mini 1.10 ± 0.056, base
   1.04 ± 0.01 (2-seed std) — ~1σ from 1. When D and D0 both approach the Hamming
   ceiling, D/D0 → 1 mechanically. The "amplification beyond the noise floor"
   reading is not statistically separated from a saturation artifact.

### W3 — The kinematics ⊥ stability decomposition (λ "model-invariant") is an artifact of reporting λ_max at one cell.
The clean claim (λ = 0.745/0.738/0.767, ~4% spread; D_norm 35% spread ⇒
orthogonal, "only stability tracks capacity") uses λ_max over a 5×2 grid, all
landing at the single saturated cell (r=8, T=0.9). **Off that cell, λ tracks
capacity exactly like D_norm:**
- (r=8, T=0.7): tiny 0.500 < mini 0.604 < base 0.662 — **24% spread** (VERIFIED).
- (r=1, T=0.7): tiny 0.284 > mini 0.184 > base 0.153 — reversed, **46% spread** (VERIFIED).
So λ is neither model-invariant nor monotone; it is coincidentally equal only at
its peak. "Only stability tracks capacity" is contradicted by the paper's own
grid. No error bars on these λ. **The decomposition must be retracted or rebuilt.**

### W4 — The AR "consistent joint" external-validity argument is overstated.
`ar_ca.py` uses an **order-r truncated causal kernel resampled in place on a
periodic ring** — *not* the model's true autoregressive joint. So **both**
constructions are non-samplers of the model's actual distribution (MLM
inconsistent; AR consistent-but-wrong-kernel, ring-wrapped). Cross-construction
replication is weaker evidence than "inconsistent vs consistent" implies. Also
the AR "5 seeds" is **n=10 pooled across two temperature regimes** (T=0.5 vals
~0.1–0.4, T=0.7 vals ~0.5–0.8) — the reported mean/SE pool a bimodal
distribution and are ill-defined. AR capacity is a clean null (ρ=0.17, p=0.29;
the one marginal step 160m→410m p=0.06 reverses at 1b p=0.86).

### W5 — Real-model census "recovery" is near-floor against a corpus the models never saw.
Ground truth exists only on the bespoke toy (C2: self-TV 0.22 vs cross-TV 0.95 —
clean, but on the toy). On real models the census is scored against **WikiText**
(BERT never trained on it): top-50 overlap 0.02–0.04, Spearman 0.14→0.21.
`tiny_census.json`'s deepest attractors are degenerate junk ("† † †",
"##osphos", bracket/year soup), not a "format skeleton." "The census reads out
what the model internalized" is licensed only on the toy.

### W6 — v ∝ r is imprecise and the finite-size "lift" is partly circular.
The reported ceiling-lift 11.5 → 23.5 → 47.5 for N=48/96/192 at r=16 is
**exactly N/4** at each N (front fills the ring in 2 sweeps → saturate_sweep=2),
i.e. these *are* the clipping ceilings, not an unclipped law. The one genuinely
unclipped r=16 point (N=384, saturate_sweep=5) is **41.1 — below the "N=192
ceiling" 47.5**. Unclipped points are **superlinear** (≈1.5·r at r=4 → ≈2.6·r at
r=16). **VERIFIED** (velocities = N/4 exactly). The paper correctly declines to
claim this as novel, but presents it as more law-like than the data support.

### W7 — The crossover rescue is single-seed and internally inconsistent.
§3.4 claims to *strengthen* the masked capacity effect ("tiny < mini < base at
every T"). **False at T=0.3: mini 0.463 < tiny 0.508** (VERIFIED). Profiles are
single-seed, no error bars; base's T_c is undefined; the AR T_c trend is ρ=−0.8,
**p=0.2, n=4**. A noisy, single-seed, internally-contradicted analysis cannot be
cited as strengthening a weak effect.

### W8 — Multiple comparisons, no correction.
A battery of tests (capacity paired test, base-vs-mini p=0.21, AR ρ p=0.29,
160m→410m p=0.06, AR T_c ρ p=0.2) with no multiplicity correction. The one
"surviving" result is the pseudoreplicated one. Garden of forking paths.

### W9 — Scale and ring size.
Headline capacity is at **N=48 only**, and the N-scan shows the effect magnitude
falls steeply with N (`repair_fss_tiny.json` tiny r=8: 0.876 → 0.554 from
N=48→192; `capacity_nscan.json` mini N48 0.76 → N96 0.48). Models ≤110M
(masked) / ≤1B (AR) — the regime where capacity conclusions transfer least.

---

## The single most damaging technical objection

The one novel scientific claim — the capacity → sensitivity axis (§3.3) — is
simultaneously **(a) pseudoreplicated** (n=2 seeds; p<10⁻⁴ merely restates a
shared sign), **(b) plausibly a normalization artifact** (denominator-driven at
large r; ">1" within 1σ of saturation), and **(c) contradicted by the paper's
own supporting data** (the λ decomposition meant to isolate it as "asymptotic,
not kinematic" evaporates off the one cell where λ_max was reported), and
**(d) does not replicate** on the consistent-joint AR construction offered as the
external-validity anchor. Strip it and what remains is a well-engineered
instrument re-measuring known CA phenomena on tiny models — valuable, but not the
empirical contribution the abstract advertises.

---

## Repair path (what would move Reject → borderline/accept)

**Tier 0 — honest reframe, no new compute (do now).**
- Demote the capacity headline; state it as a suggestive but unestablished
  2-seed effect (tiny ≪ {mini, base}), not a scaling axis.
- Retract "λ model-invariant / kinematics ⊥ stability" (fails off-peak).
- Reframe v ∝ r as apparatus kinematics; drop the "ceiling lifts" story
  (they are N/4 clipping ceilings; the unclipped law is superlinear).
- Retract the crossover "strengthens at every T" claim (false at T=0.3).
- Soften the AR "consistent joint" argument (both are non-samplers).
- Fix the two oversold abstract sentences ("exposes structure static evaluation
  misses"; "only stability tracks capacity").
- Elevate what *is* defensible: the CRN-certified null protocol, the
  ground-truth census calibration (on the toy), the confound taxonomy
  (diversity / repetition / <unk> / period-2), and the masked-vs-AR
  non-replication *as* a finding.

**Tier 1 — statistical rebuild (moderate compute).**
- Capacity test: ≥15–20 **independent** seeds; **seed-level** hierarchical /
  bootstrap test (not signed-rank over correlated cells); extend the ladder
  (mid models between mini and base; a second family, e.g. RoBERTa); multiple N
  incl. N ≫ 100; pre-register the discriminating (r,T) cell.
- Ablate the normalization: report D, D0, D_norm with seed error bars at every
  cell; swap D0 for alternative floors (CRN null; maximal-coupling); show the
  ordering is invariant; demonstrate D_norm > 1 is separable from 1.
- Re-run the full λ(r,T) grid with seeds + error bars; test orthogonality off
  the peak cell.
- Report AR per-T (never pool bimodal regimes).
- Multiplicity correction across the test battery.

**Tier 2 — the significance spine (the flagship experiment).**
- **Cross-level validation:** on a shared open-model ladder (Pythia), correlate
  the black-box token-space damping length (and/or edge-of-chaos position) with a
  **white-box** activation-space criticality measure (depth-wise Jacobian /
  Lyapunov via forward hooks; SPARC-style ρ). A positive, seed-robust correlation
  is the novel, useful result both reviewers say the paper is missing: a cheap,
  weights-free proxy for a property that currently needs internals.
- Census against **real** ground truth (Pythia vs the Pile, or a mid MLM trained
  on a known corpus), not a WikiText proxy.

Cross-level validation must itself be run with the Tier-1 statistical standard
(independent seeds, seed-level tests) or it repeats W1.
