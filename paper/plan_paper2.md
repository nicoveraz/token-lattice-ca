# Plan — second paper (universality of the token-lattice transition)

**Written 31 Jul 2026, immediately after deciding NOT to reopen the Interp4Discovery submission.**
That decision stands: the submitted paper is pinned at `submission/neurips26-i4d` and F56–F59 do
not touch any number in it. Everything below is the *next* paper.

No venue or deadline is chosen yet. That is deliberate — the claim set is not closed, and picking
a deadline before knowing which claims survive is how the first paper acquired the retractions
that the audit later had to strip out.

---

## 1. What the second paper would claim, and what actually supports it

| | Claim | Evidence | Status |
|---|---|---|---|
| **A** | The damage-spreading transition is a **genuine critical point**, not a finite-size crossover: δ and θ reach their DP values at a *common* temperature, T_c ∈ [0.4343, 0.4391] | `dp_class_n192.json` (F58) | **Solid.** Gated on DK before the LM numbers were read; robust to the fit window at both ends and to boundary saturation |
| **B** | Its **universality class is not DP** — z sits below 1.5807 | `dp_fss_z.json` (F59) | **Not supported.** z = 1.380 [1.134, 1.606] *contains* DP. The estimate is stable across ladders (1.325/1.380/1.360) and points low, but the interval cannot separate them |
| **C** | Exponent measurements on black-box LM dynamics require **gating the estimator at the measurement's own geometry**, or they produce confident wrong answers | F56, F57, F59-v1 — three retractions, each caught by re-running on DK | **Solid, and the most transferable thing here** |
| **D** | F42's unignited runs are explained: 1/3 of visit orders heal a single-site seed before it propagates, and the order was shared across the whole batch | F57 | **Solid.** Predicted deaths matched observation exactly (8/8, 5/5) |
| **E** | The transition is **universal across models / radii** | — | **No evidence at all.** Everything is pythia-410m, r=2 |

**SUPERSEDED (Aug 1) by F62–F66 — read this first.** The universality programme was run on an AR
CA at r=2, and that construction turns out to measure an out-of-distribution artifact: the frozen
phase exists only at r=2, is carried by a single token, and a one-token BOS prefix removes 50 of
its 74 points. The **masked-LM construction shows none of it** (top-1 9–14% at every temperature).
So claim A survives only as *"there is a critical point in the AR two-token construction"*, which
is not the paper anyone wants. **A second paper should be built on the MLM path**, where Phase 3's
results (F14–F18) already live and where the degeneracy is absent. Claims C and D are untouched and
remain the strongest transferable content.

**The pre-F62 position, kept for the record: A + C + D is a paper; B is not; E is the gap that
decides whether the word "universality" belongs in the title.**

A paper on A+C+D is *methods-forward*: "here is a critical point in LM token dynamics, and here is
what it takes to measure an exponent without fooling yourself." That is publishable and true. It
is not the paper that the universality-class program was aimed at.

---

## 2. What would make it the stronger paper, and what each costs

Costs are measured on this machine (M1 Pro) from the `dp_fss_z` cells: 0.084 s per site-sweep per
64 replicas. All are resumable and run in overnight batches.

| Priority | Work | Issue | Cost | What it buys |
|---|---|---|---|---|
| **1** | **Second model family** — repeat A on a different architecture | #61 | ~17 h/model | Turns "a critical point in pythia-410m" into "a critical point in LM token dynamics". Without it, claim E is empty and the title cannot say universality |
| **2** | **Transverse Lyapunov Λ** — sign separates DP from multiplicative-noise | #81 | modest | A *second, independent* class discriminator. Directly relevant now: if z really sits below DP, multiplicative noise is the natural alternative, and Λ tests it without another FSS collapse |
| **3** | **Resolve z** — more replicas at N=96 | #82 | ~12 h/cell | Either separates z from DP or shows it cannot be separated at reachable precision. Six more cells ≈ 71 h |
| **4** | **ν⊥** — off-critical temperatures | #82 | multiplies by #T | The remaining exponent. Expensive and *not* on the critical path while B is unsettled |
| **5** | Second radius r | — | ~17 h | Is the class a property of the model or of the construction? F35 makes this sharper than it looks |

**Recommended order: 1, then 2, then 3.** Reasoning: a second model changes what the paper *is*;
Λ is a cheap independent check on the one claim that failed; z is expensive and only sharpens a
number that is already reported honestly as unresolved. ν⊥ is last, not first, despite being the
obvious "complete the exponent set" move.

---

## 3. Hazards to carry forward

Written down now because all three were discovered the expensive way.

1. **Gate every estimator at its own geometry** (F56). A calibration at N=512/200 licenses nothing
   at N=96/40. `dp_calibration.py` is the single implementation; new measurements import it rather
   than copying it.
2. **Check what the independent unit actually is** (F57). Anything drawn once per batch — visit
   order, and in principle anything else — makes replicas correlated, and pooling them as
   independent shrank error bars ~8×. New scripts use `order="per_replica"`; new *estimators*
   should state their independent unit explicitly and test it (between-seed vs within-seed spread).
3. **A cost function that can shrink its own comparison window is unbounded** (F59-v1). Any fitted
   quantity gets a scan far wider than plausible, and the fit must *reject* a minimum on the scan
   edge instead of reporting it.
4. **Validate on synthetic data with a known answer** before trusting a new estimator on DK, and on
   DK before trusting it on the model. The collapse estimator was proved exact on synthetic curves;
   that is what established the failing ladder was the data and not the code.
5. **The unexplained anomaly is still unexplained.** Ladders containing N=12 recover DK's z;
   ladders starting at N=24 do not, with the corrected estimator and at 32× replicas. Any z quoted
   in the second paper has to either explain this or disclose it.

---

## 4. What is already reusable

- `experiments/dp_calibration.py` — the gate, one implementation, imported by every DP run.
- `experiments/dp_pipeline_validation.py` — "can this fitting code recover a known answer?", the
  step that gates all the others. #60 already argues this ladder is a reusable methodology rather
  than one paper's step 1; claim C is that argument made in public.
- `order="per_replica"` in `src/lattice.py`, opt-in, with the CRN null asserted under it.
- The pre-registration habit: every DP script states its primary test and its failure meaning in
  the docstring *before* the run, and writes them into the results file.

---

## 5. Open questions for the author

1. **Venue and timeline** — deliberately unset. A + C + D could go out soon; A + C + D + E needs
   the second model family first.
2. **Is claim C the headline or the framing?** A methods paper about calibration discipline in
   black-box dynamical measurement is a different submission from a physics paper about a critical
   point in LM token space. The evidence supports either; they are not the same paper.
3. **Does the RunPod option come back?** #61 (second model) is download- and disk-bound rather than
   GPU-bound, so it is a poor fit for rented compute. Resolving z is GPU-shaped but low priority.
   On current priorities, renting still does not pay.
