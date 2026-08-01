# Plan — second paper

**Rewritten 1 Aug 2026, after F62–F66 dissolved the version written on 31 Jul.** That version
planned a universality-class paper on the AR construction. F66 showed the AR construction measures
an out-of-distribution prompt artifact, so the plan is rebuilt here rather than patched. The
superseded version is in git history at `9741792`.

The submitted paper is untouched throughout and stays pinned at `submission/neurips26-i4d`.

---

## 1. What actually happened, in one paragraph

The universality programme (#80–#82) measured a damage-spreading transition in a ring CA driven by
`p(x_i | x_{i-2}, x_{i-1})` on pythia-410m. It found a critical point (F58), fitted exponents
(F59), and hit a ladder anomaly (F60). Then a second model family showed no transition at all
(F62), which led to screening nineteen models (F63/F64) and finally to two interventions (F65) and
a change of construction (F66). The result: the transition is the melting of a **single-token,
two-token-context degeneracy**, a one-token BOS prefix removes 50 of its 74 points, and the
**masked-LM construction shows none of it**. The exponents are not wrong; what they are exponents
*of* is an artifact of asking an autoregressive model to continue from two tokens.

This is not a failure of the instrument. It is the instrument's calibration discipline working:
six confident verdicts died to their own checks, each one caught before it reached a paper.

---

## 2. The claim set, rebuilt

| | Claim | Evidence | Status |
|---|---|---|---|
| **A** | Iterated-resampling probes of LMs can manufacture a phase transition from an **out-of-distribution prompt**. Shown across 19 models, isolated by three interventions: radius (exists only at r=2), token ablation (one token carries it), and construction (BOS removes 50 points; MLM shows none) | F62–F66 | **Solid, complete, no further compute needed.** The strongest single result the project has |
| **B** | Measuring exponents on black-box LM dynamics requires **gating the estimator at the measurement's own geometry**, or it returns confident wrong answers | F56, F57, F59-v1, F61 — four retractions, each caught by re-running on a system with a known answer | **Solid.** The transferable methods contribution |
| **C** | F42's unignited runs are explained: 1/3 of visit orders heal a single-site seed before it propagates, and the order was shared across the batch | F57 | **Solid.** Predicted deaths matched observation exactly |
| **D** | The **MLM construction is clean** — no single-token concentration at any temperature or radius tested | F66 | **Solid but thin.** Two models, settled-state only. No dynamics measured on it yet |
| ~~E~~ | ~~A universality class for the transition~~ | — | **Withdrawn.** There is no model-independent transition to classify |

**A + B + C is a complete paper today, with zero further compute.** It is a *negative-result plus
methodology* paper, and the negative is the interesting part: people probe language models by
iterating their conditionals, and this shows how that can produce a phase transition that is a
property of the probe rather than the model.

---

## 3. The MLM path — what it is and what it would cost

F66 establishes only that the MLM construction does not *concentrate*. That is the absence of the
pathology, not the presence of a result. Nothing about MLM **dynamics** has been measured with the
post-F57 machinery.

What already exists (Phase 3, in the submitted paper): F14 the instrument ports to
bert-tiny/mini/base with the CRN null exactly zero; F15 real MLMs are not radius-blind; F16 damage
light cones replicate with front velocity ∝ r; F17 no strong self-healing phase; F18 the
special-token scheme is a first-class apparatus factor.

What does **not** exist: any MLM damage-spreading *transition* experiment. `experiments/mlm_*.py`
covers census, damage cones, differential certification, repair and sweeps — none of them looks for
a critical point.

| Step | Question | Cost | Notes |
|---|---|---|---|
| **M1** | Does the MLM CA have a damage-spreading transition at all? Coarse T scan for where damage stops dying | ~2–4 h | bert-base is 110M vs pythia-410m, so cells are cheaper than the AR ones. Must use `order="per_replica"` — newly plumbed through `mlm_ca.run` and never used |
| **M2** | If yes: is it free of the degeneracy? Re-run the F65 interventions (radius, ablation) on it | ~1 h | Cheap, and the paper needs it: a transition in a clean construction is only worth reporting if it survives the checks that killed the AR one |
| **M3** | If M1 and M2 hold: exponents, under the F56 gate | ~20 h+ | Only worth starting after M2. The AR programme spent ~60 h before discovering the object was an artifact |

**Recommendation: M1 and M2, and stop there for now.** They cost under five hours and they decide
whether an MLM programme exists. Do not begin M3 before M2 passes — that is exactly the mistake the
AR line made, and the whole of §4 exists because of it.

**A null at M1 is also a fine outcome for the paper.** If the clean construction has no transition
either, claim A strengthens: the transition was *only ever* the artifact.

---

## 4. Hazards, carried forward

All discovered the expensive way; each cost at least one retracted verdict.

1. **Gate every estimator at its own geometry** (F56). A calibration at N=512/200 licenses nothing
   at N=96/40. `dp_calibration.py` is the single implementation; import it, never copy it.
2. **State what the independent unit is, and test it** (F57). Anything drawn once per batch makes
   replicas correlated; pooling them shrank error bars ~8×. Use `order="per_replica"` and check
   between-seed against within-seed spread.
3. **A cost function that can shrink its own comparison window is unbounded** (F59-v1). Scan far
   wider than plausible and *reject* a minimum that lands on the scan edge.
4. **Validate on synthetic data with a known answer**, then on DK, then on the model. That sequence
   is what proved the failing ladder was the data and not the code.
5. **Run the control** (F65). The radius sweep read as "the attractor survives" until the control —
   a model with no attractor — acquired one too, revealing a generic long-context effect.
6. **Vary the construction, not just the model** (F66). Nineteen models could not distinguish
   "property of LMs" from "property of the probe"; one change of CA did it immediately.
7. **The ladder anomaly is still unexplained** (F60). Ladders reaching N≥96 carry a ~7% downward
   bias on DK that survives 16× sampling. Any z quoted anywhere must disclose it.

---

## 5. Open questions for the author

1. **Is A the headline, or B?** A is the more striking result — a manufactured phase transition —
   and B is the more reusable one. They are the same paper only if the narrative is "here is how the
   discipline caught it", which is defensible and honest but makes the paper about method.
2. **Venue.** Still unset, deliberately. A + B + C needs no further compute, so timing is a writing
   decision rather than an experimental one.
3. **How much MLM work before writing?** M1+M2 is under five hours and would let the paper say what
   a clean construction does, rather than only what a dirty one does. My recommendation is to run
   them and write either way.
