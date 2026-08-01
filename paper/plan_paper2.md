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
| **A** | **Iterated-resampling probes of LMs can manufacture a phase transition from an out-of-distribution prompt.** The mechanism is identified, the boundary is sharp, and the model's *native* construction shows neither the degeneracy nor a transition | F62–F67, F69 | **CLOSED — complete, no further compute** |
| **B** | Measuring exponents on black-box LM dynamics requires **gating the estimator at the measurement's own geometry**, or it returns confident wrong answers | F56, F57, F59-v1, F61 — four retractions, each caught by a known-answer system | **Solid.** The transferable methods contribution |
| **C** | F42's unignited runs are explained: 1/3 of visit orders heal a single-site seed before it propagates, and the order was shared across the batch | F57 | **Solid.** Predicted deaths matched observation exactly |
| ~~D~~ | ~~the MLM construction is clean~~ | F66, F67 | **Absorbed into A** — its cleanliness *is* the control that makes A a claim rather than an observation |
| ~~E~~ | ~~A universality class for the transition~~ | — | **Withdrawn.** There is no model-independent transition to classify |

### Claim A, as it now reads

1. **The transition is real and precisely measurable** — δ and θ reach their DP values at a common
   T_c ∈ [0.4343, 0.4391] (F58), gated on Domany–Kinzel before the LM numbers were read.
2. **It is a property of the probe.** The frozen phase is a single-token collapse: 81 of 96 sites
   newline at T=0.02, and F58's T_c sits at 52% newline (F62).
3. **Not the corpus, not the network, not scale.** Refuted from both directions across 19 models
   (F63); granite's dense and MoE members agree within 2 points while differing 2× in width, 1.7×
   in depth, 16× in FFN and routing-vs-none; scale eliminated across a 70× Pythia ladder and a 12×
   GPT-2 ladder that never overlap (F64).
4. **The mechanism is an out-of-distribution prompt.** One BOS token takes 74.4% → 24.1%; the
   masked-LM construction, where infilling *is* the training objective, shows 9–14% at every
   temperature and radius (F66).
5. **The boundary is sharp and narrow.** Family-distinguishing degeneracy occupies r ∈ {1, 2} only;
   r=2 → r=3 drops top-1 by 52 points. The large-radius rebound appears in the control too, so it
   is a generic long-context effect and excluded (F69).
6. **The clean construction has no transition either** — surviving damage never falls below 0.547
   down to T=0.02 across two MLM models. The pre-registered good null: no absorbing state,
   therefore no absorbing-state transition (F67). There is no competing "but the clean version has
   a real one" left to explain.

Steps 4–6 are what make this a *claim* rather than a curiosity: a mechanism, a boundary, and a
control that behaves as the mechanism predicts.

**A + B + C is a complete paper today, with zero further compute, and A is now closed.** It is a *negative-result plus
methodology* paper, and the negative is the interesting part: people probe language models by
iterating their conditionals, and this shows how that can produce a phase transition that is a
property of the probe rather than the model.

---

## 3. The MLM path — answered, and folded into A

M1 (#89) ran and returned the pre-registered good null: **the clean construction has no
damage-spreading transition either.** Surviving damage never drops below 0.547 across T ∈ [0.02,
0.5] on `bert-base-uncased` and `bert-medium`, down to essentially deterministic sampling. M2 and
M3 were **skipped by the script's own gate**, not by a judgement made after seeing the numbers.

That closes the path rather than opening it. The AR frozen phase required an *absorbing state* —
every site resampling to one token regardless of context. F66 showed there is none here from the
settled composition; F67 confirmed the dynamical consequence directly. There is no MLM exponent
programme to run, and the result belongs inside claim A as its control.

**#89 can be closed.** M3 is moot.

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
3. ~~How much MLM work before writing?~~ **Answered:** M1 ran, returned the good null, and M2/M3
   were skipped by the gate. Nothing is waiting.

---

## 6. Is there a third paper?

**Maybe — and it hangs on one unresolved result, so do not plan for it yet.**

What is left over after A + B + C is a single coherent thread, and it is *not* about phase
transitions:

> Models differ, reproducibly and by a wide margin, in how they behave when handed almost no
> context. A single token dominates the two-token conditional in some models and not others —
> **6% to 98% across twenty-six models**, a *graded* quantity rather than two classes (the apparent
> bimodality of the first nineteen was a sampling artifact; see F64's correction). Attention is
> necessary and the corpus
> appears to decide (F64). The melting temperature **T\*** turns that binary into a graded scalar
> that is *tighter within a family* than the raw share and separates families the binary lumps
> together, at a cost of four settle runs per model.

That is a model-characterisation result about **out-of-distribution fallback behaviour**, which is
a different subject from paper 2's "the probe manufactures a transition". Paper 2 uses this material
as *evidence*; a third paper would make it the object.

**What it needs before it is a paper, and what is genuinely uncertain:**

| Requirement | State |
|---|---|
| An **external anchor** — T\* must predict something not measured by the CA | **Unresolved.** rho(T\*, greedy repetition) = +0.55, p = 0.11 (F68). Underpowered, not null |
| Enough **independent families** | **Weak.** The n=10 was really ~4 families — six points were Pythia sizes. Seven new families are running |
| A working account of **what determines it** | **Incomplete.** F64's "attention + corpus" is right but not sufficient: the two code models screened so far *split*, so a filler-rich corpus is not the rule |

**The honest read: if #90 resolves positively, there is a third paper. If it stays underpowered or
comes back null, there is a paragraph in paper 2 and nothing more.** T\* would then be a
well-defined property of a regime that F66 showed is an artifact — interesting to have measured,
not interesting enough to carry a submission.

Three papers out of this work would also be over-slicing unless the third has its own external
anchor. It is worth resisting the pull to make the material stretch: the project's record is built
on scoping claims down, not up.

**Decision point:** when #90's seven new families land. Not before.
