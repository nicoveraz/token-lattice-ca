# What the instrument measures — when it works, and when it fails

**Drafted 8 August 2026.** A synthesis, not a new result: every number here is already in
`findings.md` and traces to a file in `results/`. The purpose is to state, in one place, what this
tool can tell you about a language model — and to argue that its failures are measurements of the
same kind as its successes, rather than caveats attached to them.

---

## 1. There is only one quantity

Everything this project has measured is a slice of a single map:

> **Change one token of a model's context. How much does its output distribution change?**

Under the common-random-number coupling this is exact rather than estimated. Sampling is inverse-CDF
against a shared uniform, so for two conditionals `p` and `q` the probability that coupled draws
differ is a deterministic functional of the pair — `s = 1 − Σ_v |[F_p(v−1),F_p(v)) ∩
[F_q(v−1),F_q(v))|`. Two forward passes, no seed, no Monte Carlo. `λ_ca` is what that quantity does
when iterated on a ring.

Every result below is that map, read along a different axis.

| axis varied | what the response does | source |
|---|---|---|
| training step | +0.336 → −0.339 → +0.168 | F39/F46/F84 |
| temperature | collapses at T\*, which predicts greedy degeneration | F86 |
| corpus | 78.1% vs 20.4% at an identical tokenizer | F63/F64 |
| architecture | no attention → no attractor | F64 |
| window position | far 0.579 vs near 0.820 | F110 |
| context regime | 0.833–0.876 on noise; spans 0.331 on settled states | F94/F96/F99 |

---

## 2. When it measures

The instrument reports the response's **dependence on a parameter you chose to vary**, with the
confounds separated by varying the construction and the model independently. What survives that
separation is model-attributable.

**A dated training-time landmark.** λ_ca traces a reproducible curve: **+0.3363 at initialisation**
(maximally chaotic — damage ignites every time and fills the lattice), collapsing to **−0.0847 by
step 16** and **−0.3388 by step 64**, crossing back up between steps 256 and 512, then plateauing at
**+0.1683**. Replicated at two lattice sizes and across four Pythia sizes; **both endpoints reproduce
in two non-Pythia families** measured with the identical estimator — OLMo-2-0425-1B (init +0.3598,
trained +0.1813…+0.1890) and OLMo-1B-0724 (init +0.3347, trained +0.1593…+0.2087).

**A temperature that predicts a failure mode outside the instrument.** T\*, where the ring's attractor
melts, tracks greedy-decoding degeneration at family level: ρ = 0.833, n = 8, permutation p = 0.0137.
It has survived two deflation attempts — the static argmax map does *not* predict degeneration where
the CA does, and T\* is not the heat capacity of the conditional (T_V occupies [1.21, 1.81] against
T\*'s [0.25, 0.58], disjoint).

**Corpus and architecture discrimination.** `gpt-neo-125M` and `gpt2` share a tokenizer and differ
only in corpus: 78.1% vs 20.4% attractor share. Attention is necessary — RWKV, Pile-trained without
it, has none. Scale is eliminated across a 70× Pythia ladder and a 12× GPT-2 ladder that never
overlap.

**A stable attractor taxonomy.** The argmax map's fixed-point structure sorts models into
funnel / none / fragmented, stable across seeds and surviving its dedup confound — though the
correlation with training recipe is partially anticipated in the literature.

---

## 3. When it fails

Every failure has been a **regime in which the response changes character**. Locating that boundary
is the same measurement, not an apology for one.

**In deployed generation the response is not small — it is absent.** An injected token error in real
autoregressive continuation is never corrected: `P_persist = 1.000` on three models, and
distributionally `TV_norm ≈ 0.97`, meaning the twins end as far apart as two *unrelated*
continuations. The mechanism is structural: free generation never revisits a token, so an error
stays in context permanently, whereas the ring resamples every site. The instrument fails to
transfer because **the model has no correction to measure**. This is the most portable result in the
project and needs no cellular automaton to state.

**The out-of-distribution cliff is sharp and single-token.** Asked to continue from almost nothing, a
model does not fail diffusely — it falls into one high-frequency vocabulary entry. The degeneracy is
confined to r ≤ 2, one extra context token removes it, and **banning `'\n'` alone drops it from 74%
to 15% without relocating it**.

**Long-range context lives in *which* tokens receive mass.** Project the conditional onto a small
token support and the far window position's contribution collapses to 0.061 against the near
position's 0.801 — branching falls below 1 and damage walks without growing. On the full vocabulary
both positions carry real influence (0.579 vs 0.820; at r=3 the third-back token still contributes
0.535 against 0.704). Restriction does not dim the signal uniformly; it removes the long-range part
specifically.

**In activation space the analogous exponent is architectural.** White-box λ_top is flat across
training at ≈1/L — set by depth, not by learning. That is why the cross-level bridge was
structurally unbuildable rather than merely noisy.

**Modelling quality and perturbation dynamics are dissociable.** At equal bits-per-byte on a shared
Pile slice, Pythia sits in its dip (bpb 2.226, λ_ca −0.019) while OLMo-2 is already at plateau
(bpb 2.323, λ_ca +0.184). Across-family spread at matched quality is 0.0588 against a 0.0197 floor,
and matched *token count* aligns better. Loss does not determine this.

**Sensitivity is a function of input regime, not a scalar.** On uniform-random windows the response
is saturated and flat (0.833–0.876) across all of training; on the states the ring actually settles
into, the same quantity spans 0.331. It is that *function*, not any single number, that changes
during training.

---

## 4. What it cannot say

**Why the response has its shape, in terms of internals.** No single attention layer moves λ_ca
beyond seed scatter — 0 of 24 clear 2σ — yet eight together give +0.345 while the 24 singles **sum to
−0.224**, the wrong sign. The property is collective, not localised and not diffusely spread. In
activation space the door is closed structurally (§3).

**Whether the model is good.** Dissociated from loss, explicitly (§3).

**What the developmental transition *is*.** Four routes have failed to name it. The closest positive
is that the conditional itself moves across training — measured non-circularly by transplanting each
checkpoint's conditional onto every other's settled state, where the model effect is 1.8× the
ensemble effect — but with no mechanism attached.

**Whether the transition's *timing* generalises.** Not answerable by anyone: no public non-Pythia
family publishes a checkpoint inside Pythia's dip window of 0.034–1.07B tokens, across ~4,000
branches in six families. Pythia is an outlier in early-checkpoint density.

---

## 5. Hand it a black-box model

You get a temperature T\* that predicts its greedy degeneration; a slot in the funnel / none /
fragmented taxonomy; a corpus-and-architecture signature that separates models sharing a tokenizer;
and — given checkpoints dense enough, which today means Pythia — a dated training-time landmark.

You also get, stated up front and measured rather than hedged, that **none of it describes what the
model does when you actually run it**.

---

## 6. What any of it is useful for

An honest ranking, including where the usefulness is thin. Ordered by how much someone outside this
project could act on it.

### Genuinely useful

**The cautionary result, and it is the timeliest thing here.** F56–F66 is a worked end-to-end
demonstration that an iterated self-consumption probe can manufacture a **precise, reproducible,
exponent-bearing phase transition that belongs entirely to the probe** — critical point located,
exponents fitted, and then shown to be the melting of an out-of-distribution prompt degeneracy. This
matters now because self-feeding pipelines are everywhere: agent loops, model-collapse studies,
synthetic-data training. Anyone measuring a dynamical property of a model that consumes its own
output has this failure mode available. Note what did and did not catch it: **nineteen models could
not distinguish "property of LMs" from "property of the probe"; one change of construction settled it
immediately.** Varying the subject is not a substitute for varying the apparatus.

**F35, because it changes design decisions.** An injected token error in real generation is never
corrected — `P_persist = 1.000`, `TV_norm ≈ 0.97`, CRN null exactly zero — and the reason is
structural: generation never revisits a token. Anyone building agentic loops or multi-step reasoning
is implicitly assuming *some* error absorption; there is none. Stated honestly, the phenomenon is
close to folk knowledge and has adjacent literature on error snowballing. What is new is the **exact
null**: not "errors tend to persist" but 1.000, with a certified-zero control.

**The methodology, which may outlast the science.** Six confident wrong verdicts, each caught by its
own check before reaching a paper — a calibration run at the wrong geometry, error bars from
correlated replicas, a cost function that could shrink its own comparison window, an estimator that
returns ≈0 on a system whose answer is known, a control that acquired the effect, and nineteen models
that could not separate probe from model. Now packaged as `gatecheck`. Realistically few people will
install it; the transferable part is the pattern, not the package.

**F98's infrastructure finding, actionable by someone else.** No public non-Pythia family checkpoints
the first ~1B tokens densely — ~4,000 branches across six families, and the window is empty. So
**every claim about early-training dynamics outside Pythia is currently unfalsifiable**, and a lab
releasing checkpoints could close that gap cheaply.

### Where the usefulness is thin

**λ_ca itself.** It dates an event nobody has named. Four routes failed to attach it to anything; the
closest positive is that the conditional moves, with no mechanism. Without an explanandum, "λ_ca
crosses zero between steps 256 and 512" is a fact nobody can act on. It replicates, it is
well-measured, its error bars are licensed — and it currently buys no decision.

**T\* is promising but conditional.** ρ = 0.833 at n = 8 families is the only externally-predictive
result and it has survived two deflation attempts, but it is greedy-scoped and the attempt at a
second target rejected itself on dynamic range. One leg is not an anchor.

**The attractor taxonomy** correlates with training recipe, but that correlate is partially
anticipated in the literature and no downstream use has been demonstrated.

### The overall read

The project's most valuable outputs are **negative**: doors closed cheaply and precisely, with the
measurement showing where the boundary sits. That is genuinely under-supplied. Its central positive
measurement is not useful yet — which is not a failure of rigour but the state of the evidence, and
`critical_analysis.md` reached the same conclusion independently. The strongest thing that could
change it is an explanandum; after four routes, the realistic assessment is that this instrument
measures something **collective and architecture-level that does not decompose**. That is worth
knowing, and it is not what anyone hoped for.

---

## 7. Why the failures are the more portable half

The results in §2 are about a construction most people will never build. The results in §3 are
claims about language models that survive the construction being discarded — that generation has no
error-correction mechanism, that the OOD cliff is one token wide, that the activation-space exponent
is architectural, that quality and perturbation response come apart.

This is interpretability of an unfashionable kind: **negative-space**. It establishes what is *not*
explicable at a given level and measures where the boundary sits, rather than reporting a mechanism.
Most work of this sort reports what it found; very little reports what provably is not there. That
is worth saying plainly, and it is worth not overselling — none of it is a circuit, and the project
has measured that door shut twice.
