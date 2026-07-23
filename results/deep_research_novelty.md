# Novelty verification — deep-research report (evolved findings)

*Harness: deep-research (102 agents, 19 sources fetched, 25 claims adversarially
verified 3-vote, 23 confirmed / 2 refuted). Run wf_4c7dd148-169. Caveat: the three
most method-adjacent works are 2026 preprints weeks-to-months old — re-check near
submission.*

## Bottom line

The **instrument** (single trained LM iterated as a radius-r token-space CA with
async Glauber updates + common-random-number damage spreading) is **largely novel at
the method level**. But its *dynamical substrate* is **not** novel, and novelty is
**uneven across the three claims**. The paper's defensibility rests on foregrounding
the **measurement layer**, not the "LM-as-iterated-token-dynamical-system" framing.

## Per-claim verdict

| Claim | Verdict | Why / who to cite |
|---|---|---|
| (1) Damage light-cone front **velocity ∝ radius**, model-invariant | ~~NOVEL~~ → **NOT NOVEL as a phenomenon** (revised by targeted CA-literature search) | The classic CA light cone: **Bagnoli, Rechtman & Ruffo 1992** (max Lyapunov = damage-front velocity), **Lieb–Robinson 1972** (finite range → constant range-set velocity), butterfly velocity + λ(v) in chaotic CA (**arXiv:2101.01313**). Contribution = black-box **transfer to a trained LM**, model-invariance, r-parameterization, λ⊥D_norm decomposition — *not* the law. |
| (2) Diversity-controlled **damping length** + capacity→sensitivity (edge-of-chaos), finite-size Lyapunov | **PARTIALLY ANTICIPATED** | The token-space metric + capacity axis are new, but "edge-of-chaos as capability measurement" is canonical (Bertschinger & Natschläger 2004) and already shown for *trained transformers* via Lyapunov of self-attention Jacobians — **arXiv:2505.19458** (Tomihari & Karakida, NeurIPS 2025, criticality↔capability). |
| (3) **AR (Pythia) replication**, causal window | **NOVEL as instrument transfer** | SPARC (2607.09803) does AR perturbation propagation but in *activation space* (Jacobian spectral radius), not a token CA. |

## The 5 most-threatening prior works (cite + distinguish)

1. **arXiv:2605.16378 — "Mixing Times of Glauber Dynamics on Masked Language Models"**
   (Levine et al., Cornell, 2026). **Biggest threat to framing-novelty.** Recasts a
   masked-LM as iterated masked-token resampling = Glauber Markov chain on token
   sequences — the *exact substrate*. It does **not** do damage spreading, light-cone
   velocity, damping length, or a finite-size Lyapunov exponent, and it uses **maximal
   coupling** (provably distinct from our **common-random-number** coupling); it
   measures O(n log n) mixing + low-T metastability. *A stronger "it anticipates the
   framing" claim split the verifier 1–2 — this is the paper's most exposed flank; make
   the measurement-layer contribution explicit.*
2. **arXiv:2505.19458 — recurrent self-attention dynamics** (Tomihari & Karakida,
   NeurIPS 2025). **Biggest threat to claim (2).** Lyapunov exponents of trained
   self-attention Jacobians; max-Lyapunov ≈ 0 (edge of chaos) ↔ high accuracy.
   Distinguish: continuous hidden space, correlates with *performance*, not a
   *model-size→sensitivity* climb; no token-space finite-size Lyapunov / damping length.
3. **arXiv:2607.09803 — SPARC** "Spectral Origins of the Self-Correction Blind Spot in
   Autoregressive Generation" (Petrova & Vejsiu). Error-propagation operator = product
   of per-step attention Jacobians; blind spot iff spectral radius ρ(F_T) ≥ 1 (= top
   Lyapunov boundary). Activation space, deterministic-linear, AR LLMs; the
   "self-correction" terminology neighbor. **NB: 2607.09803 = SPARC, not QUIVER.**
4. **Bertschinger & Natschläger 2004** (Neural Computation 16(7)). Canonical
   edge-of-chaos-as-capability-measurement (reservoir computing).
5. **arXiv:2503.13530 (QLE)** / **arXiv:2410.02536** ("Intelligence at the Edge of
   Chaos", ICLR 2025). Lyapunov-type chaos measurement of LLMs (over depth) / edge-of-
   chaos naming space. *2410.02536 applies the CA to the training **data**, not to
   iterating the LM — does not pre-empt.*

## QUIVER (arXiv:2605.23956) — no method overlap

Formal framework for perturbation propagation in **compound-AI pipelines** (directed
computation graphs of LLM calls); classifies pipeline *edges* as amplifiers/absorbers.
Shares vocabulary ("perturbation propagation", "bifurcation") but no CA/Glauber/
damage-spreading/light-cone/Lyapunov. Legitimate related-work citation, not a threat.

## Terminology

**"repair length" collides** with "self-repair" / **Hydra effect** (McGrath et al.
2023, arXiv:2307.15771; Rushing & Nanda ICML 2024, arXiv:2402.15390) — an internal
component-compensation phenomenon, categorically different from our spatial
perturbation-damping scale. "self-correction" (SPARC; Huang 2023; Kamoi 2024 survey)
is a further adjacent collision. **Recommendation: use "damping length" /
"error-damping length"** (drop "repair") and add an explicit disambiguation paragraph.

## Open questions (harden before submission)

- ~~Targeted search for "ballistic damage spreading" / "light-cone" in probabilistic
  CA to confirm claim (1).~~ **DONE (2026-07): claim (1) is NOT novel — it is the classic
  CA light cone (Bagnoli–Rechtman–Ruffo 1992; Lieb–Robinson 1972; arXiv:2101.01313).
  Reframed as import-and-confirm; the LM-token-CA behaving as a bona-fide CA is the point.**
- Is our model-size→sensitivity **monotone climb** genuinely distinct from
  2505.19458's performance↔criticality *correlation*, or would a size-parameterized
  reanalysis of 2505.19458 reproduce it? (Inference, not verified equivalence.)
- Explicitly cite + distinguish 2605.16378 in related work (substrate identity).
- Decide: "damping length" + disambiguation paragraph, or a fresh coinage.

## Not pre-empting (checked)

2503.13530 (QLE, depth-space), 2604.13206 (floating-point rounding, single forward
pass), 2604.22771 (static single-step KL-from-uniform), 2410.02536 (CA as training
data). Refuted: the claim that any surveyed paper already does token-space CA damage
spreading (0–3).
