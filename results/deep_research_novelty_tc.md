# Novelty check — edge-of-chaos temperature T_c of LM generation (issue #18 / new-metric)

Deep-research harness (synthesis step hit a session usage limit; verdict from the verified
claims). Question: is a measured critical/edge-of-chaos sampling temperature for LM
generation (Lyapunov zero-crossing via CRN damage spreading) novel?

## Verdict: the METRIC is TAKEN; only the CRN-Lyapunov METHOD is a novel variant.

A per-model critical sampling temperature separating ordered from disordered LM generation
is already defined and measured by concurrent work:

- **arXiv:2606.06238** — measures a critical temperature T_c by sweeping softmax temperature
  and locating a susceptibility peak + order-parameter (semantic-direction) collapse below
  T_c; TwoNN intrinsic dimension minimum near T_c. A measured per-model T_c already exists.
  (Method: thermodynamic/equilibrium analogy — NOT Lyapunov / damage spreading.)
- **arXiv:2406.05335** — temperature-driven phase transition (ordered low-T repetitive →
  disordered high-T incoherent); critical point shows natural-language power laws.
  (Method: statistical/power-law properties — NOT Lyapunov.)
- **arXiv:2405.17088** — three phases vs temperature (frozen T1*≈0.02 / coherent /
  disordered T2*≈0.5), explicit 1D-Ising order-disorder analogy. (Method: dissimilarity
  peaks — NOT Lyapunov.)
- Reservoir/recurrent edge-of-chaos + Lyapunov (Bertschinger & Natschläger 2004; Boedecker
  et al.): the order-to-chaos + Lyapunov framing, but connectivity (not temperature) as the
  control axis, and reservoir systems (not autoregressive LMs).

## Implication

Do NOT claim T_c (or the token-space Lyapunov) as a *new metric*. The object is taken. The
CRN damage-spreading Lyapunov route is a methodological variant that measures an
already-defined quantity. **Cite 2606.06238 / 2406.05335 / 2405.17088** and frame the paper's
LM measurements as a CRN-certified, ground-truth-calibrated *route* to quantities others have
identified — consistent with the reframed thesis ("we measure them, not discover them").

Confirms the strategic pattern: objects keep being taken (token dynamics → 2605.16378;
activation cone → 2605.25225; critical temperature → 2606.06238). The durable, defensible
contribution is the **validated instrument** (reproduction ladder + CRN certification), not a
novel object or metric.
