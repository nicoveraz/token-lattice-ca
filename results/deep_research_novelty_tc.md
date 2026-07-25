# Novelty check — edge-of-chaos temperature T_c of LM generation (issues #12/#18)

Deep-research harness, ~100 agents, adversarially verified (2/3-refute to kill). The final
auto-synthesis step hit session/weekly usage limits on both runs; this report is synthesized
by hand from the **18 verified (3-0 / 2-0) claims across 10 sources**.

## Verdict: the METRIC is TAKEN and the METHOD is CLASSIC; only the specific
## application-combination is incremental — NOT a new metric.

### (a) The quantity — a per-model critical sampling temperature — is thoroughly TAKEN.
Five independent works define/measure a per-model critical temperature separating ordered
from disordered LM generation, all with sampling temperature as the control parameter:
- **arXiv:2606.06238** — locates T_c by sweeping softmax temperature (sharp susceptibility
  peak, order-parameter collapse; token embeddings as spins on a 1D chain).
- **arXiv:2501.16241** — defines a per-model T_c ≈ 1.2 (order-to-disorder: coherent below,
  nonsense above).
- **arXiv:2505.02879** — empirically measures a critical region T ≈ 1.25–1.38 (ChatGPT),
  Gibbs-measure / spin-glass framing.
- **arXiv:2406.05335** — temperature-driven phase transition (repetitive → incomprehensible),
  critical point via power-law statistics.
- **arXiv:2405.17088** — three phases (frozen / coherent / disordered) vs temperature via
  f-divergence distributional distances.

### (b) The method — a damage-spreading Lyapunov exponent — is CLASSIC CA physics.
- **arXiv:cond-mat/9811159** (Bagnoli et al.) — damage spreading between two CA configurations
  *defines* a maximal Lyapunov exponent (via the Boolean derivative). The
  damage-spreading→Lyapunov construction, and a state-space finite-size Lyapunov from
  perturbation growth, are decades-old in discrete dynamical systems. (Already reflected in
  the paper's `bagnoli1992damage` citation and its "we import the CA vocabulary" stance.)

So neither the target quantity nor the estimator technique is novel. The only unclaimed sliver
is the *combination*: using the classic CA damage-spreading Lyapunov to locate the
already-defined critical sampling temperature of a trained LM's token generation — an
incremental methodological variant (a different estimator for the same T_c), not a new metric.

### Controls (searched and correctly excluded as NOT anticipating)
- **arXiv:1904.09751** (nucleus sampling) — treats temperature as a quality/diversity knob;
  defines no critical temperature. **arXiv:1809.01201** (Random Language Model) — order-disorder
  transition, but control = grammar-weight broadening, not sampling temperature.
  **arXiv:2606.28103** — grammar-temperature of a WCFG ensemble (equilibrium/spin-glass), not
  generation-time sampling temperature. **Neural Computation 2004** (Bertschinger &
  Natschläger, `bertschinger2004real`) — edge of chaos via connectivity, not temperature.

## Most-threatening prior works to cite
2606.06238, 2501.16241, 2505.02879, 2406.05335, 2405.17088 (the T_c quantity);
cond-mat/9811159 / bagnoli1992damage (the damage-spreading Lyapunov method).

## Implication (unchanged, now stronger)
Do NOT claim T_c or the token-space Lyapunov as new. Cite the above and frame the LM metrics
as a CRN-certified, ground-truth-calibrated **route** to already-identified quantities —
exactly the reframed thesis ("we measure, not discover"). Confirms the strategic pattern:
objects and methods keep being taken; the durable contribution is the **validated instrument**
(reproduction ladder + CRN certification + calibration), not any novel object, metric, or
estimator.
