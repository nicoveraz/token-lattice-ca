# Novelty check — activation-lattice information-propagation cone (issue #19)

Deep-research harness, 103 agents, adversarially verified (2/3-refute to kill).
Question: is a CRN-certified, ground-truth-calibrated damage-spreading light cone over the
transformer residual stream (perturbation propagation across token positions × depth) novel?

## Bottom line: PARTIALLY ANTICIPATED, core object leaning TAKEN

The central object — an activation-space position×depth propagation cone — is largely taken.
Same pattern as the token-space paper: the framing has a novel sliver, the object does not.

## Per-element verdict

1. **Activation-space damage-spreading cone (positions×depth): PARTIALLY ANTICIPATED → TAKEN.**
   **Transformer Field Theory (arXiv:2605.25225)** already performs a localized finite-ε
   residual-stream intervention at one (layer ℓ*, token x*), R→R+εJ, measures
   δR_ℓ(x)=R^patched−R^unpatched across all layer-token sites, and reports an empirical
   anisotropic propagation cone in (Δℓ,Δx). This is element (1). SPARC (2607.09803) and
   2605.14258 also measure residual-stream perturbation propagation on adjacent axes.
2. **CRN exact-zero-null certification: NOVEL as framing** — the "differ at exactly one
   position, null is exactly 0" property is implicit in patched-minus-unpatched
   activation-patching / TFT differences, but no surfaced work names/certifies/exploits it
   as a common-random-number guarantee. Open question: does it add anything beyond nomenclature?
3. **Lieb-Robinson / butterfly-velocity / effective-receptive-field framing: PARTIALLY
   ANTICIPATED** — the chaos/Lyapunov/butterfly primitive is established (Poole 2016;
   2505.19458) but on depth/iteration axes, not position×depth. Importing L-R / butterfly
   *velocity* to the transformer cone is unclaimed. Open question / possible novel core:
   does any work EXTRACT a quantitative propagation velocity? TFT may show only qualitative
   cone broadening without naming a velocity.
4. **Calibration vs known-propagation ground truth (CA rule): NOVEL in the transformer
   domain** — CA damage-spreading + Lyapunov reference exists (Vispoel, Daly & Baetens 2024)
   but no interpretability work calibrates a transformer propagation measure against a
   synthetic known-propagation source.

## Most-threatening prior works to cite

- **arXiv:2605.25225** — Transformer Field Theory (THE threat; anticipates element 1).
- **arXiv:2607.09803** — SPARC, residual-stream error-propagation operator (depth/gen-step axis).
- **arXiv:2605.14258** — Dynamics of the Transformer Residual Stream (Jacobian eigendecomp).
- **arXiv:2403.00824** — Information Flow Routes (Ferrando & Voita); **arXiv:2005.00928** —
  attention rollout/flow (Abnar & Zuidema). Same-target / different-mechanism (attention
  aggregation, not CRN two-pass).
- **arXiv:2505.19458** — Jacobian/Lyapunov of self-attention (anticipates element 3's chaos framing).
- Vispoel, Daly & Baetens 2024 — CA damage-spreading + Lyapunov spectrum (calibration heritage).

## Caveats
- Top threats (2605.25225, 2607.09803, 2605.14258) are very recent 2026 preprints; landscape shifting.
- Search did NOT exhaustively characterize the activation-patching / path-patching / causal-
  scrubbing family — whether path patching already yields a position×layer perturbation-diff
  map is unresolved.
- Before banking element (3): direct-read TFT (2605.25225) to confirm it does NOT extract a
  quantitative butterfly velocity / Lieb-Robinson bound.

## Implication
The activation-lattice cone is NOT a clean novel win — the object is anticipated (TFT). The
defensible novel residue is thin: CRN-certification framing (possibly nomenclature),
ground-truth-CA calibration in the transformer domain, and *possibly* a quantitative
propagation-velocity bound if unclaimed. Reassess whether the front is worth pursuing given
the object is taken.
