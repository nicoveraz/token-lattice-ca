# Token-lattice CA as a structure probe — pilot findings

**Setup.** A tiny bidirectional transformer (~0.42M params, 2 layers, d=96, word-level
vocab 2000) was trained from scratch on tinyshakespeare (292k tokens) as a *windowed
conditional model*: input is a window of 2r+1 tokens with the center masked, predict the
center — i.e. the model IS the CA rule family p_r(x_i | x_{i±r}), trained for
r ∈ {1,2,4,8,16} jointly (val masked-center accuracy ≈ 0.33–0.40). The automaton is a
ring of N=48 token cells updated by that rule at temperature T: async random-order
Glauber (default) or synchronous. Damage spreading uses twin runs with common random
numbers. All code in this folder reproduces everything (`vocab.py → train.py → sweep.py
→ census.py → damage.py → analyze_figs.py`; ~25 min total on 2 CPU cores).

## Findings

**F1 — The temperature phase structure reproduces the literature anchors.**
The order parameter (fraction of lattice bigrams that exist in the corpus) falls from
1.00 at T=0.3 to 0.14 at T=2.5, with the steep drop between T=1.0 and T=1.5 — matching
the reported autoregressive criticality at T_c≈1 (arXiv 2406.05335) and the MLM-Glauber
slow/fast-mixing crossover at τ≈1.5–2 (arXiv 2605.16378). The instrument passes its
sanity anchor. (`fig/phase_curves.png`, `fig/spacetime.png`)

**F2 — Static observables are radius-blind; damage transport is not.**
Equilibrium curves collapse across rule radius r=1…16 (phase curves are nearly
identical), consistent with the model's masked-center accuracy barely improving with
radius: its learned conditionals are dominated by nearest neighbors. But the *dynamical*
probe separates what statics could not: damage cones spread at a velocity set by r
(≈1 site/sweep at r=1; whole ring within ~3 sweeps at r=16), and in the ordered phase
(T=0.3) perturbations **heal** for r≤4 (total damage 0.02–0.03, cone contained at
7–14 sites) while r=16 leaks damage globally (0.34). Interaction range is readable from
transport even when invisible in order parameters. (`fig/damage_cones.png`)

**F3 — The attractor census partially recovers the training distribution (ground-truth
validation).** Running 32 lattices from random-token soup to quasi-stationarity and
censusing trigrams: top-50 overlap with the corpus's top-50 trigrams = 0.40–0.50
(random-lattice baseline 0.00), Spearman ρ = 0.49–0.56 on shared trigrams. The deepest
priors surfaced are the play's *format skeleton* (speaker-name + colon + newline) and
formulaic address ("my lord , my lord"). Validation is partial, and one artifact is
instructive: the `<unk>` token (8% of the stream, absorbing 12k rare types) acts as
artificial attractor material, inflating `<unk>`-rich patterns — vocabulary compression
distorts the prior map. (`fig/census_validation.png`)

**F4 — The ordered phase is metastable churn, not frozen fixed points.**
Even at T=0.3 activity floors at ~0.14–0.24 and no lattice froze or entered an exact
cycle in async mode (0/32 in 60-sweep windows). This matches the metastable-trap picture
of MLM Glauber dynamics (finite escape times) rather than deterministic-CA absorbing
states.

**F5 — The synchronous update scheme manufactures period-2 oscillators.**
At T=0.3, the fraction of sites in exact period-2 alternation is **0.84 under sync vs
0.05 under async** updates. The 2-cycle attractors reported for successive LLM
paraphrasing (arXiv 2502.15208) have a structural analog that is partly a property of
the *iteration scheme*, not only of the model — a methodological warning for
iterated-map probes.

**F6 — Dynamics preserve the measure, not the sample.**
Real corpus text melts under its own model's rule even at T=0.3 (site identity 0.26
after 5 sweeps, 0.17 after 60) while the lattice's corpus-bigram fraction stays at 1.0:
the specific text is forgotten, its local statistics are perfectly retained. At this
scale the model stores distributional structure, not retrievable literal strings.
(`fig/melting.png`)

**F7 — Structure crystallizes early; prediction quality keeps improving long after.**
Running the full instrument on every training checkpoint (step 0 = random init, then
1000…6000): the dynamical order (corpus-bigram fraction at T=0.3) jumps from 0.01 to
0.95 by step 1000 and saturates at 1.0 by step 2000, and census recovery jumps 0.00 →
0.42 in the same window — while val accuracy climbs only 0.26 → 0.32 over the remaining
5000 steps. The CA-visible structure (local grammar, format skeleton, deep priors) forms
in the first ~sixth of training; the rest is fine-grained predictive refinement that
barely moves the dynamical phase. (`fig/crystallization.png`)

**F8 — Self-healing is a *learned* property, and damage ignition is all-or-nothing.**
The untrained rule (step 0) is chaotically fragile: a 3-site flip infects 83% of the
ring at T=0.3 (random logits + low-temperature sharpening amplify any context change).
By step 1000 the same probe collapses to 0.06 — the ordered basin that heals damage
forms together with the grammar. Near the transition (T=0.7) susceptibility stays high
and noisy at every checkpoint. Mechanistic nugget: because the rule masks the center,
a flipped site cannot re-assert itself — its own next resampling ignores its value — so
single-site damage spreads only if a *neighbor* updates first AND decouples under
shared randomness. Ignition is therefore a rare event and bimodal: identical settings
gave total damage 0.33 (seed 21) vs 0.00 (seed 51) at T=0.7, r=1. Single-flip damage
probes on this rule class must report ignition probability separately from spread.

**F9 — Instrument and signal separate under coupled differential runs — but only at
the statistics level.** Twin runs sharing initial state, update order, and uniforms,
differing in exactly one factor: the null arm (nothing differs) gives exactly zero
divergence, validating the coupling. Trajectory-level divergence, however, saturates
for *any* difference — model arm 0.77–0.89, distribution-identical coupling swap
0.74–0.93 — chaos amplifies signal and apparatus alike, so endpoint state comparison
cannot attribute. Differencing *statistics* separates cleanly: the apparatus arm
vanishes (Δ order parameter ≤ 0.008, as it must — same distribution) while the model
arm survives exactly where models truly differ (Δ = 0.052 for final vs step-1000 at
T=0.3; ≈ 0 at T=0.7, consistent with F7). Certification rule: a reading is model
signal iff it nulls under apparatus swaps and moves under model swaps.
(`differential.py`, `results/differential.json`)

## Caveats

Tiny model, single small corpus, N=48, B=8–32 lattices per condition, no seeds swept;
MLM local conditionals are globally inconsistent (proven for real MLMs in 2605.16378),
so the CA is a well-defined stochastic dynamical system but not an exact sampler of any
joint distribution; `<unk>` compression biases the census; melting/census used r=2 only.

## Next steps

BPE vocab to remove the `<unk>` artifact; ignition-probability-aware damage statistics
(larger lattice ensembles); scale model/corpus; point the instrument at a real
pretrained small LM (weights would need to be provided to this sandbox); compare
against a Claude-as-rule qualitative run (LOGOS-CA style) with matched protocol.

## Files

- `fig/phase_curves.png` — order parameter & activity vs T, radius collapse
- `fig/spacetime.png` — space-time diagrams at T=0.3 / 1.0 / 2.5
- `fig/damage_cones.png` — damage light cones across (T, r)
- `fig/melting.png` — corpus-text melting curves
- `fig/census_validation.png` — census-vs-corpus trigram recovery
- `fig/crystallization.png` — all probes vs training checkpoint (F7, F8)
- `results/` — raw npz for every run + `summary.jsonl`, `census.json`, `analysis.json`
- `model.py ca.py train.py sweep.py census.py damage.py analyze_figs.py` — full harness
