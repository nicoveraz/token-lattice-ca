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

## Phase 2 findings — hardening (word-level toy, still)

**F10 — BPE removes the `<unk>` artifact, and reveals the pilot's low-T census
recovery was partly artifact-to-artifact matching.** Replacing the word-level
2000-type vocab (8% `<unk>`) with a 4096 byte-level BPE (100% coverage, no
`<unk>`; model retrained, val masked-center acc ≈0.25–0.29) changes the census
*qualitatively*. Word-level top-15 attractor trigrams were **11–13/15
`<unk>`-containing** ("`<unk> , <nl>`", "`<unk> <unk> ,`", "`the <unk> <unk>`");
since the corpus itself is 8% `<unk>`, those matched trivially and inflated the
top-50 overlap. BPE top attractors are **0/15 artifact** and are real formulaic
Shakespeare ("`my lord,`", "`, my lord`", the speaker-name+colon+newline
skeleton). Raw top-50 overlap therefore *drops* at T=0.3 (0.40→0.26) — the
artifact correction — while genuine recovery is *higher* at T=0.7 (0.50→0.60),
and the recovery peak shifts T=1.0→T=0.7. Trigram units differ across
tokenizations, so the number is not directly comparable; the finding is that BPE
recovery is artifact-free and trustworthy. Confirms F3's mechanism; the fix works.
(`fig/census_bpe.png`, `results/census_bpe.json`)

**F11 — Radius collapse and the phase curves survive proper seed statistics.**
≥5 independent seeds per condition (different init, update order, and uniform
stream): error bars on the order parameter are tiny — std ≤0.001 deep in the
ordered phase, peaking at only 0.026 near the drop (T≈1.25), max 0.026 over all
conditions. The radius collapse (F2) holds: at T=1.0 the spread across r=1…16 is
0.034, within ~2× the seed noise. The pilot's single-seed curves were not a lucky
seed. (`fig/phase_curves_multiseed.png`, `results/sweep_multiseed.jsonl`)

**F12 — The temperature "phase transition" is a finite-size CROSSOVER, not a true
transition.** Finite-size scan N∈{48,96,192}, r=2, fine T grid, ≥5 seeds: the
order_mean(T) curves **overlay** across N (T=1.0: 0.763 / 0.759 / 0.762), the
maximum slope |d·order/dT| is **constant** (0.86 → 0.85 → 0.84 — no steepening),
and the susceptibility (variance of the per-lattice order parameter across the
ensemble) **self-averages as χ_peak ∝ 1/N** (0.0068 : 0.0035 : 0.0018 ≈ 4:2:1
for N=48:96:192). At a genuine continuous transition the susceptibility peak would
*grow* with N and the drop would *steepen*; here it does the opposite. So the
T-driven order/disorder change of F1 — and the literature anchors T_c≈1
(2406.05335) and τ≈1.5–2 (2605.16378) — are **crossover scales** at this
radius-windowed toy scale, not critical points. Any "phase transition" language
must be downgraded to "crossover" unless larger-N or real-model data show χ growth.
This is the single most important correction to the pilot's framing.
(`fig/finite_size.png`, `results/finite_size.json`)

**F13 — Block-flip damage with ignition probability cleanly separates the
all-or-nothing Bernoulli from the spread magnitude (F8 done right).** Single-site
flips have all-or-nothing ignition (F8), so one number conflates P(ignite) with
spread. With **3-site block flips and B=64** the two separate: in the ordered
phase (T=0.3) ignition probability rises with radius (0.52 at r=1 → 0.72 at r=16)
and so does conditional spread (0.16→0.49) — small-radius damage usually heals
(mean 0.083), large-radius leaks (0.353), so interaction range is readable from
ignition statistics; near the transition (T=0.7) ignition is 0.83 (r=1) → 1.0
(r≥4), spread 0.60→0.85; disordered (T=1.5) ignition 1.0, spread ≈0.95. Across
training, the healing basin forms with the grammar: the untrained rule (step 0)
ignites **every** lattice (P=1.0, mean damage 0.88 at T=0.3); by step 1000 mean
damage collapses to 0.086 (P_ignite 0.58) as bigram order jumps 0.01→0.945 — F8's
"0.83→0.06" reproduced and decomposed into a falling ignition probability and a
contained conditional spread. (`fig/damage_ignition.png`, `fig/crystallization.png`,
`results/damage_block.json`)

## Phase 3 findings — real pretrained MLMs (the actual test)

The instrument is ported to HuggingFace masked LMs (prajjwal1/bert-tiny 2L/128H,
prajjwal1/bert-mini 4L/256H, bert-base-uncased 12L/768H; fp16 on MPS; shared
bert-base-uncased WordPiece vocab, 30522). The rule wraps the 2r+1 window in
[CLS]…[SEP] with the center masked (an *apparatus* choice; a no-special-tokens
variant is in the apparatus arm). Census validation is now **proxy-based** against
a WikiText-103 slice (238k tokens) — BERT was not trained on WikiText, so
overlap/ρ are a lower bound on "recovers natural English", not ground truth.

**F14 — The CA instrument ports cleanly to real MLMs and the CRN harness stays
exactly zero.** All three models drive the ring CA on MPS; coupled twin runs
sharing model, init, order, and uniforms diverge by **exactly 0** (the null test,
on the torch path too). Low-T lattices are locally coherent English (bert-base at
T=0.7: *"i don't. … is written by christian for mother. … he hoped to increase
access to the trampoline"*), high-T is token soup. The instrument works.
(`experiments/mlm_ca.py`, `mlm_smoke.py`)

**F15 — Real MLMs are NOT radius-blind (contrast with toy F2).** Where the toy's
equilibrium statistics collapsed across r=1…16, the MLMs' do not. Local bigram
order is already r-dependent for the larger models (k2 spread across r = 0.09 tiny,
0.27 mini, 0.18 base), and longer-range corpus consistency (k3, k4 overlap) rises
with r and peaks at an *intermediate* radius (r≈4 for mini, r≈8 for tiny/base)
before falling at r=16 — larger windows can build longer-range structure, but a
too-large window relative to N=48 degrades it. Caveat: the k-gram overlap conflates
"corpus-consistent long-range structure" with repetitive attractor loops (tiny's
high k4≈0.22 at r=8 owes partly to repeats like "hong kong hong kong"), so read the
*presence of r-dependence* as the finding, not the absolute magnitudes.
(`fig/mlm_radius.png`, `results/mlm/*_sweep.json`)

**F16 — Damage light cones replicate on real MLMs, and the front velocity is
model-invariant (set by r, not scale).** Block-flip CRN cones give a front velocity
of **1.6 → 3.5 → 7.7 → 11.5 → 11.5 sites/sweep for r = 1,2,4,8,16, essentially
identical across tiny/mini/base** (saturating at ≈N/2 per sweep once r≥8 fills the
ring in ~2 sweeps). The toy's F2 light cone (velocity∝r) reproduces on real models,
and the velocity is an interaction-range (apparatus) property independent of model
scale. (`fig/mlm_damage.png` left)

> **Novelty note (targeted CA-literature search, do it 2026-07):** `v∝r` is **NOT
> novel** — a ballistic damage front whose velocity is set by the neighborhood range is
> the *classic CA light cone*: Bagnoli, Rechtman & Ruffo 1992 (max Lyapunov = damage-front
> velocity), the Lieb–Robinson bound (finite range → constant range-set velocity), and
> butterfly velocities of decorrelator fronts in chaotic CA (arXiv:2101.01313, which also
> has the velocity-dependent λ(v)). This *strengthens* the framing: the LM token-lattice
> behaves as a bona-fide CA (it shows the light cone). The contribution is the black-box
> **transfer to a trained LM + model-invariance + the r-parameterization + the λ⊥D_norm
> decomposition**, not the law. F16/F21/F23 reworded to import, not discover. Even the
> F23 λ result (λ = kinematic velocity axis) is anticipated by Bagnoli–Rechtman 1992; what
> is new there is the *contrast* with the capacity-tracking asymptotic D_norm.

**F17 — Real MLMs lack a strong self-healing ordered phase in the radius-windowed
CA; the healing/spreading boundary sits *below* the full-context τ≈1.5–2 crossover.**
At r=4 the mean block-flip damage never heals to near-zero even at the lowest T
tested: T=0.5 gives mean damage 0.39 (tiny) / ~0.58 (mini, base), rising to ≈1.0 by
T≈1.3. So the healing→spreading boundary is T≈0.5–0.8, whereas the toy healed to
0.02–0.03 (T=0.3, r≤4) — the real MLMs are markedly **more damage-fragile**. This
sits well below the slow/fast-mixing crossover τ≈1.5–2 reported for *full-context*
MLM-Glauber (arXiv:2605.16378). The setups differ deliberately — they condition on
the full sequence, we condition on a radius-r window — and the comparison suggests
that removing long-range context (windowing) destabilizes the dynamics and pushes
the fragility boundary to lower T. This is the clearest divergence from the toy.
(`fig/mlm_damage.png` right)

**F18 — Differential certification (F9) holds on real MLMs and exposes scale-
dependent *apparatus* sensitivity.** Statistics-level Δ(order parameter): the null
arm is exactly 0 for all three models; distribution-preserving apparatus swaps
(update order, CDF-ordering) sit at a floor of Δ≈0.009–0.021; **model** swaps move
it above the floor (tiny↔mini 0.072, tiny↔base 0.066, mini↔base 0.023) — so the
certification rule (a reading is model signal iff it nulls under apparatus swaps and
moves under model swaps) is satisfiable. Two cautions: (i) tiny→mini changes the CA
equilibrium *more* than mini→base (0.072 vs 0.023) — the dynamical structure
saturates early with scale, echoing F7 on the training axis; (ii) the special-token
scheme is **not** a negligible apparatus — swapping CLS/SEP for no-special-tokens
moves Δorder by 0.042 (tiny), **0.135 (mini)**, 0.045 (base), i.e. for mini the
windowing choice moves the order parameter *more than a model change does*. Headline
claims must therefore hold the special-token scheme fixed and be certified against
it. (`fig/mlm_differential.png`, `results/mlm/*_diff.json`, `model_arm_*.json`)

**F19 — Proxy census: scale improves recovery, and bert-base recovers WikiText's
*format skeleton* — the same phenomenon as the toy's Shakespeare skeleton (F3).**
Top-50 trigram overlap with the WikiText proxy is low for all (0.02–0.04, vs the
toy's ground-truth 0.40–0.60) but Spearman ρ improves with scale (0.14 tiny → 0.21
base) and bert-base's lattices are coherent English. Its deepest attractors are
WikiText's format markers (abbreviation/list patterns ". a.", ". c.", sentence
boundaries) — structurally the same "deepest prior = document format skeleton"
result as the toy's speaker-name+colon+newline (F3), now on a real model. The low
absolute overlap is the stated proxy limitation, not weak recovery.
(`results/mlm/*_census.json`)

## Phase A findings — hardening the real-MLM headline

Three publication-blocking threats to F15/F16 closed, at a fixed special-token
scheme (cls_sep) with ≥5-seed error bars, at T=0.7 (the ordered regime).

**F20 — F15's radius-profile difference is a certified *model* effect at a fixed
scheme — but the special-token scheme is a first-class apparatus of comparable or
larger magnitude.** Cross-model order-profile shift (mean_r |Δorder|, fixed
cls_sep, 5 seeds): tiny↔mini **0.140 ± 0.033**, tiny↔base 0.115 ± 0.023, mini↔base
0.053 ± 0.027 — all far above the distribution-preserving apparatus floor
(update-order / CDF-ordering swaps) of **0.014**. So the radius profile genuinely
differs between models; F15 is not an artifact of the distribution-preserving
knobs. *However*, swapping cls_sep → no-special-tokens moves the same profile by
0.083 (tiny), **0.324 (mini)**, 0.155 (base) — for mini the scheme swap exceeds any
model change. Conclusion: cross-model radius claims are valid *only at a fixed
scheme*; the CLS/SEP choice is a first-class apparatus that must be held fixed and
certified against (confirming and quantifying F18 at the profile level). The
certification protocol is what makes F15 reportable. (`fig/phaseA_radius.png`)

**F21 — The r≥8 damage-velocity plateau (F16) was finite-size wraparound; the
velocity∝r law continues.** Finite-size scan N∈{48,96,192,384} at r∈{4,8,16}: the
"11.5 sites/sweep ceiling" on N=48 lifts with N — at r=16, v = 11.5 → 23.5 → 47.5 →
41 for N = 48,96,192,384; at r=8, 11.5 → 15.7 → 14 (settles once unclipped); at r=4
(never clipped) v ≈ 6–7, stable across all N. The clean unclipped law is
v ≈ (1.5–2.7)·r sites/sweep, growing monotonically with r. F16's velocity-set-by-
radius result holds; the saturation was the ring filling in ~2 sweeps at small N.
(Apply finite-size scaling before calling anything saturated — the F12 lesson, now
applied to a velocity.) (`fig/phaseA_velocity.png`, `results/mlm/phaseA_velocity_*.json`)

**F22 — F15's raw "long-range structure grows with r" was partly repetition; the
repetition-robust signal is an *intermediate*-radius optimum (r≈4).** The raw
k-gram overlap conflates corpus-consistent structure with degenerate repetition (a
lattice looping one corpus bigram scores high). Two controls: (i) a
distinct-corpus-k-gram count (a repeated k-gram counts once) and (ii) the
distinct-token fraction. Under the distinct metric the 4-gram signal **peaks at
r≈4 for all three models** (tiny 0.018, mini 0.041, base 0.036) and *falls* by
r=16 — an intermediate conditioning-radius optimum, not monotonic growth. It is
genuine (not repetition) for mini and base, whose distinct-token fraction stays
high across r (mini ≈0.5, base ≈0.7); tiny (weakest) degenerates into repetition at
large r (distinct 0.20 → 0.16), so *its* raw large-r rise was repetition — the
confound is itself model-dependent. The coarse-grained MI-decay length is
repetition-confounded in the same direction (long for repetitive tiny, short for
diverse base) and must not be read as a structure measure. **Reframed F15:** real
models are not radius-blind (vs toy F2) — they show a genuine intermediate
conditioning-radius optimum (~r=4) for corpus-consistent structure — but there is
no monotonic long-range growth, and the effect must be read with a
repetition-robust metric. (`fig/phaseA_repetition.png`)

## Phase B/C findings — the damping length, and external validity

Naming/positioning (per novelty check, `results/deep_research_novelty.md`): we call the
perturbation-damping scale the **damping length** / **error-damping length** (symbol
`ξ_repair` in code). We deliberately avoid **"repair"** (collides with *self-repair* /
the *Hydra effect* — internal component-compensation, arXiv:2307.15771, 2402.15390) and
**"self-correction"** (the AR/CoT reasoning-error literature and SPARC, arXiv:2607.09803);
both are disambiguated below. Our damping length is a *spatial/dynamical* scale over
which a token-space perturbation is absorbed — a categorically different object. The
contribution is the **black-box token-lattice CA instrument** and the specific
quantities it measures (the radius law F15, velocity∝r F16/F21, the damping length,
and its capacity scaling); the *phenomena* those land on — criticality, computation
at the edge of chaos — are decades old (Langton; Bertschinger & Natschläger 2004,
reservoir computing; *Intelligence at the Edge of Chaos* arXiv:2410.02536). We
**measure and quantify a long-hypothesized picture from the outside and add a
capacity-scaling result**; we do not claim to discover the edge of chaos.

**F23 — A diversity- and velocity-controlled damping length exists, and it shrinks
with conditioning radius; larger-capacity models sit closer to the chaotic side.**
The raw asymptotic damage D (fraction of sites still differing at long time) is
*diversity-confounded*: a degenerate low-entropy lattice snaps a perturbation back
trivially (a deep narrow attractor with nowhere to differ), scoring low D for the
wrong reason — the stability analog of the F22 repetition confound. We control with
the **diversity floor** D0 = unperturbed drift of twins sharing the settled init but
with **independent** noise and **no** flip; because D0 is a full radius-r run it
propagates at the same velocity∝r as the perturbed run, so the normalized
`D_norm = D/D0` cancels **both** the diversity term **and** the kinematic term. Then:
- **D_norm rises with r** (tiny 0.43→0.82, mini 0.74→1.03 over r=1…16): the
  error-damping length shrinks as the window widens. The raw-D "recovery" at large r
  was the lattice collapsing into repetition (distinct-token frac ↓), deflating raw D
  while D_norm stayed high — the confound, caught.
- **Capacity separates it — but the climb saturates early.** Mean D_norm: tiny 0.672,
  mini 0.880, base 0.904. A paired test across the 15 (r,T) cells shows **mini ≫ tiny**
  is decisive (15/15 cells, gap +0.208, Wilcoxon p<10⁻⁴) but **base vs mini is NOT
  significant** (8/15 cells, gap +0.024, p=0.21). So the honest claim is **tiny ≪
  {mini, base}, with sensitivity saturating by mini** — not a monotone 3-point ladder.
  This matches F18 (tiny→mini changes the dynamics more than mini→base): the dynamical
  structure crystallizes early on the scale axis. Only the two larger models cross
  **D_norm > 1** (amplification beyond the floor); tiny stays sub-critical (max 0.82).
  The separation *fights* the normalization (a more diverse model's higher floor should
  deflate its D_norm, yet it is higher) and is velocity-immune (matched r ⇒ matched
  propagation speed), so it is neither diversity nor kinematics. **N-robust:** at the
  discriminating radius r=2 the full ordering tiny < mini < base holds at both N=48
  (0.56 < 0.76 < 0.90) and N=96 (0.36 < 0.48 < 0.58); the mini–base gap is visible at low
  r but washes out in the all-cell average because large-r cells saturate (hence p=0.21).
  (`results/mlm/capacity_nscan.json`)
- **Kinematics ⊥ stability (the Lyapunov confirms it)**: the finite-size Lyapunov
  exponent λ (early log-separation slope of the CRN twins) is **model-invariant** —
  λ_max = +0.745 / +0.738 / +0.767 for tiny / mini / base, all at (r=8, T=0.9), a
  ~4% spread against D_norm's 35%. λ is universally *positive* (every model
  transiently spreads a perturbation) and set by the radius (velocity∝r, F16/F21),
  not the model. So the early-time dynamics are a kinematic axis shared by all
  models; the capacity signal lives *entirely* in the **asymptotic** persistence
  D_norm. Velocity/λ and stability/D_norm are orthogonal — and only stability tracks
  capacity. This is the clean kinematic⊥stability decomposition.
- **Reading (edge-of-chaos, as measurement)**: D_norm > 1 (mini at large r) is a
  chaos signature — the flip is amplified *beyond the model's own noise floor*. tiny
  reads "stable" because it is *frozen* (heals by collapsing to a dead repetitive
  attractor); mini reads "sensitive" because it is *expressive* (rich dynamics carry
  a perturbation). So the capacity axis traces a **stability↔expressiveness
  tradeoff**: more capable models run nearer criticality, where expressiveness lives.
Rigor: D_norm is monotone in r for mini and base but **tiny dips at r=16**
(0.82→0.75) — the dip is *model-dependent*, arguing collapse-residual over pure ring
geometry (a geometric finite-size effect would hit all models equally). The
finite-size N-scan (N∈{48,96,192}) settles it in favor of **geometry, not collapse**:
the dip is present only at N=48 — where the r=16 window (w=33) spans 69% of the ring —
and *vanishes* at N=96 and N=192, where D_norm is monotone in r (N=192:
0.18→0.26→0.55→0.62). The underlying monotone rise of D_norm with r is therefore
N-robust (the real damping-length signal); only the small-ring r=16 dip was
ring-geometry. (The absolute D_norm level falls with N — a perturbation reaches a
smaller *fraction* of a larger ring — but the shape is stable.)
(`fig/repair_grid.png`, `fig/repair_scale.png`,
`results/mlm/repair_*.json`, `lyapunov_mlm_*.json`, `repair_fss_tiny.json`)

**F24 — Both load-bearing measurements replicate on an autoregressive model, so
they are not artifacts of the MLM's globally-inconsistent construction (external
validity).** The instrument is ported to Pythia-160m as a one-sided **causal**
window p(x_i | x_{i-r..i-1}) — an order-r Markov approximation, the AR analog of the
MLM's symmetric masked window (`src/ar_ca.py`; null CRN divergence exactly 0). On
this consistent-joint model: (a) **velocity∝r replicates** (v = 5.8, 7.7, 11.5,
11.5 for r = 2,4,8,16, same N/2 saturation as F21; r=1 does not propagate at all),
and (b) the **damping length replicates** — D_norm rises 0.001 (r=1, fully damps) →
0.98 (r≥4, fully decorrelates), the same climb as the MLM. Because the MLM joint is
globally inconsistent (2605.16378) while the AR joint is consistent, the phenomena
being construction-independent is the single most important external-validity check.
We hold the MLM↔AR comparison at the level of the **trend**, not absolute numbers:
AR healing is causal-context (one-sided), a different object from bidirectional MLM
healing. **The capacity→sensitivity effect does NOT robustly replicate on the AR
construction.** Firmed up with four Pythia sizes and 5 seeds each (mean D_norm at r=2,
±SE): **70m 0.41±0.08 ≈ 160m 0.41±0.08 < 410m 0.56±0.06 > 1b 0.47±0.08** — it is
*non-monotone*, peaking at 410m and *dropping* at 1b. The size-rank correlation is
insignificant (**Spearman ρ=0.17, p=0.29** over 40 points); only the 160m→410m step is
even marginal (gap +0.15, p=0.06), and 1b→410m reverses (p=0.86). So the masked-side
capacity result (F23, tiny≪{mini,base}) does **not** carry over to the AR causal window
as a monotone scaling law. Honest reading: the capacity→sensitivity effect is a jump at
*some intermediate scale* that neither construction extends monotonically — the masked
side saturates (mini≈base), the AR side reverses (410m>1b). The *instrument* replicates
on AR (velocity∝r, damping length rises with r); the *capacity-scaling* of the damping
length is masked-specific in this data, not construction-independent. This is the check
catching an overclaim: the earlier 2-seed "160m→410m climb" did not survive a 4th size.
(`fig/ar_capacity.png`, `results/mlm/ar_capacity.json`)

**Crossover-relative robustness (is the plateau/non-monotonicity a fixed-(r,T) sampling
artifact?).** A fixed probe could sample different points of each model's phase diagram, so
we swept T at r=2 and located each model's heal→spread crossover (`fig/crossover.png`,
`results/mlm/crossover.json`). It cuts both ways. **Masked:** the D_norm(T) profiles are
cleanly ordered **tiny < mini < base at *every* T** — base's whole profile is shifted up (it
amplifies at lower T) — so the mini≈base "plateau" (F23) was an artifact of averaging over
*large* radii where both saturate; **at the discriminating radius r=2 the masked capacity
effect is monotone and real.** **Autoregressive:** the ordering does *not* hold — Pythia-410m
is an outlier (profile shifted up, lowest crossover) while 70m≈160m≈1b — so the **AR
non-monotonicity is genuine, not a sampling artifact.** Caveat: the extracted crossover
temperature is noisy in the high-variance low-T healing regime (bert-tiny's profile is
non-monotone there), so we compare *profiles at fixed T*, not the crossing. Net:
crossover-relative probing *strengthens* the masked capacity effect (monotone at r=2) and
*confirms* the AR non-replication (410m genuinely anomalous).
*Trim logged*: the 410m r=16 cells were dropped — 410m fp16 + the largest window
OOM'd on 16 GB; velocity and the D-grid through r=8 (the discriminating range)
completed. (`results/mlm/ar_pythia-160m.json`, `ar_pythia-410m.json`)

**F25 — The damping length has a non-monotone *developmental* trajectory: chaotic
init → early ordering collapse → edge-of-chaos climb; and structure crystallizes
before the sensitivity does (the real-training echo of F7).** Running the instrument
across Pythia-160m's public training checkpoints (`step0/512/4000/32000/143000`),
`D_norm` at the discriminating radius r=2 traces a U in *training time*: the
untrained network is maximally chaotic (`D_norm≈0.99` — a random net amplifies every
perturbation, sitting above the edge of chaos); early training **collapses it to an
ordered/contractive minimum** (`0.27` at step 512, the strongest damping); then
sensitivity **climbs monotonically back toward the edge** as capability grows
(`0.27→0.55→0.65→0.71`). This is the *training-time analog of F23's capacity→sensitivity
size climb*: on both the size axis and the training axis, more capable models sit
nearer the chaotic side. Order of acquisition: the order parameter (bigram overlap vs
the proxy) forms **early** — it is already 0.16 by step 4000 while `D_norm` is still
climbing — so local structure crystallizes *before* the dynamical sensitivity settles,
the real-model echo of the toy's F7/F19 early crystallization. Caveats: the proxy
top-50 census is too weak here to add signal (overlap 0; only ρ moves, 0.15→0.18); r=4
is already saturated (`D_norm≈0.99`) so r=2 is the informative radius; this is one
model on five checkpoints. (`fig/pythia_dev.png`, `results/mlm/pythia_dev.json`)

> Novelty TODO before submission (from the check): direct-read arXiv:2607.09803 and
> QUIVER; keep the instrument (not "dynamical-systems analysis of LLMs") as the
> claimed novel core.

## Adversarial audit (pre-NeurIPS) — retractions and demotions

Two independent adversarial reviews (originality + rigor lenses) were run against
the draft and the raw JSON; both recommend Reject (4/10, 3/10). Every rigor
objection below was re-verified against the JSON and **confirmed**. See
`paper/REVIEW.md` and repository issues #1–#12. The paper (`paper/paper.tex`) and
this log now reflect the corrected claims.

- **A1 — Capacity→sensitivity is 2-seed and its significance was pseudoreplicated.**
  `repair_{tiny,mini,base}.json` all use `seeds:[21,22]` (n=2). The "15/15 cells,
  p<10⁻⁴" signed-rank is over a smooth correlated (r,T) grid from the same 2 seeds
  → effective n=2; p<10⁻⁴ merely restates a shared sign. **Retracted as a
  significance claim.** Defensible residue: tiny ≪ {mini, base} (base-vs-mini null,
  8/15, p=0.21). Demoted from "capacity axis" to "suggestive gap." (F23)
- **A2 — λ "model-invariant" / kinematics⊥stability is cherry-picked.** The
  0.745/0.738/0.767 (~4% spread) are λ_max, all at the single saturated cell
  (r=8,T=0.9). Off-cell λ tracks capacity: (r=8,T=0.7) 0.50/0.60/0.66 = 24% spread;
  (r=1,T=0.7) reversed, 46% spread. **Orthogonality decomposition withdrawn.**
- **A3 — D_norm large-r rise is denominator-driven.** tiny T=0.7: raw D falls
  0.537→0.496→0.391 (r=4→16) while D_norm rises 0.748→0.803→0.810 because D0 falls
  faster. "Damping shrinks with r" partly = floor collapse. Also numerator (CRN) and
  denominator (independent noise) use different couplings; D_norm>1 is ~1σ from the
  saturation value 1. Now stated as a controlled *diagnostic*, not a clean ratio.
- **A4 — v∝r overclaimed.** The N-scan "lifts" 11.5→23.5→47.5 at r=16 are exactly
  N/4 (clipping ceilings, saturate_sweep=2). The one unclipped point (N=384) = 41.1
  < 47.5; unclipped points superlinear (1.5r→2.6r). Reframed as "velocity grows
  monotonically with r, model-invariant" — no clean proportional law claimed. (F21)
- **A5 — Crossover "rescue" false at T=0.3.** "tiny<mini<base at every T" is wrong
  (T=0.3: mini 0.463 < tiny 0.508); profiles single-seed. Downgraded to a plateau
  *diagnostic*, not a confirmation of a monotone axis. (crossover.json)
- **A6 — AR "consistent joint" overstated + bimodal pooling.** `ar_ca.py` is a
  truncated order-r causal kernel resampled in-place on a ring — not the model's true
  AR joint; both constructions are non-samplers. AR "5 seeds" = n=10 pooled across
  two T regimes (bimodal) → ill-defined mean/SE; per-T reporting is the fix. (F24)
- **A7 — Census recovery real only on the toy.** Quantitative recovery (self-TV 0.22
  vs cross 0.95) is synthetic-only; real-model overlap is near-floor (0.02–0.04 vs
  WikiText, out-of-training); tiny_census deepest attractors are degenerate fragments.
- **Path to significance (flagship, issue #4):** cross-level check — does the
  black-box token-space damping length agree with a white-box activation-space
  criticality measure (Jacobian/Lyapunov; SPARC ρ(F_T)≥1) across a Pythia ladder,
  at the seed-level standard of A1. This is the missing external-validity result.
  **Run (F26 below).**

## Phase D findings — cross-level validation (issue #4)

### F26 — token-space vs activation-space criticality: matched-type is suggestive, mismatched is null
Six-model Pythia ladder (14M/31M/70M/160M/410M/1B). WHITE-BOX: finite-depth
top-Lyapunov λ_top = (1/L)·log ρ(J_{emb→h_L}), depth-normalized log spectral radius
of the embedding→final-hidden Jacobian (= SPARC's ρ(F_T); finite-difference-JVP
power iteration, fp32). BLACK-BOX (same CA): asymptotic damping length D_norm and
the token-space finite-size Lyapunov λ_ca (early damage-growth slope), r=2, T=0.7,
5/2 seeds. (`results/mlm/crosslevel.json`, `fig/crosslevel.png`,
`experiments/crosslevel.py` + `crosslevel_lyap.py`.)

- **A clean white-box scaling law:** λ_top falls monotonically with size (1.106,
  0.925, 0.771, 0.312, 0.187, 0.235 for 14M→1B; Spearman ρ=−0.94, **p=0.005**) —
  larger models sit closer to critical, approaching λ_top=0 from the supercritical
  (expanding) side. Stands on its own, independent of the cross-level question.
- **Mismatched pairing is null.** White λ_top vs black **D_norm** (a growth rate vs
  an asymptotic *persistence*): Pearson r=−0.29, p=0.57; Spearman ρ=−0.49, p=0.33.
  D_norm is flat across size (ρ=0.31, p=0.54) — it does not scale, so it cannot
  track a scaling white-box quantity. Expected once framed by type.
- **Matched pairing is suggestive but underpowered.** White λ_top vs black **λ_ca**
  (Lyapunov vs Lyapunov): **Pearson r=+0.71 (p=0.11), Spearman ρ=+0.60 (p=0.21)** —
  right sign, moderate-strong effect, a clean flip from the mismatched pairing, but
  **not significant at n=6**, and leveraged by the two smallest models. Reported as
  *suggestive, not established*. (λ_ca_max is rail-bound flat ~0.90 at the grid edge
  r=8,T=0.9 for all models — apparatus-saturated, its correlation is noise; λ_ca at
  the informative r=2 cell carries the effect.)
- **Reading:** token-space and activation-space criticality appear to share a
  *growth-rate* axis but not a *persistence* axis — enough to motivate, not certify,
  black-box λ_ca as a weights-free proxy for internal criticality.
- **Path to significance (not yet run):** longer ladder + second family (GPT-2) for
  more points, and — most powerfully — a *within-model* design correlating λ_ca(T)
  against a temperature-dependent white-box ρ(F_T) over a swept T, giving many
  matched points per model instead of n=6 cross-model scalars.

### F27 — ground-truth calibration: λ_ca recovers the known criticality ordering of classical CA rules (issue #14)
Drove the SAME damage-spreading + Lyapunov estimator (async updates, CRN twins,
single-site flip, `lyap_from_cone`) with classical Elementary CA rules of known
class (k=2, radius 1), same protocol as the instrument. Group-mean λ_ca: ordered
{128,232,4} = **−0.315** < edge/complex {110,54} = **+0.185** < chaotic {150,30,22}
= **+0.256**. **Pre-registered ordering ordered<edge<chaotic RECOVERED.** This is the
criticality-side analog of the census calibration (which recovers a known transition
matrix): it validates that black-box λ_ca measures *criticality*, not an apparatus
artifact. **[RETRACTED — see F34.** This entry originally claimed that linear Rule 90
"correctly reads marginal despite wide spread", i.e. that λ_ca separates exponential growth
from mere ballistic spreading. That reading was an **ignition-averaging artifact**: 25% of
Rule 90 runs never ignite, and averaging them with the 75% that do produced the ≈0 value.
Conditional on ignition Rule 90 grows at **+0.276**, like a chaotic rule. The claimed
velocity⊥Lyapunov separation is **not** demonstrated.**] Individual-rule λ_ca is noisy (3 seeds); the class-level ordering is the
robust claim. (`results/eca_calib.json`, `experiments/eca_calib.py`.)

### F28 — the cross-level proxy does NOT hold: an honest negative (issues #4, #5, #15)
Full investigation across two families, two within-model axes, and a ground-truth
calibration. Verdict: **the black-box token-space criticality instrument does not
robustly proxy white-box activation-space criticality.**
- **Second family (GPT-2 small→xl, issue #5).** The suggestive Pythia cross-model
  correlation (white λ_top vs black λ_ca, r=+0.71, F26) does **not** replicate: GPT-2
  gives r=−0.43. Pooling the two families gives r=+0.70, p=0.025 — but that is a
  **Simpson's-paradox pooling artifact** (the families disagree), NOT significance.
  What *does* replicate cross-family: the standalone white-box scaling law (λ_top
  falls monotonically with size — Pythia ρ=−0.94; GPT-2 0.152→0.059).
- **Within-model across T (issue #4).** Correlating λ_ca(T) vs white ρ(F_T) over
  swept T gives a uniform r≈−0.9 for **all six models, both families** (pooled −0.90,
  p=4e-16). This is a **mechanical confound**, not signal: raise T and the CA explores
  more (λ_ca↑) while the softmax flattens (ρ(F_T)↓), so they anti-correlate by
  construction. The uniformity across wildly different models is the tell.
  (`crosslevel_within.json`.)
- **Within-model across r, de-confounded (issue #15).** At fixed T=0.7, correlating
  λ_ca(r) vs white ρ(F_r) over swept radius gives per-model r=[0.16,−0.08,−0.55,0.73,
  −0.37,0.28] — median +0.04, no consistent sign → **null**. Reason: **λ_ca(r) is
  model-invariant** (≈[−1.13, 0.15, 0.49, 0.89] for r=1,2,4,8 across all six models,
  both families) — its radius-dependence is pure light-cone kinematics, carrying no
  model signal, while white ρ(F_r) varies by model. (`crosslevel_radius.json`.)
- **Reading:** token-space and activation-space criticality are **distinct,
  non-proxying levels** in trained LMs. The instrument measures a real, ECA-calibrated
  (F27) token-space criticality, but it is not a weights-free proxy for the
  activation-space Jacobian. A clean negative that kills a natural hypothesis.

### F29 — mining for a positive cross-level proxy: exhausted; the negative is structural (issue #4)
Following F28, we mined the principled angles for a positive black-box→white-box
proxy. All negative, with a **unifying mechanistic cause: the white-box depth-Lyapunov
λ_top is an architectural quantity** (flat across training, ≈1/L across models), so it
cannot proxy the learned token dynamics.
- **Developmental** (`crosslevel_dev.json`): across 10 Pythia-410m checkpoints
  (step256→143000), white λ_top is nearly flat (~0.13–0.19; vs-step ρ=−0.15) while
  black λ_ca / D_norm undergo a clean order→chaos phase transition around step~1000
  (λ_ca −0.08→+0.19; D_norm 0.10→0.71). They do not co-evolve (r=−0.21; partial|step
  −0.48, n.s.). White λ_top is set by architecture, not learning. (Silver lining: the
  black-box developmental phase transition at step~1000 is a real standalone signal.)
- **Masked ladder** (`masked_ladder.json`): the one strong signal — 6 BERT depths
  (L=2..24), white λ_top vs black D_norm r=−0.92, p=0.009 — is **depth-mediated**:
  white-vs-depth r=−0.92, black-vs-depth r=+0.76. The two L=4 models (mini, small) are
  the tell: white λ_top ≈ identical (0.965, 0.978) while D_norm differs (0.786, 0.886)
  — at fixed depth white does NOT track black. The "surviving" partial correlation
  (r=−0.90) is a linear-fit artifact on a 1/L relation. Depth is public → useless as a
  weights-free proxy. (Initial n=3 in `masked_crosslevel.json`.)
- **Conclusion:** no *useful* weights-free proxy for internal criticality exists with
  these measures. The negative is structural, not a measurement failure: white-box
  depth-Lyapunov is architectural (≈1/L); black-box λ_ca is either kinematic (AR) or
  reflects learned dynamics decoupled from λ_top. Further measure-swapping would be
  p-hacking. This is the honest, mechanistically-explained answer to the
  external-significance question the instrument raises.

## Phase 0 findings — correctness audit of the validation ladder

### F30 — the logistic-map "validation" was CIRCULAR; demoted to a smooth-limit unit test
**Self-verified before acting** (external audit of `dd302df` flagged it; confirmed here):
in `results/reproduce_lyapunov.json` the `crn` array is **bit-identical to `exact` at all
61 points** (0 differing elements, max|diff| = 0.0). Cause, in `lyap_crn`
(now `lyap_tangent_fd`): `xp = x + d0*np.sign(xp - x)` re-anchors the twin to the
**reference orbit every step**, along the same orbit/seed/burn-in as `lyap_exact`. Hence
`log(d/d0) = log|f'(x)| + O(d0)` — the estimator **is** a finite-difference evaluation of
the analytic derivative it was being compared against. It reproduced the bifurcation
diagram because it *is* the bifurcation diagram.
- **Independently reproduced the O(d0) scaling** (`results/logistic_epsilon_sweep.json`,
  `fig/logistic_epsilon.png`): mean|err| = 0.000000 (d0=1e-9), 0.000003 (1e-6), 0.003351
  (1e-3), 0.039985 (1e-2), 0.256042 (1e-1); **log-log slope 0.79** ⇒ error is O(d0).
  Perfect agreement is a property of the infinitesimal limit and nothing else.
- **A token flip is O(1) in a discrete alphabet — there is no ε→0 limit in token space.**
  So this rung certified exactly the one regime the instrument can never occupy.
- **The docstring claim "the SAME primitive the LM instrument uses" was false at code
  level.** `reproduce_lyapunov.py` imports only `pathlib/json/numpy`; the LM estimator is
  `lyap_from_cone` (`experiments/lyapunov.py:22`). Zero shared code, and structurally
  different estimators (renormalized tangent growth vs a windowed polyfit on a saturating
  discrete damage count).
- **The instrument's actual regime, measured** (`--finite-perturbation`,
  `results/logistic_finite_perturbation.json`): no renormalization, finite d0 →
  mean|err| = 0.147 (d0=1e-6), 0.161 (1e-3), 0.200 (1e-2), 0.308 (1e-1). The bias
  **does not vanish as d0→0** (≈0.15 floor at d0=1e-9 too): it comes from the windowed fit
  on a *saturating* separation, not from perturbation size. This is the honest analogue of
  what `lyap_from_cone` does on a discrete damage count.
- **Action:** `lyap_crn` → `lyap_tangent_fd` with an honest docstring; the logistic and CML
  rungs are relabelled **smooth-limit arithmetic unit tests**, not validation of the
  instrument, in `paper/paper.tex`, `README.md`, and here. The **weight-bearing rungs are
  the ECA class recovery (F27) and the census** — both discrete, finite-perturbation,
  no renormalization. New JSONs written to new files; prior results untouched (audit trail).

### F31 — repo-wide hunt for the same failure mode: one more hit, the LM path is clean
Grepped every twin/perturbed/damaged trajectory for renormalization, re-anchoring, or
resetting to the reference, and every comparison sharing a seed/orbit/burn-in.
**Listed, not fixed** (fixes deferred to Phase 2).

**CIRCULAR — same failure mode as F30 (1 hit):**
- `cml_lyap` (`reproduce_lyapunov.py:144`): `xp = x + (xp-x)*(d0/(d+1e-300))` rescales the
  twin every step ⇒ tangent-space (infinitesimal) estimate, not finite damage spreading.
  The CML ladder rung inherits F30's caveat exactly. Its ε>0 values additionally have **no
  ground truth** (only ε=0 ↔ ln2). *Fix owed: Phase 2.3 Benettin/QR reference.*

**GENUINE — finite perturbation, twins never re-anchored (4):**
- `mlm_damage.block_damage`: twins = `base.copy()` + block flip, evolved under a **shared
  tiled uniform stream**, never re-anchored; damage = Hamming diff of snapshots. ✅
- `ar_probe.block_damage`: identical structure. ✅
- `eca_calib.damage_cone` (**the F27 weight-bearing rung**): single-site flip, shared visit
  order + shared per-site uniforms, no renormalization. ✅
- `differential.coupled`: shared init + shared stream; null arm identical ⇒ exactly 0. ✅

**Shared seed / burn-in — intentional and correct, not circular (2):**
- `block_damage` and `drift_floor` both derive `base` from `run(..., seed=seed)`: the *same*
  settled state. That is matched-pairs design (D and D0 measured from one starting state),
  which is what you want, not a leak.
- Known coupling mismatch (audit W2) persists: D uses **shared** uniforms (CRN), D0 uses
  **independent** uniforms. Not circular, but D/D0 is a ratio across two couplings — and
  Phase 2.2 (Domany–Kinzel) is precisely the experiment that shows the damage boundary is a
  property of *(model, coupling)*.

**New mechanism for the cross-level negative (F28/F29) — regime mismatch, not just level:**
- `crosslevel.white_box` λ_top uses power iteration with `v ← w/‖w‖` each step. This is
  **not** circular (power iteration is the standard spectral-radius algorithm and is not
  compared against its own derivative) — but it makes λ_top an **infinitesimal /
  tangent-space** quantity, while λ_ca is a **finite, O(1), un-renormalized** one. F30's
  ε-sweep shows these two regimes disagree systematically (≈0.15 nats/step bias floor)
  *even in a system where both are well-defined and ground truth is known*. So the
  cross-level null had a second, deeper cause than "λ_top is architectural": the two
  quantities were never in the same regime. This strengthens the negative and should be
  stated in the paper.

### F32 — the F27 ordering is robust to the estimator's branch constants, but NOT to an arbitrary fixed window
`lyap_from_cone`'s four magic constants are now named kwargs (`sat_threshold=3.5`,
`frac_of_max=0.5`, `max_sweeps=8`, `min_sweeps=3`) plus a `fit_window=(start,end)` that
bypasses the data-dependent branch. Sensitivity run over 5 seeds × 8 rules, simulation
cached so the sweep isolates the *estimator* (`results/lyap_fit_sensitivity.json`).

- **Adaptive branch: ordering recovered in 54/54 parameter settings** (all combinations of
  sat_threshold ∈ {2.5,3.5,5.0} × frac_of_max ∈ {0.3,0.5,0.7} × max_sweeps ∈ {5,8,12} ×
  min_sweeps ∈ {2,3}). ordered<edge, ordered<chaotic and edge<chaotic each hold in every
  setting; the edge→chaotic margin ranges +0.023 … +0.324. So the branch constants are
  **not** a garden-of-forking-paths risk for F27.
- **Fixed windows are a different story.** `ordered < {edge, chaotic}` survives all four
  tested windows, but **edge < chaotic INVERTS for any window extending past ~3 sweeps**:
  (0,3) ord −1.451 < edge +0.235 < chaos +0.249 ✓; (0,5) edge +0.172 > chaos +0.137 ✗;
  (0,8) +0.125 > +0.078 ✗; (1,6) +0.093 > +0.028 ✗.
- **Mechanism:** chaotic rules saturate *fast*, so a long fixed window averages the
  post-saturation plateau into the slope and deflates their exponent; edge rules grow
  slowly and keep growing through the window. The adaptive rule avoids this precisely
  because it ends the window at a fraction of the run's own maximum.
- **Consequence for Phase 2.1 (plan change).** A single global pre-registered fixed window
  is the **wrong** pre-registration for this estimator: different classes saturate at
  different sweeps, so any one window privileges one class. The defensible pre-registration
  is a *saturation-relative* window with constants fixed in advance (demonstrably robust,
  54/54), or a short fixed window (≤3 sweeps) selected on held-out rules.
- **Honest residue:** the coarse claim (`ordered` ≪ everything else) is solid; the
  **edge-vs-chaotic separation is the fragile comparison** and should be reported with the
  window rule stated, not as a bare ordering. Rule 90 (linear) reads λ=−0.186 at 5 seeds
  with dmax_frac=0.266 — which at the time appeared to confirm the F27 nuance, but is an
  ignition artifact (see F34); the F27 nuance is retracted.

### F33 — hardened ECA rung (19 rules x 12 seeds): the 3-class ordering does NOT survive; ordered-vs-rest does
Phase 2.1. Pre-registered before running: Wolfram (1984, Physica D 10:1-35) class
assignment; the **saturation-relative** fit window with `lyap_from_cone` defaults frozen in
advance (chosen over a fixed window because F32 showed a fixed window inverts edge-vs-chaotic
3 times out of 4); primary claim `ordered < chaotic`, secondary `ordered < edge < chaotic`.
Rule 90 held out as a linear reference. (`results/eca_calib_hardened.json`.)

- **PRIMARY CONFIRMED, decisively.** ordered −0.663 [CI −0.921,−0.400] < chaotic +0.240
  [+0.055,+0.370], bootstrap p=0.0000 (rule as the unit of analysis, not seed —
  avoiding the pseudoreplication of W1). ordered < edge also p=0.0000.
- **SECONDARY DEMOTED.** edge +0.143 vs chaotic +0.240: **p=0.167, not significant.**
  F27's headline "ordered < edge < chaotic" 3-class ordering **does not survive** 12 seeds
  and rule-level bootstrap. Only the coarse `ordered ≪ {edge, chaotic}` separation is
  supported. Dropping the two disputed rules (106, 62) does not rescue it (edge +0.154).
  This is the demotion F32 predicted from the window-sensitivity analysis.
- **Rule 90 (linear reference) reads λ=−0.023, CI [−0.330,+0.236]** — which looked like
  the F27 nuance surviving. It does not: the wide CI was the tell, and F34 shows the value
  is an ignition-averaging artifact.
- **New concern, reported not hidden: Rule 30 reads λ=−0.243, CI [−0.850,+0.195]** — the
  canonical chaotic rule reads *negative* with a very wide interval, i.e. its damage
  measurement is unstable across seeds. It was +0.234 at 3 seeds (F27). This inflates the
  chaotic group's variance and is part of why edge-vs-chaotic fails. Worth diagnosing
  before the rung is used as evidence for anything finer than ordered-vs-rest.
- Ordered rules 0/8/32/128/160 all pin at exactly −0.921 with zero-width CI: damage dies
  immediately, so the estimator is at its floor for them (not a meaningful spread).

### F34 — the ECA rung had an ignition confound; Rule 30 explained, the Rule 90 nuance RETRACTED
`eca_calib.damage_cone` averaged the damage cone over all B lattices *before* fitting λ.
Single-site damage in a discrete CA is **bimodal** — it ignites or dies — so that average
measures the mixture, not a growth rate. This is precisely the lesson the LM path already
learned (F8: "ignition is a rare event and bimodal … must report ignition probability
separately from spread"; F13, via `block_damage`'s `ignition_prob`/`cond_spread`); the ECA
rung never received it. Fixed in `experiments/eca_calib_ignition.py`
(`results/eca_calib_ignition.json`, 19 rules × 12 seeds).

| rule | P(ignite) | λ (averaged, old) | λ \| ignited |
|---|---:|---:|---:|
| 30 (canonical chaotic) | 0.209 | −0.243 | **+0.447** |
| 90 (linear reference) | 0.750 | −0.023 | **+0.276** |

- **Rule 30 resolved.** Its negative reading (F33) was entirely the confound: only 21% of
  lattices ignite, and the extinguished 79% dragged the mean below zero. Conditional on
  ignition it is strongly chaotic.
- **RETRACTION (F27's showcased nuance).** Rule 90 read ≈0 only because 25% of its runs
  never ignited. Conditional on ignition it grows at +0.276 — indistinguishable from a
  chaotic rule. **λ_ca does not demonstrate a separation of exponential growth from
  ballistic spreading.** Removed from `paper/paper.tex`, this log, and `README.md`.
- **What the rung actually discriminates: ignition probability**, not λ — ordered **0.046**
  (range 0.000–0.178) vs edge **0.668** vs chaotic **0.682**. Ordered rules 0/8/32/128/160
  have P(ignite) = 0.000 exactly.
- Conditional on ignition, edge-vs-chaotic remains non-significant (p=0.067, vs 0.167
  unconditional), consistent with F33: only the coarse ordered-vs-rest split is supported.
- **Caveat, recorded not hidden:** the script's `ordered < chaotic p=0.0000` are
  **NaN-comparison artifacts** (λ|ignited is undefined for rules that never ignite) and must
  not be quoted. Ordered-vs-rest should be tested on ignition probability instead.

**Framing correction (the mixture is a composite, not merely an artifact).** An earlier draft
of this entry called the ignition-averaged λ an "artifact". That is too strong, and the data
say so: λ_all separated ordered from chaotic at **p<10⁻⁴** (F33), whereas λ|ignited is
*undefined* for the ordered group (those rules never ignite). For the one comparison that
survives, the mixture was the **more** usable statistic. The error was never that λ_all fails
to discriminate — it is that we **labelled a composite with a component's name** and read it
as a growth rate, so Rule 30's −0.243 was interpreted as "not chaotic" when it is strongly
chaotic in the 21% of runs where damage ignites.
- Two real residues of that mislabelling: the mixture is **not injective** (rule 90 at
  P=0.75, λ|ign=+0.28 and rule 4 at P=0.15, λ|ign=+0.05 both land near −0.02 from entirely
  different dynamics), and its weight is **apparatus-sensitive** (it moves with N, sweeps and
  the ignition threshold — rule 30 went +0.234 at 3 seeds to −0.243 at 12).
- **Resolution: report the pair (P_ignite, λ|ignited).** It is strictly more information than
  the mixture, which is a lossy projection of it; it stays interpretable; and it retains the
  discriminating power, since P_ignite alone separates ordered 0.046 from edge 0.668 /
  chaotic 0.682. Decomposing did not discard the discriminator — it located it.
- **P_ignite is the physically right object, not a nuisance.** Damage spreading in a discrete
  CA is directed-percolation-class, and DP transitions are characterised by *survival
  probability*. This is the same quantity whose boundary the Domany–Kinzel rung (Phase 2.2)
  is meant to reproduce, so the ECA and DK rungs should report a common order parameter.

## Caveats

Several pilot caveats are now *addressed*: seeds are swept (F11, ≥5), the `<unk>`
census bias is removed (F10, BPE), damage uses block flips with ignition
probability (F13), the "transition" is re-characterized as a crossover (F12), and
the instrument is validated on real pretrained MLMs (F14–F19).

Remaining limitations. **General:** MLM local conditionals are globally
inconsistent (proven for real MLMs in 2605.16378), so the CA is a well-defined
stochastic dynamical system but *not* an exact sampler of any joint distribution —
all findings are phrased as properties of the dynamics, not of a sampled
distribution. **Toy:** single small corpus, N≤192, radius-windowed conditionals.
**Phase 3:** census is *proxy* (WikiText-103, which BERT was not trained on) so
overlap/ρ are lower bounds; the k-gram overlap metric conflates corpus-consistent
long-range structure with repetitive attractor loops (F15); MLM sweeps used a
single seed and N=48; bert-base ran at reduced settings (sweeps 25–30, B=12–24) for
tractability; the finite-size crossover result (F12) was established on the toy
only — it is not yet checked on the real models.

### F35 — real generation does NOT absorb a single-token error: the ring CA's "healing" is a property of the CONSTRUCTION
The instrument's load-bearing assumption is that its error-propagation numbers say something
about the *model*. Every number so far is measured on a ring CA whose stationary measure is
not the model's generative distribution. This tests the assumption directly, on real
autoregressive generation, with the same CRN discipline
(`experiments/real_generation_damage.py`, `real_generation_reconvergence.py`).

**Certification first:** in both experiments the null arm (no injection, shared uniform
stream) diverges by **exactly zero** — asserted in code, not assumed. So the protocol
transfers to free generation and the numbers mean what they say.

**Two independent readouts agree that absorption is zero.** Token identity, all three
models, 4 prompts × 8 seeds = 32 trials each:

| model | P_persist | P_reconverge | divergence | null |
|---|---:|---:|---:|---:|
| pythia-70m | **1.000** | 0.000 | 0.945 | 0.0 |
| pythia-160m | **1.000** | 0.000 | 0.958 | 0.0 |
| pythia-410m | **1.000** | 0.000 | 0.951 | 0.0 |

The error is *never* absorbed, at any scale tested, and the effect does not weaken with model
size. **Distributional readout** (the fairer test, since after an injection the sequences are
no longer positionally aligned): TV between the twins' next-token distributions normalised by
the TV between two *independent* continuations of the same prompt —
**TV_norm = 0.967** on pythia-70m (TV_damage 0.941 vs floor 0.973). The twins end up ~97% of
the way to complete decorrelation. So the saturation is not a harsh-metric artifact: there is
no meaningful reconvergence in distribution either. (The residual 3% is the only trace of
coupling that survives, and it is not a recovery signal.)

**Mechanism (structural, not empirical accident).** Free generation **never resamples a
token**: once the wrong token is in the context it stays there permanently, and the two
continuations are simply different samples thereafter. The ring CA is the opposite — every
site is revisited repeatedly, so a perturbed site *can* be overwritten back. **In-place
resampling is what creates the possibility of healing, and it is exactly what real
generation lacks.**

**Consequence for the paper (this reframes it).** The damping length / repair length /
`D_norm` measure a property of the **iterated-resampling construction**, not of the model's
generative process. This is not a new failure — it retroactively explains the whole pattern:
the light cone is kinematic (F16/F21), λ_ca(r) is model-invariant (F28), and the white-box
proxy failed structurally (F29/F31). The instrument characterises *(model, construction)*,
with the construction carrying more of the signal than the paper has been claiming.

**Status: established on the token-identity readout** (3/3 models, 96 trials, all nulls
exactly zero, no size dependence) and **corroborated distributionally** on pythia-70m
(TV_norm 0.967); the 160m/410m distributional arms are still running. The mechanism is
structural rather than statistical, which is why three models suffice. The honest framing for the paper is a precise
statement of what the instrument does and does not measure, made *before* a reviewer made it.

### F36 — the ECA class separation, tested on the right statistic (issue #24)
Re-runs the class test on **ignition probability** instead of λ, per F34. This replaces the
λ-based ordered-group p-values, which were NaN-comparison artifacts (λ|ignited is undefined
for rules that never ignite) and could not be quoted. Rule is the unit of analysis.
(`results/eca_ordered_vs_rest.json`.)

| class | P_ignite | CI95 | rules |
|---|---:|---|---:|
| ordered | **0.046** | [0.000, 0.102] | 7 |
| edge | 0.668 | [0.392, 0.944] | 4 |
| chaotic | 0.682 | [0.486, 0.867] | 7 |

- **Primary claim, now on a valid statistic: ordered < rest, p=0.0000, Cohen's d = 3.03.**
  Non-overlapping CIs and a very large effect. This is a *stronger* result than the λ-based
  version and, unlike it, is not an artifact.
- **Edge vs chaotic is definitively dead: p=0.470** — a coin flip. Under λ it looked
  marginal (p=0.067 conditional, 0.167 unconditional); on the correct statistic there is no
  separation at all. The three-class ordering is not recoverable by this instrument.
- Reference rule 90 sits at P_ignite=0.750, inside the chaotic range — consistent with F34's
  retraction of its "marginal" reading.
- **Net:** the ECA rung supports exactly one claim — *the instrument separates rules whose
  damage dies from rules whose damage survives* — and supports it decisively. That is the
  DP-class survival transition, and it is the order parameter the DK rung (#22) should share.

### F37 — the CML rung's eps>0 values are correct, but the rung stays a smooth-limit check (issue #23)
The coupled-map-lattice rung had ground truth only at eps=0; its eps>0 values were compared
against nothing. Computed the maximal Lyapunov exponent by the standard **Benettin** method
— tangent vector evolved under the *exact analytic Jacobian* `C diag(f'(x))`, renormalized
each step — as a genuine reference. (`results/cml_benettin.json`, 5 seeds, 20k steps.)

| eps | Benettin (exact J) | `cml_lyap` (finite-diff) | \|diff\| |
|---|---:|---:|---:|
| 0.0 | 0.6932 ± 0.0000 | 0.6936 ± 0.0007 | 0.0003 |
| 0.1 | 0.4499 ± 0.0008 | 0.4490 ± 0.0013 | 0.0009 |
| 0.2 | 0.3671 ± 0.0008 | 0.3674 ± 0.0041 | 0.0004 |
| 0.3 | 0.3636 ± 0.0009 | 0.3648 ± 0.0023 | 0.0011 |
| 0.4 | 0.3691 ± 0.0009 | 0.3683 ± 0.0011 | 0.0007 |

- **Implementation self-check:** at eps=0 the lattice decouples and Benettin returns
  0.6932 against the analytic ln2 = 0.6931 (error 1e-4), so the reference is trustworthy
  before it is used to judge anything.
- **`cml_lyap` is correct: max |Benettin − cml_lyap| = 0.0011** across the whole eps range.
  The gap flagged in F31 was that the values were unverified, not that they were wrong.
- The exponent is **non-monotone in coupling** (0.693 → 0.450 → 0.367 → 0.364 → 0.369,
  minimum near eps≈0.3), which the earlier "known monotone reduction with coupling" phrasing
  in the paper got wrong; it decreases then turns back up.
- **What this does NOT do.** Benettin is itself a *tangent-space* computation, so the
  agreement is two infinitesimal methods agreeing with each other. It does **not**
  rehabilitate the CML as a validation of the token instrument, which lives in the finite,
  discrete regime (F30/F31/F35). The rung stays labelled a smooth-limit arithmetic check —
  it is now a *verified* one.

### F38 — the Domany–Kinzel rung passes, and its exact anchor holds bit-for-bit (issue #22)

`src/dk.py`, `experiments/dk_calib.py` → `results/dk_calib.json`, `fig/dk_ladder.png`;
`tests/test_dk_damage_identity.py` (26 tests). Design rationale in the literature check below.

DK is the **only rung in the regime the instrument actually lives in** — stochastic *and*
discrete. The logistic map and the CML are smooth and infinitesimal (F30/F31/F37); the ECA
rung is discrete but deterministic. It is also the only rung whose damage behaviour has a
published boundary rather than an analogy.

- **The exact identity holds.** On the `p2=0` line the CRN damage field is itself a DK
  automaton at the same `p1` (Kohring–Schreckenberg). Predicted by an independent
  simulator and compared cell by cell: **0 mismatching cells** at p1 ∈ {0.2, 0.5, 0.75,
  0.8087, 0.95, 1.0}, ring 4096, 1500 steps, 3 seeds each — and **16 mismatches** in the
  off-line control at (0.6, 0.5), so the test is not vacuous.
- **Why this matters more than a critical point.** The identity is run *through*
  `lattice.run` in the test file — the same loop that produces every language-model number
  in this project. It verifies the window indexing, the shared-uniform consumption order,
  the inverse-CDF sampling and the synchronous update **bit-exactly, with no error bar**.
  Nothing else in the ladder does that; every other rung agrees to within a fitted constant.
- **Critical points recovered.** Local-slope method (`δ_eff → δ_DP = 0.159464`), 2000 seed
  runs × 512 steps on a ring of 1100:

  | line | estimate | published | off by |
  |---|---|---|---|
  | site DP (`p1=p2`) | 0.7065 | 0.705489(4) | **0.15%** |
  | W18 (`p2=0`), activity | 0.8092 | 0.8087(5) HWD / 0.801(2) Zebende–Penna | **0.06%** / 1.02% |
  | W18 (`p2=0`), damage | 0.8089 | 0.8087(5) HWD / 0.801(2) Zebende–Penna | **0.02%** / 0.98% |

  These are ~1%-accurate calibrations at modest sizes, not measurements of DP exponents,
  and are reported as such.
- **The literature disagreement is NOT resolved by this — correcting an earlier overreach
  in this entry.** The two published `p2=0` values are 0.96% apart, and our own stated
  accuracy is ~1%, so 0.8089 cannot discriminate 0.8087(5) from 0.801(2): both sit inside
  our error bar. The first draft of this finding said the disagreement "resolves toward
  HWD"; that claim exceeded the method's precision and is withdrawn. We report both and
  claim neither.
- **A second exact identity rides inside Part A, free.** At `p1=1, p2=0` the update is
  deterministic XOR — Rule 90 — so a single-seed damage field holds exactly
  `2^popcount(t)` live cells (odd entries in row t of Pascal's triangle). `popcount(1500)
  = 7`, so 128 cells, and `128/4096 = 0.03125` — exactly the reported
  `final_damage_density`. The cone spans 3001 < 4096 cells, so there is no wraparound and
  the infinite-lattice count applies. A different closed form from the one Part A was
  designed to test, and it also matches to the digit.
- **The corollary check passes.** Because the damage field *is* the automaton on `p2=0`,
  the damage and activity transitions must coincide. Measured independently: **|gap| =
  0.0003**.
- **W2 — partially answered, and the first version of this bullet was WRONG.** It claimed
  CRN is HWD's *maximal-correlation* member of the admissible coupling family, hence that
  every damage number here is a lower bound over that family. **That holds only on a binary
  alphabet.** See F41: inverse-CDF sampling from a shared uniform is the *monotone*
  coupling, which coincides with maximal at |V|=2 — which is exactly why this rung is exact
  and unaffected — but not at |V|=30522. The inequality was backwards for the LM backends
  and is retracted there. What survives: the coupling-dependence itself is published
  (Kohring–Schreckenberg 1992; Grassberger 1995), so it is a known property being handled
  rather than a defect discovered, and on DK the bound is real.
- **What this does NOT do.** It validates the *apparatus*, not the LM claims. F35 stands:
  the instrument characterises the iterated-resampling construction, and real AR generation
  does not absorb an injected token error. A correct instrument can still be pointed at a
  process whose numbers do not transfer.

### F43 — three citations were carrying invented titles (Phase 4.3)

Found while cutting the paper: `plainnat` prints `note=` fields, so the compiled bibliography
literally read **"Title/authors to verify"** on five entries. Verified all five against arXiv.
**Three of the placeholder titles were wrong** — not approximations, different papers' worth of
wrong:

| key | title as cited | actual title (verified) |
|---|---|---|
| `critical_temp` | *A Critical Sampling Temperature in Large Language Model Generation* | **Generative Criticality in Large Language Model Temperature Scaling** (Ruan, Li, Guo, Wang 2026) |
| `critical_temp2` | *A Critical Sampling Temperature for Coherent Text Generation* | **Phase Transitions in Large Language Models and the O(N) Model** (Sun & Haghighat 2025) |
| `critical_temp3` | *A Temperature-Driven Phase Transition in LLM Text Generation* | **(Un)biased data and spin glasses reveal clustering for Turing phase transitions within human–transformer interactions** (George, Yusaf, Zoltick, Huynh 2025) |
| `tft` | *Transformer Field Theory* | **Transformer Field Theory: A Response-Theoretic Approach to Mechanistic Interpretability** (Olivieri & Pérez Rodríguez 2026) |
| `residual_dynamics` | *Dynamics of the Transformer Residual Stream* | **…: Coupling Spectral Geometry to Network Topology** (Fernando & Guitchounts 2026) |

The arXiv IDs were right in every case — `results/deep_research_novelty*.md` had recorded the
IDs and what each paper *does*, but the titles had been written from the descriptions rather
than read off the source, and then never reconciled.

**A claim moved as a result.** The paper said "a critical sampling temperature is defined in at
least five prior works." `critical_temp3` is about temperature-induced phase transitions in
*human–transformer interaction* (Turing tests, spin-glass framing), not a per-model sampling
$T_c$. The sentence now reads "temperature-driven order–disorder transitions in LLM generation
are already studied in several independent works," which is true of all five. Note the direction:
this is a **prior-art disclaimer**, so the error was over-crediting others, not inflating us —
but citing a paper for something it does not say is wrong either way.

**Rule-2 case, one level out from the data.** The standing rule is "never fabricate a number";
this is the same failure applied to a citation. A `note` field saying "to verify" is an
admission that the entry was written on trust, and it survived into a compiled PDF where a
reviewer would have seen it before we did. Anything self-flagged as unverified must either be
verified or removed before it can be printed.

### F44 — the open question in F42 answered, and the obvious answer is wrong

F42 recorded an open empirical question rather than asserting one: *do unignited runs get more
common at larger N?* Three lattice sizes have now been run under an identical protocol, so it is
answerable. `experiments/ignition_vs_size.py` → `results/ignition_vs_size.json`.

**The raw answer says yes, and the raw answer is a confound.** At matched checkpoints
(steps 256 and 512, the only ones present at every size):

| step | N=48 | N=96 | N=192 |
|---|---|---|---|
| 256 | 0/8 | 1/8 | 2/8 |
| 512 | 0/8 | 0/8 | 3/8 |
| **total** | **0/16** | 1/16 | **5/16** |

Fisher exact, smallest vs largest: **p = 0.043**. Stopping there would have produced a finding.

**But B is halved as N doubles** — 16, 8, 4 — to fit the 16 GB budget, and a run is recorded as
unignited only if **all B lattices** die. So under a *constant* per-lattice death probability `d`
with **no N dependence at all**, P(run unignited) = `d^B`, which grows as B shrinks. Fitting one
`d` to all three sizes:

| N | B | d^B | expected unignited | observed |
|---|---|---|---|---|
| 48 | 16 | 0.0057 | 0.09 | 0/16 |
| 96 | 8 | 0.0757 | 1.21 | 1/16 |
| 192 | 4 | 0.2751 | 4.40 | 5/16 |

**d = 0.7242, χ² = 0.244 on 2 df, p = 0.8851.** A single constant reproduces every
size. **These data show no evidence of an N effect on ignition at all** — the apparent trend is
exactly what a pure batch-size effect predicts.

**Why this is recorded as a finding rather than a null.** The Fisher test is the analysis anyone
would run, it is significant, and it is wrong — the design's memory-budget compromise (halve B as
N doubles) is confounded with the variable of interest by construction. The same compromise is
what made the `D_norm == 0` fallback margin constant across sizes (F42 §5.3): N·B is held fixed
at 768, which is convenient there and misleading here.

**What it does not touch.** F42's exclusion rule is unaffected — λ is undefined without a cone
whether or not the rate depends on N. Nothing in F39 depends on this. What it changes is that
a fourth lattice size run at *fixed* B would be needed to separate the two, and until then no
N-dependence of ignition may be claimed.

### F42 — λ_ca is emitted for runs where damage never ignited, and it is uninterpretable there

Found before the N=192 analysis, deliberately: deciding how to treat non-ignited runs *after*
seeing all 24 values is the degrees-of-freedom problem this project keeps catching in itself.
The rule below was written with 4 of 24 runs in.

- **The defect.** `lyap_from_cone` returns a finite λ when damage never ignites. λ_ca is the
  growth rate of a damage cone; with no surviving damage there is no cone and no rate, but
  the estimator fits against its `DAMAGE_CLAMP` floor and emits a number anyway.
- **It is wildly unstable for physically identical runs.** Phase 3 has exactly one unignited
  run (N=96, step256, seed 22): `D_norm = 0.0`, `λ = −0.1649`, against a global minimum of
  −0.2156 across all 96 runs — invisible. N=192 run 1/24 is the same physical outcome at
  **λ = −1.7130**, an order of magnitude away.
- **`is_dead_damage_floor` catches neither** — both are far from −0.9210. F40 named one
  specific sentinel; this is the general case it does not cover.
- **A mechanism I checked rather than assumed.** The natural explanation is that the bogus
  value grows with lattice size. **It does not.** `lyap_from_cone` is N-independent for a
  fixed cone — a 3-site seed dying immediately returns −0.9943 at N=48, 96 and 192 alike;
  the fit window is `min(max_sweeps=8, len(d)−1)` with no N in it, and N enters only the
  second return value `dmax/N`. The spread comes from the **cones differing** — how
  gradually damage decays before vanishing — not from N entering the estimator. Whether
  unignited runs get more common or more extreme at larger N is an open empirical question
  and is not asserted here. *The reason to exclude these runs does not depend on the
  mechanism*: λ is undefined without a cone.
- **Magnitude of the exposure.** One unignited run displaces a 16-run pre mean by ≈ −0.108,
  which is **73% of N=96's entire pre→plateau λ gap** as measured *with* that run included
  (0.1489; the corrected gap is 0.1366). At N=192 a single such run
  among eight would drag the step256 cell mean to ≈ −0.24 and make the retention read ~173%
  — the transition apparently *sharpening* with size, as a pure artifact.

**5.1 — F42 applied retroactively to Phase 3 (this moved F39's committed numbers).**
The one unignited run sits *inside* the N=96 pre cell, so `dev_transition_shape.json` was
stale the moment the rule existed — a file predating a rule its own data violates. Re-run,
not hand-edited. Machine-written result:

| | before F42 | after F42 |
|---|---|---|
| N=96 pre λ mean | +0.0197 (n=16) | **+0.0320 (n=15)** |
| N=96 gap | 0.1489 | **0.1366** |
| N=48 gap | 0.1436 | 0.1436 (zero unignited at N=48) |
| retention (96/48) | 104% | **95%** |
| N=96 λ Cohen's d | 1.76 | **1.71** |
| sign agreement, N=96 pre | 8/16 negative | **7/15 negative** |

Two things worth recording. **Cohen's d fell (1.76 → 1.71), it did not rise** — dropping an
outlier shrinks the sd as well as the gap, so the direction had to be recomputed rather than
inferred. And **the headline is untouched**: it is ordinal (sign counts and a rank test), so
0/48 plateau runs negative stands exactly as before. That is the argument for having made it
ordinal in the first place — a metric-definition change of this size moved every mean in the
analysis and left the claim alone.

**5.2 — the rule is asymmetric between the two metrics, deliberately.** For λ, zero damage
means *no cone*, so the value is **undefined** → drop the run. For D_norm, zero damage means
the ratio is **genuinely zero**, a true measurement → keep it. Dropping unignited runs from
D_norm as well would raise its pre level and shrink its gap (N=96: 0.1030 → 0.1099, retention
53% → 51%), i.e. silently bias the metric that is *not* broken. The two bases are recorded in
the emitted `_definitions` and the asymmetry is asserted in the suite so a refactor cannot
collapse them.

**5.3 — the `D_norm == 0` fallback has a margin, and the margin is what is asserted.** The
fallback is what Phase 3's records actually use, since they predate `mean_damage`. Measured
rather than assumed: the smallest nonzero `mean_damage` is **1/(tail·N·B)**. It is equal at
N=48/96/192 in this project **only because the design holds N·B = 768 fixed** (B is halved as
N doubles for the 16 GB budget). It is neither N-independent nor 1/N — it is 1/(N·B), so a
design with fixed B would halve it at every doubling. The fallback dies once
`round(quantum/D0, 5) == 0`, i.e. beyond N·B ≈ 25,000 (at D0=1.0) to 250,000 (at D0=0.1); the
current design sits at 768, a 30–300× margin. The test asserts the *formula and the headroom*,
and fails if the design stops holding N·B fixed — asserting today's 2.7e-4 would have passed
right up to the configuration where it silently stopped being true.

**The rule (pre-registered, before the N=192 analysis).**

1. `ignited` is a property of the **damage**, not of λ: `is_unignited(mean_damage)` in
   `experiments/lyapunov.py`, keyed on the raw quantity. It cannot key on λ's sign or
   magnitude — N=192 seed 23 has `λ = −0.2197, D_norm = 0.0250`, **negative and ignited**,
   a real measurement that must be kept.
2. **Ignition fraction is reported per cell as its own observable.** It is the DP survival
   order parameter, which this project already uses for the ECA rungs (F34/F36) — the same
   framing, reused, not reinvented. `block_damage` was *already* computing `ignition_prob`;
   `measure()` was discarding it. It is now recorded.
3. λ **means, sds, Cohen's d, gaps and retention over ignited runs only**, with `n` stated
   in every cell. **No censoring to `DEAD_DAMAGE_FLOOR`** — issue #28 records what a constant
   sentinel does to a group mean.
4. **The rank test keeps all runs.** Mann–Whitney uses only ranks, and a dead run's −1.713
   versus −0.1649 does not move a rank. So the pre-registered test and the λ plateau
   prediction were never at risk; every *mean-based* number was.
5. Same rule the ECA rungs already needed for `n_seeds_ignited = 0` (#27).

**This is a rule-8 case**: the estimator emitted a number its own definition does not
sanction, and nothing asserted the precondition. `tests/test_results_self_consistency.py`
now asserts it, along with the F39 design check (`n_pre == len(PRE) × n_seeds`) that a prose
grep could not have caught, and a guard on the N-independence claim above so the docstring
cannot drift from what was verified.

### F41 — our CRN is the *monotone* coupling, not the maximal one (retracts part of F38)

`experiments/coupling_gap.py` → `results/coupling_gap.json`. Raised in review; verified, and
the correction is real.

- **The claim that broke.** F38 argued that one shared uniform thresholded against every
  probability gives HWD's *maximal-correlation* coupling, hence that all damage numbers here
  are a lower bound over the admissible family. Sampling in this project is
  `(cdf < u).sum()` — inverse CDF — which is the **monotone (quantile)** coupling.
- **It coincides with maximal only at |V| = 2.** Verified: 200,000 random binary pairs,
  `max |maximal − quantile| = 0.0`. So **the DK rung is untouched and stays exact** — that
  is precisely why the identity holds there. At |V| > 2 they diverge: `p=(.5,.5,0)`,
  `q=(0,.5,.5)` gives maximal agreement 0.5 and quantile agreement **0**. Over 20,000 random
  8-way pairs the quantile coupling is strictly worse in 99.9% of cases (mean gap 0.215).
- **The direction matters and was backwards.** Maximal coupling maximises agreement, so it
  *minimises* damage. Our LM damage numbers are therefore **not** an extremum of the family —
  they sit inside it. The "lower bound" claim is retracted for every backend with |V| > 2.
- **Measured, not hedged**, on real conditionals from a live bert-tiny damage run (384 (p,q)
  pairs per temperature, taken through the same adapter the loop uses):

  | T | mean disagreement, maximal | inverse-CDF | inflation | near-agreement subset (TV<0.05) |
  |---|---|---|---|---|
  | 0.7 | 0.7717 | 0.7818 | 1.013× | 0.00136 → 0.00188 (**1.38×**) |
  | 0.9 | 0.8042 | 0.8477 | 1.054× | 0.00437 → 0.00505 (**1.16×**) |

  The review's synthetic estimate put the near-agreement inflation at 3–6.6×; measured at the
  real operating point it is **1.16–1.38×**. The qualitative correction stands regardless of
  magnitude — the inequality direction was wrong — but the effect is smaller than feared.
  One model, one lattice size, one settled configuration; treat as an order estimate.
- **What is untouched.** The exact-zero null (p ≡ q gives agreement 1 under any coupling),
  the DK rung, the ECA rungs, and every **relative** comparison — checkpoint-to-checkpoint,
  across radii, rule-to-rule — because the coupling is a common mode. The developmental
  headline is unaffected. What weakens is the *absolute* reading of `D_norm`, whose numerator
  is coupling-inflated toward the independent-noise denominator.
- **The replacement argument, which is true and costs nothing.** Inverse-CDF is
  **replica-independent**: each replica's next state is a function of (its own state, the
  shared noise) alone, never of its twin. A maximal coupling is defined only pairwise — it
  needs both `p` and `q` at construction and does not extend consistently to three replicas
  or to a self-consistent damage field. That is a principled reason to use it, and at |V|=2
  it coincides with maximal anyway.
- **Future work, explicitly not for this deadline.** The Gumbel-max coupling with shared
  per-token Gumbels is replica-independent, ordering-invariant, and much closer to maximal.
  Switching couplings now would invalidate every measurement in the repo.

### F39 — the developmental transition SURVIVES at both lattice sizes, and it is not a step

96/96 runs complete. `experiments/dev_transition_phase3.py` → `results/dev_transition_phase3.json`
(pre-registered test, BH-FDR); `experiments/dev_transition_shape.py` →
`results/dev_transition_shape.json` (shape, effect size, W9). 6 checkpoints × 8 seeds ×
{N=48 B=16, N=96 B=8}, Pythia-410m, run-level statistics.

**Verdict: (b) rise → overshoot → plateau.** The headline survives; the framing and the
effect size both needed rewriting, and have been.

**Pre-registered primary — all four survive BH-FDR:**

| family member | pre → post | p_raw | p_BH |
|---|---|---|---|
| N=48 λ_ca | +0.0247 → +0.1743 | 0.00001 | **0.00002** |
| N=48 D_norm | +0.1865 → +0.6008 | 0.00000 | **0.00000** |
| N=96 λ_ca | +0.0197 → +0.1692 | 0.00002 | **0.00002** |

*(This table is the pre-registered **rank** test and keeps all 96 runs. Mann–Whitney uses only ranks, so the unignited run's magnitude cannot move it — F42 leaves every p-value here unchanged, which is exactly why the ordinal headline was chosen.)*
| N=96 D_norm | +0.1030 → +0.3192 | 0.00000 | **0.00000** |

**The headline number, measured against the pooled plateau (2000/8000/143000), not the
step-1000 peak:**

| N | metric | pre {256,512} | plateau | Cohen d | p | vs peak | from 256 only |
|---|---|---|---|---|---|---|---|
| 48 | λ_ca | +0.0247 | +0.1683 | **1.59** | 5.9e−05 | 1.16 | *2.74* |
| 48 | D_norm | +0.1865 | +0.5689 | **2.88** | 7.2e−07 | *3.94* | *3.69* |
| 96 | λ_ca | +0.0320 | +0.1686 | **1.71** | 1.3e−04 | 1.86 | *2.65* |
| 96 | D_norm | +0.1030 | +0.3062 | **3.07** | 3.0e−07 | 3.56 | *3.91* |

**Both ends of this contrast can be inflated, and the first version of this entry inflated
one of them.** Taking step 1000 as the post value quotes the transition's peak for its level.
Taking step 256 *alone* as the pre value does the identical thing at the other end — and it
is worse: **1.72×** on λ_ca at N=48 and **1.58×** at N=96, against 1.37× for the peak error
that was caught first. The pre-registered pre set is `{256, 512}` and that is what the table
now uses. Italicised columns are the unregistered variants, retained in the JSON under
`_INFLATED` / `_UNREGISTERED` keys purely so the difference stays auditable — they must not
be quoted.

**"λ_ca crosses zero" is withdrawn; it fails twice.** Under the pre-registered split the
pre-group mean is **+0.0247** (N=48) and **+0.0320** (N=96, ignited runs, F42) — both positive, so there is no
crossing at group level. And taking cell means, the crossing sits between **256 and 512**
(−0.0185→+0.0679; −0.0307→+0.0702), not between 512 and 1000; that interval is the
pre-registration boundary, not the crossing point.

**The replacement is stronger and needs no pre/post choice.** Before the transition seeds do
not agree on the *sign* of λ_ca — 6/16 negative at N=48, 7/15 at N=96, spanning −0.216 to
+0.320. After it, **not one of 48 plateau runs is negative** (minimum +0.1074). That uses all
96 runs, is immune to where the split is drawn, and merges the headline with the
variance-collapse observation below.

- **The overshoot is weak, and on the spine quantity it is essentially absent.** The range
  is **+1.4% to +22.4%**, not 14–22% as first written. Broken out: λ_ca overshoots +14.3% at
  N=48 (p_BH 0.114) and **+1.4% at N=96** (d=0.06, p_BH 0.78); D_norm +22.4% and +17.0%
  (p_BH 0.047 and 0.114). It survives correction in **1 of 4 cells**, and that cell is
  D_norm. Since λ_ca now carries the claim, the honest reading is that **non-monotonicity is
  largely a D_norm phenomenon**; the spine quantity shows no overshoot at the larger lattice.
  Describe the shape as non-monotone, give the range, and do **not** claim step 1000 as a
  distinct developmental phase.
- **The transition is durable, which is the substantive point.** The fully-trained
  checkpoint (step 143000) is not a decay back toward the initial state: D_norm 0.61 (N=48)
  and 0.33 (N=96) against step256's 0.13 and 0.07. Whatever happens at step ~1000 persists
  to the end of training.

**W9 — the size question, and the answer is split.** This is the objection that killed the
capacity claim, so it gets reported plainly in both directions:

| metric | gap N=48 | gap N=96 | retention | plateau level N48 vs N96 |
|---|---|---|---|---|
| λ_ca | +0.1436 | +0.1366 | **95%** | +0.1683 vs +0.1686 |
| D_norm | +0.3824 | +0.2033 | **53%** | 0.5689 vs 0.3062, **p=1.3e−08** |

- **λ_ca is size-robust, stated as a bound rather than a null result.** The effect does not
  shrink much (**95%** retention after F42; 104% before it) and the plateau levels differ by **−0.0003, 95% CI
  [−0.0229, +0.0223]** on a plateau of 0.168 — the two lattice sizes **agree to within ±14%**.
  A confidence interval is the right form here; "p=0.91 therefore the same" is an argument
  from a null result and a reviewer can decline it.
- **D_norm is size-dependent.** The gap roughly halves and the plateau level differs
  decisively. Standardised, the effect is undiminished (d 3.69 → 3.91) because the variance
  shrinks too — so the *discrimination* survives while the *absolute scale* does not.
  D_norm's absolute reading is therefore an N-relative quantity and must never be quoted as
  a lattice-free property of the model. This compounds F41: its numerator's coupling is
  already known not to be extremal, and now its scale is known to move with N.
- **Consequence for the paper:** λ_ca carries the developmental claim; D_norm is reported
  alongside it as a same-direction corroboration at a stated lattice size, not as a second
  independent number.

**Variance collapse replicates at both sizes** (observation, not pre-registered): sd(λ) falls
0.1363 → 0.0366 at N=48 (3.7×, Levene p=1.7e−04) and 0.1238 → 0.0395 at N=96 (3.1×,
p=7.7e−07). Before the transition seeds disagree about the *sign*; after it every run is
positive. **They do not agree "to a few percent"** — the plateau per-seed CV is 21.9% (N=48)
and 25.4% (N=96). The few-percent agreement is between checkpoint *means* and between lattice
*sizes* (0.1683 vs 0.1686, 0.2%), which is a different and weaker statement. Reported as an
observation, but it replicates independently at both sizes.

### F40 — the ordered-group λ is an estimator floor, not a measurement (Phase 4.1)

Found while auditing the paper against `results/`, not by a reviewer.

- Five of the seven ordered ECA rules (0, 8, 32, 128, 160) report
  `lambda_all = -0.9210340371976184` in **both** `eca_calib_hardened.json` (as `mean`) and
  `eca_calib_ignition.json`, with `sd = 1.2e-16` and a **zero-width** bootstrap CI
  `[-0.921, -0.921]`. All five have `n_seeds_ignited = 0`.
- That value is exactly **−0.4·ln 10**, and the mechanism is now traced: `lyap_from_cone`
  clamps the damage count at `1e-6` before taking a log, so a cone whose damage dies
  immediately gives the fitted sequence `[1, 1e-6, 1e-6, …]`, and the least-squares slope
  over the default 9-point window is a **constant** — independent of rule, seed, model and
  lattice size. Reproduced directly:
  `lyap_from_cone(dead_cone, 64) → -0.9210340371976186`.
- **Consequence:** the ordered group mean of **−0.32 is 5/7 a constant**. Any "ordered <
  edge < chaotic" ordering built on it is partly arithmetic rather than measured. This is an
  independent second reason the three-class ordering had to go, on top of its failing the
  significance test (F33 λ_all p=0.17; F36 P_ignite p=0.470).
- **Fix landed.** `experiments/lyapunov.py` now names `DAMAGE_CLAMP` and
  `DEAD_DAMAGE_FLOOR` with the derivation in a comment, and exports
  `is_dead_damage_floor(lam)` so callers can exclude the sentinel before averaging. The
  returned values are **unchanged** — verified — so no downstream number moves; this is
  documentation plus a predicate, not a behaviour change.
- The paper now states this explicitly where it previously quoted the λ ordering, and reports
  the coarse split on ignition probability instead.

**Related quoting hazard, also fixed.** `eca_calib_ignition.json`'s *ordered* p-values are
NaN-comparison artifacts: `lambda_cond` is `NaN` for every rule with `n_seeds_ignited = 0`,
so comparisons against it are meaningless. They must not be quoted anywhere. Three different
edge-vs-chaotic p-values exist across the result files — 0.1665 (λ_all), 0.0665 (λ|ignited),
**0.46985 (P_ignite, the correct one)** — and the paper had been quoting the middle,
most-favourable value for a sentence whose subject was ignition probability. Now quotes 0.470.

## Literature check — Domany–Kinzel rung (issue #22; the report that shaped F38)

Standing rule: check before you build. This is the report as written *before* any code;
the rung it specified is now implemented and its results are F38 above. Nothing DK-related
existed in the repo when this was written (grep: zero hits outside planning prose).

**The prior art is large and settles most of the design.**

- Domany & Kinzel, *Equivalence of cellular automata to Ising models and directed
  percolation*, PRL 53, 311 (1984); Kinzel, Z. Phys. B 58, 229 (1985). Two-parameter
  synchronous PCA on a diagonal lattice: `P[1|0,0]=0`, `P[1|0,1]=P[1|1,0]=p1`,
  `P[1|1,1]=p2`. Implemented with **one uniform `z_i` per site per step**.
- Martins, de Resende, Tsallis & de Magalhães, PRL 66, 2045 (1991) — first observed damage
  spreading in DK; split the active phase into "chaotic" and non-chaotic.
- Zebende & Penna, J. Stat. Phys. 74, 1273 (1994); Kohring & Schreckenberg, J. Phys. I
  France 2, 2033 (1992); Grassberger, J. Stat. Phys. 79, 13 (1995); Bagnoli, J. Stat. Phys.
  85, 151 (1996); Hinrichsen, Weitz & Domany, cond-mat/9611085, J. Stat. Phys. 88, 617
  (1997). Review: Hinrichsen, *Adv. Phys.* 49, 815 (2000), cond-mat/0001070, §5.

**W2 is a published, named problem — not a defect this project invented.** The DK damage
boundary was shown to *move* when the algorithm coupling the two replicas changes, while a
single replica is completely insensitive to that choice. Grassberger's verdict, quoted in
Hinrichsen's review:

> "it is misleading to speak of different phases in the DK automaton, ... instead these are
> different phases for very specific algorithms for simulating pairs of such automata"

Hinrichsen–Weitz–Domany's resolution: restrict to the family of couplings that leave
single-replica dynamics and its symmetries intact, then classify a point by how the *whole
family* behaves — damage spreads for all couplings (phase 1), heals for all (phase 2), or
spreads for some and heals for others (phase 3). Under that definition the DK active phase
has **three** sub-phases, and boundaries `B_max`/`B_min` bracket the coupling-dependent
region. **This is exactly the right frame for our W2 disclosure**, and it upgrades the
statement from "we picked a coupling" to "damage spreading is a property of (model,
coupling); here is which member of the family CRN is."

**Which member is CRN? The maximally-correlated one — by construction, not by choice.**
HWD parametrise couplings by correlations `α̃ = <r01 r11>`, `β̃ = <r01 r10>`; drawing a
single `z` per site and thresholding it against every probability gives
`α̃ = min(p1,p2)`, `β̃ = p1`, which they name *maximal correlation*. Our lattice draws one
uniform per site per sweep, shared by both twins, and samples by inverse CDF — that is
literally eq. (21) of HWD. Larger correlation ⇒ smaller damage, so **CRN sits on the
damage-minimising edge of the family**, `B_max`. Every damage number in this repo is a
lower bound over the admissible couplings. That is a sharper and more defensible statement
than the current disclosure.

**The rung has an exact, statistics-free anchor** (Kohring & Schreckenberg; extended by
HWD §IV A). On the `p2 = 0` line the damage field is *itself* a DK automaton at the same
`p1`. Derivation: with `p2=0`, `s'_i = (s_{i-1} ⊕ s_{i+1})·θ(p1 − z_i)`, so for CRN twins

    d'_i = s'_i ⊕ t'_i = θ(p1 − z_i)·[(a⊕c) ⊕ (b⊕d)] = θ(p1 − z_i)·(d_{i-1} ⊕ d_{i+1})

**Verified numerically** (scratch, pure numpy, N=512, 400 steps, p1 ∈ {0.3, 0.6, 0.8087,
0.95}): `max |(a XOR b) − DK(d)| = 0` at every site and step. Off the line the identity
breaks as it must — control at (p1,p2)=(0.6,0.5) gives 15 mismatching sites.

This is worth more than a critical-point comparison: it makes the DK rung a **bit-exact
golden test of the entire CRN damage machinery** (window indexing, shared-uniform
consumption order, inverse-CDF sampling, sync update) with no error bars, in the same style
as `tests/test_golden.py`. A critical-point check can only ever agree to ~1%.

**Published numbers available as anchors.**

| quantity | value | note |
|---|---|---|
| site DP (`p1=p2`) | 0.705489(4) | density transition; 7 digits |
| bond DP | p1=0.6447001(1), p2=0.8737620(2) | density transition; 7 digits |
| compact DP | (1/2, 1) | exactly solvable |
| W18 line (`p2=0`) | 0.801(2) (Zebende–Penna) vs **0.8087(5)** (HWD) | ~1% spread — a *loose* anchor, quote both |
| DK triple point | p1=0.744(10), p2=0.526(10) | terminus of the coupling-dependent region |
| DP density exponent β | 0.277(1) | HWD measured 0.279(10)–0.302(30) for damage |

The `p2=0` disagreement is real and should be reported as a range, not collapsed to one
number. The tight anchors are *density* transitions, which coincide with the damage
transition only on the `p2=0` line (by the mapping above) — so they validate the
survival-probability estimator, not the coupling.

**Design consequences (supersedes the sketch in issue #22).**
1. Primary deliverable is the **exact XOR identity as a bit-identical test**, not a
   critical-point estimate. Cheap, deterministic, and it tests the machinery the LM numbers
   actually depend on.
2. Secondary: `P_ignite` vs `p1` along `p2=0` and along the site-DP line, against the table
   above — the shared DP order parameter F34/F36 established.
3. `src/lattice.py` needs **no changes**: DK is a `Rule` whose `probs` reads positions 0 and
   2 of an `r=1` symmetric window and ignores the centre, run in existing `mode="sync"`.
   `N` must be even (the diagonal lattice decouples into two sublattices under a plain ring).
4. Nothing here has been done for language models — no prior work found applying DK-style
   damage spreading to LM token dynamics. The rung is a validation instrument, not a claim.

## Audit ledger — verdicts on every reviewer objection (W1–W9)

Status of each objection in `paper/REVIEW.md` as of Phase 3. "Resolved" means the paper no
longer makes the offending claim or the claim now matches the data; "stands" means the
objection is still live and is disclosed rather than fixed.

| # | Objection | Verdict |
|---|---|---|
| W1 | Capacity claim pseudoreplicated (n=2 seeds) | **Resolved by retraction + re-test.** Capacity dropped from the paper entirely; the surviving headline is being re-tested at 8 seeds × 2 lattice sizes (`dev_transition_phase3.py`). |
| W2 | D_norm coupling mismatch; denominator-driven rise; ">1" within 1σ | **Stands, disclosed.** All three sub-claims verified. The paper now states the coupling mismatch and the denominator-driven rise plainly and no longer reads `D_norm>1` as amplification. **Not fixed:** the alternative floors (CRN-null, maximal coupling) are unrun. **Answered in part by the DK rung (F38).** Coupling-dependence of damage boundaries is a known, named result (Kohring–Schreckenberg 1992; Grassberger 1995), with a standard resolution in Hinrichsen–Weitz–Domany 1997: classify by the behaviour of the whole admissible coupling family. On DK (binary alphabet) our CRN provably *is* their maximal-correlation member, so the bound is real there. **On the LM backends it is not** — inverse-CDF is the *monotone* coupling (F41), so those damage numbers sit inside the family rather than at its damage-minimising edge; the earlier "lower bound" wording had the inequality backwards and is retracted. Measured excess disagreement 1.3–5.4% overall, 1.16–1.38× in the near-agreement regime. The defensible property is **replica-independence**, not extremality. **Still not fixed:** the alternative floors themselves are unrun on the LM backends. |
| W3 | λ "model-invariance" rests on one saturated cell (off-cell spread 24–46%) | **Resolved by retraction.** Verified: (r=8,T=0.7) spread 24%, (r=1,T=0.7) 46% and reversed. The kinematics⊥stability decomposition is withdrawn from the paper. F31 adds the deeper reason the cross-level pairing was ill-posed: λ_top is a *tangent-space* quantity and λ_ca a *finite* one. |
| W4 | AR "consistent joint" overstated; bimodal T-pooling | **Resolved for surviving claims.** The paper now says both constructions are windowed, in-place-resampled rings — neither samples the model's joint. The bimodal pooling affected the AR *capacity* numbers, which were dropped with the capacity claim, so it no longer touches anything the paper asserts. |
| W5 | Census near floor on real models (0.02–0.04 vs an out-of-training proxy) | **Stands, scoped.** Quantitative recovery is claimed **only** on the synthetic toy; the real-model numbers are reported as near-floor. The real fix (Pythia vs the Pile) is tracked as issue #6 under *Future work*. |
| W6 | v∝r "lifts" are exactly N/4 clipping ceilings; unclipped is superlinear | **Resolved.** Verified the velocities equal N/4 exactly and the one unclipped point (N=384, 41.1) sits *below* the N=192 "ceiling" (47.5). The paper claims only that front velocity grows monotonically with r, and states the superlinearity. |
| W7 | Crossover "strengthens at every T" is false at T=0.3; single-seed | **Resolved by retraction.** Verified (T=0.3: mini 0.463 < tiny 0.508). Downgraded to a plateau *diagnostic*; the "strengthens" claim is gone. |
| W8 | No multiplicity correction | **In progress.** BH-FDR implemented and verified against known values; applied across an explicitly stated family in the Phase 3 run. Note the *central* validation claims are reproductions of known values, not NHT, so multiplicity does not apply to them. |
| W9 | N=48 only; effect shrinks with N | **In progress.** N=96 arm running in Phase 3 (B halved 16→8 for the 16 GB budget; trade recorded). |

**Pattern worth naming.** W2 and F34 are the same class of error: a statistic averaged over
two populations that behave differently (CRN vs independent coupling; ignited vs
extinguished damage). F8/F13 identified this on the LM path years earlier. Any new metric in
this project should be checked for a mixed population *before* it is reported.

## Next steps

**Phase 3 is complete (96/96).** The pre-committed decision rule did NOT fire: all four
family members survive BH-FDR at both lattice sizes. See **F39** for the verdict, the
plateau-based effect sizes, and the split W9 answer (λ_ca size-robust, D_norm size-dependent).

**Blocking the paper (in order).**
1. Phase 4 — rebuild the paper around whatever survives; delete the stale `paper/paper.md`;
   build the PDF (never yet built) and cut to ≤5 pages; double-blind pass; responsible-use
   statement (its absence is an automatic desk reject).

**Done since this list was written.** ECA rebuilt on ignition probability (F36); Phase 2.3
Benettin reference for the CML rung (F37); Phase 2.2 Domany–Kinzel rung, literature check
then build (F38) — which also converts W2 from a concession into a bounded statement.

**Deferred, tracked as issues.** Phase 1.5 duplication hoisting (the `ca.DATA_DIR` mutable
global is a genuine cross-experiment hazard); real-corpus census (#6); the compositional-
complexity axis (#13, #20); greedy/T→0 limit (#17); activation-lattice cone (#19, object
already taken by arXiv:2605.25225 — cite, don't claim).

**Do not re-run.** The cross-level λ_ca vs λ_top hypothesis is settled and explained
(F26/F28/F29 + the tangent-vs-finite regime mismatch in F31). Further measure-swapping there
would be p-hacking.

## Files

Toy (Phases 1–2):
- `fig/phase_curves.png`, `fig/phase_curves_multiseed.png` — order & activity vs T (F1, F11)
- `fig/spacetime.png` — space-time diagrams; `fig/damage_cones.png`, `fig/damage_ignition.png` — damage (F2, F13)
- `fig/melting.png` (F6), `fig/census_validation.png`, `fig/census_bpe.png` (F3, F10)
- `fig/crystallization.png` — probes vs training checkpoint (F7, F8); `fig/finite_size.png` — crossover (F12)
Real MLMs (Phase 3):
- `fig/mlm_radius.png` (F15), `fig/mlm_damage.png` (F16, F17), `fig/mlm_phase.png`, `fig/mlm_differential.png` (F18)
- `fig/mlm_spacetime.png` — bert-base space-time: random soup → ordered English vs churn (F14)
Code & data:
- `src/` (model.py, ca.py, mlm_ca.py), `experiments/` (pipeline scripts), `tests/`
- `results/` — raw npz/json for every run; `results/mlm/` — per-model MLM probes
- `ckpt/` word-level + `ckpt_bpe/` BPE checkpoints; `data/`, `data_bpe/`, `data_mlm/`
- See `README.md` for install, per-phase repro commands, and the M1 runtime tables.
