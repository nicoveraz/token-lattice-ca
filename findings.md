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

## Next steps

- **Full-context vs radius-windowed** head-to-head on the same MLM, to test the
  F17 conjecture that windowing is what makes the dynamics fragile.
- Multi-seed + finite-size (N scan) on the real MLMs; is the MLM temperature
  behavior also a crossover (F12) or a transition?
- Disentangle repetition from genuine long-range structure in the radius test
  (F15) — e.g. an overlap metric that discounts periodic attractors.
- More of the scale ladder (small, medium, large) to test the tiny→mini→base
  "early saturation" hint (F18); RoBERTa/other families as a model-family arm.
- Claude-as-rule qualitative run (LOGOS-CA style, arXiv:2602.00036) with matched
  protocol.

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
