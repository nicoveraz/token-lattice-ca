# A Token-Lattice Cellular Automaton: Black-Box Measurement of Dynamical Structure in Language Models

*Draft — token-lattice-ca. Findings F1–F25 in `findings.md`; figures in `fig/`.*

## Abstract

We turn a language model into a **cellular automaton over token space** and use it as
a measurement instrument. A ring of *N* token cells is updated in place by the
model's own windowed conditional `p_r(x_i | x_{i±r})` at temperature *T* (async
random-order Glauber); the model *is* the update rule, and the radius *r* of the
conditioning window is a swept knob. Reading the model out as a dynamical system —
rather than as a next-token predictor — exposes structure that static evaluation
misses. We report four black-box measurements, each certified against apparatus
confounds by a common-random-number differential protocol. (1) A **radius law**:
equilibrium *local* statistics are nearly radius-blind, but corpus-consistent
*longer-range* structure has an intermediate-radius optimum — visible only through a
repetition-robust metric. (2) **Damage light cones** whose front velocity is set by
the conditioning radius, `v ∝ r`, and is *model-invariant*. (3) A **damping length**:
the asymptotic damage from a perturbation, once controlled for the model's intrinsic
diversity, shrinks as the window widens and grows with model capacity — the larger
models amplify perturbations beyond their own noise floor. A finite-size Lyapunov
exponent shows the *early* dynamics are a purely kinematic axis shared by all models,
so the capacity signal lives entirely in the *asymptotic* persistence: velocity and
stability are orthogonal, and only stability tracks capacity. (4) **External
validity**: both the velocity law and the damping length replicate on an
autoregressive model driven through a one-sided causal window, so they are not
artifacts of the masked LM's globally-inconsistent joint. We calibrate the instrument
against synthetic sources with known transition matrices, where the attractor census
recovers ground truth quantitatively and discriminates sources. The contribution is
the **instrument and the quantities it measures**; the phenomena those quantities
land on — criticality, computation at the edge of chaos — are decades old, and we
measure and quantify them from the outside rather than claim to discover them.

## 1. Introduction

Language models are evaluated almost entirely as functions: given a context, how good
is the next-token distribution? But a masked or autoregressive conditional also
*defines a dynamics* — iterate it in place over a lattice of tokens and you get a
stochastic cellular automaton (CA) whose rule is the model. Iterated-map views of LLMs
have appeared qualitatively (natural-language CAs with an LLM rule; telephone-game and
paraphrase attractors) and as mixing-time studies of masked-LM Glauber dynamics. What
has been missing is a *measurement instrument*: a small, controlled, calibrated
apparatus that reads structural properties of the model out of its dynamics, with the
apparatus confounds explicitly certified away.

We build that instrument and sweep the one knob a token-space CA has that a
next-token evaluation does not: the **conditioning radius** *r* (the half-width of the
window the rule sees). Sweeping *r* and the temperature *T*, and perturbing the
lattice under common random numbers, turns qualitative "LLMs as dynamical systems"
intuition into numbers with error bars. Crucially, a CA has a native chaos vocabulary
— damage spreading, light cones, Lyapunov exponents, attractor censuses — developed
over forty years of complex-systems research, and we import it wholesale.

Our claims are deliberately modest about *what* is found and precise about *how*. The
edge-of-chaos and criticality ideas the results touch are old (Langton's λ parameter;
Bertschinger & Natschläger's reservoir-computing analysis of computation at the edge
of chaos; and, for trained networks, *Intelligence at the Edge of Chaos*). We do not
discover them. We provide a *black-box, token-space, generation-time* measurement of
them in real trained language models, plus a capacity-scaling result we did not find
taken. The novel core is the instrument and its measurements.

## 2. The instrument

**The rule.** A window of `w = 2r+1` tokens with the center masked is fed to a masked
LM; the center distribution, tempered by *T*, is the transition kernel
`p_r(x_i | x_{i±r})`. For an autoregressive model (§3.4) the window is the *r* cells
to the left and the rule is the next-token distribution — an order-*r* Markov
approximation, the causal analog of the symmetric masked window.

**The automaton.** *N* cells on a ring are updated by async random-order single-site
Glauber (the default; synchronous updates manufacture period-2 blinkers and are only
ever an apparatus arm). Special/placeholder tokens are forbidden as emissions.

**Common-random-number (CRN) coupling.** Twin lattices sharing initial state, update
order, and the uniform stream that drives inverse-CDF sampling differ only in the
factor under test. The *null* arm — nothing differs — must diverge by **exactly zero**;
it does, on both the masked-LM and autoregressive paths, which certifies the coupling.

**The four pillars.** (i) *Phase sweeps* over `T × r`. (ii) *Damage spreading*: block
flips, CRN twins, ignition probability reported separately from spread. (iii)
*Attractor census*: run from random soup to quasi-stationarity, census n-grams,
validate against a reference corpus. (iv) *Differential certification*: attribute at
the level of *statistics*, never trajectories — under chaos, trajectory divergence
saturates for signal and apparatus alike, so only a statistic that nulls under
apparatus swaps and moves under model swaps is a model reading.

**Two lessons that shape every metric.** *Repetition confound*: a lattice that loops a
single corpus n-gram scores high on raw overlap for free, so structure metrics use a
repetition-robust variant (distinct corpus n-grams). *Diversity confound* (its
stability analog): a degenerate low-entropy lattice snaps a perturbation back
trivially — low damage for the wrong reason — so damage is normalized by the model's
own diversity floor (below). Both are the same hazard: a quantity that looks like
structure/stability but is really collapse.

We validate the whole apparatus on a tiny transformer (~0.42 M params) trained from
scratch on tinyshakespeare — where the census can be checked against the *actual*
training corpus — then scale to `bert-tiny → bert-mini → bert-base-uncased` and to
`Pythia-160m`. A key control comes for free: the temperature "phase transition" is,
on a finite-size scan, a **crossover, not a true transition** (the order-parameter
susceptibility self-averages, χ_peak ∝ 1/N), which disciplines all "transition"
language and motivates finite-size scaling on every headline (F12).

## 3. Results

### 3.1 A radius law (F15, F22)

Sweeping *r* at fixed temperature, the *local* equilibrium statistic (fraction of
lattice bigrams present in the corpus) is nearly radius-blind — echoing the toy, whose
curves collapse across `r = 1…16`. But *longer-range* corpus consistency is not. Raw
3- and 4-gram overlap rises with *r*, which naively reads as "larger windows build
longer-range structure." Under the repetition-robust metric (distinct corpus 4-grams),
however, the signal **peaks at an intermediate radius (r≈4) and falls by r=16**: the
apparent large-*r* growth was the lattice collapsing into repetition (distinct-token
fraction drops), inflating raw overlap. The intermediate-radius optimum is genuine
for the larger models, whose lattices stay diverse; the weakest model degenerates,
and the confound is itself capacity-dependent. Certification (§2): holding the
special-token windowing scheme fixed, the cross-model radius-profile difference far
exceeds the distribution-preserving apparatus floor — so the profile is a model
effect — but the scheme itself moves the profile as much as a model change, so every
cross-model claim must hold the scheme fixed. *This is a certified conditional claim,
not an unconditioned one.*

### 3.2 Damage light cones, velocity ∝ r (F16, F21)

A 3-site block flip under CRN produces a damage cone whose front advances ballistically
at a velocity set by the conditioning radius: `v ≈ 1.6, 3.5, 7.7, …` sites/sweep for
`r = 1, 2, 4, …`. An apparent plateau at large *r* on small rings is a finite-size
artifact — the front simply fills the ring in ~2 sweeps — and it lifts cleanly on a
`N ∈ {48,96,192,384}` scan (at r=16, `v = 11.5 → 23.5 → 47.5`). The unclipped law is
`v ∝ r`, and it is **model-invariant**: tiny, mini, and base share the same velocity
profile. Interaction range is a kinematic property of the apparatus, not of the model.
We do **not** claim `v ∝ r` as new — a ballistic damage front whose velocity is set by
the neighborhood range is the **classic CA light cone**: the maximal Lyapunov exponent
equals the damage-front velocity (Bagnoli, Rechtman & Ruffo 1992), finite-range
interactions give a constant range-set velocity (Lieb–Robinson 1972), and butterfly
velocities of decorrelator fronts are standard in chaotic CA (arXiv:2101.01313). The
point is that the LM token-lattice **behaves as a bona-fide CA** and that this velocity
is a *model-invariant kinematic axis* (contrast the capacity-dependent §3.3).

### 3.3 The damping length, and a kinematic ⊥ stability decomposition (F23)

Perturb, evolve under CRN, and ask how much damage *persists* asymptotically. The raw
answer is diversity-confounded, so we normalize by a **diversity floor** `D0`: the
asymptotic drift of twins that share the settled initial state but evolve under
*independent* noise with *no* flip — how far the model decorrelates on its own. Because
`D0` is a full radius-*r* run, it propagates at the same `v ∝ r`, so the normalized
`D_norm = D/D0` cancels **both** the diversity and the kinematic terms. Then:

- **`D_norm` rises with radius** (tiny 0.43→0.82, mini 0.74→1.03): the damping length —
  the conditioning radius at which `D_norm` crosses half — shrinks as the window
  widens. The raw-damage "recovery" at large *r* was collapse-into-repetition; the
  control removes it.
- **Capacity orders it, monotonically** (mean `D_norm` tiny 0.672 < mini 0.880 <
  base 0.904). The larger models cross `D_norm > 1` — the flip is amplified *beyond
  the model's own noise floor* — while the smallest stays sub-critical. The ordering
  fights the normalization (a more diverse model's higher floor should deflate its
  `D_norm`) and is velocity-immune (matched *r*), so it is neither diversity nor
  kinematics.
- **Kinematics ⊥ stability**, made rigorous by a finite-size Lyapunov exponent λ (the
  early log-separation slope of the CRN twins). λ is **model-invariant**
  (`λ_max ≈ 0.74–0.77` for all three, all at the same `(r,T)`) and universally
  *positive*: every model transiently spreads a perturbation, at a rate set by the
  radius. The capacity signal is entirely in the *asymptotic* persistence, not the
  early growth. Velocity/λ and stability/`D_norm` are orthogonal axes; only the second
  tracks capacity.

**Reading (as measurement, not discovery).** `D_norm > 1` is a chaos signature. The
smallest model reads "stable" because it is *frozen* — it heals by collapsing to a
dead repetitive attractor; the larger models read "sensitive" because they are
*expressive* — rich dynamics carry a perturbation rather than crushing it. The
capacity axis thus traces a **stability ↔ expressiveness tradeoff**: more capable
models run nearer the chaotic side, where expressiveness lives. This is the
long-hypothesized edge-of-chaos picture (reservoir computing; *Intelligence at the
Edge of Chaos*) measured black-box, in token-generation dynamics, with a
capacity-scaling axis added.

### 3.4 External validity: the autoregressive port (F24)

Both load-bearing measurements replicate on `Pythia-160m` driven through a one-sided
*causal* window (the AR analog of the masked window; null CRN divergence exactly
zero). Velocity∝r holds (`v = 5.8, 7.7, 11.5` for `r = 2,4,8`; `r=1` does not
propagate — a single causal token cannot carry the flip), and the damping length holds
(`D_norm` climbs 0.001 → 0.98 across *r*). Because the masked-LM joint is globally
inconsistent while the autoregressive joint is consistent, replication across the two
constructions is the strongest available evidence that the phenomena are properties of
trained-LM token dynamics, not artifacts of the masked-LM construction. The
**capacity→sensitivity climb replicates on the AR construction** too: at the
sub-saturation radius r=2, mean `D_norm` rises 0.64 (Pythia-160m) → 0.73
(Pythia-410m), the larger causal model damping less — mirroring the masked ladder. We
compare at the level of the *trend*, not absolute numbers: causal-context healing is a
different object from bidirectional healing.

### 3.5 Calibration against ground truth (F19, C2)

The census's validity rests on recovering *known* structure. On three synthetic
first-order Markov sources sharing a vocabulary but with different sparse transition
matrices `P_a/P_b/P_c`, a small model trained on each and then censused yields an
empirical bigram transition `Q_X` that matches its own `P_X` (row total-variation 0.22
vs a random-lattice baseline of 0.91 — a 4× move toward ground truth) and *not* the
others (cross-TV 0.95). The census recovers each model's own priors quantitatively and
discriminates sources. On natural corpora the recovery is qualitative but structurally
identical: the deepest attractors are the corpus's *format skeleton* (Shakespeare's
speaker-name+colon+newline; WikiText's abbreviation/list markers), recovered from
random soup. Vocabulary-compression artifacts corrupt this — a word-level `<unk>` token
becomes attractor material and inflates recovery through artifact-to-artifact matching;
byte-level BPE removes it (top attractors go from 11–13/15 `<unk>`-laden to 0/15 real
text).

## 4. Related work

A 102-agent adversarial novelty check (`results/deep_research_novelty.md`) places the
contribution precisely: the **instrument** is largely novel at the method level, but
its **substrate is not**, and novelty is uneven across claims.

**Shared substrate — the exposed flank.** *Glauber dynamics on masked LMs*
(arXiv:2605.16378) already recasts a masked LM as iterated masked-token resampling — a
Glauber Markov chain on token sequences, the exact substrate we iterate. It measures
mixing time / metastability, **not** damage spreading, light-cone velocity, a damping
length, or a finite-size Lyapunov exponent, and it uses **maximal coupling** (provably
distinct from our common-random-number coupling). Our novelty rests on the
**measurement layer**, not the LM-as-iterated-token-CA framing, which we do not claim.

**Edge of chaos — §3.3 is partially anticipated.** Edge-of-chaos-as-capability is
canonical (Langton; Bertschinger & Natschläger 2004, reservoir computing) and has been
shown for *trained transformers* via Lyapunov exponents of self-attention Jacobians
(arXiv:2505.19458, Tomihari & Karakida, NeurIPS 2025), and named for LLMs generally
(*Intelligence at the Edge of Chaos* arXiv:2410.02536; QLE arXiv:2503.13530). We claim
neither the concept nor the criticality↔capability link — only the *token-space*
finite-size Lyapunov / damping-length measurement and the explicit **model-size →
sensitivity** axis, which arXiv:2505.19458 (continuous hidden space, performance-
correlated) does not report.

**The CA light cone is classic — §3.2 is import, not discovery.** A ballistic damage
front whose velocity is set by the neighborhood range, and the identity
Lyapunov = front-velocity, are textbook CA / statistical physics (Bagnoli, Rechtman &
Ruffo 1992; Lieb–Robinson 1972; butterfly velocity in a Kauffman CA, arXiv:2101.01313).
We import them and show the LM token-lattice exhibits them (behaves as a bona-fide CA)
with a model-invariant velocity — we do not claim `v ∝ r` as a new law.

**Perturbation propagation.** *SPARC* (arXiv:2607.09803 — **not** QUIVER) formalizes an
error-propagation operator on AR residual streams with a ρ(F_T)≥1 criticality threshold
(the top-Lyapunov boundary) in *activation space* — adjacent, not a token-space CA;
*QUIVER* (arXiv:2605.23956) is compound-AI pipeline graphs, sharing vocabulary but no
method. **Terminology:** we avoid "repair" (taken by *self-repair* / the *Hydra effect*,
arXiv:2307.15771, 2402.15390, for internal component compensation) and "self-correction"
(SPARC), both categorically different from our spatial damping length.

**Other.** *Sampler-centric oracle* work is the calibration kin to §3.5 (our
synthetic-Markov recovery is its trained-model analog); temperature-criticality work
makes *T* a calibration anchor; the census descends from Hanson & Crutchfield's basin
portraits; early low-order crystallization is consistent with distributional simplicity
bias, read black-box; the *telephone-game / paraphrase 2-cycle* line is where our
sync-update period-2 caution applies.

*(Pre-submission: full-paper reads of arXiv:2605.16378 and QUIVER; verify all
2026-preprint titles/IDs; re-check near submission — these are weeks-old preprints.)*

## 5. Limitations

Masked-LM local conditionals are globally inconsistent, so the CA is a well-defined
stochastic dynamical system but not a sampler of any joint — every claim is phrased as
a property of the dynamics, not of a sampled distribution. The census on real models is
*proxy*-validated (WikiText, which the models were not trained on), a lower bound on
recovery; the ground-truth calibration is on synthetic sources only. Rings are small
(`N ≤ 384`), the model ladder short (three masked, two autoregressive). The
autoregressive capacity trend rests on two models and only the sub-saturation radius
discriminates (both saturate by r≥4); the 410m r=16 cell OOM'd on 16 GB and was
dropped. The special-token windowing scheme is a first-class apparatus factor that
must be held fixed for cross-model claims. Temperature scales are not comparable across
the masked and autoregressive constructions.

## 6. Conclusion

Reading a language model as a cellular automaton over its own token space, with the
conditioning radius as a swept knob and common-random-number perturbations as the
probe, yields calibrated, apparatus-certified measurements that static evaluation does
not: a radius law, a model-invariant damage velocity, and a diversity- and
velocity-controlled damping length that shortens with capacity as the dynamics move
toward the chaotic side — replicated on an autoregressive model and quantified against
ground truth. The instrument is the contribution; the phenomena it measures are old and
deep, and that is exactly why measuring them cleanly, from the outside, is worth doing.
