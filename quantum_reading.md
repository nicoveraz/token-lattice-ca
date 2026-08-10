# A quantum-theory reading of the instrument — what is exact, what is vocabulary, what is empty

**Drafted 10 August 2026** by the session analyst (companion to `critical_analysis.md` and
`what_it_measures.md`). Origin: a remark of the author's — *tokens behave like a wavefunction that
collapses at the end* — taken seriously for one full pass. This is an **analogy audit**, not a
physics claim: every proposed mapping between the project's constructs and quantum measurement
theory is tagged **EXACT**, **VOCABULARY**, **EMPTY**, or **OPEN**, and the document's job is to
keep the four apart. Nothing here is a new result; every project number cited traces to
`findings.md`.

---

## 0. The rule this document runs under

This project has burned twice on analogies that were exact only where they were trivial. F30: the
logistic rung reproduced the bifurcation diagram because it *was* the bifurcation diagram — "an
analogy validated only in the limit where it is empty is empty." F102: annealed mean field, put
through the ladder like any estimator, was qualitatively right on known answers and cleanly wrong on
the model. The same discipline applies to a formalism as to an estimator: state where the mapping is
exact, calibrate it on a system where the answer is known, and refuse to quote it outside that
regime. A quantum reading of this instrument must survive the same three questions as any rung —
*what does it predict, where is it exact, and what would kill it.*

---

## 1. Formal placement: the process is classical, exactly

The next-token distribution is a probability vector over the vocabulary. In density-matrix
language it is ρ = Σ_v p(v)|v⟩⟨v| — **diagonal in a fixed basis, with no off-diagonal terms.**
There are no amplitudes, no relative phases, and therefore no interference: paths to the same
outcome add probabilities and can never cancel. Every observable in the system commutes with every
other. A state of this form is precisely the definition of a *classical* mixture, and a process
built from sequential sampling of such states is a Kolmogorov stochastic process — fully described
by classical probability, with no remainder for quantum theory to explain.

This is worth stating as sharply as it deserves: **the airtight version of "tokens are a
wavefunction that collapses" is airtight because it is the definition of sampling.** Quantum
mechanics strictly contains classical probability; the analogy is exact exactly where the quantum
content is absent. What quantum theory adds beyond Kolmogorov — interference, contextuality,
non-commuting observables, violations of the law of total probability — is absent from the
sampler *by construction*, and §5 is the only place in this document where that could even in
principle change.

One nuance, flagged and dismissed: logits combine additively and can cancel, which resembles
destructive interference. It is not. Cancellation in a log-odds score happens *within one
outcome's evidence*, not *between alternative histories of the process*, and it can never produce
the interference signature (probabilities below either branch's contribution). Evidence
combination is classical inference, not superposition. **Status: EMPTY.**

---

## 2. The mapping audit

| Mapping | Content | Status |
|---|---|---|
| Sampling = projective measurement | True, of a diagonal state in one fixed basis — i.e., classical sampling. Adds no structure Kolmogorov lacks | EXACT (and classical) |
| F35 = irreversible measurement record | "Decoding is a measurement whose record is never revised." P_persist = 1.000; the collapsed token enters the record permanently because generation never re-measures a site | EXACT restatement — the best sentence the analogy buys |
| Commitment schedules | AR decoding: immediate, permanent collapse per token. The ring: collapse revoked every sweep (re-preparation + re-measurement — which is *why* healing exists there and nowhere in deployment). Diffusion LMs: collapse on a schedule. Speculative decoding: provisional collapse with rollback — the one deployed decoder where collapse is reversible | VOCABULARY, and the useful kind: it names the axis separating the project's constructions (*when does a distribution become a record*) better than any current term |
| Temperature = measurement strength | T→0 is fully projective; the T=0 limit of the construction is the argmax map, whose fixed points F70/F84/F85 censused. Higher T = weaker, noisier measurement | VOCABULARY; the projective-limit identification of the argmax map is exact |
| Greedy degeneration ≈ Zeno freezing | Measure maximally hard at every step and the dynamics freeze into loops; weaken the measurement (nucleus) and the freezing vanishes — F93's "degeneration is decoder-induced," restated. The near-prediction (degeneration monotone in measurement strength) is already known folklore | VOCABULARY; predicts nothing new |
| Attractor = pointer state; T\* = pointer delocalization | The attractor token as the einselected stable state of repeated interaction; the funnel / none / fragmented taxonomy as pointer-basis structure (one pointer state / none / many); T\* as the temperature where the pointer state delocalizes | VOCABULARY; evocative, decorative |
| MSS chaos bound (λ ≤ 2πT/ℏ) vs λ_ca(T) | A temperature-dependent bound on Lyapunov growth exists in quantum many-body physics and the resonance with a sampling-temperature-dependent λ_ca is seductive. It is dimensional nonsense here: sampling temperature is not a bath temperature and there is no ℏ | EMPTY — recorded because it is the most seductive wrong rung available |
| Contextuality / total-probability violations *on the ring* | The sampler is a defined classical process; a Kolmogorov-violation test run on it **cannot fail**, and by the project's own anti-vacuity rule a test that cannot fail is not a rung | VACUOUS on the ring; see §5 for the one non-vacuous host |

---

## 3. The one exact bridge, and it is already named: distinguishability dynamics

Followed rigorously instead of poetically, the quantum lens does not land on measurement collapse.
It lands somewhere better — on a body of physics where the classical limit of a quantum object *is
this instrument*.

**The core quantity is a trace-distance response.** The maximal coupling of two conditionals p and
q realizes P(coupled draws differ) = TV(p, q) exactly; the instrument's monotone (inverse-CDF)
coupling sits a measured 1.3–5.4% above that on LM backends and coincides with it at |V| = 2
(F41 — which is why the Domany–Kinzel rung is exact). Total variation is precisely the classical
restriction of the **trace distance**, quantum information theory's canonical distinguishability
measure, with the Helstrom bound giving its operational meaning: the best achievable one-shot
probability of telling two states apart. So, to measured accuracy:

> **s is the one-token distinguishability of the conditional, and λ_ca is the exponential growth
> rate of the distinguishability of two branches under repeated local measurement and
> re-preparation.**

That is not an analogy; it is a translation, exact in the classical restriction. And it gives F35
its cleanest form yet: deployed generation drives two branches to *maximal* distinguishability
immediately and never lets it decay (TV_norm ≈ 0.97 — as distinguishable as unrelated
continuations).

**The damage field has a name in the chaos literature.** Two replicas driven by common noise,
differing by one local flip, with the spread of their disagreement tracked in space and time — that
construction exists in classical many-body physics as the **decorrelator**, the standard probe of
the classical butterfly effect (studied on spin chains, with a measured butterfly velocity and
Lyapunov regime), and the decorrelator is in turn the recognized classical limit of the
**out-of-time-order correlator (OTOC)**, quantum many-body theory's scrambling diagnostic. The
project's damage cone is a token-space decorrelator; its light-cone velocity is a butterfly
velocity; λ_ca is the classical Lyapunov rate of that literature. The paper already cites
Lieb–Robinson for the cone; this is the rest of that neighborhood.

**One correspondence worth savoring**: in the scrambling literature, the butterfly *velocity* is
set by the geometry and coupling structure (Lieb–Robinson-bounded) while the Lyapunov *rate*
carries the system's dynamics. The project measured exactly this split without the vocabulary:
the cone's shape is apparatus, and only its growth rate is measurement. The instrument recovered
the v_B-versus-λ_L division of labor from scratch. **Status: EXACT (classical restriction), with
named prior art** — the physics home of the response family is scrambling/decorrelation, not
measurement theory, and any future physics-facing write-up should say "token-space decorrelator"
rather than "wavefunction."

> **CORRECTION, 10 August 2026 — caught in the author's audit; original citation preserved
> struck.** The paragraph above first cited ~~F16/F21/F28~~ for the kinematic-cone claim. That
> string was the F55 vector twice over: F21 *retracts* F16's velocity plateau (finite-size
> wraparound), and F28's headline is the cross-level negative — the intended referent was only
> F28's within-r clause (λ_ca(r) model-invariant), which citing the bare number does not convey.
> The claim survives on cleaner evidence: **F116/F119** — cone area and fill track λ while
> front_width has no readable span, i.e. the cone's *shape* carries nothing beyond its growth
> rate. One caution back at the correction: area–λ at **+1.000** should be quoted as
> near-definitional, not as a discovery — at fixed geometry the pre-saturation area is close to a
> deterministic functional of the growth rate, two readouts of one count. The load-bearing half of
> F116/F119 is front_width's *null*, and the split should be cited by that.
>
> **And a scope split, per F128/F129 (one day younger than this document):** "the Lyapunov rate
> carries the system's dynamics" now needs saying precisely. λ_ca carries the **training**
> dynamics of a single model (+0.336 → −0.339 → +0.168); it carries **no usable cross-model
> ranking** (spread 0.051 against a construction-induced range of 0.68, seed-stability 0.030, and
> blind to RWKV — the one architectural contrast the instrument establishes best). The decorrelator
> translation is exact; the quantity it translates is developmental, not model-characterising.
> That sharpens §6's verdict rather than weakening it.

Citations to verify through `audit_refs` before any of this is quoted: the classical-butterfly /
decorrelator spin-chain literature (Das–Dhar–Huse–Moessner et al., PRL ~2018), OTOC reviews for
the quantum side, Helstrom for the distinguishability bound, Busemeyer–Bruza and the
Wang–Busemeyer order-effect work for §5. None of these has passed the repo's citation gate yet.

---

## 4. F122 in this light: not interference — collision

F122 measured two damage clouds superposing sub-additively (−2.52, −1.17, −0.51 damaged sites at
separations 6, 12, 24, against a causally-disconnected control at −0.016 ± 0.010) while the
underlying two-token response is additive (F114). The tempting word is *interference*, and this
document exists partly to withhold it: interference requires amplitudes and phases, and §1
established there are none. What F122 measured is **front collision** — competition for sites and
shared healing — the standard nonlinearity of classical spreading processes; colliding damage
fronts in directed-percolation-class systems are sub-additive for the same reason two fires burn
less than twice one fire.

Which suggests the rung this finding still needs: **the interaction rung.** Two-seed damage
interaction is computable on Domany–Kinzel — the known-answer system where single-seed damage is
already bit-exact — so the sub-additivity of colliding fronts can be calibrated where the answer
is known before the LM number is interpreted. The design writes itself in the repo's idiom: same
statistic, same separations, DK first, NOT DECIDABLE if the DK gate fails. Until that runs, F122
is "the lattice is not reducible to its local response," full stop, and the mechanism label
(competition vs shared healing) stays open. **Status: VOCABULARY withheld, calibration proposed.**

---

## 5. The one place the lens could earn compute

There is exactly one falsifiable research question the quantum perspective generates, and it is
not about the ring — it is about the model's *judgments*, and it inverts the usual direction of
the analogy.

Quantum cognition (Busemeyer and colleagues) argues that human judgment data — question-order
effects, conjunction fallacies, and specifically the **QQ equality**, a symmetry in
order-effect data confirmed across dozens of national surveys — violates classical probability in
a patterned way that quantum probability models fit naturally. The claimed evidential force runs:
these signatures are hard for Kolmogorov models, so cognition may be quantum-like.

An LLM is a *provably Kolmogorov* system trained on the outputs of the humans who produce those
signatures. So: **do LLM judgments, probed with the same paired-question protocols, reproduce the
order-effect patterns and the QQ equality?** Both outcomes are informative. If yes — a certified
classical process reproduces the signatures said to evidence non-classical cognition, and their
evidential force collapses to "properties of the data-generating culture, learnable by a
conditional model." If no — LLMs systematically deviate from a robust human regularity, which is a
finding about LLMs. The experiment is prompting only; the instrument's disciplines transfer
directly (dynamic-range gate first: the paraphrase-instability floor must be measured before any
order effect is read against it — F93's lesson; family as the unit; prereg with kill conditions).

Two gates before a token is spent. **Prior-art gate, mandatory and likely fatal:** the
LLM-as-survey-participant literature is large post-2023, and "QQ equality in LLMs" is exactly the
kind of study that may already exist — this goes through the deep-research check *first*, per the
F90/F91 lesson. **And a framing note:** a positive result would not show LLMs are quantum; it
would show the *signatures* aren't quantum. The apparatus of this project is not needed to run it
— which, by the standards of `what_it_measures.md` §6, is a point in its favor as knowledge and a
point against it as a use of this instrument. **Status: OPEN — the lens's only live branch.**

> **AMENDMENT, 10 August 2026 — the gate fired the day this was written.** The experiment exists:
> *"Auditing Question-Order Effects in Large Language Models with the QQ Equality: Mechanism
> Characterization and a Saturation Caveat"* (arXiv:2607.17219, July 2026 — the ShortOPD pattern
> again, weeks not years). A first-signal pilot on one open-weight instruction-tuned model, reading
> forced-binary next-token log-probabilities as survey-response distributions. **Status revised:
> OPEN → PARTIALLY TAKEN.**
>
> What they found is the better half of the story: the audit largely could not run, because
> **17/18 and 7/8 item pairs were saturated** — near-deterministic response distributions with no
> room to vary — and they conclude that forced-binary next-token log-probs "were thus inadequate
> for distribution-level QQ audits," recommending "pre-specified saturation diagnostics" for any
> such protocol. That is this project's F89/F93 defect class — *a statistically-shaped criterion
> applied to a quantity with no room to vary* — and gatecheck's dynamic-range gate, independently
> reinvented by strangers within weeks of this repo naming it. The convergence is worth more than
> the priority loss: the failure mode is real enough that two unrelated groups hit it from
> opposite directions in the same month.
>
> What remains unclaimed, honestly sized: (i) they did **not** make the epistemic argument — their
> framing is methodological, not "a Kolmogorov system reproducing the signatures deflates their
> evidential force" — but they also *could not*, since the measurement never succeeded; the framing
> is only available to whoever gets past saturation. (ii) The route past saturation is this
> project's home move: when a level saturates, measure the **response** — sweep temperature and
> framing until the distribution has dynamic range, gate on it, then audit. One model was tested;
> families were not; sampled behavioral responses (rather than logprob readouts) were not. So the
> residual is execution, not the question — and by this repo's own rule (plan_paper2: "one search
> away"), execution-residuals are **cite-don't-claim** unless run and landed. The lens's only live
> branch is now a possible second paper *to someone else's pilot*, entered with eyes open or not
> at all.

---

## 6. Verdict

Keep four pieces of vocabulary, because they name structure the project's current language does
not: **commitment schedule** (the axis from AR's instant permanent collapse through diffusion's
scheduled collapse to the ring's revoked collapse), **the record sentence** for F35 ("decoding is
a measurement whose record is never revised"), **pointer state** for the attractor and its melting,
and — from the bridge in §3, which is not vocabulary but translation — **token-space decorrelator /
butterfly velocity** for the damage field and its cone, which additionally connects the work to a
literature that will recognize it.

Drop the ontology. There is no wavefunction: the state is diagonal, nothing interferes, everything
commutes, and the one construction in this repository that genuinely holds cells in superposition
and collapses them one at a time is not the language model — it is the texture-synthesis algorithm
that already took the name (WaveFunctionCollapse, whose constraint-propagating tile grid is this
ring's deterministic cousin). The quantum reading, run through the project's own ladder, resolves
the same way the mean-field reading did: qualitatively generative, quantitatively empty — except
for one exact classical bridge that hands the response family its proper physics home, and one
cheap behavioral experiment that was required to survive a prior-art gate before existing — and
did not: the gate fired the same day (§5 amendment), leaving an execution-residual on someone
else's three-week-old pilot.

The author's remark survives in edited form, which is this repository's house style: tokens do not
behave like a wavefunction that collapses at the end. They behave like a measurement record that
never gets to collapse twice — and the instrument's whole subject, seen from this angle, is what
becomes measurable in the one construction where it does.
