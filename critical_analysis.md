# Critical analysis: textca as an interpretability tool

**Revision 2 — 5 August 2026.** Supersedes revision 1 (1 August 2026), which remains in git history.
**Scope:** an outside critical read with one question in front: how far is this from being a *tool for
interpretability*, and what would developing it into one require. Basis for this revision:
`findings.md` (F1–F96), `paper/paper.tex` and the camera-ready tag, `paper/plan_paper3.md`,
`fingerprint/PROGRAM.md` with `gate1/2/3.json`, `paper/NOTES.md`, `assembly_theory.md`, the `src/`,
`tests/` and `experiments/` trees (119 scripts, 125 results files, 13 test modules), and the
`gatecheck/` package. Weaknesses the project already records about itself are credited as such.

**What changed since revision 1:** nineteen new findings (F77–F96) in four days, three fingerprint
gates run to completion, a camera-ready revision of the paper, and the first prior-art collisions the
project has hit. Three of those findings (F77, F78, F80) cite revision 1 of this document by name, and
two more (F79, F88) are downstream of the routes and issues it named — so this revision opens with a
scorecard on its own predecessor before adding anything new.

---

## 1. Verdict in brief

Revision 1 concluded that the project had built a validated black-box measurement apparatus and a
working falsification discipline, but not an interpretability tool, and that the gap was *referent*
and *user* — something the numbers are about that lives inside or generalizes beyond the
construction, and someone outside the project who can act on them.

Four days of intense work have changed that picture in two opposite directions, and the honest
summary is that **the strategic case has improved for measurement and worsened for
interpretability.**

Improved: the flagship developmental transition survived the sharpest attack revision 1 named and
came out larger (F77); the transition was decisively separated in time from the F62–F70 artifact,
killing the most damaging alternative reading (F84); an external anchor landed for the first time in
the project's history, at family level with the right unit of analysis (F86, ρ = +0.833, n = 8
families, p = 0.0137); and the fingerprinting capability survived its own deflationary gate on pair
separation (Gate 1) and its controlled-pair test (Gate 2).

Worsened: the explanandum programme — the three bridges revision 1 named from the flagship's *when*
to a *what* — was executed and returned nothing. Route 3 (ablation) returned a pre-registered null
and is closed (F80). Route 1 (context-use co-timing) returned "neither confirmed nor eliminated"
(F78). Route 2 (induction heads) was killed by arithmetic for the early dip and remains unrun for the
crossing (F83). The author's own summary is the correct one: **"λ_ca still dates an event nobody has
named."** The cheap bridges revision 1 recommended have now been tried, and the interpretability
framing is harder to sustain today than it was on 1 August — not because the science got worse, but
because the honest negative answers arrived.

Meanwhile two genuinely new problems appeared, neither of which existed in revision 1's picture: the
project hit its **first real prior-art collisions** (F90/F91), including a paper three weeks older
than the taxonomy it nearly scoops; and the **research layer has outrun the verification layer** —
eleven self-caught analysis defects in sixteen findings, with the same defect class committed twice in
three days, once wasting the single measurement the third-paper decision was gated on.

The conclusion revision 1 reached still holds, restated more sharply: this is an excellent black-box
*measurement and characterization* programme wearing an interpretability label that its own evidence
no longer supports. The recommendation that follows from four days of negative results is not to try
harder on the bridges; it is to **change the label**.

---

## 2. Scorecard on revision 1

Revision 1 made three specific criticisms with testable content and seven recommendations. This is
what happened to each — an unusually clean natural experiment, since the project acted on the document
directly.

| Rev-1 claim | Outcome | Verdict |
|---|---|---|
| §3: the flagship lives at r=2, "the radius your own interventions condemned" — a skeptic's cheapest attack | **Closed decisively** by F77. The transition survives at r=3 and r=4 with *tighter* BH-FDR (9.1e-07, 6.9e-07 vs r=2's 1.3e-05), 48/48 ignition, and the crossing bracket does not move | Criticism was correct and is now retired |
| §3: λ_ca is "a detector without an explanandum" | **Confirmed, and worse.** Three routes run, none delivered. F80 closes route 3 by pre-registered null; F78 returns "neither"; F83 forecloses induction heads for the dip | Criticism stands, strengthened by evidence |
| §4: single family, confounded LR/size axis | **Untouched.** F88's attempt to dissolve it via loss-alignment returned NOT DECIDABLE (residuals 0.0254 vs 0.0243 against a 0.0247 seed floor). #61/#83 still open | Oldest unaddressed critique in the project |
| §6.2: "give the thermometer a disease" (#58/#84) | **Partially delivered by a different route.** #84 itself returned NOT DECIDABLE; but T\* → greedy degeneration (F86) is the disease, found via #90/#101 | Achieved, though not the way recommended |
| §6.3: induction-head co-timing (#69/#70) | **Foreclosed for the dip** (F83: step ~1000 against a dip at step 32–512, "one to two orders of magnitude off"); still unrun for the crossing | Recommendation partly dead |
| §6.4: second family / iso-LR width scan (#61/#83) | Not run. F81 used the existing width ladder for the *dip*, not the crossing | Open |
| §6.5: diffusion LMs (#59) | Not run | Open |
| §6.6: engineering debt | Minimally addressed (one new guard test). Still no packaging, no CI, 119 scripts, `gatecheck` unadopted | Open |
| §6.7: ship the discipline | **Done** — `gatecheck/` exists as an installable package with 42 tests | Delivered, unadopted |
| §3: the fingerprint capability (gpt-neo vs gpt2 "used only as a control") | **Developed into a three-gate programme** with a frozen preregistration; Gates 1–3 all run | Delivered |

Two things are worth extracting from this table. First, the analysis was *useful* — it named an
exposure that the project then closed, and named a gap that the project then tried three ways to fill.
Second, and more important: **every recommendation that pointed at the interpretability framing came
back negative, and every recommendation that pointed at measurement and characterization came back
positive.** That is a signal about where the work actually lives, and §9 takes it seriously.

---

## 3. What is genuinely strong (updated)

The falsification discipline did not merely survive four days of high-velocity work; it was the thing
producing the value. Sixteen findings contain eleven self-caught analysis defects, and in almost every
case the catch changed a reported conclusion before it left the project.

The highest-quality instances are worth naming because they are teachable. F80's verdict logic
initially manufactured a positive out of a null — reporting "LOCALISED: L20, L23" — through two
compounding errors, taking `|z|` on a directional hypothesis and computing a ratio before any noise
gate; the author diagnosed the second by measuring it (`Spearman(Δloss, |per_nat|) = −0.472, p =
0.02`, i.e. the ranking was significantly driven by its own denominator). F83 found that
`(d["gap_in_steps"] or 99) <= 1` silently discards a gap of zero, so "the perfect match was the one the
guard threw away" — and then swept the same idiom out of three other files. F88 caught a knife-edge
verdict decided on a margin of 0.0011 against a noise floor of 0.0247 and converted it to NOT
DECIDABLE. F89's own control caught that its registered criterion tested separation where the question
required retention, so the "significant" result was a difference between 92% and 97% erasure. F85 ran
a zero-difference reproduction check across 18 checkpoints × 7 fields *before* reading a probe it had
just modified.

Two disciplinary refusals deserve particular credit because both were tempting to skip. F81 declined
to pool pythia-410m into the width ordering despite it sitting "exactly in the predicted direction —
which is what makes pooling it tempting and wrong," on the grounds that it carries a depth *and* a
learning-rate confound. F86's Gate B failed its benchmark primary on coverage (11 < 16 families) and
the run was re-scoped *before its data existed*, with the benchmark correlation demoted to labelled
exploratory rather than quietly promoted afterwards.

And the camera-ready revision was executed with a verification invariant — a numeric-multiset diff
proving that numbers moved between body, tables and appendix without changing value — with the single
exception adjudicated and logged as an erratum. That is a standard almost no solo paper meets.

The critique that follows takes all of this as given. The problem with the project is not rigor.

---

## 4. The central weakness, revisited: the explanandum programme came back empty

Revision 1's core charge was that λ_ca flags that *something* reorganized during training without
saying what, and that interpretability begins where that sentence ends. The project took the charge
seriously and mounted a three-route programme against it. All three routes are now resolved or
foreclosed, and none produced an explanandum.

**Route 3 (ablation) is the most informative failure**, because it was the only route that
*attributes* rather than correlates — hold the black-box measurement fixed, manipulate the internals,
see if the reading moves. The design was right: selectivity measured as Δλ per nat of loss damage, so
that a raw λ drop from a degraded model proves nothing; the confound measured before the grid was
chosen. F79's group ablations produced a suggestive dissociation — `attn_early` recovers 86% of the
distance back to pre-crossing while `mlp_all`, which costs twice the loss damage, moves λ_ca by 0.021
— but the declared statistic returned max z = +1.49 against a pre-registered 2.0, and the author
correctly demoted it to "underpowered, not null" only after finding the z-normaliser contaminated by
its own candidates. F80's follow-up then killed localisation outright: 0 of 24 single attention layers
clear 2σ, the largest single-layer effect (L16, |Δλ| = 0.0577) is smaller than its own seed spread
(0.0611), and — the decisive number — eight layers ablated together give Δλ = +0.345 while the 24
singles sum to **−0.224, the wrong sign**. The effect is not localised and not diffuse; it requires
removing most of the attention stack at once.

That non-additivity is a genuine fact, and it is worth having. But it is, as the author writes,
"closer to F64's architecture-level statement than to a mechanism." A tool that can say "this depends
on the attention stack collectively" is not telling anyone anything they can act on about a specific
model.

**Route 1 (F78) is a near-miss that should not be counted as one.** The co-timing test between
context-use onset and the λ_ca crossing returned a null on its declared statistic, which the author
then correctly refused to accept as clean: "largest single rise" on a log-spaced grid splits a
two-interval ramp arbitrarily, the onset spans steps 128→512 as +0.267 then +0.237 — together 74% of
the total span — and the crossing sits *inside* that ramp. The honest verdict is "neither confirmed
nor eliminated," and the results file's verdict string still says "cleanly eliminated," which is a
small but real instance of the correction living in prose while the machine-readable artifact carries
the superseded claim.

**Route 2 (induction heads) — revision 1's recommended cheap bridge — is partly foreclosed.** F83
eliminates it for the early dip by arithmetic: induction-head formation lands near step 1000 against a
dip at steps 32–512, one to two orders of magnitude off. It remains unrun for the *crossing*, which is
the version revision 1 actually recommended, and that is now the last cheap bridge standing. But the
expected value has fallen: two of three routes returned negative, and the one positive-adjacent result
(F78) is a correlation the author himself flags as insufficient — "even a perfect match would show
that two events coincide in one model family, not that λ_ca measures context use."

The residual value here is real but modest, and it is epistemic rather than interpretive: F82 and F83
between them eliminated three candidate mechanisms for the extinction window, which as the author
notes leaves it "genuinely open rather than merely unexamined." That is a better state than before.
It is not an explanandum.

**Assessment.** Revision 1 treated the missing explanandum as a gap to be filled. Four days of
competent, well-designed, honestly-reported work says it may instead be a property of the
measurement. λ_ca is a coarse, collective, architecture-level dynamical readout; the evidence now
available says it does not decompose. A project that wants to be interpretability needs a different
observable, not more attempts on this one.

---

## 5. The surviving science, and where it now stands

### 5.1 The flagship is materially stronger and its scope is unchanged where it matters

F77 is the best experiment in this block. It answered a named exposure directly, ran the protocol by
*import* rather than reimplementation (`measure` and `bh_fdr` from `dev_transition_phase3`, F42 filters
from `lyapunov`), and returned the opposite of what an artifact story predicts: the transition is
present at r ∈ {2,3,4}, largest at r=3, and the crossing bracket does not move. Two readings formed
off the partial grid were withdrawn before recording, with the lesson stated as "check the claim you
are about to contradict, in the file that makes it, before contradicting it."

F84 is the second-best, and its payoff is an ordering: funnel onset at step 8 ≪ extinction window at
step 32 ≪ λ_ca crossing at steps 256–512. Because the degeneracy is more than an order of magnitude
older than the crossing, the developmental transition **cannot be** the formation of the F62–F70
artifact. That kills the most damaging alternative reading available to a reviewer, and it does so by
dating rather than by argument.

One awkwardness surfaced that the paper should absorb. At N=48 the r=2 median λ_ca is never negative
(step256 reads +0.0083), so the genuine *median* sign change exists only at r=3 and r=4; at the
paper's own radius the sign change is carried by run-level disagreement. The paper's ordinal framing
("not one of 48 plateau runs is negative", 6/16 negative before) remains accurate as written — F77
checked this explicitly and withdrew its own contrary reading — but the stronger evidence now sits at a
radius the paper does not report. **The camera-ready should include the radius replication**: it
converts the sharpest available attack into a supporting result, and it costs a sentence plus a table
row.

What has not moved is generality. The transition remains a **Pythia fact measured in a narrow
temperature band** — one family, two grid points of temperature (T ∈ {0.5, 0.7}), with the LR/size
confound intact after F88 returned NOT DECIDABLE on the loss-collapse test. This was revision 1's §4
criticism, it was in `REVIEW.md` before that, it is in the paper's own limitations, and it is now the
project's oldest open debt. F88's diagnosis is worth noting for planning: the fix is "more checkpoints
per size (finer loss spacing), not more sizes," which is cheaper than a second family but does not
substitute for one.

### 5.2 T\* is the best result the project has produced, and it has four caveats

F86 delivers what revision 1 §6.2 asked for and what two adversarial reviewers said the paper was
missing: the instrument predicts something measured outside itself. ρ(T\*, rep_4) = +0.833 over n = 8
families with permutation p = 0.0137, with family as the unit — the correct unit, adopted precisely
because F68's earlier model-level version (ρ = +0.552, p = 0.107, six of ten points being Pythia
sizes) was pseudoreplicated. Measured properly, the effect got *larger* and significant. That is the
right shape for a real result.

The caveats are stated by the project itself and all four are load-bearing:

It is **conditional** (F87): the claim holds among attractor-bearing families only, and two attempts
to extend it across regimes returned clean nulls (Gehan tau = +0.10, p = 0.72; threshold-free ladder
AUC ρ = +0.03, p = 0.93). The cause is diagnosed rather than hand-waved — polyglot-ko and Minerva sit
near the top of the repetition range with no attractor at all, degenerating by a route the axis cannot
see. F87's instruction that "the sensitivity nulls publish with the anchor, not after a reviewer runs
them first" is exactly right.

It is **decoder-scoped** (F93): nucleus sampling removes greedy degeneration on all 15 families
(spread 0.052 against greedy's 0.541), so the anchor predicts degeneration *under greedy decoding*,
not degeneration in general.

It has **one leg**. The second behavioural target was run and rejected itself on dynamic range, so no
verdict on F86 is licensed from it in either direction. Finding a valid second target — a degeneration
measure that survives nucleus sampling, which by construction excludes most repetition metrics — has
gone from "a day's work" to "a design question."

And **n does not grow cheaply**: 7 of 15 families have no attractor, additional in-band members of
measured families share corpora and add zero independent pairs, and growth requires new independent
corpora. One family moving would cost the word "significant."

There is also a quiet dependency worth flagging: the n = 8 exists partly because two harness bugs were
recovered from — Qwen1.5-1.8B's spurious `too_slow` flag and LFM2's transient OSError each added a
family to the primary on retry. Had either not been retried, the anchor would be n = 6 or 7 and
probably not significant. That is not misconduct; it is a reminder of how thin the margin is.

Finally, F92's deflationary result — the static argmax map carries no information about degeneration
(ρ = −0.12) while T\* does (+0.83) — was described in `plan_paper3.md` as "branch A's strongest single
argument," and then **did not reproduce on the second target** (modal +0.405 against T\*'s +0.429, a
tie). The arm was rejected, so this is not evidence against F92 either. But the deflationary question
is open again, and the project should stop leaning on F92 until a valid second target exists.

### 5.3 The fingerprint programme: honest gates, and one result the prereg's logic rescued

All three kill conditions were written to fire and none did. The programme's own §7 correctly declines
to read that as success, and the most important structural fact is inside Gate 1: **the static
baseline beat the CA on attribution** (5/14 vs 4/14) and K1 survived only because it required both
limbs of an AND. The gate says so itself — "the honest reading is that neither battery attributes
families from four numbers." What the CA does carry is pair separation (0.5× vs 2.4×) and
within-family coherence (no static feature survives BH correction; the CA's attractor share was p ≈
3×10⁻⁴). That is a real asymmetry and it licensed Gate 2, where corpus separates at 3.7× and
post-training at 2.3×, with the direction now measured (instruction tuning *removes* the attractor:
top-1 0.85 → 0.23, fixed-point fraction 0.875 → 0.0).

Gate 3 produced the most interesting single measurement in the programme and it was not the one
registered. The expected hazard was chat templating; the actual hazard is the **plain-text
round-trip**, where a two-token window decoded to text and re-tokenized merges into a single token —
63% of calls for gpt2 against 13% for pythia-410m — so the endpoint silently runs the CA at a smaller
radius and *inverts* which model looks attractor-bearing. That is a genuinely novel, precisely
measured, and practically important artifact for anyone probing models through text APIs, and the
programme identified it only because realized context widths were recorded per cell.

Three caveats the programme does not fully surface. Gate 3 ran 2 seeds where the preregistration froze
4, and two temperatures of four, with no note of the deviation in the file. Every Gate-2 manipulation
is n = 1 — one distilled model is not a fact about distillation. And no third-party endpoint was ever
contacted, which the programme states plainly but which means the "characterises models behind an API"
affordance remains, as revision 1 put it, asserted rather than demonstrated — now with a local harness
standing behind it, which is progress, but not with a real endpoint.

---

## 6. New weakness: the first prior-art collisions, and how they were found

This is the most serious new problem, and it is structural rather than local.

F90 ran a novelty check on the argmax-census taxonomy and it came back **incomplete** — a session
limit killed 32 of 100 agents and the synthesis. Tellingly, "the agents verifying the
distillation/pruning literature — precisely claim 3's threat — are the ones that died." The check that
failed was the check that mattered. F91's scoped re-run (104/104 agents, clean) then found what the
first pass missed, and the results are not comfortable:

The pruning arm is **taken outright** — Wang et al. (COLM 2026) quantify pruning-induced looping with a
Loop Fraction metric rising 0.3 → >0.8 after a single pruned layer, in a section titled "Repetitive
Reasoning Loops after Layer Pruning," and a second paper already cites it as settled background. The
taxonomy is **nearly scooped by three weeks** — ShortOPD (14 July 2026) reports three regimes under
structured depth pruning that "structurally rhyme with funnel/none/fragmented." The argmax-map framing
is **pre-empted** by "The Benchmark Illusion," whose description of the top-1 map degrading relative
to the distribution it is read from is, in the author's own words, "our mechanism." And fixed points of
LM token maps, and LLMs-as-dynamical-systems-with-attractors, were already prior art at F90.

Worse than the collisions themselves is what surfaced them. Kim & Rush (2016) — a directional prior
from the literature, not an internal gate — predicted that distillation makes the student's
distribution *more* peaked, the opposite sign to F90's reading. Re-sorting the census by fixed-point
abundance confirmed the literature and broke the project's own claim: gemma-2 (distilled) has the
third-highest fixed-point abundance measured, so F90's "no modified model is a funnel" **pools two
opposite mechanisms** — distillation raises abundance while fragmenting basins, whereas
pruning-plus-distillation and annealing eliminate fixed points entirely. That is, as the author
immediately recognized, "F87's own defect — 'no attractor is two mechanisms' — committed again one
level up, and only the literature's directional prior surfaced it."

**The generalizable criticism:** this project applies its calibration discipline rigorously to
*measurements* and belatedly to *claims of priority*. Novelty checks happen after the work is done,
they are run once and sometimes incompletely, and in this case the internal gates could not have
caught the defect because the missing information was in the literature. For a project whose entire
methodology is "reproduce a known answer before believing an unknown one," the literature is a body of
known answers it has been reading last rather than first. The fix is cheap and mechanical: a
prior-art gate at the *hypothesis* stage, not the write-up stage — the same shape as
`dp_calibration`, refusing to license work until the adjacent literature is on record.

Note also that `fingerprint/PROGRAM.md` §1 mandates exactly such a check for the fingerprint
programme — "Prior-art check is mandatory before any write-up… run the deep-research workflow before
drafting anything" — and it **has not been run**. The only novelty check on record covers the
taxonomy. Model-equality testing, API model verification, and output-based attribution are live
adjacent literatures, and the programme's most novel result (the tokenizer-merge mechanism) is
precisely the one nobody has checked.

---

## 7. New weakness: the research layer has outrun the verification layer

Sixteen findings in four days, with eleven self-caught analysis defects. The catch rate is
extraordinary and the discipline is working. But the *rate of defects* is itself the finding, and
three signals say the pace has begun to cost real things.

**The same defect class twice in three days.** F89 registered a separation criterion where the
question required retention; F93 registered a correlation target without applying a dynamic-range
gate. The author names the pattern precisely — "a statistically-shaped criterion applied to a quantity
with no room to vary" — and notes that the gate F93 needed had been built two weeks earlier for a
different run and simply not applied. The cost was not academic: F93 was *the* measurement the
third-paper decision was gated on, and it was wasted.

**A hazard written down and shipped anyway.** F80's denominator problem was flagged in prose before
the run, and the declared statistic was correctly left untouched mid-run — but "only the *statistic*
was protected; the *verdict that consumes it* was not." The project's guard machinery protects
measurements; it does not yet protect the verdict logic that reads them.

**The bookkeeping layer has fallen behind.** The audit ledger W1–W9 still reads "as of Phase 3" with
W8/W9 marked "In progress" — untouched across sixteen findings. The "Next steps" section still lists
building the PDF as blocking work, though the PDF was built on 3 August and a camera-ready tag exists.
There is no post-F92 caveats section. `findings.md` is now 3,733 lines and its own navigational
apparatus no longer describes it. For a project whose credibility rests on a ledger a reviewer can
audit, a ledger that has stopped tracking its own contents is a real cost — and it is the same failure
mode as the retracted claim that survived in §4's opening sentence (F55), one level up.

None of this is a rigor failure. It is a throughput failure: verification, bookkeeping, and
prior-art checking are all running slower than experiment production, and the gap is where the
recent costs have landed.

---

## 8. Engineering (updated: unchanged, plus one unadopted asset)

Revision 1's §5 stands almost verbatim, which is itself the finding. There is still no
`pyproject.toml`, no pinned environment, no CI. `experiments/` has grown from 110 to **119 scripts**;
`results/` holds 125 files. Configuration is still constants at the top of scripts, `ca.DATA_DIR` is
still a mutable module global (issue #25, open), device handling is still Apple-Silicon-first, and
golden files are still machine-locked to one MPS machine. One new guard test (`test_precommit_guard.py`)
was added — a real improvement, aimed at the class of defect that produced the F80 verdict — bringing
the suite to 13 modules.

The notable new fact: **`gatecheck/` is in the repository and textca does not import it.** A grep
across `experiments/`, `tests/` and `src/` returns nothing. The package that was extracted precisely
to hold this project's discipline in reusable form is sitting beside the project as dead weight, while
the same patterns continue to be hand-written inside it. `DESIGN.md` §8 laid out an incremental
adoption path that invalidates no existing results — `provenance` as a drop-in, `dp_calibration` as a
thin wrapper around `Gate`, staleness sweeps delegated to `testing.assert_fresh`. Adopting even the
first of those would have caught F80's unguarded verdict logic by construction, since `gatecheck.fits`
exists to reject exactly that shape of statistic.

Two housekeeping items: a stray zero-byte `.findings_tail_tmp.md` sits untracked at the repository
root (an artifact of this analysis; delete it), and the results files still carry no environment
fingerprint, so a numpy or torch upgrade can move a number without anything noticing —
`gatecheck.provenance` closes that hole and is, again, unadopted.

---

## 9. What developing this into a tool would take — revised

Revision 1 offered seven recommendations. Four days of evidence justifies restructuring them around a
sharper claim: **the interpretability framing should be retired, and the measurement framing promoted
in its place.**

**9.1 Change the label.** Revision 1 asked the project to choose between three products: a
training-dynamics monitor, a probe-artifact audit kit, and a mechanistic bridge. The mechanistic
bridge has now failed twice — structurally in F26–F29 and empirically in F78/F79/F80 — and the
project's own summary is that λ_ca "dates an event nobody has named." Continuing to present the work
under an interpretability heading invites exactly the question the evidence cannot answer, and a
reviewer at an interpretability venue will ask it in the first paragraph. What the project *has* is a
validated black-box instrument that (i) dates a real, replicated, radius-robust training-time
transition, (ii) predicts a real failure mode in a stated regime, and (iii) documents how iterated
probes manufacture phenomena. Every one of those is a measurement claim. Naming them as such is not a
retreat; it is the same move F35 and F87 already made at claim level, applied to the project's
identity.

**9.2 Close the generality debt before anything else.** One checkpointed non-Pythia family is now the
single highest-value experiment available, and it has been the highest-value experiment since
revision 1. It gates the flagship, it gates any interpretability or measurement framing equally, and
it is the objection every reviewer will reach first. F88 supplies a cheaper partial: finer
loss-spacing within Pythia would resolve the loss-versus-step alignment that returned NOT DECIDABLE.
Do the cheap one to sharpen the design, then spend the GPU-days on the second family.

**9.3 Treat T\* as the product, and protect it.** It is the only result in the project that predicts
something outside itself. Three actions follow. Publish the F87 sensitivity nulls alongside it, as the
project already intends. State the greedy-decoding scope in the claim sentence, not in a limitation.
And find the second target properly: the requirement — a degeneration measure that survives nucleus
sampling — is a literature question before it is an experiment, and the right first move is a scoped
search of the degeneration/decoding literature rather than another day of GPU time. Until it exists,
the anchor has one leg and `plan_paper3.md` is correct to leave the recommendation struck out.

**9.4 Institute a prior-art gate at the hypothesis stage.** This is the cheapest high-value change
available and it follows directly from F91. The project already owns the machinery: a gate that
refuses to license work until a known-answer check passes. Point it at the literature instead of at
Domany–Kinzel, run it when a thread *opens* rather than when it is written up, and require the check
to complete — F90's partial run is exactly the "vacuous pass" failure mode the project's own
anti-vacuity discipline exists to prevent. The fingerprint programme's own mandated check is overdue
and should be the first one run.

**9.5 Protect the verdict layer, not just the statistic.** F80's meta-defect is the specification for
this: the declared statistic was correctly frozen mid-run while the verdict logic that consumed it was
not. Every registered criterion should ship with a dynamic-range check on its own target (F93), a
noise gate before any ratio (F80), a directional test where the hypothesis is directional (F80), and
an explicit NOT-DECIDABLE branch (F88). All four now exist as one-off fixes in individual scripts;
they belong in one place, and that place already exists in `gatecheck.gate` and `gatecheck.fits`.

**9.6 Adopt gatecheck inside textca.** Not for its own sake — because the specific defects of the last
four days are the ones it was built from. This is a day of work with no scientific risk, and it
converts a spun-off asset into a live one.

**9.7 Keep the two genuinely novel measurements visible.** Two results in this block are, on current
evidence, nobody else's: the **tokenizer-merge mechanism** in Gate 3 (a text-in/text-out endpoint
silently runs a windowed probe at a smaller radius, model-dependently, inverting the reading), and the
**abundance-versus-concentration distinction** in the argmax census (two properties that "peakedness"
conflates). Both are small, both are checkable, and both are more defensible than the framings around
them. The first in particular is a genuine service to anyone probing models through APIs, and it
currently lives in a JSON verdict string.

**9.8 Slow the cadence to match the verification layer.** Four days produced nineteen findings, eleven
defects, one wasted decision-gating measurement, and a ledger that no longer describes itself. The
discipline caught everything, but it caught things later and later in the pipeline. Bringing the
audit ledger, the caveats section and the "next steps" list current is half a day and restores the
artifact a reviewer would actually audit.

**9.9 Addendum, 5 August (F94–F96): two of the items above are now closed, and one is sharpened.**

*The prior-art gate (9.4) has been run, and it relocates the fingerprint claim rather than killing
it.* F95 discharges the check `fingerprint/PROGRAM.md` made mandatory and had never executed. Generic
black-box model identification is **taken** (Model Equality Testing, ICLR 2025; IRIS at 0.99 AUROC),
quantization detection is taken, and the tokenizer round-trip artifact is substantially anticipated —
`encode(decode(t)) != t` is already formalised, `token healing` is a shipped mitigation, and a
model-dependent rate table already exists. But the state of the art is uniformly **single-shot
scoring of externally supplied text**: no published method feeds a model its own output back in.
**The defensible novelty is the dynamics, not the fingerprint**, and distillation and pruning remain
untouched. Any write-up must be pitched there. This vindicates 9.4's argument at cost: the check was
cheap and it moved the claim, exactly as predicted, and it should have been run at the hypothesis
stage rather than after three gates.

*T\*'s second leg (9.3) had its literature question answered, and the answer is a formula.* 9.3 said
the second target was "a literature question before it is an experiment." It is, and F95 found it:
IRIS derives that decoding temperature is a rank-one on-family move, so two temperatures separate
only at second order — `I* ≈ (1/8)(Δβ)²·V` with `V = T³ dH/dT`, the heat capacity of the next-token
distribution — with the strong signal at T→0 support collapse (AUROC ≈ 0.99), which is the same
physics as an attractor melting. That is a **closed-form external prediction for how much signal a
temperature sweep can carry**, and T\* is defined at a melting point rather than by adjacent-T
comparison. It is now the cheapest available test of the project's best result, and it displaces
F93's rejected second target as the next move on T\*.

*The explanandum gap (§4) is narrowed rather than closed, and the fourth route was the informative
one.* F94 tried to **derive** λ_ca instead of correlating it, via annealed mean field on the full
ladder, and the registered deflationary outcome — that a few thousand forward passes might reproduce
what the ring measures — **did not fire** (residual 0.445 against a 0.023 seed floor). That is the
strongest defence of the construction's expense the project has, and unlike the Gate-1 K1 answer it
is a statement about magnitude against a measured floor with the prediction frozen in advance. It
belongs wherever the instrument is justified, not in a route-4 footnote. F96 then found that F94's
own input was measured off-distribution: on the states the ring actually occupies, `s` spans 0.331
rather than 0.071 and the predictor finally clears a range gate. Bounded by degeneracy and
circularity, so not a positive result — but it makes the next experiment well-specified for the first
time since the explanandum search began.

*One item gets worse, not better.* The recurring defect — **a statistically-shaped criterion applied
to a quantity with no room to vary** — now has a fourth instance (F94's own Spearman, retro-gated),
and F96's registered primary died at a distinct-context floor that only existed because the defect
had been anticipated. 9.5 and 9.6 are therefore upgraded from housekeeping to the highest-value
*engineering* item: this class is being caught by hand, every time, and it should be a `gatecheck`
primitive.

---

## 10. Bottom line

Revision 1 said the gap between this project and an interpretability tool was *referent* and *user*.
Four days later, the referent question has been answered — negatively, three times, by the project's
own well-designed experiments — and the user question has been answered positively in an adjacent
domain: T\* predicts a real failure mode, the fingerprint battery separates real manipulations, and the
API-port measurement identifies a real hazard. **The work has found its subject; it is just not the
subject on the label.**

What now stands is a validated black-box measurement programme with one replicated training-time
phenomenon (radius-robust, artifact-separated, single-family), one conditional external anchor
(family-level, greedy-scoped, one-legged), a cautionary result about iterated probes that is complete
and publishable, and a falsification discipline that is the best thing in the repository and remains
under-exported. Against that: an explanandum that three routes failed to find, a generality debt now
older than any other open item, the first genuine prior-art collisions with one near-scoop three weeks
old, and a research cadence that has begun to outrun its own verification layer.

The single most consequential decision in front of the project is not which paper to write. It is
whether to keep describing this as interpretability. The evidence assembled since 1 August says no —
and says so clearly enough that continuing would be the one place where this project, which has been
scrupulously honest about every measurement it has made, would be overclaiming about itself.
