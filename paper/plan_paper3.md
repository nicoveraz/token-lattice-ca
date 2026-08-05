# Plan — the third-paper decision

**Written 4 Aug 2026, the day F86 landed.** `plan_paper2.md` §6 has gated this decision on one
thing since July: *"if #90 resolves positively, there is a third paper. If it stays underpowered
or comes back null, there is a paragraph in paper 2 and nothing more."* #90 resolved positively.
This document exists so the decision is made over two concrete objects rather than a mood, and it
is a **decision document, not a plan** — nothing below is committed work.

The decision point was set at "after Gate 2" and then "when the anchor lands". It has landed.
Per the same plan's caution: **three papers out of this material is over-slicing unless the third
has its own external anchor.** The question below is whether F86 is that anchor or the best
paragraph paper 2 will ever get.

---

## 1. What exists today, by claim

| claim | evidence | state |
|---|---|---|
| **T\* predicts neural text degeneration** at family level | F86: ρ(T\*, rep_4) = +0.833, n = 8 families, permutation p = 0.0137; pre-registered primary, promoted at Gate B *before* data | **The anchor.** Significant, independent families, behavioural target sharing no machinery with the ring |
| T\* is a well-behaved instrument | F68 (tighter within family than raw share), F64 correction (graded, not bimodal), band census 8/0/7 | Solid |
| The fingerprint battery separates known manipulations | Gates 1–3: corpus 3.7×, post-training 2.3× (instruction tuning *removes* the attractor), distillation 5.7× on the radius/BOS arms; K1/K2/K3 none fired | Solid, each n = 1 manipulation |
| The battery survives the API abstraction | Gate 3: token-id endpoints carry it, text endpoints destroy it via silent radius reduction (measured, 63% vs 13% merge rates); chat templating does *not* break it | Solid, local harness only |
| Corpus direction (Pile one-way inference) | was 2 families; band screen: 15 families measured, 8/7 attractor split | Materially improved, direction analysis not yet re-run |
| F64 scale boundary | binary extends to 1.5–3B (5 pairs, 0 flips); *level* drifts (max 0.475) | Solid, with the refinement |
| The argmax-map account | F70, F84, F85: funnel by step 8, contested basin, predates crossing by 30× | Solid, one model dated |

## 2. What the anchor's fragility actually is

- **The claim is CONDITIONAL, and the sensitivity analyses proving it are already run (F87):**
  treating no-attractor families as left-censored (all 15, Gehan tau = +0.10, p = 0.72) or using a
  threshold-free melting scalar (ρ = +0.03, p = 0.93) both return nulls, because families like
  polyglot-ko degenerate heavily with no attractor at all. T\* predicts degeneration *within the
  attractor-bearing regime*, full stop. Branch A's thesis must say "conditional"; the nulls ship
  with the anchor.
- **n = 8 finite-T\* pairs**, and this number does not grow cheaply: 7 of 15 measured families
  have no attractor (no T\* exists); second in-band members of measured families share corpora
  and add **zero** independent pairs; the five failed architectures are kernel problems, not
  measurement problems. Growing n means **new independent corpora** — 7B-band families on other
  hardware, or non-English families (polyglot-ko measured; it has no attractor).
- One family moving could soften p = 0.0137. The result would survive as "large effect,
  suggestive" but lose the word *significant*.
- rep_4 is one behavioural target. A reviewer's first ask will be a second target measured the
  same way (e.g. repetition under nucleus sampling, or MAUVE-style degeneration) — cheap
  (generation only, models cached) and it either corroborates or kills.

## 2b. The taxonomy line, after its novelty check (F90 → F91)

The argmax-map taxonomy was briefly the most interesting thing on the board. Its scoped novelty
check (104/104 agents) returned **PARTIALLY ANTICIPATED** and, worse for us, a directional prior
(Kim & Rush 2016) that exposed F90's recipe association as **pooling two opposite mechanisms** —
distillation *raises* fixed-point abundance while fragmenting basins; pruning and annealing
*eliminate* fixed points. What survives is narrower: the census separates **abundance** from
**concentration**, which "peakedness" conflates. At n = 2 distilled models that is a hypothesis.

**Consequence for this decision:** the taxonomy cannot carry branch A. It needs Wang et al.
(COLM 2026), ShortOPD, the Benchmark Illusion and Kim & Rush cited as prior art, and its own
contribution reduces to an instrument plus an unpowered association. **F86 remains the anchor**,
and the recommendation below is unchanged.

## 3. Branch A — the third paper

**Thesis:** *a cheap black-box battery, validated by known-answer gates, characterises language
models where weights are unavailable — and its central scalar predicts a real failure mode.*
Spine: validation-gate methodology (inherited from paper 1's ladder) → the battery → Gates 1–3 →
T\* → F86. The OOD-artifact line (F62–F70) appears as the *mechanism* of the instrument, told
from strength: the artifact was dissected until it became a measurement.

**What it needs before submission-grade, costed:**

| item | cost | risk |
|---|---|---|
| Second behavioural target for T\* | ~1 GPU-day, models cached | Could kill the anchor — which is the point |
| Corpus-direction analysis re-run at n = 15 | zero compute (data exists) | Could weaken the one-way claim |
| One real completion endpoint for Gate 3 | dollars + a token, ~an hour | K3 scoping already covers failure |
| Per-manipulation n ≥ 2 (one more distilled, one more instruct pair) | ~2 h GPU each | Discovery either way |
| Venue | — | Workshop-scale unless the second target lands clean |

**Honest odds:** if the second behavioural target corroborates, this is a real paper with a real
anchor. If it does not, branch A collapses into branch B *having spent one GPU-day*, which is
exactly the cheap-kill structure every gate in this repo uses.

## 4. Branch B — fold into paper 2

Paper 2 (A+B+C) gains a section: "the instrument's central scalar predicts degeneration"
(F86, stated with n = 8), plus the battery as an applications subsection. Paper 2 goes from
*negative-result + methodology* to *methodology + one positive external validation* — arguably a
stronger single paper than either alone, and immune to the over-slicing critique by
construction.

**Cost:** ~2 days of writing into an existing complete draft plan. **What is lost:** the
fingerprint capability framing (Gates 1–3) compresses to a paragraph; the API-port measurement
(the tokenizer-merge mechanism, which is genuinely novel) probably doesn't fit at all.

## 5. The recommendation

**Run the second behavioural target first (~1 GPU-day), then decide.** It is the only item that
changes the decision rather than decorating it: corroboration makes branch A's anchor
two-legged and the paper real; a null makes branch B automatic and costs a day. Every other
branch-A item is spendable after the decision.

The one calendar fact: nothing here interacts with the Sept 29 notification. Paper 2's venue
decision remains independent and remains open.
