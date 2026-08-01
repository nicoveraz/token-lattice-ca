# Assembly theory as a structure read-out for the token lattice

**Written 1 Aug 2026, and corrected three times the same day by the experiments it specified.**
§1.5/§3.5/§5.3/§5.5/§7 for F71/F72; §0/§3.1/§3.2/§4.1/§5.1/§5.2 for F73/F74. A third pass then swept
the whole document for claims the first two had left stranded: §3.4's pole symmetry (withdrawn per
F73), §3.5's internal contradictions, §5.2's `[z, RePair]` bracket and §6's exactness claim (both
still carrying the F74 unit error), §4.4 and §6's stop conditions (§5.2 has since answered them), and
**§5.3, which had its radii backwards** — r=2 is the instrument-selection rung, not the artifact to
exclude. §8 is new: a tracker review, since the issue list had drifted from the findings ledger.

A first-principles analysis of what this project has established, followed by a concrete program for
issue **#20** (assembly theory as a compositional-complexity axis), including a **pilot that was
actually run** against the data already in `results/`.

Read §0 if you read nothing else: it states what the pilot found, including the part that does not
work.

---

## 0. Summary

**The project's achievement is not a result, it is an instrument plus a discipline.**
Seventy-five findings, fifty-eight of them written up in `findings.md`, with every retraction,
correction and demotion still visible there. The headline dynamical claims were dissolved by the project's own
checks; what survives is a
black-box measurement apparatus for language models, a validation ladder with one **bit-exact** rung,
and a documented method for not fooling yourself that caught six confident wrong verdicts before any
of them reached a paper.

**Assembly theory fits a gap the project has already identified in writing.** The docstring of
`experiments/novelty_structure.py` states the problem exactly: "entropy is MAXIMISED BY NOISE… the
quantity that peaks at 'interesting' is excess entropy… It also cannot be estimated here." Excess
entropy needs block statistics over a 50,000-token vocabulary from a few thousand samples, which
measures the sample size rather than the system. **Assembly theory's ensemble quantity is a
computable non-monotone structure measure that needs no distribution estimate at all**, because it is
built from exact combinatorial counts on observed strings.

**What the pilot established, by running it:**

| | Result | Status |
|---|---|---|
| 1 | A **free calibration rung** exists: two string families have provable exact assembly indices. The estimator is **sound on both** — never below a proven bound — and **exact on the no-reuse family** at every n to 256 | **Solid, and now gated** (`experiments/assembly_calib.py`). But an exhaustive sweep **corrected this row**: RePair is *not* exact on `a^n` (75/127), see F73 |
| 2 | The **raw assembly index carries no word-order information.** Under length and multiset control it does not separate real text from word-shuffled text (0.4725 vs 0.4765 per character) — and neither does LZ77, gzip or entropy | **Solid.** The critique's home ground, conceded on our own data |
| 3 | The **exponential weighting is load-bearing.** Tempering `e^{a}` to `e^{a/2}` or to linear `a` *inverts* the ordering, so degenerate repetition beats real text | **Solid, and non-obvious.** This is the thing that separates AT from a compression measure |
| 4 | The right statistic is a **contrast against a matched word-shuffle**, `Δ = log A(text) − ⟨log A(shuffled)⟩`. It reads **exactly +0.00 on pure degenerate repetition** and **+0.00 on random soup**, against **+6.87 on real text**, at matched length | **Solid, with the symmetry withdrawn** (F73). The degenerate zero is measured; the noise zero is *definitional* (log floor). The substantive fact there is `A(text) = 0` — where entropy is maximal — now reported separately as `A_is_zero` |
| 5 | Applied to the existing CA output, Δ is large at low temperature and **exactly zero** at T ≥ 0.9 for both constructions, with MLM above AR | **Suggestive only — NOT established.** Window-to-window spread overlaps zero in several cells; effective object count is 1–2 |
| 6 | **No FLAT compression or entropy baseline reproduces Δ's ordering.** 11 of 12 peak on degenerate repetition; Δ alone peaks on real text, by 20× | **Solid** (F74, `assembly_baselines.py`). The difference is in *shape*, which no correlation coefficient explains away |
| 7 | **The assembly index plays no role in row 6.** Substituting character length, an LZ77 count, a constant, or a **random** weight for `a_i` all reproduce the ordering — random scores *highest*. Δ reduces to "real text has more repeated n-gram types than its own shuffle", exponentially amplified | **Corrects row 6's framing, and shrinks it** (§3.7). Not a result about assembly theory; a standard shuffled-background enrichment control under another name |

**The honest read on (5):** the direction is consistent across twelve cells and both constructions,
and the two zero-poles behave as designed. But every applied number rests on **one settle run per
cell**, and Δ is a tail statistic — the effective number of contributing objects is 1 to 2, so a
single chance repeat moves it. This is precisely the shape of F23 (pseudoreplication) and F57 (the
independent unit was not what it appeared to be). **It must not be quoted as a finding until it is
re-measured with the seed as the independent unit.**

**What the literature check found** (§4, full citations there). Applying assembly theory to text or
LLM output is **unoccupied** — zero prior art. So is testing the ensemble quantity at a critical point.
But **AT applied to cellular automata is not novel**: `AssemblyCA` (Patarroyo, Sharma, Walker & Cronin,
ALOE @ NeurIPS 2023) did it for 2D discrete CA, and must be read and cited. And the critique is
sharper than issue #20 assumed — the string assembly index is now *proven* to equal the smallest
straight-line-program size (NP-complete, APX-hard), so there is no room to claim independence from
grammar compression for the raw index.

**The rebuttal literature supplies this project's defence, and the pilot arrived at it
independently.** Kempes et al. (2025) showed the critics' assembly↔LZW correlations are **length
artifacts**: under a fixed-multiset permutation control the correlation collapses from ~0.9 to 0.25.
This project measures a **ring of fixed N**, so it is natively in that regime — and §3.2's
real-versus-word-shuffled test *is* that permutation control, applied to text for the first time.

**The single experiment worth building** is in §5.3: the AR construction has a *known* degenerate
pole at low temperature (F62–F70) and a known noise pole at high temperature, and Δ must go to zero
at both. The MLM construction has **no** degenerate pole (F67), so it must not turn over at low T.
That is a pre-registered differential prediction with a control that should behave differently, on a
system where both poles were established independently and in advance. It is the strongest test of
assembly theory available anywhere in this repository, and it needs no new compute beyond settle runs
the project already performs.

The result that would survive review, stated now so it cannot be constructed later: **Δ non-monotone
in temperature while every compression baseline stays monotone.** A difference in *shape* is not
explainable by a correlation coefficient. Its null — Δ monotone too — closes the thread cleanly.

---

## 1. The project from first principles

### 1.1 What the object is

A ring of *N* cells, each holding one token. Pick a cell, show a language model the neighbouring
tokens, ask what goes here, sample from the answer at temperature *T*, write it back. Repeat forever
in random visit order. That is a cellular automaton whose transition rule is a language model.

The point of the construction is that it converts a language model — normally studied by prompting
and reading output — into a **dynamical system**, which can be perturbed, which has phases, and to
which the whole apparatus of statistical physics applies.

### 1.2 What was actually built

Three things, in increasing order of durability.

**The apparatus.** One simulation loop (`src/lattice.py`) shared by every backend — a toy JAX
transformer, an autoregressive HuggingFace model, a masked LM, the Domany–Kinzel PCA, and a stub used
in tests. Because they share the loop, a test on the stub exercises the same code path as a
measurement on pythia-410m *by construction*. The core probe is common-random-number damage
spreading: two replicas sharing model, initial state, uniform stream and visit order must differ in
**exactly zero** cells, and that null is asserted on every backend in both update modes.

**The validation ladder.** Before measuring a system whose answers are unknown, reproduce systems
whose answers are known. Elementary CA rules (ordered vs chaotic separates at p = 0.0000, d = 3.03,
on ignition probability); synthetic Markov sources with known transition matrices; a coupled-map
lattice against an exact Benettin reference; the logistic map as a smooth-limit arithmetic check —
explicitly demoted to a unit test after F30 showed its estimator was circular.

**The one rung that is exact.** On the Domany–Kinzel `p2 = 0` line, the CRN damage field is provably
*itself* a DK automaton at the same `p1`. So the damage machinery has an independent prediction to
check against, and it reproduces it **bit-for-bit: zero mismatching cells, no error bar**, through
the same loop that produces every language-model number. Published critical points come back to
0.15% and 0.06%. This is the credibility spine, and it is worth more than any measurement in the
repository.

### 1.3 What was established, sorted by what it can bear

**Durable — methods.**

- Damage spreading, ported to real language models, with the exact-zero null holding throughout.
- The DK identity as a bit-exact anchor for the whole measurement chain.
- A gating discipline: every estimator re-validates on a known-answer system **at its own geometry**
  before it is allowed to report, and returns `NOT DECIDABLE` rather than a number when it fails.
  Three confident verdicts died to this gate, none of which was visible from the language-model
  numbers themselves.

**Durable — negative and delimiting results.** These are the strongest scientific content here.

- **F35.** Real autoregressive generation does not absorb a single injected token error:
  `P_persist = 1.000` across three models, `TV_norm ≈ 0.97`. Free generation never resamples a token,
  so an error stays in context permanently; the ring revisits every site, which is what makes healing
  possible at all. **Therefore every repair/damping number characterises the construction, not the
  model's generative process.** This retroactively explains three earlier puzzles.
- **F26/F28/F29.** Black-box token-space criticality does not proxy white-box activation-space
  Lyapunov exponents. The negative is structural, not a power problem: white-box `λ_top` is
  architectural (flat across training, ≈ 1/L).
- **F62–F70, the artifact line.** A damage-spreading transition was found at T_c ∈ [0.4343, 0.4391],
  its exponents fitted, and then it dissolved. A second model family had no transition (F62);
  nineteen models refuted a corpus explanation (F63); attention is necessary and corpus determines,
  with scale eliminated over a 70× and a 12× ladder (F64); the frozen phase exists only at r ≤ 2 and
  one vocabulary entry carries it (F65/F69); one BOS token removes two thirds of it and the masked-LM
  construction shows none (F66); the clean construction has no transition either, the pre-registered
  good null (F67); and the mechanism is **an attracting fixed point of the argmax map** — 18 of 24
  random starts reach `'\n'` on pythia-410m, while gpt2-medium's map has none and wanders to 11
  endpoints (F70).

**Not established, and correctly labelled so.**

- **T\***, the melting temperature of the single-token degeneracy, as a predictor of neural text
  degeneration. Model-level ρ = +0.575, p = 0.028 over 15 models; **family-level ρ = +0.483,
  p = 0.189 over 9 families.** Six of the fifteen points are Pythia sizes. Roughly 16 independent
  families would settle it. Worth reading F68 in full: it is the clearest statement in the repo that
  "collect more data until p drops" would have declared victory at n = 15.
- The universality class of the transition: **withdrawn**, because there is no model-independent
  transition to classify.
- The ladder anomaly of F60 — a ~7% downward bias on DK for ladders reaching N ≥ 96, surviving 16×
  sampling, with six hypotheses refuted — remains **unexplained** and must be disclosed by anything
  quoting a dynamic exponent.

### 1.4 The discipline is the asset

Eleven retracted claims, eight retractions, six corrections, five amendments, three demotions — all
still visible in `findings.md`, deliberately. The transferable rules, each bought with at least one
dead verdict:

1. **Gate every estimator at its own geometry** (F56). A calibration at N=512/200 sweeps licenses
   nothing at N=96/40 — the tolerance so derived rejects Domany–Kinzel itself.
2. **State what the independent unit is, and test it** (F57). One visit order was drawn per *batch*,
   so 512 replicas carried the weight of one draw and error bars were ~8× too small.
3. **A cost function that can shrink its own comparison window is unbounded** (F59-v1).
4. **Run a control that should show nothing** (F65). The radius sweep read clean until the control
   acquired the effect too.
5. **Vary the construction, not only the subject** (F66). Nineteen models could not distinguish "a
   property of language models" from "a property of the probe"; one change of CA did it immediately.
6. **Evaluate an estimator in the regime the system actually runs in** (F70). Measuring top-1
   probability mass at T=1.0 refuted the mechanism; measuring the argmax map at T=0.02 — where the CA
   actually runs — confirmed it.

Anything proposed below has to survive these six. §5 is organised so that it does.

### 1.5 Where the work stands, as of today

`paper/plan_paper2.md` reports claims **A + B + C complete with zero further compute**, and claim A
closed. The open frontier is three uncommitted or just-landed experiments:

- **#93 `novelty_structure`** — does the CA create, recall, or randomise? Measured on a two-axis
  plane: word-n-gram novelty against per-token NLL from a *third* model (`gpt2-large`, which never
  generates). Result (**F71**, after two corrections): **the MLM construction reaches +0.669 at
  r=8, T=0.3 — 94% of the way to shuffled on novelty while only 27% of the way on
  unpredictability**, with no cell excluded. **The AR construction shows none** — 7 of 12 cells are
  whitespace padding, 3 more are *more unpredictable than word-shuffled text*, and the best
  survivor is −0.021. An earlier "+0.157 for AR" was withdrawn: the gap statistic is scale-free and
  stayed positive when both fractions exceeded 1, so a cell worse than shuffling on both axes was
  being scored as structured. Cells must now lie **between** the references.
- **#94 `basin_dependence`** — **F72, and it reverses the reading this document was drafted under.**
  **The prompt is erased.** Random and corpus seeds settle to the same composition (max top-1 gap
  0.053 AR, 0.022 MLM) and only 2% / 12% of a corpus seed survives in place, so #93's
  novelty-from-noise is *representative* rather than one basin among several. The first verdict said
  "INITS DIVERGE, basins are real, novelty is prompt-relative"; that spread was computed across the
  degenerate fixed-point seed, which never moves. **Separately**, a ring filled with the attractor
  token is a trap in 2/6 AR and 6/6 MLM cells — self-sustaining, but with a basin so small neither
  random nor text seeds reach it.
- **#90 T\*** — awaiting roughly seven more independent families.

**The gap #93 leaves open is the one assembly theory addresses.** Its structure axis is a **NLL from
another neural model**. That is a defensible choice and the circularity was thought about — but it is
still a model-relative quantity, it truncates at 512 scorer tokens, and it was already shown to be
gameable by whitespace padding, which had to be patched with `collapse_ws` and a word-density filter
after the confound was caught. A **model-free, combinatorial** structure axis would be an independent
line of evidence on exactly the question #93 asks.

---

## 2. What assembly theory is, and the one thing that matters

The **assembly index** `a(x)` of an object is the minimum number of joining steps needed to build it
from basic building blocks, where **any object already built may be reused at no further cost**.

For a string this has an exact reading: the assembly index is the size of the smallest binary
straight-line program deriving the string — each step concatenates two things already in the pool.
`aaaaaaaa` costs 3 steps (`aa → aaaa → aaaaaaaa`). A string of eight all-distinct symbols costs 7,
because nothing repeats so nothing can be reused. Computing it exactly is the smallest-grammar
problem and is NP-hard, but a greedy pairing algorithm (RePair) *exhibits a pathway*, so it is a
**certified upper bound** rather than a fitted estimate.

**The naive move fails, and the pilot confirms it.** Raw assembly index behaves like every other
compression measure: low for periodic strings, high for random ones, monotone in disorder. It is
therefore maximised by noise, which is the exact defect that ruled entropy out for this question. §3.2
shows it failing to separate real text from word-shuffled text on our own data.

**The quantity that does not fail is the ensemble one.** Assembly theory does not score single
objects; it scores an ensemble, combining assembly index with **copy number**:

```
        A  =  SUM over object types i  of   exp(a_i) * (n_i - 1) / N_T
```

where `n_i` is how many copies of type *i* were observed and `N_T` the total. The `(n_i − 1)` factor
is the whole idea: **an object with a high assembly index appearing many times cannot have arisen by
chance**, because high-`a` objects are exponentially unlikely to be hit twice independently. So:

- **Random ensemble** — every object unique, so `n_i − 1 = 0` and **A = 0**, no matter how high the
  assembly indices are. This is the property entropy does not have.
- **Degenerate ensemble** (the newline attractor) — enormous copy number, but `a_i ≈ log(length)`, so
  `exp(a_i)` stays small and **A stays small**.
- **Structured ensemble** — objects of substantial assembly index recurring, so **A is large**.

That is a non-monotone structure measure computed from exact counts, with no distribution to
estimate. It is what `novelty_structure.py` says it needs and cannot have.

---

## 3. The pilot — what was run, and what it showed

All of §3 was executed against `results/novelty_structure.json` and `data/shakespeare.txt`.

**The scripts are no longer a pilot.** §5.1 and §5.2 have since been built as gated experiments
(`experiments/assembly_calib.py`, `experiments/assembly_baselines.py`), the estimator lives in
`assembly_calib.py` as the single implementation, and `_assembly_pilot.py` imports it rather than
carrying a copy. Two of the claims below were **corrected by the experiments they specified** —
§3.1 on exactness (F73) and §3.2's clause about compression baselines (F74) — and both corrections
are in place inline rather than appended.

### 3.1 A calibration rung, and it is free

The project's rule is that an estimator earns the right to report by reproducing a known answer
first. Assembly index admits two string families with **provable** exact values:

- `a^n` — the exact index is the minimal **addition-chain length** for *n*, computable by breadth-first
  search over reachable pools. (Each step at most doubles the longest object.)
- **All-distinct symbols**, length *n* — the exact index is `n − 1`. No substring repeats, so no reuse
  is possible; the pathway is a binary tree with *n* leaves.

RePair against those references:

```
  family 1:  a^n                          family 2:  all-distinct
     n   exact   repair   gap                n   exact   repair   gap
     2       1        1     0                2       1        1     0
     3       2        2     0                4       3        3     0
     5       3        3     0                8       7        7     0
     7       4        4     0               16      15       15     0
     8       3        3     0               32      31       31     0
    24       5        5     0               64      63       63     0
   100       8        8     0               90      89       89     0
   128       7        7     0
```

**This table is a 14-point sample, and the claim first drawn from it — "exact at every size, on both
families" — is false.** `assembly_calib.py` sweeps *every* n from 2 to 128 and finds RePair exact on
**75 of 127** and overshooting on 52. The smallest failure is **n=15**, where the minimum is 5
(1,2,3,6,12,15) and greedy halving finds 6 — the textbook smallest n at which the binary method is
not an optimal addition chain, so it was always going to be there. The sampled values above happened
to contain none of the 52. That is the F64 failure mode exactly: a property of the sample stated as a
property of the phenomenon.

**What survives is the property that was load-bearing anyway.** RePair never returns a value below a
proven lower bound — **127/127** — which is what makes it a *certified upper bound* rather than a
fitted estimate, and family 2 is **exact at every n to 256**. Exactness on `a^n` was never needed
downstream, and the overshoot is in the conservative direction: it inflates `a_i` and therefore
`e^{a_i}`, which is what would make a *degenerate* ensemble look more structured — and the degenerate
pole still reads +0.00 (§3.4). This is a genuine new rung on the validation ladder that costs
CPU-seconds and needs no model — cheaper than any rung currently on it.

### 3.2 The raw assembly index carries no word-order information — the critique's home ground

The standing objection to assembly theory is that the assembly index is a repackaged compression
measure (§4.2). The decisive test is the **length- and multiset-controlled** one, because
word-shuffling preserves length, vocabulary and unigram frequencies exactly while destroying all word
order:

```
  case                     a_hat   a/len   lz77_z   gzip_bits   H_bits
  periodic (ab)*              15  0.0075       38         192     2000
  real text (Shakespeare)    945  0.4725      766        8000     9093
  word-shuffled text         944  0.4765      770        8120     8773   <-- indistinguishable
  char-shuffled text        1384  0.6920     1166       10232     9093
  uniform random (26 sym)   1451  0.7255     1201        9952     9390
```

Two statements, and they must be kept apart.

**Across regimes that differ in disorder, the raw index and LZ77 track each other closely.** That is
consistent with the correlations the critics report (Pearson 0.874–0.99 against LZW) — and it is
measured in the same confounded way, since these regimes differ in far more than word order.

**Under length and multiset control, the raw index is null.** Real and word-shuffled text differ by
**0.004 in assembly rate**. The raw assembly index carries no information about word order at this
granularity. Reporting it as a "complexity" read-out of generated text would be reporting gzip with
extra steps. §5.2 confirms this against a *20-shuffle ensemble* rather than the single shuffle used
here: the raw index reads **z = −1.78**, inside noise.

**"— and so does every compression baseline" was the overstatement, and F74 withdraws it.** That
clause rested on comparing single numbers without asking whether the difference was large against
shuffle-to-shuffle scatter. Against an ensemble, **gzip separates real text from its own shuffles at
z = −8.13**, lzma at −7.23, LZ77 at −5.30 and block entropy at −3.89. Compressors are *not* blind to
word order under multiset control. The case for the ensemble quantity therefore cannot be "compression
cannot see this"; it rests on the ORDERING across regimes, which is where §5.2 finds the separation.

**The ensemble quantity separates the same pair by 6.87** (§3.4). That gap between the two rows is
the entire case for assembly theory here, and it lands exactly where the rebuttal literature says it
should — see §4.2.

### 3.3 The exponential weighting is load-bearing — the non-obvious result

`exp(a_i)` is aggressive, so the natural instinct is to temper it. **Tempering destroys the
property that makes the measure useful.** At matched length, over the five reference regimes:

```
  weighting        ordering of log A                                                verdict
  e^{a}   (as defined)  real 4.61 > cycle 3.04 > single 2.17 > soup 1.58 > shuf -0.62   CORRECT
  e^{a/2}               cycle 1.52 > real 1.19 > single 1.09 > soup -0.58 > shuf -1.71  INVERTED
  a       (linear)      cycle 0.84 > single 0.70 > real -0.73 > soup -1.67 > shuf -2.10 INVERTED
```

Under linear weighting the measure is pure copy number and **degenerate repetition wins outright** —
`"the of the of…"` scores above real text. Only the full exponential penalises trivially-assembled
objects hard enough to keep real text on top. So the `exp` is not decoration; **it is the entire
difference between assembly theory and a repetition count**, and that is a concrete empirical
contribution to a debate that has been conducted mostly on theoretical grounds.

The cost is that `A` becomes a **tail statistic**. Measured effective object count is 3.4 of 18
repeated types for real text, and 1.0 for several CA cells — meaning one repeat carries the number.
At n=4 the top object carried **100%** of `A` in one cell (`'still lifes and portraits'`, appearing
twice). Any use of `A` must report the effective object count next to it.

### 3.4 The statistic to use: a contrast against a matched shuffle

Because `A` grows with sample length (Shakespeare: log A = 3.87 / 4.61 / 5.20 at 3.5k / 7k / 20k
characters) and is tail-dominated, the absolute value is not comparable across cells. The contrast is:

```
        Delta  =  log A(text)  -  < log A(same words, shuffled) >
```

The shuffle preserves length, vocabulary and unigram frequencies exactly, so it is the tightest
available control — the analogue of this project's CRN null. Length-matched to 440 words, median of
5 contiguous windows, 6 shuffles each:

```
  case                Delta      [min,max]
  real text           +6.87  [+5.75,+8.33]
  degenerate x1       +0.00  [+0.00,+0.00]     "the the the ..."
  degenerate x2       +0.35  [+0.35,+0.35]     "the of the of ..."
  random soup         +0.00  [-1.08,+0.00]
  unique tokens       +0.00  [+0.00,+0.00]
```

Real text sits far above both poles. That is the non-monotone behaviour the project needs, and it is
a property of the statistic rather than of a tuned threshold.

**But "both poles pin at zero automatically" oversold it, and F73 withdraws the symmetry.** The two
zeros are not the same kind of fact:

- **Degenerate pole — a genuine measurement.** Shuffling a string of identical tokens returns the
  same string, so Δ = 0 is something the estimator *found*. A repetition detector could not do this.
- **Noise pole — definitional, not measured.** Both the text and its shuffle have `A = 0`, and the
  log floor maps both to the same constant, so Δ = 0 follows from the arithmetic whatever the input.

The substantive fact at the noise pole is therefore **`A(text) = 0` itself** — and that is the
stronger claim anyway, because it is exactly where **entropy is maximal**. It is now reported
separately as `A_is_zero` rather than being folded into Δ, so the two cannot be read as one result.

### 3.5 Applied to the CA — direction consistent, power insufficient

Length-matched to 440 words, as emitted by `experiments/_assembly_pilot.py matched`:

```
  cell             Delta      [min,max]      cell             Delta      [min,max]
  ar|ref           +5.03  [+4.61,+5.25]      mlm|ref          +4.85  [+2.84,+5.25]
  ar|shuf          +0.00  [-2.13,+0.00]      mlm|shuf         +0.00  [-1.13,+2.96]
  ar|r3|T0.3       +3.16  [-2.06,+5.48]      mlm|r3|T0.3      +8.70  [-0.64,+9.48]
  ar|r3|T0.5       -1.41  [-2.29,+3.92]      mlm|r3|T0.5      +3.69  [-0.71,+3.85]
  ar|r3|T0.7       +0.00  [-0.92,+0.00]      mlm|r3|T0.7      +7.03  [+4.27,+7.25]
  ar|r3|T0.9+      +0.00  [+0.00,+0.00]      mlm|r3|T0.9      +0.00  [+0.00,+5.14]
  ar|r8|T0.3       +1.65  [+0.60,+2.85]      mlm|r8|T0.3      +7.39  [+6.38,+8.46]
  ar|r8|T0.5       +2.30  [+1.72,+2.66]      mlm|r8|T0.5      +5.13  [+4.98,+9.07]
  ar|r8|T0.7       +1.70  [+0.78,+2.26]      mlm|r8|T0.7      -0.57  [-1.72,+7.62]
  ar|r8|T0.9+      +0.00  [+0.00,+0.00]      mlm|r8|T0.9+     +0.00  [-0.64,+0.00]
```

Three things are consistent and one is not.

**Consistent.** The references behave: real round-tripped text reads +5.03 / +4.85, and its shuffle
reads +0.00 in both. Every cell at T ≥ 0.9 reads **exactly +0.00** in both constructions — at high
temperature the ring contains no repeated 3-gram at all, which is the noise pole reached from data.
And **MLM sits well above AR** at low temperature (+7.4 to +9.0 against +2.1 to +2.4), agreeing in
direction with F71's independent finding that the MLM construction is the one showing structured
novelty **and the AR construction shows none**, from a statistic that shares no machinery with it.
The agreement tightened when F71 was corrected: AR's apparent +0.157 was withdrawn, and the pilot
already had AR near zero.

**Not consistent — and this is the reason nothing here is a finding.** The window-to-window spreads
are enormous and several overlap zero: `mlm|r3|T0.3` is +8.70 over a range of [−0.64, +9.48];
`mlm|r8|T0.7` is −0.57 over [−1.72, +7.62]. Effective object counts are 1.0–2.3. There is **one
settle run per cell**, so the independent unit is a single draw of the thing that decides the outcome
— the identical structure as F57. The convergence with F71 is encouraging and is *not* evidence until
it is re-measured with seeds as independent units, which is what §5.3 is running.

**A direct demonstration of that, found while consolidating the pilot scripts.** The first pass and
`_assembly_pilot.py` differ only in how the shuffle RNG is seeded across windows — nothing about the
CA, the texts, or the estimator changed. The **controls and references were unmoved** (real text
+6.87 both times, degenerate exactly +0.00, `ar|ref` +5.03, `mlm|r8|T0.3` +7.39), but individual CA
cells shifted by up to **0.8** (`ar|r3|T0.3` +2.36 → +3.16, `ar|r8|T0.5` +1.87 → +2.30, `ar|r8|T0.7`
+2.26 → +1.70). The poles are stable; the cells are not. That is the underpowering made visible, and
it is the reason §5.3 specifies eight seeds rather than one.

**One shape worth flagging in advance, because it contradicts the obvious hypothesis.** Issue #20
predicts assembly peaks at the edge of chaos. Over the tested range Δ is **monotone decreasing in
temperature** for MLM at r=8 (+7.39 → +5.13 → −0.57 → +0.00) and roughly flat then zero for AR.
There is no interior peak — because the sweep starts at T = 0.3 and the AR degeneracy lives *below*
it. Finding the peak requires going down to T ≈ 0.02, which is what §5.3 does — and note that the
degeneracy lives at r ≤ 2, not at the r ∈ {3, 8} tabulated here, which is the correction §5.3 had to
make to its own grid.

### 3.6 Confounds found in the pilot that any real experiment must handle

1. **Ring rotations.** The lattice is periodic, so a phrase and its rotations are distinct linear
   substrings. Two of the top-contributing objects in one cell were `'he thought berlin germany'` and
   `'germany he thought berlin'` — the same content counted twice.
2. **Replica concatenation.** `full_text` joins all 16 replicas, so cross-replica convergence is
   scored as within-text structure. Those are different claims and must be separated.
3. **Novelty may be overstated in the co-measurement.** Top contributors read like memorised corpus
   fragments (`'still lifes and portraits'`, `'confession of faith'`, `'the church of st'`) while
   scoring `novel_4gram ≈ 0.95` against a 3000-document reference. A 3000-document sample of the Pile
   cannot rule out recall. High Δ with high measured novelty may be **recall the reference is too
   small to see**.
4. **Length.** `A` grows with word count; cells hold 443–1137 words. Matched truncation is mandatory.
5. **Sample size vs n.** At n = 4 real text scored `A = 0` in a 606-word sample — an undersampling
   artifact, not a property of real text. Use n = 2 and 3 at these lengths, or raise the budget.

### 3.7 The substitution control — the assembly index is NOT what carries the result

**This overturns how §5.2's result was being described, and it was found by asking the obvious
question one step later than F74 asked it.**

F74 compared Δ against **flat** measures — gzip, lzma, LZ77 and the rest applied to the whole text
and contrasted against shuffles. Eleven of twelve peaked on degenerate repetition where Δ peaked on
real text, and that was read as evidence that assembly theory differs from compression. But Δ stacks
four things, and F74 varied **none** of them:

```
  (1) a per-object complexity   a_i
  (2) an exponential weight     e^{a_i}
  (3) copy-number coupling      (n_i - 1)
  (4) a matched-shuffle contrast
```

So F74 establishes that the **stack** beats flat compression. It does not establish that (1) has to
be the assembly index. The decisive control keeps (2), (3) and (4) exactly and substitutes a
non-assembly per-object term. At 440 words, 3-grams, against the five reference regimes:

```
  regime           a_i (assembly)   len_i (chars)   z_i (LZ77)   const (no complexity term)
  real text             +6.09           +6.40         +6.47            +1.30
  degenerate x1         +0.00           +0.00         +0.00            +0.00
  degenerate x2         +0.35           -0.10         +0.35            +0.01
  random soup           -0.70           -0.80         -0.75            -0.10
  unique tokens         +0.00           +0.00         +0.00            +0.00

  peaks on REAL TEXT:   all four
```

**Every substitution reproduces the ordering, and character length and LZ77 do it marginally
better than the assembly index does.** Even a *constant* weight — no complexity term at all, pure
copy number against a shuffle — peaks on real text, at +1.30. The exponential **amplifies** the
effect (6.09 against 1.30) but does not create it.

**And one more substitution collapses it further: a RANDOM per-object weight.**

```
  regime          a_i (assembly)   RANDOM e^U(0,20)   const   # repeated 3-gram types
  real text            +6.09            +6.34         +1.30            4
  degenerate x1        +0.00            +0.00         +0.00            1
  degenerate x2        +0.35            -1.74         +0.01            2
  random soup          -0.70            -0.15         -0.10            0
  unique tokens        +0.00            +0.00         +0.00            0
```

`e^{U(0,20)}` — a weight drawn from **noise**, carrying no information about the object at all —
peaks on real text at **+6.34, above the assembly index's +6.09**. So the per-object term is not
interchangeable, it is **irrelevant**. What produces the ordering is the last column: real text has
more repeated 3-gram types than its own shuffle does, degenerate text has the same as its shuffle,
and noise has none. Any heavy-tailed weighting amplifies that count difference into a large
log-scale gap; the assembly index is one such weighting and has no privileged role.

**So the honest statement shrinks again, to something close to a tautology:**

> Real text contains more repeated n-gram types than a word-shuffle of itself. Degenerate
> repetition does not (shuffling it returns it). Noise contains none. Δ detects that, and the
> exponential turns a count difference of 4-vs-1 into six orders of magnitude.

That is a legitimate structure measure with both failure poles pinned at zero, and it is **not** a
result about assembly theory. It is also standard practice elsewhere under another name —
enrichment of repeated motifs against a shuffled background is the ordinary control in corpus
linguistics and in bioinformatics.

**Note the sample size in that last column.** Four repeated 3-gram types in 440 words of
Shakespeare. Every number in §3 rests on counts of that order, which is the quantitative form of
the tail-domination warning in §3.3 and the reason §5.3 runs eight seeds.

**What this changes.** The claim is not "assembly theory separates from compression." It is:

> A **copy-number statistic read against a matched-multiset shuffle** separates real text from
> degenerate repetition and from noise, where flat compression and entropy measures do not. The
> per-object complexity term inside it is interchangeable.

That is narrower, and it is the honest version. It also **re-opens the per-object InChI-length
confound** that §5.2 claimed was disarmed: length is held constant *in aggregate* by the shuffle,
but the per-object correlation between an n-gram's length and its assembly index is untouched, and
`len_i` works just as well. §5.2's "provably cannot operate here" was true of the global confound
and false of the per-object one.

**What still stands.** §3.2 (the raw index carries no word-order information, z = −1.78) is
unaffected — this is the same conclusion reached from a second direction. F74's finding that flat
measures peak on degeneracy is unaffected, and it remains the useful result for §4.5. And §3.3's
weighting sweep is unaffected *within* the A-family, though it now reads as a fact about the
exponential rather than about assembly.

**Power, stated because the direction matters more than the numbers.** One text, five regimes,
eight shuffles, one seed, 3-grams only — the same underpowering §3.5 is flagged for. The ordering is
unambiguous across four weightings, but the *magnitudes* should not be quoted. `experiments/_assembly_substitution.py` reproduces it; §5.2 should absorb it as a
gated arm.

---

## 4. Prior art and novelty

A literature check was run alongside the pilot (arXiv API sweep over all papers matching
`all:"assembly index"`, 2019–2026, plus targeted searches). Items marked **[unverified]** could not be
fetched directly and must be checked before citing.

### 4.1 The theorem that settles the estimator question

**The string assembly index is exactly the smallest straight-line-program size.** Masierak,
*Computational Complexity of Determining the Assembly Index* (arXiv:2604.16302), Lemma 1: `ASI(w) =
SLP(w)`; Theorem 1: the decision problem is **NP-complete**; Theorem 2: the optimisation version is
**NP-hard and APX-hard** — no polynomial-time approximation scheme unless P = NP.

This is good news for §3.1, not bad. It means the estimator design is forced and defensible: RePair
*exhibits a grammar*, so it is a **certified upper bound**.

**The lower-bound half of this paragraph had a unit error, corrected in F74.** `z ≤ g` (Rytter 2003 /
Charikar et al. 2005 — still **[unverified]**) is stated for `g` = the **total length of all
right-hand sides**. A binary SLP with `r` rules has total RHS length `2r`, and the assembly index *is*
the binary rule count, so `g = 2·ASI` and the theorem gives **`z ≤ 2·ASI`**. The lower bound is
**`z/2`**, not `z`, and `[z, RePair]` is **not a bracket** — z exceeds RePair on ordinary text (11 vs
10 on `"abracadabra"×6`, 20 vs 19 on repeated English). Report **`[z/2, RePair]`**, which holds with
room to spare, and claim exactness only on the no-reuse family of §3.1, where it holds.

The AT camp has itself converged on grammar algorithms: Siebert, Chowdhury, Slocombe & Walker,
*Assembly Spaces: Formal Definitions and Fast Methods for Approximating Assembly Indices*
(arXiv:2606.15499, 2026), "show how formal grammar algorithms can be adapted to efficiently bound
assembly index calculations". There is **no published general-purpose exact string-assembly tool** —
the `assembly-theory` Rust crate (Vimal et al., JOSS, DOI `10.21105/joss.09318`) is molecules-only.
Ours had to be written, and now is: `experiments/assembly_calib.py` (§5.1).

### 4.2 The contested status, stated fairly

**The critique.** Abrahão, Hernández-Orozco, Kiani, Tegnér & Zenil, *PLOS Complex Systems* 2024 (DOI
`10.1371/journal.pcsy.0000014`, arXiv:2403.06629): "We prove the full equivalence between Assembly
Theory and Shannon Entropy via a method based upon the principles of statistical compression renamed
'assembly index' that belongs to the LZ family." Supporting empirics (Ozelim et al., *npj Complexity*
2026, DOI `10.1038/s44260-026-00088-w`, arXiv:2408.15108): LZW vs assembly Pearson **0.874**;
Spearman **1.00** on fixed-length strings; **0.95 between InChI string length and assembly index**;
and on mass-spec biosignature classification LZW and BDM reach **AUC 1.00** against assembly's
0.43–0.77. In its home domain, assembly lost the head-to-head.

**The rebuttal, and it contains the defence this project should adopt.** Kempes, Lachmann,
Iannaccone, Fricke, Chowdhury, Walker & Cronin, *npj Complexity* 2025 (DOI
`10.1038/s44260-025-00049-9`, arXiv:2406.12176): assembly index is NP-complete while Huffman coding
and Shannon entropy are in P, so "any complexity measure that lives in P cannot be strictly
equivalent". More usefully, they run **a length- and multiset-controlled experiment**: over 10,000
permutations of the fixed multiset `aaaaaabbbbbbcccc`, the assembly↔LZW correlation collapses from
~0.9 to **0.25**. Their reading — which the InChI-length correlation of 0.95 supports — is that the
critics' high correlations are **length artifacts**.

**Why this matters here, and it is the strongest structural point in this document.** This project
measures a **ring of fixed N**. It is natively in the length-controlled regime where the two measures
decorrelate. And the pilot's control was arrived at independently: §3.2's real-versus-word-shuffled
comparison **is** the Kempes permutation test, applied to text. It reproduces their result in a new
domain — under multiset control the raw index goes null — and then shows the ensemble quantity
separating the same pair by 6.87. That is the shape of a defensible contribution rather than a
position in someone else's argument.

Note also Łukaszyk, *IPI Letters* 3(1) 2025 (DOI `10.59973/ipil.157`), which disputes the rebuttal's
own complexity claim. The debate is live; do not present either side as settled.

### 4.3 What is novel here, precisely

| Claim | Novel? | Prior art |
|---|---|---|
| AT applied to natural language / LLM-generated text | ✅ **Yes** — zero prior art found | The AT reviews *name* language as an anticipated domain. Cite that, own it, claim execution not conception |
| **Assembly A tested at a critical point / edge of chaos** | ✅ **Yes** — zero prior art found | Nearest: arXiv:2602.15185, a *compression*-based complexity peaking at the 2D Ising T_c |
| AT ↔ Crutchfield statistical complexity / excess entropy | ✅ **Yes** — no connection established anywhere | — |
| AT applied to cellular automata | ❌ **No** | **AssemblyCA** — Patarroyo, Sharma, Walker & Cronin, ALOE @ NeurIPS 2023 |
| AT applied to strings, formally | ❌ **No** | Masierak (arXiv:2602.04889, 2604.16302); Łukaszyk & Bieniawski, *Mathematics* 12(10):1600 |
| AT outside chemistry at all | ❌ **No** | Organisational design, DOI `10.1007/s41469-024-00182-0` |

**`AssemblyCA` must be read and cited before anything is written.** It applies assembly index and copy
number to 2D discrete cellular automata to quantify open-endedness, using a Hashlife-style
"Hash Assembly" approximation. Reported differences, from the project site and repository rather than
the PDF (**[unverified]** — read it): it works on **2D spatial patterns**, not 1D token strings; it
does **not** use the exponential assembly equation `A = Σ e^{a_i}(n_i−1)/N_T`; and it does **not**
discuss criticality, edge of chaos, or Wolfram classes. Those three gaps are the differentiation, but
they have to be confirmed from the paper.

**The defensible novelty is the conjunction, not any ingredient:** *1D token strings generated by an
LM-driven CA, measured with the full assembly equation rather than the index alone, tested for
non-monotonicity across a transition whose two poles were independently established by damage
spreading.* Do not claim "first application of AT to strings" or "first AT + CA" — both are false and
both are one search away.

### 4.4 The two objections a reviewer will raise, and the answers

**"You have re-badged grammar compression."** Answer in three parts, in this order: (i) concede it for
the raw index and show the concession in your own data (§3.2) — the project's record is built on
conceding early; (ii) the object of study is **A**, the exponentially-weighted copy-number-coupled
ensemble quantity, which has no compression analogue, and §3.3 shows the exponential is load-bearing
rather than decorative; (iii) the experiments live at fixed N, the regime where Kempes et al. showed
the measures decorrelate.

**"Lindgren & Nordahl (1988) already gave one-hump complexity measures for cellular automata, and
Crutchfield's statistical complexity already peaks at criticality — what does AT add?"** This is the
sharper objection and it has no rhetorical answer, only an empirical one. §5.2 ran it: **neither
`C_μ` nor excess entropy peaks on real text** (ρ = −0.15 and +0.77 against Δ; they peak on random
soup and on degenerate repetition respectively). So Δ is **not** a redescription of statistical
complexity *as estimated here* — but "here" is a coarse, frequency-bucketed estimator on 440 words.
A proper CSSR reconstruction could still peak with Δ, and this experiment could not tell. **The
objection is weakened, not retired**, and Lindgren & Nordahl remains the one to answer.

**The result that would survive review**, stated in advance so it cannot be constructed afterwards:
Δ is **non-monotone in temperature with an interior peak, while LZ77, LZ78/LZW, gzip and the entropy
rate are monotone across the same sweep**. A qualitative difference in shape is not explainable by a
correlation coefficient. §5.2 has already delivered half of it — the ordering across regimes — and
§5.3 is running the other half, the shape across temperature. Its null is the case where Δ is
monotone too.

### 4.5 The external framing target — an assumption in the literature that §5.3 measures

Mohsin et al., *On the Fundamental Limits of LLMs at Scale* (TMLR 07/2026,
`openreview.net/forum?id=BIRDGVrom8`), §2.4, posits a **creativity–factuality trade-off**:

> low temperatures (T ≈ 0.1–0.5) produce repetitive, conservative text with high factual accuracy
> but limited novelty. High temperatures (T ≈ 1.0–2.0) yield diverse, creative outputs but also
> frequent hallucinations

formalised as a constrained capacity `A(θ) + α·C(θ) = κ`, giving `dA = −α·dC` — a **monotone**
trade-off with constant α. Their own Table 1 files this under *"modeling assumption / framework"*,
not as proved and not as an empirical observation. **So it is a named assumption in a current
journal paper, awaiting evidence.**

Two points of contact, and the second is the sharper one.

**§5.3 measures the shape.** Its temperature grid spans both regimes §2.4 names, and its
pre-registration is precisely whether the structure measure is non-monotone with an interior
maximum — which a constant-α scalar trade-off cannot produce. This is the **external anchor** F68
lacked and issue #20 asked for: it turns "does Δ peak?" from an internal curiosity into a test of a
stated assumption in the literature.

**F74 says their `C` is not instantiable with an off-the-shelf FLAT metric.** The paper defines
`C(θ)` as "a creativity metric measuring diversity, novelty, or originality" and treats it as a
well-defined scalar. §5.2 measured twelve candidates for exactly that metric — LZ77, LZW, RePair,
gzip, bz2, lzma, unigram and block entropy, entropy rate, excess entropy, coarse `C_μ`, integrated
MI — and **eleven of twelve peak on degenerate repetition** rather than on real text. Instantiating
`A + αC` with any of them makes creativity maximal on `"the of the of…"`.

**But the repair is not "use the assembly index" — see §3.7.** The substitution control shows the
per-object assembly index is interchangeable with character length or an LZ77 phrase count, and
that what carries the ordering is the **ensemble construction**: copy-number coupling read against
a matched-multiset shuffle. So the contribution to this framework is *"`C` must be an ensemble
statistic, not a flat one"* — which is narrower than it first appeared and does **not** license any
claim that assembly theory beats compression here.

**One boundary, held firmly.** F71's plane — 94% of the way to shuffled on novelty at only 27% of
the way on unpredictability — sits well off any linear frontier and looks like a counterexample.
It is not one yet: NLL under `gpt2-large` is **predictability**, and the paper's `A` is factual
accuracy. Predictability is not factuality. F71 is suggestive of the shape and must not be quoted
as refuting the trade-off.

Two further connections worth keeping, neither load-bearing. Their Theorems 1–3 prove errors are
*inevitable*; **F35 measured that errors are also unrecoverable** (`P_persist = 1.000`,
`TV_norm ≈ 0.97`), and their Table 1 lists "exposure bias compounding" as a modeling assumption —
F35 is a measurement of it. And their Lemma 3 (positional undertraining, effective context below
nominal) is the long-context mirror of F69's short-context measurement, where r=2 → r=3 drops top-1
by 52 points.

---

## 5. The follow-up program

Staged so each step gates the next, in the project's own idiom. Steps 1 and 2 were free, had to pass
before any model was loaded, and **both have now run**: §5.1 passed its gate after correcting one of
its own assertions (F73), and §5.2 returned the result the program needed (F74). §5.3 is in flight.

### 5.1 `experiments/assembly_calib.py` — the rung *(free, CPU-seconds)*

**BUILT, and it corrected its own specification** — see F73. The three assertions as drafted asked
for exactness on `a^n`; the exhaustive sweep showed that is false, so the gate asserts the two
properties that hold and *measures* the third:

- **G1, soundness.** RePair ≥ minimal addition-chain length at every n in 2..128. Holds 127/127.
  This is the load-bearing one: a value below a proven bound would mean it is not an upper bound and
  nothing downstream is usable.
- **G2, exactness on the no-reuse family.** RePair = `n − 1` at every n to 256. Holds throughout.
- **G3, the poles.** Δ within ±0.5 on degenerate repetition and random soup, and ≥ +4 on real text,
  at matched length. Reads +0.00 / +0.00 / **+6.87**.
- **Measured, not gated:** the exactness rate on `a^n` (75/127) with its failure list, so the
  overshoot is on the record and a refactor that changes it is visible.

It emits `NOT DECIDABLE` and refuses to report if any gate fails, in the manner of
`dp_calibration.py`, and it is the **single implementation** of the estimator — `_assembly_pilot.py`
imports it rather than carrying its own copy, with a test asserting the two are the same function
objects. `tests/test_assembly_calib.py` asserts every property directly, including the *non*-exactness
at n=15, 23 and 63, so the withdrawn claim cannot quietly return.

### 5.2 `experiments/assembly_baselines.py` — the head-to-head *(free)*

**BUILT AND RUN — see F74. The program continues, and it cost this section two claims on the way.**
Thirteen baselines against Δ across six length-matched reference regimes, every one reported as a
contrast against its own 20-shuffle ensemble:

```
  Delta (log A)          peaks on REAL TEXT   +6.87  vs +0.34 degenerate, +0.00 noise
  11 of 12 baselines     peak on DEGENERATE REPETITION
  closest to Delta       repair_size, rho = -0.88, inside the +/-0.90 redescription threshold
  C_mu / excess entropy  peak on random_soup / degenerate_x2 -- NOT on real text
```

**No baseline reproduces Δ's ordering, and the difference is in *shape*.** Every compression and
entropy measure responds more strongly to a two-word cycle than to real English; Δ is the only one
that inverts it, by a factor of **20**. That is §3.3's tempering result — the exponential is
load-bearing — now shown against the full suite rather than against three tempered versions of
itself, and a difference in shape is not explainable by a correlation coefficient.

The critique in §4.2 is not a footnote, so it was run as an experiment rather than answered in prose,
and built so that the critics' answer could win. Every text measured anywhere in the program carries
**all** of these, and no cell ever reports Δ alone:

| Baseline | Why it is mandatory |
|---|---|
| **LZ77 factor count `z`** | `z ≤ g = 2·ASI`, so **`z/2`** is the lower bound — see F74 for the unit error in the original wording. Omitting it is indefensible |
| **LZ78 / LZW dictionary size** | The exact measure the critics claim equivalence to |
| **RePair size, and Sequitur** | `ASI = SLP`, so these are upper bounds *on our own quantity*. Report the bracket **`[z/2, RePair]`** — `[z, RePair]` is not a bracket, see F74 and §4.1 |
| **gzip / zstd length** | The "does a zip file do this too" test. Cheap; omitting it looks evasive |
| **Entropy: unigram `H₀`, block `H_k`, entropy rate `h`** | The base claim is "not entropy". Prove it here rather than asserting it |
| **Crutchfield statistical complexity `C_μ`, excess entropy `E`** | The established one-hump measure, and the sharpest objection (§4.4) |
| **Raw length and type count** | The InChI-length confound (r = 0.95) says the trivial baseline is the dangerous one |
| `coarse_mi_decay` (`experiments/mlm_lib.py`) | Already in the repo, already shuffle-debiased — the closest existing long-range structure estimator |

`C_μ` and `E` come with the same undersampling problem `novelty_structure.py` names, so estimate them
on the **frequency-bucketed** alphabet that `mlm_lib.freq_buckets` already builds for `coarse_mi_decay`,
and report them as bounded/coarse rather than as the true values.

The pre-registered question was: **does any baseline reproduce Δ's ordering across the reference
regimes?** If yes, assembly theory adds nothing here and the output is a short negative note. **The
answer was no**, so the program continues — and the second half of the question, whether the
baselines stay monotone in temperature where Δ turns over, is what §5.3 is now running.

**A defect in this section's own design, caught by its own kill condition.** The first version ranked
measures on a **z-score** rather than on the contrast, and reported that Δ peaked on degenerate
repetition — the pre-registered kill. The cause was the normalisation: `z = contrast/sd` explodes
when the *control* has little variance (shuffling a two-word cycle barely changes it, so sd = 0.0153
turned a contrast of +0.34 into z = 22.2, above real text's +6.87 at z = 3.2) and is **undefined**
when the control has none, silently dropping two poles. Δ is *defined* as a contrast, so ranking on
its z-score ranked something that is not Δ. A smaller second instance: `n_words`, `n_types` and `H₀`
are exactly invariant under a word shuffle — functions of the multiset alone — so an argmax over
their all-zero contrast vector returned whichever regime sorted first, and they were being counted as
peaking on real text. Both are the same failure: a formula applied where its denominator is
degenerate.

**That invariance disarms the critics' most dangerous baseline at the GLOBAL level only, and §3.7
shows the per-object version is alive.** Total length and type count are held *exactly* constant by
the shuffle control rather than partialled out, so the InChI-length confound (r = 0.95 between
string length and assembly index) cannot operate on the document as a whole. It was overstated here
as "provably cannot operate", full stop. **It can, and does, operate per object:** the shuffle does
not hold an individual n-gram's length constant, and substituting `len_i` for `a_i` inside the
ensemble formula reproduces Δ's ordering — marginally better than the assembly index itself. See
§3.7. This section's conclusion about *flat* baselines is untouched; its implied conclusion about
the assembly index is withdrawn.

### 5.3 `experiments/assembly_temperature.py` — **BUILT, and it had the radii backwards**

**Running as of 1 Aug 2026.** Two corrections were made to this section by the act of building it,
and the second is the more instructive.

**Correction 1 — the subject.** As drafted this section was about *assembly theory*: "is Δ
non-monotone?" The built script reframes it onto the CA, which is the thing actually worth knowing:
**does a language model driven as a cellular automaton produce structure that is more than
recombination, and at what temperature?** Δ is one candidate instrument for reading that, and §5.2
measured thirteen of them against each other. If gzip reads it, use gzip. Nothing here defends a
measure; the measures are apparatus and the CA is the subject.

**Correction 2 — r=2 is the rung, not the artifact.** As drafted, r ∈ {3, 8} was the headline and
r=2 was "the artifact regime (F69), never pooled." That is backwards, and a probe caught it before
the run. The interior-peak prediction rests **entirely** on the low-T pole, and F69 established that
pole exists only at **r ≤ 2** — the r=3 ring at T=0.02 sits at 21% on its top token, not degenerate.
So at r ∈ {3, 8} there is only **one** pole, and with one pole a monotone curve is exactly what you
would expect. Predicting a peak there, and reading monotonicity as refutation, would have tested a
prediction nothing implies.

**Being an out-of-distribution artifact is precisely what makes r=2 useful**: its answer is known in
advance, which is the definition of a calibration rung. It becomes an **instrument-selection rung for
the whole suite** — any measure claiming to track complexity must be non-monotone on a system known
to be degenerate at one end and random at the other, and this one is, on both poles, established by
damage spreading long before any of this. A measure that runs monotonically through a known
non-monotonicity cannot be believed at radii where the answer is unknown. It costs nothing extra:
the same settles carry every measure.

```
  AS BUILT

  r=2      INSTRUMENT SELECTION. Both poles known -> every usable measure must be non-monotone.
           Runs FIRST and gates the rest AS CONTROL FLOW, not as a judgement made after seeing
           numbers -- the discipline that made F67's skipped M2/M3 a result rather than an excuse.
  r=3,8    THE MEASUREMENT. Only the high-T pole is known. Whatever the surviving instruments
           read here is the finding about the CA.

  Selection (r=2). Which measures show an INTERIOR maximum in T, judged against BETWEEN-SEED
           spread? A measure whose peak sits inside its own noise is recorded as monotone,
           not as a peak.
  Null.    No measure is non-monotone at r=2. Then nothing in the suite can be trusted to read
           complexity here, and that closes the thread cleanly. A NULL IS A GOOD RESULT.

  Grid.    T in {0.02, 0.1, 0.2, 0.3, 0.436, 0.52, 0.7, 0.9, 1.1} -- spans both poles, and
           includes F58's T_c and F68's T*. 2 constructions x 8 seeds (11..18).
           144 settles for the r=2 rung, 288 for the measurement.
  Power.   The SEED is the independent unit (F57). Between-seed and within-seed spread both
           reported, and the peak test is judged against the former.
  Orientation. Declared in ORIENT before the run -- smaller compressed size means MORE
           structure, larger Delta means more structure -- so "which measure peaks where"
           cannot be tuned afterwards.
  Contrast. EVERY measure is a contrast against a WITHIN-REPLICA shuffle, which does three
           jobs at once (F74): it is the Kempes fixed-multiset control; it cancels the
           finite-sample bias that makes C_mu and excess entropy unusable at 440 words; and
           because it permutes within each replica, the cross-replica redundancy of 16 similar
           rings sits in both terms and CANCELS -- so no measure can score replica convergence
           as within-text structure (S3.6 confound 2). Its size is reported separately via
           Delta's within/cross copy-number split.
  Rotations. Delta reported both plain and with n-grams canonicalised to their minimal rotation
           (S3.6 confound 1), because canonicalisation is right for a ring and wrong for linear
           reference text, and the size of that difference is what nobody has measured.
  Seeding. ONE arm. F72 measured the basins: the prompt is ERASED, so random seeding answers
           the whole question, halves the grid, and avoids the attractor trap (S5.5).
```

**Cost.** The settle runs are the ones the project already does (N=96, B=16, 16 sweeps): 432 settles
in total, resumable and keyed per cell, at roughly 42 s each — about five hours.

**Why it is the right experiment.** It satisfies all six rules of §1.4: it is gated at its own
geometry (§5.1), it names the seed as the independent unit, it has no fitted window, **it runs a
rung whose answer is known in advance**, it varies the construction rather than only the model, and
it evaluates the estimator in the regime the system runs in — which is precisely the low-T argmax
regime F70 showed is the one that matters. The reframing in correction 2 also *strengthens* it: a
prediction that rests on a known pole is testable, and one that rests on hope is not.

### 5.4 `experiments/assembly_developmental.py` — the usefulness test *(second tier)*

Δ across Pythia training checkpoints, against the developmental transition of F25/F39. If Δ tracks a
transition the project already measures by other means, assembly becomes a *cheap black-box*
instrument for it — a real methods contribution, since Δ needs no damage runs, no CRN twins and no
scorer model, just settled rings. If it does not track it, that is a clean boundary on what the
measure sees. **Do not start this until §5.3 returns.**

### 5.5 What to do with `basin_dependence` first

**This section was drafted under #94's first verdict, which was wrong, and the correction removes
the work it asked for.** F72: the prompt is **erased** — random and corpus seeds settle to the same
composition (max top-1 gap 0.053 / 0.022), with only 2% / 12% of a corpus seed surviving. So Δ
measured from random seeding is not "the narrower question"; it is the whole question, because the
settled state does not remember what it started from.

**§5.3 therefore needs one seeding arm, not two** — 288 settles rather than 576. Report the
initialisation anyway, since the claim now rests on a measurement rather than an assumption, and
cite F72 for it.

The one design constraint that survives is the **trap**. A uniform ring is self-sustaining in 2/6 AR
and 6/6 MLM cells, so §5.3 must never seed from the attractor token — it would read Δ ≈ 0 by
construction and look like the low-T pole while measuring nothing. Random seeding avoids it, which
is the second reason to use it.

---

## 6. Risks, and what should stop this

| Risk | Severity | Mitigation, and where |
|---|---|---|
| **Δ is a tail statistic** — effective object count 1–3, one chance repeat moves it | **High.** The single biggest threat | Report effective object count beside every Δ; ≥8 seeds with the seed as the independent unit; per-replica measurement (§5.3) |
| **AT is contested**, and importing it imports the fight | **High → answered** | §5.2 ran it. No baseline reproduces Δ's ordering (closest ρ = −0.88); 11 of 12 peak on degenerate repetition where Δ peaks on real text. §3.2's concession for the *raw* index stands and is now ensemble-confirmed (z = −1.78), while the overclaim that compressors are blind to word order is withdrawn (gzip: z = −8.13) |
| **`C_μ` already peaks at criticality** (Lindgren & Nordahl 1988; Crutchfield), so Δ may be a redescription | **High → partially answered** | §5.2: neither `C_μ` nor excess entropy peaks on real text (ρ = −0.15 and +0.77). But that is a **coarse, undersampled** estimator on 440 words, not `C_μ` itself — a proper CSSR reconstruction could still peak with Δ and this experiment could not tell. The objection is weakened, **not** retired |
| **AssemblyCA is prior art for AT-on-CA** | **Medium** | Read the PDF before writing; differentiate on 1D-vs-2D, the full assembly equation vs the index alone, and criticality — but confirm all three from the paper, not the project page |
| **Δ may be a repetition detector in disguise** | **High** | The degenerate control reads exactly +0.00 (§3.4), which a repetition detector could not do. But it must be reported every time, not cited once |
| **Novelty may be recall the reference is too small to see** (§3.6) | **Medium–High** | Enlarge the novelty reference well past 3000 documents before any joint novelty × assembly claim; treat top contributors as memorisation candidates and check them |
| **Ring rotation and replica concatenation inflate copy number** | **Medium.** Mechanical | Canonicalise rotations; measure per replica (§5.3) |
| **Exact assembly index is NP-hard** | **Low** | Never claim exactness on long strings. RePair is a *certified upper bound* — it exhibits a pathway, and soundness holds 127/127. Exactness is claimed only on the **no-reuse** family; on `a^n` it is 75/127 and is *measured and reported*, not asserted (F73) |
| **Over-slicing into a third paper** | **Medium** | `plan_paper2.md` already warns against this. Assembly is a *section* of the structure paper unless §5.3 returns the non-monotone result with power |

**The two conditions that should stop the program — one resolved, one live.**

*First, resolved.* If §5.2 had shown gzip and LZ77 reproducing Δ's ordering, the measure would add
nothing and the output would be a paragraph. **It did not**: 11 of 12 baselines peak on degenerate
repetition, Δ alone peaks on real text, and the closest baseline sits at ρ = −0.88, inside the
pre-registered ±0.90 redescription threshold. The program passed its first stop condition.

*Second, live.* If §5.3's r=2 rung returns no non-monotone measure at all, then nothing in the suite
can be trusted to read complexity in this system, and the thread closes.

**Both of those are good outcomes.** This project's record is built on scoping claims down, and the
value of §5.3 as built is that a null is as informative as a positive — r=2 has **both** poles known
in advance, so a measure that runs straight through them condemns the measure rather than leaving the
question open. That is what makes it an instrument-selection rung rather than a hypothesis test.

---

## 7. What to do next, in order

**Done since the first draft** — struck through rather than deleted, because what each one *cost* is
the useful record:

1. ~~Build `assembly_calib.py` (§5.1).~~ **DONE.** The estimator earned the right to report, and
   falsified one of the three assertions that specified it — RePair is not exact on `a^n` (75/127).
   The 14-point pilot sample contained none of the 52 failures (F73).
2. ~~Build `assembly_baselines.py` (§5.2).~~ **DONE, and the program passed its first stop
   condition** (F74). It cost two of our own claims: §3.2's "and so does every compression baseline"
   is withdrawn, and §4.1/§5.2 had a unit error in `z ≤ g`.
3. ~~Register `("assembly_*.json", "assembly_*.py")` in `_STALENESS_PAIRS`.~~ **DONE** — all three
   are registered, and `tests/test_assembly_calib.py` / `test_assembly_baselines.py` exist.
4. ~~Build `assembly_temperature.py` (§5.3).~~ **BUILT AND RUNNING.** It corrected its own
   specification twice before starting: the subject is the CA rather than assembly theory, and
   **r=2 is the instrument-selection rung, not the artifact to exclude**.

**Still open, in order:**

5. **Let §5.3 finish, and read the r=2 rung first.** It gates the r ∈ {3, 8} measurement as control
   flow. Do not read the measurement radii before the rung, and do not reinterpret if the rung comes
   back with nothing non-monotone — that is the null, and it closes the thread.
6. **Do not quote §3.5.** The direction agrees with F71 from an independent statistic, which is
   genuinely encouraging, but it is one settle per cell on a tail statistic. §5.3 supersedes it with
   eight seeds; when it lands, §3.5 becomes history rather than evidence.
7. **Read the `AssemblyCA` paper** (ALOE @ NeurIPS 2023). Closest prior art, and the three
   differentiators in §4.3 still rest on the project page rather than the PDF. A half-hour that
   decides whether the framing survives.
8. **Verify the load-bearing citations** flagged in §4.1 — Rytter 2003 and Charikar et al. 2005 for
   `z ≤ g`. F74 corrected the *units* (`g = 2·ASI`, so the bound is `z/2`), but the sources
   themselves are still unread. `experiments/audit_refs.py` and `tests/test_refs_match_arxiv.py`
   exist for exactly this.
9. **Update issue #20** with §3, §5.1 and §5.2 — particularly §3.3 and F74, which together are a
   real finding *about assembly theory itself* and worth recording whether or not the program
   continues.
10. ~~Close the issues whose findings have landed.~~ **DONE — eight closed** (#61, #83, #87, #88,
    #91, #92, #93, #94), each with the verdict and its backing results file in the closing comment.
    Eighteen remained. Two follow-ups fell out of it, both in §8: the small-model **dip** measurement
    is now filed as **#95**, and `dev_transition_width_early.json` still carries a verdict its own
    commit disowns — which no test can catch, because the disownment lives in a commit message.

---

## 8. Tracker review — the issue list has drifted from the findings ledger

Twenty-six issues were open, and **eight of them were not open work** — the answer had been recorded
in `findings.md` or in a commit, but the tracker never learned it. **All eight are now closed**, each
with a comment carrying the verdict and its backing results file, so the answer is findable from the
tracker rather than only from the ledger. Eighteen remain.

**Closed — a numbered finding exists and names the issue.** These four were the clearest: each has a
`### F## — … (#NN)` header in `findings.md`.

| Issue | Finding | What settled it |
|---|---|---|
| **#91** | **F69** | The degeneracy is confined to r ≤ 2; r=2 → r=3 drops top-1 by 52 points |
| **#92** | **F70** | Not data sparsity — an attracting fixed point of the argmax map |
| **#93** | **F71** | MLM shows structured novelty (+0.669); the AR probe shows none |
| **#94** | **F72** | The prompt is erased; the absorbing state's basin is negligible |

**Closed — the verdict was in the commit history but not as a numbered finding.** These needed the
commit quoted into the issue, or the answer was reachable only by `git log --grep`.

| Issue | Where | Verdict |
|---|---|---|
| **#61** second model family | `4dc5b06`, then **F62** | Done, and it is what broke the transition open — gpt2-medium has none |
| **#83** LR/size confound | `90a390b`, resolved via #87 | Its own run returned `NOT DECIDABLE` (1 of 3 widths); the corrected grid supplied the answer |
| **#87** width-scan grid | `b26a72e`, `def98fd` | Grid fixed; the curve is non-monotone and "crossing bracket" is the wrong instrument for it |
| **#88** onset vs recovery | `cadad64` | The paper's transition is a **recovery from total extinction**, not an onset |

**One caution surfaced while closing these, and it is worth more than the closes.**
`results/dev_transition_width_early.json` carries the printed verdict **"WIDTH EFFECT AT FIXED LR"**,
and commit `def98fd` **disowns it in its own message** — the 14m bracket rests on a step512 cell with
4 of 8 runs ignited, passing the guard by exactly one run, and `crossing_interval` searches for the
first *upward* crossing on a curve whose first sign change is downward in every model, so it reports
the recovery rather than an onset. A results file whose verdict string its own author does not stand
behind is the F45/F46 hazard in a new form: the disownment lives in a commit message, which no test
reads. **What survives is that the dip moves earlier with width** — 70m at step64, 31m at step128,
14m never reaching zero — monotone across a 4× range with depth, LR, batch and data order fixed.

**The work those closes orphaned is now #95.** `def98fd` states that the right observable is the
**dip itself** — depth, timing, and whether it reaches total extinction — and that this is a different
measurement rather than a re-run with a nicer threshold. #88 did it for 410m (total extinction at
step32, 0/8 ignited); the small models were left unrepresented once #87 and #83 closed. #95 files it
with three design points worth noting here, because each is a hazard this project has already paid
for:

- **The metric must be `D_norm` and ignition fraction, not λ.** λ is *undefined* exactly where the
  dip is deepest (F42; F40's dead-damage floor) — 410m's step32 row reads `EXTINCT` with 0/8 — while
  `D_norm` traces the bottom continuously because zero damage is a true zero. The F42 asymmetry doing
  the job it was built for.
- **No threshold anywhere.** `MIN_IGNITED = 4` is the guard that failed in #87, set "at the value it
  needed to exclude" and then passed by exactly one run. Ignition fraction is reported continuously
  and never classified on.
- **The clean width axis stops at 4×, and 410m may not be pooled into it.** 14m/31m/70m share depth 6
  and LR 1.0e-3; 160m is depth 12 at 6.0e-4 and 410m depth 24 at 3.0e-4, so extending upward
  reintroduces the LR confound (#66) *and* a depth confound at once. 410m's extinction sits in the
  predicted direction, which is precisely what makes it tempting and wrong.

**Still open, and superseded rather than answered — a separate judgement, so left for the author.**
These are premised on an object F62–F70 dissolved, but nothing *answered* them, so they were not
included in the close.

- **#80** universality-class program, and **#82** extract θ, δ, z, β, ν⊥, ν∥. There is no
  model-independent transition to classify. #80 is worth keeping as the *pre-registration record* —
  close it with a pointer to F56–F70 rather than deleting it, since the hazards it registered in
  advance are what make the retractions credible.
- **#86** citation ledger for the universality line — reduces to whatever paper 2 actually cites.

**Genuinely open, no work started:** #84 (λ_ca-vs-loss collapse on a fixed Pile slice) and #85
(coupling robustness) have no commits referencing them at all.

**Genuinely live:** **#20** (this document), **#90** (T\*, blocked on ~16 independent families
against the 9 in hand — F68), plus the standing methods threads
(#13, #16, #58, #59, #60, #63, #65, #69, #70) and the hygiene items (#6, #25).

**One recommendation beyond closing.** #20's original text predicted the LZ-equivalence fight and
asked for baselines; F74 ran it and the answer came back favourable, while F73 falsified the
exactness assumption the same issue relied on. Both belong in the issue as comments — the tracker is
currently the only project artifact that still carries #20's *pre*-F73 assumptions.

---

## 9. Bibliography

Identifiers as returned by the literature check. **[unverified]** means the source could not be
fetched directly — check before citing.

**Assembly theory, primary**
- Sharma, Czégel, Lachmann, Kempes, Walker & Cronin (2023). "Assembly theory explains and quantifies
  selection and evolution." *Nature* 622:321–328. DOI `10.1038/s41586-023-06600-9`. arXiv:2206.02279
- Marshall, Murray, Cronin et al. (2022). "Formalising the pathways to life using assembly spaces."
  *Entropy* 24(7):884. DOI `10.3390/e24070884`. arXiv:1907.04649
- Marshall et al. (2021). *Nature Communications* 12:3033. DOI `10.1038/s41467-021-23258-x`

**Strings and complexity — the estimator's foundation**
- Masierak. "Computational Complexity of Determining the Assembly Index." arXiv:2604.16302 —
  `ASI(w) = SLP(w)`; NP-complete; APX-hard. *(the arXiv ID and stated submission date disagree)*
- Masierak. "Templated Assembly Theory." arXiv:2602.04889 — Defs 2.1/2.2, the no-trash condition
- Siebert, Chowdhury, Slocombe & Walker (2026). "Assembly Spaces: Formal Definitions and Fast Methods
  for Approximating Assembly Indices." arXiv:2606.15499
- Łukaszyk & Bieniawski (2024). "Assembly Theory of Binary Messages." *Mathematics* 12(10):1600.
  DOI `10.3390/math12101600` **[unverified]**
- Rytter (2003); Charikar et al. (2005), "The smallest grammar problem", *IEEE Trans. Inf. Theory*
  51(7):2554–2576 — the `z ≤ g` bounds. **[unverified, and load-bearing]** F74 corrected the *units*:
  `g` is total right-hand-side length, so `g = 2·ASI` and the usable lower bound is **`z/2`**. The
  sources themselves are still unread
- Larsson & Moffat (2000), RePair; Nevill-Manning & Witten (1997), Sequitur **[unverified]**

**The critique**
- Abrahão, Hernández-Orozco, Kiani, Tegnér & Zenil (2024). *PLOS Complex Systems*.
  DOI `10.1371/journal.pcsy.0000014`. arXiv:2403.06629
- Uthamacumaran, Abrahão, Kiani & Zenil (2024). *npj Systems Biology and Applications* 10:82.
  DOI `10.1038/s41540-024-00403-y`. arXiv:2210.00901
- Ozelim et al. (2026). *npj Complexity*. DOI `10.1038/s44260-026-00088-w`. arXiv:2408.15108

**The rebuttal — source of this project's defence**
- Kempes, Lachmann, Iannaccone, Fricke, Chowdhury, Walker & Cronin (2025). "Assembly theory and its
  relationship with computational complexity." *npj Complexity*. DOI `10.1038/s44260-025-00049-9`.
  arXiv:2406.12176. PMC12408342 — **the fixed-multiset permutation experiment**
- Łukaszyk (2025). *IPI Letters* 3(1). DOI `10.59973/ipil.157` — disputes the rebuttal in turn

**External framing target — verified**
- Mohsin, Umer, Bilal, Memon, Qadir, Bhattacharya et al. (2026). "On the Fundamental Limits of LLMs
  at Scale." *Transactions on Machine Learning Research* 07/2026.
  `openreview.net/forum?id=BIRDGVrom8` — §2.4's creativity–factuality trade-off is the assumption
  §4.5 and §5.3 test. Verified against a local PDF; record and hash in `paper/refs_manual.json`

**Closest prior art — read before writing**
- **Patarroyo, Sharma, Walker & Cronin (2023). "AssemblyCA: A Benchmark of Open-Endedness for Discrete
  Cellular Automata." ALOE @ NeurIPS 2023.** `openreview.net/forum?id=5cEQ4ZOsIN` ·
  `assemblyca.github.io` · `github.com/KeithPatarroyo/assemblyca` **[unverified — the PDF has never
  been obtained; everything §4.3 claims about it comes from the project page. Record and the action
  that closes it in `paper/refs_manual.json`]**
- Champagne-Ruel, Kempes & Mathis (2025). arXiv:2509.04547

**Comparison targets**
- "Finding the Edge of Chaos in a Ferromagnet: Quantifying the 'Complexity' of 2D Ising Phase
  Transitions with Image Compression." arXiv:2602.15185 — compression-based complexity peaking at T_c
- Lindgren & Nordahl (1988). "Complexity Measures and Cellular Automata." *Complex Systems*
  2(4):409–440 — the canonical one-hump prior art **[unverified]**
- Feldman & Crutchfield. arXiv:cond-mat/9702191

**Coverage caveat.** The negative result in §4.3 rests on an arXiv API sweep plus web search;
Semantic Scholar rate-limited during the check. It is a strong negative, not an exhaustive one — a
non-arXiv journal paper applying AT to text could exist.

---

## Appendix — reproducing the pilot

Every table in §3 is reproduced by one script, added with this document:

```bash
.venv/bin/python experiments/_assembly_pilot.py            # all four stages, ~7 min
.venv/bin/python experiments/_assembly_pilot.py calib      # S3.1  the provable rung
.venv/bin/python experiments/_assembly_pilot.py discrim    # S3.2  raw index vs LZ77/gzip/entropy
.venv/bin/python experiments/_assembly_pilot.py weight     # S3.3  exp vs e^{a/2} vs linear
.venv/bin/python experiments/_assembly_pilot.py matched    # S3.4/S3.5  length-matched Delta
```

**It is deliberately named `_assembly_pilot.py`, not `assembly_pilot.py`.** It writes nothing to
`results/`, carries no provenance stamp and is not registered in `_STALENESS_PAIRS`, because it is
not a gated experiment and must not be mistaken for one.

**It no longer owns the estimator.** `assembly_calib.py` (§5.1) is the single implementation and the
pilot imports `repair_assembly_index`, `addition_chain_length`, `A_exp` and `delta` from it, with
`assembly_baselines.py` supplying `lz77_phrases`, `gzip_bits` and `shannon_bits`. Two copies of an
estimator can drift, and a drifted estimator is indistinguishable from the defect the gate exists to
catch — hazard 1, F56. A test asserts the two modules share the same function objects, and it caught
a shadowed duplicate on its first run. Pilot output is bit-identical to the pre-refactor baseline.

**What the tables here are now for.** §3.1's `a^n` row is superseded by the exhaustive sweep in
`assembly_calib.py` (75/127 exact, F73) and is kept only to show what a 14-point sample looked like.
§3.2's compression columns are superseded by the 13-measure ensemble comparison in
`assembly_baselines.py` (F74). The stages that remain the primary source are `weight` (§3.3) and
`matched` (§3.4), and §3.5 will be superseded by §5.3 when it lands.
