# PROTOCOL — is your loop measuring the model, or itself?

A template for anyone who runs a language model in a **self-feeding loop** — a ring, a
self-consistency vote, an iterated-refinement chain, a transmission chain, a multi-turn agent — and
computes a statistic of the loop's behaviour.

The statistic you report is a property of *model × construction*. Most such statistics turn out to
be properties of the **construction**: they move when you change the loop and barely move when you
change the model, and nothing in a correct calculation objects. This protocol separates the two.
Copy the checklist, fill in the blanks, and keep the filled copy beside your results.

> **The one-sentence version.** Vary the construction with the model fixed, vary the model with the
> construction fixed, and report which axis moves your readout more — *after* establishing that
> your readout has room to move at all, and that its model ranking survives a change of seed.

---

## 0. Write down the loopness vector

The construction is not "my setup". It is a point in a space, and the space has axes. Fill this in
for **every** arm you run, including the ones you think are obvious:

| Axis | Meaning | Yours |
|---|---|---|
| `radius` | how many neighbours a site conditions on (`None` = whole prefix) | |
| `temperature` | sampling temperature, or `None` if deterministic | |
| `scheme` | `sync` / `async` / `ordered` / `none` — the visit order | |
| `commitment` | `in_place` / `scheduled` / `rollback` / `append_only` / `free_ar` | |
| `masking` | whatever masking policy applies | |
| `domain` | what CONDITIONS the state: `raw` / `bos` / `system_prompt` / `chat_template` / `few_shot` / `custom` | |
| `domain_tokens` | how many tokens that conditioning occupies | |

`commitment` is the axis that separates the constructions people actually use, and it is ordered:

```
in_place ── scheduled ── rollback ── append_only ── free_ar
   ring       diffusion   speculative   growing      ordinary
                          decoding      transcript   generation
```

A ring **revokes** every commitment each sweep, which is why healing exists there and nowhere in
deployment. Ordinary generation revokes none. If you have never written down where your loop sits
on this axis, that is the first thing this protocol buys you.

```python
from gatecheck import Loopness
con = Loopness(radius=2, temperature=0.2, scheme="async", commitment="in_place")
```

`Loopness` rejects a `commitment` it does not know, on purpose: a free-text label cannot be swept
along a gradient, and the gradient is the point.

**Why `scheme` is not cosmetic.** With asynchronous updating the within-sweep reach of an
influence is set by the visit order, not by the radius. A project that assumed the synchronous
bound published a cone width that was wrong by a factor of about six.

**Why `domain` is the axis you are most likely to leave implicit, and least safe to.** It was added
last because it turned out to move a readout further than anything else in this vector. The same
greedy map, same weights, same estimator, same seeds, measured with nothing before the state and
then behind each model's own chat template:

```
  nine template tokens   fixed-point fraction 0.948 -> 0.000   and the class changed
  eleven template tokens fixed-point fraction 0.615 -> 0.844   and the class did not
  one BOS token          frozen fraction      74.4% -> 24.1%
```

Two things follow. The domain is a **construction parameter** — two runs differing only in it are
two constructions, and this protocol will treat them as such. And the direction was
**model-specific**: structure destroyed in one model, reinforced in another. That means it cannot be
corrected away with a factor; it has to be varied like any other axis. If your loop runs behind a
system prompt or a chat template in deployment and you measure it raw, you are measuring a different
construction and this vector is where you say so.

---

## 1. Build the grid

You need **three** varying things. Each has a job, and dropping any one makes the test unrunnable:

| | Minimum | Why |
|---|---|---|
| models | 4 | below that a rank correlation takes a handful of discrete values and cannot fail informatively |
| constructions | 2 | the construction axis *is* the test |
| seeds | 2 | without a second seed there is no noise floor, so "the models differ" cannot be told from "the seeds differ" |

```python
observations = {(model, construction, seed): value, ...}
```

**Use the family as the independent unit wherever models vary.** Ten checkpoints of one model are
not ten models. If your grid is five sizes of one family plus one other model, you have two units,
not six — say so in your boundary statement.

**Store the state, not just the statistic.** Whatever your readout reduces — a lattice, a
transcript, a trajectory — write it to the results file under a size cap. Step 4 cannot run without
it, and every later question about the run costs a full re-run without it. This is the single
cheapest thing on this page and the one most often skipped.

---

## 2. Run the discriminator

```python
from gatecheck import discriminate
rep = discriminate(observations, readout="my_statistic")
print(rep.summary())
print(rep.verdict.reason)
results["discriminator"] = rep.block()
```

The gates run **in this order**, and the order is not negotiable:

**0 — Anti-vacuity, on the construction axis.** If varying the construction does not move the
readout, then "the model ranking survives construction change" is true because nothing changed, and
invariance passes *vacuously*. A pinned observable is the single most flattering failure available.

**1 — Signal.** Per construction, the across-model spread must exceed the across-seed spread by
`noise_factor` (default 2×), on a majority of constructions. If model identity does not move the
readout beyond seed noise, there is nothing for the construction axis to be tested against.

**2 — Seed stability.** The model ranking must reproduce across seeds (default Spearman ≥ 0.6)
*before* any ranking is compared to any other ranking. **This step is the one people skip and it is
the one that bites.** A real spread can carry no reproducible ordering at all — the project this
package came from measured a spread that cleared its noise floor by 30× and ranked models at
Spearman 0.030. An invariance statistic computed on that is a correlation between two noise vectors.

**3 — Invariance.** *Only now*: do different constructions produce the same model ranking?

The verdict is one of:

- `MODEL_DETERMINED` — signal, a stable ranking, and the ranking survives construction change. Your
  readout is about the model **across the constructions you tested**. Never write "in general".
- `CONSTRUCTION_DETERMINED` — signal and a stable ranking, but each construction produces its own
  model ordering. Your readout is about the apparatus.
- `NOT_DECIDABLE` — the grid could not answer. This is a *result*, and `rep.verdict.reason` names
  the binding step. Publish it.

`rep.range_ratio` reports how much more the construction moves the readout than the model does. It
is reported always and gates nothing — it is the magnitude, not the verdict.

---

## 3. Sweep the gradient (optional, and where the interesting answers are)

Run step 2 at several points along `commitment`, from your tightest loop toward free generation, and
record **where each observable's model-attribution dies**.

- An observable that is `CONSTRUCTION_DETERMINED` at every loopness is **kinematic** — it measures
  your apparatus and always did.
- An observable that is `MODEL_DETERMINED` at the ring and dies as commitment tightens tells you the
  attribution *depends on revocability*, which is a finding about the observable.
- An observable that survives to `free_ar` is the one worth reporting to people who do not run your
  loop.

---

## 4. The nuisance gate — the one no other check can do for you

A readout can pass every step above while being an **arithmetic function of something nobody meant
to measure**. Range gates cannot see it (the values are well separated), stability gates cannot see
it (the values are perfectly reproducible), and a correct calculation will not object.

The instance that produced this gate: an attractor-share readout `top1` — the largest token's share
of a settled lattice — measured on a six-word alphabet. A lattice that crystallises into a period-`p`
orbit reads `top1 = 1/p` **exactly**, whatever the model does. Three cells came back at 0.3333 and
0.2500 to the last digit; they were period-3 and period-4 crystals. The campaign had already run to
completion.

So: write down what your readout would equal under an explicit nuisance hypothesis, compute that
prediction **per cell from the stored state**, and check how often the readout matches it.

```python
from gatecheck import nuisance_identity
g = nuisance_identity(values, predicted, name="top1", nuisance="1/period of the settled orbit")
if not g.usable:
    ...  # the readout is the nuisance on those cells; no verdict is available from them
```

Pass it to `discriminate(..., nuisance_prediction={key: predicted})` to have it block the verdict.

**You cannot run this gate on a results file that stored only scalars.** That is the reason step 1
asks you to store the state.

---

## 5. Write the boundary statement

Every verdict is a statement about *your grid*. Name, in one paragraph:

- how many **families** (not checkpoints) varied, and their size range;
- how many constructions, and where they sit on the commitment axis;
- what did **not** vary (tokenizer, corpus, scale) and therefore could be carrying the effect;
- the thresholds you used, and that you set them before looking.

A worked template: *"N families spanning A–B parameters, C constructions all at `in_place`
commitment, one alphabet, lattice size L. A readout that survives here is model-attributable across
THESE constructions, not in general. Corpus was not varied and dominates this readout elsewhere, so
the architectural reading is corpus-controlled or it is not made."*

---

## Worked example

`experiments/discriminator_demo.py` in the [token-lattice-ca](https://github.com/nicoveraz/token-lattice-ca)
repository runs one `discriminate` call, with identical thresholds and no per-readout tuning, over
two readouts on the same models, seeds and lattice:

```
lambda_ca  -> NOT_DECIDABLE      signal on only 2 of 4 constructions
top1       -> MODEL_DETERMINED   6/6 signal, seed-stable ranking +0.848, invariance +0.752
```

Same apparatus, opposite verdicts. The module is checked against those two known answers before it
is used on anything, which is the discipline the rest of this package exists to enforce: **an
estimator that has not reproduced a known answer does not get to report a verdict on an unknown
one.**

One honest note from that run, kept here because it is the kind of thing a protocol should not hide:
`lambda_ca` was *expected* to fail at step 2 (unrankable spread) and actually fails at step 1
(signal), one step earlier. Same label, different reason. The reason is the part a reader needs, so
the report prints the binding step rather than the author asserting it in advance.
