# Black-box fingerprinting from OOD fallback dynamics — a pre-registered program

**Date:** 1 August 2026; **status updated 2 August 2026 — Gates 1–3 have now run.** Gate 0
(reanalysis of existing screen data) complete and exploratory; **Gates 1, 2 and 3 complete, and
none of K1/K2/K3 fired.** Per §7 that is the all-gates-pass branch — but see §7 for why the
binding constraint is no longer gate-shaped. The outcome contract is frozen in
[`prereg.json`](prereg.json), sha256 `5a15d2e26abe…`, created 2026-08-01T21:34:59Z, *before*
any Gate-1 data existed; verify with `gatecheck.verify_block`. **Origin:**
`critical_analysis.md` §3 ("the paper's stated affordance is asserted, not demonstrated…
a real black-box inference about training data from behavior — currently used only as a
control"), findings F62–F70, and the paper-3 thread of `paper/plan_paper2.md` §6, to which
this program adds a capability framing and a deflationary baseline.

## 1. The claim, if it survives

A battery of cheap black-box probes — the settled attractor share and its temperature
profile, the dominant-token identity, the melting temperature T\* with its censoring state,
and the probe's sensitivity to one extra token of context (radius r=2→3, BOS prefix) —
constitutes a *fingerprint* of a language model that can be read through a raw completion
API. If the gates pass, the capability is model characterization where weights are
unavailable: family attribution, corpus-class inference (one-directional), and detection of
post-training interventions (distillation, instruction tuning) — the demonstrated version of
the affordance the submitted paper only asserts. The scientific object underneath is F70's:
the fingerprint reads the structure of the learned two-token conditional and its argmax map,
a region of the model's function that normal evaluation never visits.

**Prior-art check is mandatory before any write-up.** Adjacent literatures exist: API model
verification / model-equality testing ("which model is this endpoint actually serving"),
output-based authorship attribution of generated text, and watermarking. None of these is
known to use OOD fallback dynamics or to target *corpus* inference, but that sentence is one
search away from being wrong — run the deep-research workflow before drafting anything
(textca's own rule: "both are false and both are one search away" was written about exactly
this kind of claim).

## 2. Gate 0 — what the data already in `results/` says

Reanalysis of `attractor_corpus_screen.json` (26 models, 16 families),
`degeneration_vs_tstar.json`, and `evidence_falloff.json`; zero new compute; code in
[`reanalysis.py`](reanalysis.py), stamped output in [`reanalysis.json`](reanalysis.json).
Exploratory throughout — family labels were known to the analyst, so nothing here is a
finding; it is a feasibility read.

Three things came out in favor. The CA-derived signature *coheres within families*: the
within-family/between-family dispersion ratio for the attractor share is 0.218 (permutation
p ≈ 3×10⁻⁴, family-size-preserving label shuffles), and 0.335 for finite T\* (p ≈ 0.015,
thin subset — only pythia and granite contribute within-pairs). A generic degeneration
metric does *not* cohere (rep_4: ratio 0.745, p ≈ 0.13) — whatever the signature reads, it
is not "how repetitive is this model", which is the nearest boring explanation. And the
controlled pair is quantitatively strong: gpt-neo-125M vs gpt2 (identical tokenizer,
different corpus) separate by 0.577 in attractor share, **2.4× the worst within-family
range** in the whole screen (pythia's 0.237), while their rep_4 gap is 0.043 — and their T\*
lands at *opposite censoring ends* (gpt-neo's attractor never melts on the probed grid;
gpt2's never exists).

One thing came out against, and it sets Gate 1's job. Sixteen-way leave-one-out family
attribution from the four-temperature profile alone is weak: 4/14 correct against ~0.9
expected by chance — above chance, far from a capability. The profile is effectively
low-dimensional (it encodes little more than attractor strength and melting point), so it
bands models into strong/weak-attractor groups rather than identifying them. The battery
must widen before any attribution claim; the frozen feature list in `prereg.json` is that
widening.

Two standing cautions carry over unchanged. The corpus direction is currently supported by
**2 independent families** (all seven Pile+attention models have the attractor, min share
0.744 — but six are Pythia sizes and the seventh is gpt-neo), which is F68's
pseudoreplication hazard arriving again; the author's own power note (~21 families) applies.
And the one-token marginal does not explain the signature (Spearman 0.238 against the CA
share, n=8, consistent with `evidence_falloff`'s verdict) — but the marginal was never the
serious rival. The serious rival is Gate 1.

## 3. Gate 1 — the deflationary baseline runs FIRST

Before any new CA compute: compute **direct two-token-conditional statistics** — argmax-map
fixed-point census (F70's probe, already prototyped), mean top-1 share and entropy of
p(·|x₁,x₂) over sampled contexts, and the by-length falloff curve — for every screened
model, and run the same coherence/attribution/pair protocols on them. No lattice, no
dynamics, one forward pass per context; CPU-days for the small models.

**RESULT: K1 DOES NOT FIRE** (`gate1.py`, `gate1.json`). Attribution is a tie within noise —
5/14 for the baseline against 4/14 for the CA, a one-model margin, both far above the ~0.9/14
expected by chance and far below a capability; neither battery identifies families from four
numbers. The baseline loses decisively on the pair: **0.5× the worst within-family range against
the CA's 2.4×**, i.e. its best gap is *half* the within-family spread. Coherence goes the same
way — three baseline features clear an uncorrected p<0.05, but after Benjamini–Hochberg over the
four tests actually run **none survives**, while Gate 0's CA attractor share was p≈3e-4.

Two defects were fixed before those numbers were used. The argmax census initially tested a fixed
point as (a,b)→b, which only says the trajectory reached the diagonal; it scored gpt2-medium at
0.96 fixed where F70 establishes **none**, and would have inverted the most discriminating feature
for every model. F70 is now wired in as a gate that refuses to report. And the coherence verdict
flipped between two runs of identical code (BH-adjusted 0.0525 then 0.0477) because `perm_p` drew
from a module-level RNG consumed across calls; it now uses a per-feature RNG at 100k permutations
and records a feature within 2 SE of the boundary as **not decidable** rather than calling it.

This is F75's lesson imported: the assembly thread died when a random weight reproduced the
ordering the assembly index was credited with. If direct conditional statistics fingerprint
as well as the CA-derived features (kill condition K1), the CA machinery *exits the
capability* — the product reframes as "conditional-statistics fingerprinting", cheaper and
simpler, and the dynamics story returns to being a paper-2 control. That outcome is a
success for the tool and a demotion for the instrument; the program is written so that
either way, something true and useful survives.

## 4. Gate 2 — controlled pairs

| Pair | Manipulation isolated | Registered expectation |
|---|---|---|
| gpt-neo-125M vs gpt2 (4 seeds/side) | corpus, tokenizer held identical | replicates F64's 78/20 split beyond within-family spread |
| gpt2 vs distilgpt2 | distillation | uncertain — a discovery either way |
| one base-vs-instruct sibling (e.g. Qwen2.5-0.5B vs -Instruct) | post-training | separation expected; direction unregistered |
| pythia-160m vs pythia-160m-deduped | corpus deduplication only | **discovery probe, not a requirement** — dedup is a far weaker manipulation than a different corpus |
| Cerebras-GPT (retry; was HTTP-401) | third Pile family | raises corpus-direction families from 2 toward the needed ~21 |

Protocol frozen in `prereg.json`: 4 seeds per side, gap read against the worst within-family
range (the Gate-0 statistic), family as the unit, battery and baseline computed together so
K1 is evaluated on the same runs.

**RESULT: K2 DOES NOT FIRE** (`gate2.py`, `gate2.json`; 288 cells, 7 models).

```
  corpus         gpt-neo-125M vs gpt2         3.7x   REPLICATES, as registered
  post-training  Qwen2.5-0.5B vs -Instruct    2.3x   SEPARATES; direction was unregistered and
                                                     is now measured: instruction tuning REMOVES
                                                     the attractor, 0.853 -> 0.228 at T=0.02
  distillation   gpt2 vs distilgpt2           0.42x  null on the frozen statistic, BUT 5.7x on
                                                     the radius/BOS arms (see below)
  dedup          pythia-160m vs -deduped      0.8x   null, as registered for a probe
```

**Each manipulation separates on a different part of the battery** — corpus on temperature,
distillation on radius/BOS *only* (invisible across all four temperatures: 0.024/0.045/0.002/
0.015), post-training only below T=0.436. Gate 0's "the profile is low-dimensional, the battery
must widen" is thereby vindicated: two of three manipulations are undetectable in the
four-temperature profile alone.

Two corrections the run forced. First, a **matched-geometry arm** was added: the prereg froze
B=16/16 sweeps while the within-family-range denominator comes from a screen run at B=8/12, and
the shift between them is not uniform — it lands on models that *have* an attractor, because more
sweeps settle a ring further into one. Mixing them would have moved the ratios by corpus +1.25 and
post-training +0.34, *upward exactly where the claim benefits*. That is F56, so both are reported
and the matched pair is what the verdict uses. Second, restricting the statistic to matched
features silently dropped radius_drop and bos_drop, which have no screen-wide denominator — and
distillation lives entirely there, with a radius gap of 0.782 at **103 seed-sd**. The dedup pair
supplies the missing reference (same family, same geometry, same arms), giving distillation 5.7×
and post-training 4.9×; that is reported as **supplementary**, post-hoc and n=1, and is kept out
of K2.

**Cerebras-GPT-111M failed to load again**, so the third Pile family was not obtained and the
corpus direction still rests on the **same 2 independent families** as at Gate 0. F68's
pseudoreplication hazard is undischarged and the ~21-family note stands; the corpus pair
replicating does not touch it.

## 5. Gate 3 — the API port

The battery needs: single-token completions, a **temperature parameter with softmax
semantics and truncation off** (top_p=1, no top_k — a truncated sampler is a *different
construction* and would be textca's F66 lesson repeated at the sampler), and the model's
tokenizer (public for every target considered). Cost is small: one settle at the registered
geometry is N×sweeps×B ≈ 6k single-token calls at B=4, so the full battery including a T\*
ladder is order 10⁴–10⁵ calls of ~3 tokens each per model — dollars and under an hour with
modest parallelism, per model. The hazard is chat templating: F66 showed a *single* BOS
token moves the attractor share 74%→24%, and chat endpoints prepend far more than one token.
So the registered scope is raw completion endpoints (open-weight serving, base-model APIs);
whether any signature survives template wrapping is a separate registered question (K3
scopes, it does not kill).

**RESULT: K3 DOES NOT FIRE — AND THE HAZARD REGISTERED ABOVE WAS THE WRONG ONE** (`gate3.py`,
`gate3.json`). A real completion endpoint was stood up over HTTP against local weights and the CA
driven through it one token at a time, on the pair whose answer is known.

```
                     pythia-410m@0.02   gpt2@0.02   separation (local 0.596)
  ids  (fidelity)         0.786           0.197            0.589
  nospecial               0.805           0.216            0.589
  text                    0.321           0.974           -0.653   INVERTED
  chat                    0.673           0.206            0.467
```

Chat templating does **not** break the port — the chat arm keeps a full-width context and still
separates the models. **The plain-text round-trip does**, and it was not registered as a hazard at
all. The mechanism is measured rather than argued: a two-token window decoded to text and
re-tokenized **merges into one token** — gpt2 in 63% of calls, pythia-410m in 13% — so the
endpoint silently runs the CA at a *smaller radius* than requested, and F69 established r=1 is the
degenerate regime. Being tokenizer-dependent it does not cancel; it inverts which model looks
attractor-bearing. The chat template *helps*, because its wrapper text keeps the window tokens
from merging. Not being able to forbid special tokens, which no real endpoint exposes, costs
nothing (`nospecial` reproduces).

So the capability is scoped by **interface type, not by templating**: token-id endpoints carry it,
text endpoints destroy it, and a text endpoint might be repairable by separating the window so it
cannot merge — a testable claim this gate did not test.

Three harness artifacts were found and removed first, each of which manufactured the effect it
then measured: BOS-padding short windows (the ring collapsed to 95.8% newline, and a
uniform-window check said the pad fired 0.0% of the time — it only fires once the ring has
drifted, which is F70's lesson turned back on the harness); forcing a common context width across
the batch (one merged prompt among sixteen truncated *every* replica to r=1); and a
`local_reference` that read Gate 2 only, which has no pythia-410m rows, so the fidelity arm had
nothing to compare against and would have passed vacuously. Realized context widths are now
recorded per cell so this class of drift is visible rather than inferred.

**SCOPE: local harness only.** No third-party endpoint was contacted — this machine has no API
credentials and obtaining them was not authorised. This establishes what §6 requires *before* an
unknown endpoint may be characterized; it does not establish that any commercial endpoint behaves
this way.

## 6. Statistics, units, and the gate around the gate

Family is the independent unit everywhere — six Pythia sizes are one observation for every
claim in this program (F68). Attribution is leave-one-out nearest-family-centroid over the
all-families candidate set, reported as counts against chance. Pair verdicts are gaps over
worst within-family range. And the calibration-gate pattern applies to the capability
itself: before the battery is trusted on any *unknown* endpoint, it must recover the known
family labels of held-out *public* models at the same call budget and sampler settings — the
known-answer reference here is the public model zoo, and a battery that cannot re-identify
pythia-410m through its own API harness has no business characterizing a closed endpoint.

## 7. Kill conditions and exits

Frozen in `prereg.json`: **K1** — the direct-conditional baseline matches or beats the CA
features on both attribution and pairs → the CA exits the capability, which reframes as
conditional-statistics fingerprinting. **K2** — neither the neo/gpt2 replication nor any of
the distillation/instruct pairs separates beyond within-family spread → corpus/post-training
inference dies and F64 stays what it is today, a control in a negative-result argument.
**K3** — no raw completion endpoint reproduces the local signature → capability scoped to
open-weight serving. Exits by outcome: all gates pass → a standalone audit capability and
the demonstrated version of the paper's affordance sentence; K1 fires alone → a simpler
tool, same uses; K2 fires → this folds back into paper 2 as a paragraph, and the program's
value was the ~2 GPU-days it spent to avoid a much longer illusion.

**WHERE THIS ACTUALLY LANDED (2 Aug 2026).** All three kill conditions were written to fire and
none did, which is nominally the all-gates-pass exit. It should not be read as the capability
being demonstrated, because **the binding constraint is no longer gate-shaped**:

- **Corpus inference rests on 2 independent families**, unchanged since Gate 0, because Cerebras
  failed twice. This is the same F68 hazard, and no gate can retire it — only families can.
- **Attribution is 5/14 and 4/14.** That is coherence, not identification. Gate 0 said the battery
  must widen before any attribution claim; the widened battery separates *known* pairs but has not
  been shown to *identify* an unknown one.
- **Every Gate-2 manipulation is n=1.** One distilled model is not a fact about distillation.
- **Gate 3 never touched a real endpoint.**

So the honest position is: a battery that separates three known manipulations, each on a different
arm, with a measured interface-type scope constraint. What it needs next is **breadth** — families,
one real endpoint — and none of that is a gate. Set against the F77/F78 line, which needs *depth*
and has an explanandum a reviewer can attack, this program is the weaker candidate for the third
paper's external anchor until the family count moves. Per plan_paper2's own
caution, three papers out of this material would be over-slicing unless this program's
external anchor lands — the decision point is after Gate 2, not before.

## 8. What this program does not claim

Nothing here reaches model internals; a passing program yields behavioral characterization —
family, corpus class, post-training status — not interpretation, and the write-up must say
so (this is the same boundary `critical_analysis.md` §3 draws for the project at large). The
fingerprint reads a two-token OOD regime that F65/F66 established is a property of the
*(model, construction)* pair; its use as an identifier is legitimate precisely because
identification does not require the regime to be representative — only stable, cheap, and
model-discriminating, which is what Gate 0 suggests and Gates 1–2 will decide.
