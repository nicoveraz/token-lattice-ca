# Black-box fingerprinting from OOD fallback dynamics — a pre-registered program

**Date:** 1 August 2026. **Status:** Gate 0 (reanalysis of existing screen data) complete and
exploratory; Gates 1–3 registered, not run. The outcome contract is frozen in
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
value was the ~2 GPU-days it spent to avoid a much longer illusion. Per plan_paper2's own
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
