# The delta paragraphs — paper 4

F186 made these mandatory before any write-up: an explicit delta against `arXiv:2410.06287` **and**
against our own paper 1, with the novelty pitched as the vocabulary-wide set-valued destination map,
never as "degenerate probes" or "fixed points of greedy decoding". These are the paragraphs, ready
to drop into a related-work section, followed by what each sentence rests on.

---

## 1. Against Hammouri et al., `arXiv:2410.06287`

```latex
\paragraph{The probe is not new; the index set is.}
The degenerate diagonal input and its fixed-point reading are published.
\citet{hammouri2025nonhalting} feed a model its own token repeated, formalise the
temperature-zero condition --- at $\tau = 0$, a fixed point $x$ of $f$ with
$f(x_1,x_2,x_3) = x_1,x_2,x_3$ yields an output that never halts --- and tabulate, for
$100$ randomly chosen words across five aligned models, how many repetitions each word
needs before the model stops emitting its end-of-string token. They further observe that a
fixed point found in a base model transfers to models derived from it, which is the same
invariance our attribution rests on. We claim none of this.

Three things separate the present work from theirs, and only the third is a claim about
novelty rather than scope. First, the measurement: theirs is behavioural and requires
generation, since non-halting is defined by what the model does over many steps, while
ours is a single forward pass per token and generates nothing. Second, the sample: one
hundred words against five aligned models --- two open-weight, three served only through an
API --- against an exhaustive sweep of $50$k-token
vocabularies across twelve base models trained on one fixed corpus. Third, and this is the
delta: their table enters $0$ wherever the model halts normally, and discards what it
emitted instead. Those cells are where our measurement begins. We ask which token the
model produced in place of the repetition, read that answer across the whole vocabulary,
and compare it between models as a decoded string. Every result below rests on the
contents of their zeros.
```

## 2. Against our own paper 1, `arXiv:2608.10986`

```latex
\paragraph{What our own earlier work established, and what it did not.}
The argmax map over token space is not introduced here. Paper~1 \citep{veraz2026probes}
identifies it as the mechanism behind the transition it reports and states the per-model
contrast directly: for \texttt{pythia-410m} the map sends $18$ of $24$ random starts to
the newline token, a genuine fixed point, while \texttt{gpt2-medium} has no such point and
wanders to $11$ distinct endpoints. It also shows the map's behaviour is a property of its
domain rather than its parameters, one prepended token moving the frozen fraction from
$74.4\%$ to $24.1\%$. The object, the mechanism, and the per-model contrast are therefore
already ours and already in print, and this paper claims none of them.

What paper~1 measures is a trajectory census: twenty-four random starts, iterated, with
the number of distinct endpoints as the readout. It reports how many places trajectories
land, and never which tokens are fixed points, because twenty-four starts cannot enumerate
a vocabulary. This paper inverts that. It evaluates every token of every model's
vocabulary exactly once, keeps identity rather than count, and keeps it for the tokens
that are \emph{not} fixed points as much as for those that are. The difference is not one
of scale. A census of endpoints and a map defined on the whole vocabulary are different
objects, and only the second supports the comparison the results turn on --- which tokens,
agreeing between which models.
```

## 3. The claim, stated narrowly (F186's mandated pitch)

```latex
\paragraph{The claim, stated narrowly.}
What is new here is a vocabulary-wide, set-valued destination map: for every token in a
model's vocabulary, where its own two-token diagonal state is sent by the argmax of the
model's conditional, decoded to a string so that models with different tokenizers are
comparable at all. Not the probe, which is \citet{hammouri2025nonhalting}'s. Not the
fixed-point framing, which is theirs and paper~1's. Not the agreement statistic, which is
the established estimand of the model-provenance line \citep{stemma2026,mpt2025}. Not the
leave-one-out family-attribution protocol, which is already published for a single-shot
probe battery \citep{onetoken2026}. The index set, and what is recorded at each of its
elements.
```

---

## What each sentence rests on

| claim | basis |
|---|---|
| Hammouri's $\tau=0$ fixed-point formalism, verbatim | **LOCAL FULL TEXT** — v2 PDF fetched to this machine, `pdftotext -layout`, quote read in the extraction |
| "100 randomly chosen words", five aligned models (Gemini Pro 1.5, Claude-3.5-Sonnet, Gemma-2-9B-it, ChatGPT-4o, Llama-3.1-8B-it), repetitions-needed table, `0` = halts normally | **LOCAL FULL TEXT** — Tables IV/V and the caption *"A zero means the model does not produce a non-halting response for the corresponding word."* |
| lineage transfer of fixed points, quoted verbatim | **LOCAL FULL TEXT** — *"once we identify a cycle in the base model for some LLM, we may transfer the same cycle to target a different aligned model"* |
| non-halting requires generation, not a logit read | **LOCAL FULL TEXT** — the anomaly is defined by the model never sampling `<eos>` over many steps |
| paper 1's 18/24, 11 endpoints, 74.4→24.1 | **SELF-CITATION**, read in `paper_arxiv/main.tex` in this repository |
| agreement statistic is standard (Stemma, MPT) | **GATE FULL TEXT** (F186, `results/prior_art_selfcont_gate.json`) |
| LOO family attribution published for a single-shot battery | **GATE FULL TEXT** (F186) |

**Why the two GATE-basis entries are acceptable here and would not be elsewhere.** `CITATIONS.md`'s
escalation rule exists to stop a citation being *over-read* — a claim resting on more than the source
says. Both gate-basis citations are used to **concede** priority, not to claim it. Conceding on
weaker evidence than you strictly need costs the reader nothing and costs us only credit we were not
entitled to assume. They should still be fetched before submission, but nothing in the argument
breaks if they turn out to concede too much.

## What these paragraphs may not say

Binding, from F186:

- Never pitch the work as **"degenerate probes"** or as **"fixed points of greedy decoding"**. Both
  are taken — the first by `2410.06287`, the second by `2410.06287` and paper 1 together.
- Never claim the **agreement statistic** or the **attribution protocol** as novel.
- The reframing "the object is dynamical even though the measurement is single-shot" is available,
  but it runs into paper 1, which already banked that object. It may be used to explain, never to
  claim.
- Paper 1's related-work section asserts on the record that single-shot feature sets are what the
  identification literature already does. A paper 4 that pitched itself as single-shot novelty would
  be contradicting its own companion in print.

## Bib entries these paragraphs need

```bibtex
@inproceedings{hammouri2025nonhalting,
  title     = {Non-Halting Queries: Exploiting Fixed Points in {LLM}s},
  author    = {Hammouri, Ghaith and Derya, Kemal and Sunar, Berk},
  booktitle = {IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
  year      = {2025},
  eprint    = {2410.06287}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
```

`veraz2026probes` (`2608.10986`) and `veraz2026domain` (`2608.21315`) already exist in
`paper3_arxiv/refs.bib` and can be copied unchanged. `stemma2026` (`2607.25880`), `mpt2025`
(`2502.00706`) and `onetoken2026` (`2607.10252`) need entries, and each needs its own `CITATIONS.md`
row recording GATE basis until it is fetched.
