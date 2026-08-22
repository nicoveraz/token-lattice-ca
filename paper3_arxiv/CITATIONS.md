# Citation ledger for paper 3

Why this file exists: F157's prior-art gate **refuted 13 of 74** extracted claims for overreaching
their own sources. Gate summaries are therefore not citable. Every entry in `refs.bib` used by
`main.tex` is verified against the source, and the supporting quote is recorded here with the
**basis** of that verification, so the next reader can check the citation without repeating the work.

**Two bases appear below and they are not equivalent.** `LOCAL FULL TEXT` means the PDF was fetched
to this machine and extracted with `pdftotext -layout`, and the quote was read in that extraction.
`GATE FULL TEXT` means the quote comes from the protocol-depth prior-art gate (F177,
`results/prior_art_paper3_gate.json`), which fetched and extracted the source and put the claim
through 3-vote adversarial verification. The gate is a strong basis but it is not my own read, and
entries resting on it are marked so that a future reader can escalate them if a claim becomes
load-bearing.

Status: **10 works cited, 10 verified.** `\citepend{}` is kept in the preamble as a tripwire — it
renders red in the PDF — and there are zero uses.

---

## 1. `veraz2026probes` — arXiv:2608.10986 (our own paper 1)

- **Cited in:** §Setup, for the measurement itself.
- **Our claim:** the argmax map and the contrast between a model with a dominant attracting fixed
  point and one without are **already published by us**, and this paper claims neither.
- **Source (paper 1, §"The mechanism is an attracting fixed point of the argmax map"):** "For
  `pythia-410m` this map sends 18 of 24 random starts to the newline token --- a genuine fixed point.
  For `gpt2-medium` it has no such point and wanders to 11 distinct endpoints."
- **Verification:** SELF-CITATION, read directly in `paper_arxiv/main.tex` in this repository.
- **Why it is here at all:** the protocol-depth gate (F177) found that this paper's sharpest
  novelty constraint was our own prior publication, and recorded it as kill condition K10. Paper 3
  may claim the 17-model scale, the four-way class with 17/17 seed stability, and the
  corpus-versus-weights attribution — **not the census method**. §Setup says so explicitly.

## 2. `fu2021repetition` — arXiv:2012.14660 (AAAI 2021)

- **Cited in:** §Introduction (the data-side account), §Setup (the funnel geometry), §E2.
- **Our claim:** they attribute degeneration to high-inflow words, define inflow as the probability
  sum over predecessors, and distinguish high-inflow from high-frequency.
- **Source (body, §2.2, Corollary 1.2 discussion):** "the inflow for a word is the probability sum of
  all words that take it as the subsequent word. If it is too big, the upper bound can be magnified
  extensively... high inflow words are more likely to go back to itself and cause the repetition
  problem." And: "it is not the high-frequency words, but the high inflow words that really lead to
  repetition."
- **Verification:** LOCAL FULL TEXT (arXiv PDF v4, 22 Mar 2021, `pdftotext -layout`). Recorded in
  F170.
- **Note:** the second quote is what E2's frequency-matched control is built on — the control tests
  their own stated distinction, not one we imposed.

## 3. `li2023repetition` — arXiv:2310.10226 (NeurIPS 2023)

- **Cited in:** §Introduction, §E2, §Limits.
- **Our claim:** they train GPT-2 on repetition-sorted shards of five datasets, demonstrate causation
  by attention dropout, and subsume the high-inflow account into repetition-in-training-data.
- **Source (abstract):** "Our preliminary investigation reveals a strong correlation between the
  degeneration issue and the presence of repetitions in training data. Subsequent experiments also
  demonstrate that by selectively dropping out the attention to repetitive words in training data,
  degeneration can be significantly minimized."
- **Source (§6.2, the subsumption our §E2 depends on):** merging only the repetitive high-inflow
  pairs (8.1% of training words) "achieves performance comparable to the original HI-RE method"
  (31.1% of words), while merging random high-inflow pairs of the same size "cannot alleviate the
  degeneration". Their conclusion: "penalizing repetitions in data is critical in the success of Fu
  et al."
- **Source (Related Work — the gap this paper addresses):** "The model architecture and size may also
  contribute, but the two factors have not been quantitatively evaluated."
- **Verification:** LOCAL FULL TEXT (`pdftotext -layout`, 921 lines). Recorded in F175.
- **CORRECTION this ledger forced.** An earlier draft of the plan described the data-side camp as
  having "no cross-model comparison". That is **false** and would have been refutable by their
  Figure 2(b), an off-the-shelf OPT size ladder. §Introduction now claims only the absence of a
  cross-architecture/cross-corpus comparison and of a non-generation readout.
- **Also recorded:** `Zihao Fu`, first author of entry 2, is a **co-author** of this paper. The two
  data-side works are not independent, and §Introduction does not present them as rival camps.

## 4. `hernandez2022repeated` — arXiv:2205.10487 (Anthropic, 2022)

- **Cited in:** §E1, for why a dedup null is weaker than it looks.
- **Our claim:** repeated data damages copying circuits, and the damage is non-monotonic with a
  specific damaging range.
- **Source (abstract):** "We find a strong double descent phenomenon, in which repeated data can lead
  test loss to increase midway through training. A predictable range of repetition frequency leads to
  surprisingly severe degradation in performance." And: "data repetition disproportionately damages
  copying and internal structures associated with generalization, such as induction heads."
- **Verification:** LOCAL FULL TEXT (`pdftotext -layout`). Recorded in F176.

## 5. `aoyama2026induction` — arXiv:2511.16893

- **Cited in:** §Introduction, as the second data-to-weights bridge.
- **Our claim:** surface bigram repetition statistics govern induction-head formation.
- **Source (abstract):** "surface bigram repetition frequency and reliability strongly affect the
  formation of IHs, and we find an effective decision boundary in terms of these two values."
- **Verification:** LOCAL FULL TEXT (`pdftotext -layout`). Recorded in F176.
- **Scope note:** their 35 and 60 models are **trained by them**, natural and synthetic. It is not a
  census of pretrained models and §Introduction does not present it as one.

## 6. `michaelov2024recurrent` — arXiv:2404.19178 (COLM 2024)

- **Cited in:** §Related work, conceding that the same-corpus size-matched design is established.
- **Our claim:** 14 off-the-shelf Pile-trained checkpoints across Pythia, RWKV and Mamba, matched by
  weight class, with the explicit purpose of measuring the effect of architecture at fixed corpus.
- **Source (Methods):** "All models are trained on the Pile, a 300B token English-language
  dataset... For each architecture, we selected models of comparable size (i.e., weight class)" and
  "Since all models were trained on the same dataset and have comparable numbers of parameters, we
  are able to measure the effect of architecture on the extent to which a language model's
  predictions correlate with metrics..."
- **Verification:** GATE FULL TEXT (F177, 3-0 adversarial vote, full-text extraction).
- **Why it matters:** this is why §E3 presents its design as adopted rather than new, and why the
  size-matched arm was run at all.

## 7. `wang2025universality` — arXiv:2410.06672 (ICLR 2025)

- **Cited in:** §Related work.
- **Our claim:** Pythia against Mamba at matched size on the same corpus and tokenizer, on induction
  circuits, reporting substantial cross-architecture similarity.
- **Source:** "We choose to study Pythia-160M and an open-sourced version of Mamba. These two models
  are of near size i.e. 160M and 130M, respectively. They adopt the same tokenizer and both are
  trained on the Pile dataset." And: "Cross-arch SAE MPPC has an average of 0.74, compared to 0.76
  for model seed variant."
- **Verification:** GATE FULL TEXT (F177, 3-0 adversarial vote).
- **Why it matters:** it supplies the quantified *opposing* prior — changing architecture costs about
  what changing a seed costs — which §Related work concedes and §E3 argues against.

## 8. `michaelov2025phases` — arXiv:2510.24963 (NeurIPS 2025)

- **Cited in:** §Related work (the invariance prior) and §E2 (frequency dominates).
- **Our claim (a):** its title and result assert consistency of behavioural phases across
  architecture, training data and scale.
- **Source (abstract):** "across architecture (Transformer vs. Mamba vs. RWKV), training dataset
  (OpenWebText vs. The Pile), and scale (14 million parameters to 12 billion parameters),
  autoregressive language models exhibit highly consistent patterns of change in their behavior."
- **Our claim (b):** unigram frequency dominates word-level behavioural variance.
- **Source (§5):** "the regressions best predict language model log-probability (R2 = 0.86 − 0.98
  depending on the model) when the effect of unigram log-probability on language model
  log-probability is at its peak, after which fit decreases sharply."
- **Verification:** LOCAL FULL TEXT (`pdftotext -layout`, 3492 lines), read to resolve kill condition
  K12.
- **PRECISION NOTE, and it changed how we cite this.** The abstract's "up to 98%" is a **peak across
  training**, not a general figure; fit later falls and "does not fall below R2 = 0.5". §E2 therefore
  says frequency "dominates a large share" rather than quoting 98%, because quoting the peak as a
  headline would repeat exactly the over-reading this ledger exists to prevent.

## 9. `du2025correlation` — arXiv:2510.21258 (NeurIPS 2025)

- **Cited in:** §Related work, as the nearest published object and the source of the framing.
- **Our claim:** they frame degeneration as collapse onto a lower-dimensional attractor and measure a
  degeneration-detecting dynamical property across several independently trained families.
- **Source (§5):** "Conceptually, degeneration is viewed as a sudden collapse from a
  higher-dimensional trajectory in the model's state space into a lower-dimensional attractor. Such
  collapses are generally irreversible, mirroring the boundary crisis phenomenon in chaotic
  dynamical systems."
- **Verification:** GATE FULL TEXT (F177, verified with term censuses: zero occurrences of *argmax*,
  *greedy*, *fixed point*, *basin* or *self-map* in their text).
- **Why it is cited prominently:** kill condition K11. The dynamical-systems vocabulary is **not
  ours**, and §Related work says so and states the differences rather than leaving a reader to find
  the overlap.

## 10. `biderman2023pythia` — arXiv:2304.01373 (ICML 2023)

- **Cited in:** §E1, twice — for the pair's existence and for its confound.
- **Our claim (a):** the suite was released in duplicate expressly so deduplication could be studied.
- **Source (§2):** "We train one suite of 8 models on the Pile, and the other on a copy of the Pile
  after applying near-deduplication with MinHashLSH and a threshold of 0.87", framed to "allow users
  of the Pythia suite to study deduplication in greater detail".
- **Our claim (b), which is the load-bearing one:** the deduplicated Pile is ~207B tokens while both
  suites train to ~300B, so the deduplicated models run ~1.45 epochs.
- **Source (§2.2):** "the deduplicated Pile is approximately 207B tokens in size, compared to the
  original Pile which contains 300B tokens."
- **Verification:** GATE FULL TEXT (F177, 3-0 adversarial vote; raised independently by two
  verifiers as a precision defect in our own framing).
- **Why this entry exists in this form:** the gate found that our draft described the pair as
  "differing only in deduplication", which this source shows is **false**. §E1 now carries the epoch
  confound in the body text rather than in a footnote.

---

## Method notes for whoever checks this next

- Entries 1–5 and 8 are LOCAL FULL TEXT: fetched and extracted on this machine. Entries 6, 7, 9 and
  10 are GATE FULL TEXT: extracted and 3-vote verified by the protocol-depth gate recorded in
  `results/prior_art_paper3_gate.json`. If any of those four becomes load-bearing for a claim beyond
  what is quoted here, fetch it directly before strengthening the sentence.
- The gate also recorded **7 refuted claims**, kept in the results file rather than deleted. Two of
  them concerned papers that a weaker check would have cited as threats.
- Three works were checked and are **deliberately not cited**: arXiv:2601.04854, arXiv:2605.02236 and
  arXiv:2504.15471 were cleared by the gate as non-threatening, and none is needed for any sentence
  in this manuscript.
