# Citation ledger for paper 2

Why this file exists: F157's prior-art gate **refuted 13 of 74** extracted claims for overreaching
their own sources. Its summaries are therefore not citable. Every entry in `refs.bib` used by
`main.tex` must be verified against the source itself, and the supporting quote recorded here so the
next reader can check the citation without repeating the fetch.

Status: **all 12 `\citepend{}` placeholders (8 distinct works) are resolved.** Fifteen works are now
cited by `main.tex` and all fifteen are verified below. The `\citepend` macro is kept in the preamble
as a tripwire — it renders red in the PDF — so a future unresolved citation cannot slip through.
There are none now, and `tests/test_paper2_citations.py` fails if one appears.

Entries 1–4 were verified in 3d64a66; entries 5–12 on 18 Aug 2026; entry 13 on the
restructure; entries 14–15 on 21 Aug 2026, for the submission edits (the funnel credit in Setup
and the adjacent-repetition sentence in the introduction).

---

## What each entry records

- **Our claim** — what `main.tex` asserts on the strength of the citation.
- **Source** — the quote, marked *(abstract)* or *(body, §)*, and whether it was fetched from the
  abstract page or the full text.
- **Corrections** — where the draft, or the F157 gate, said something the source does not support.

---

## 1. `cao2024worstprompt` — arXiv:2406.10248 (NeurIPS 2024)

Cao, Cai, Zhang, Zou, Lam, *On the Worst Prompt Performance of Large Language Models*, 2024.

- **Our claim (§intro-known):** per-prompt performance rankings agree across models only weakly,
  Kendall's $W = 0.238$.
- **Source (body, full text HTML v2):** "The consistency between all models is significantly lower"
  at $W = 0.238$; Llama family $W = 0.443$, Gemma $W = 0.548$.
- **Note:** the abstract does NOT contain this number — it is in the body. The first fetch of the
  abstract alone did not support the claim, which is exactly the failure mode this ledger exists
  for. Also supports (unused): worst-prompt overlap at $k{=}1$ is 2% (Llama) / 13% (Gemma).

## 2. `alzahrani2024benchmarks` — arXiv:2402.01781 (ACL 2024)

Alzahrani et al., *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model
Leaderboards*, 2024.

- **Our claim (§intro-known):** semantically neutral reformatting reorders leaderboards by up to
  8 positions.
- **Source (abstract, verbatim):** "minor perturbations to the benchmark, such as changing the order
  of choices or the method of answer selection, result in changes in rankings up to 8 positions."
- **Correction applied:** our draft previously said "11-model leaderboard". The abstract does not
  state the model count, so that number is not in the text. Verify before reintroducing it.

## 3. `xiao2023streamingllm` — arXiv:2309.17453 (ICLR 2024)

Xiao, Tian, Chen, Han, Lewis, *Efficient Streaming Language Models with Attention Sinks*, 2023.

- **Our claim (§sink):** initial tokens dominate a scalar readout; the mechanism is positional, not
  semantic; substituting the first four tokens with the linebreak token restores perplexity nearly
  as well as the originals ($5.60$ vs $5.40$ on Llama-2-13B, vs $5158.07$ with those positions
  dropped).
- **Source (abstract, verbatim):** "the emergence of attention sink is due to the strong attention
  scores towards initial tokens as a 'sink' even if they are not semantically important."
- **Source (body, §3.1 + Table 1, verified 18 Aug 2026 via ar5iv full text):** "we conduct
  experiments (Table 1), wherein the first four tokens are substituted with the linebreak token
  '\n'. The observations indicate that the model still significantly emphasizes these initial
  linebreak tokens. Furthermore, reintroducing them restores the language modeling perplexity to
  levels comparable to having the original initial tokens." Table 1, Llama-2-13B PPL on the first
  book (65K tokens) of PG19: `0+1024` (window) 5158.07, `4+1020` 5.40, `4"\n"+1020` 5.60.
- **This clears the PARTIALLY UNVERIFIED flag** the previous version of this ledger carried on that
  sentence. Corrections applied to the text: "the first tokens" → "the first four tokens", and the
  perplexity figures and model are now named, since 5.60/5.40 are Llama-2-13B-specific.
- **Still gate-sourced, not re-verified here:** §sink's "cross-model variation is in magnitude and
  saturation point, not in sign", which the gate grounded in Table 2 (Falcon-7B 17.90 → 12.12,
  saturating at one token; Llama-2-7B 3359.95 → 9.59, needing four). The direction of that table is
  consistent with what we saw of it in the full text, but the per-model numbers were not re-fetched.

## 4. `voronov2024format` — arXiv:2401.06766 (Findings of ACL 2024)

Voronov, Wolf, Ryabinin, *Mind Your Format: Towards Consistent Evaluation of In-Context Learning
Improvements*, 2024.

- **Our claim (§intro-known):** the best prompt templates do not transfer between models, even
  within a single model family.
- **Source (abstract, verbatim):** "the best templates do not transfer between different setups and
  even between models of the same family". 21 models, 770M–70B, 4 classification datasets.
- **Correction applied:** the abstract supports **non-transfer**, which is what our sentence now
  says. It does not by itself support the stronger "best for one, among the worst for another"
  phrasing — that was the gate's wording, and it is not in the text.

## 5. `lu2022ordered` — arXiv:2104.08786, ACL 2022, pp. 8086–8098

Lu, Bartolo, Moore, Riedel, Stenetorp, *Fantastically Ordered Prompts and Where to Find Them:
Overcoming Few-Shot Prompt Order Sensitivity*. doi 10.18653/v1/2022.acl-long.556.

- **Resolves:** the `prompt certification / non-transferability` placeholder (§interaction) — one of
  the four that must not be guessed, because §interaction concedes priority to it.
- **Our claim (§interaction):** that prompt quality does not certify across weights is in the
  abstract; prompt rankings correlate at $0.05$ between their 2.7B and 175B models; one fixed
  permutation moves accuracy from $88.7\%$ to $51.6\%$ between two sizes of GPT-2.
- **Source (abstract, verbatim, fetched from arXiv):** "We analyse this phenomenon in detail,
  establishing that: it is present across model sizes (even for the largest current models), it is
  not related to a specific subset of samples, and that **a given good permutation for one model is
  not transferable to another**."
- **Source (body, via the F157 gate's quotes):** "the 175B and 2.7B model only has a correlation of
  0.05, this means a good permutation for the 2.7B model is in no way guaranteed that it will also
  yield good performance for the 175B model." And: "a specific permutation's performance may drop
  from 88.7% to 51.6% by changing the underlying model from GPT2-XL (1.5B) to GPT2-Large (0.8B)."
- **Correction applied:** the draft said "a prompt cannot be certified independent of the model
  appears in the abstract of at least one prior work". *Certified* is our word, not theirs. The text
  now quotes the non-transferability sentence and attributes only that.
- **Bib detail source:** ACL Anthology page for 2022.acl-long.556 (authors, venue, pages, DOI).

## 6. `chen2023mapo` — arXiv:2407.04118, **Findings of EMNLP 2023**, pp. 3279–3304

Chen, Wen, Fan, Chen, Wu, Liu, Li, Liu, Xiao, *MAPO: Boosting Large Language Model Performance with
Model-Adaptive Prompt Optimization*.

- **Resolves:** the `MAPO, Chen et al. 2024` placeholder (×2: §intro-known and §interaction) — a
  priority claim, one of the four that must not be guessed.
- **Our claim:** the earliest anchor for prompt effectiveness being model-specific.
- **Source (abstract, verbatim):** "The existing research primarily emphasizes the importance of
  adapting prompts to specific tasks, rather than specific LLMs. However, a good prompt is not
  solely defined by its wording, but also **binds to the nature of the LLM in question**. In this
  work, we first quantitatively demonstrate that different prompts should be adapted to different
  LLMs to enhance their capabilities across various downstream tasks in NLP."
- **CORRECTION — the year.** The draft note said "Chen et al. 2024" because the arXiv posting
  (2407.04118) is dated 4 July 2024. The paper is **Findings of EMNLP 2023**, pp. 3279–3304
  (confirmed on the ACL Anthology page, `2023.findings-emnlp.215`, and by the arXiv comment line
  "Accepted to EMNLP 2023 (Findings)"). Cite it as **2023**. This matters: the sentence calls it the
  *earlier* anchor, and the 2023 date is what makes that true.
- **Note:** PromptBridge (entry 7) itself credits "Chen et al. (2024)" for MAPO, i.e. it cites the
  arXiv posting year. Do not inherit that error.

## 7. `wang2025promptbridge` — arXiv:2512.01420 (1 Dec 2025)

Wang, Liu, Wang, Li, Wei, Liu, Bao, *PromptBridge: Cross-Model Prompt Transfer for Large Language
Models*.

- **Resolves:** the `model drifting` placeholder (×2: §intro-known and §interaction) — we credit a
  specific name, so this is one of the four that must not be guessed.
- **Our claim:** a prompt optimised for one model is suboptimal on another, and the phenomenon is
  *named* Model Drifting.
- **Source (abstract, verbatim):** "Yet prompts are highly model-sensitive: reusing a prompt
  engineered for one model on another often yields substantially worse performance than a prompt
  optimized for the target model. **We term this phenomenon Model Drifting.** Through extensive
  empirical analysis across diverse LLM configurations, we show that model drifting is both common
  and severe."
- **Note:** the name is coined here, in the abstract, in so many words. That is exactly the standard
  a naming credit needs.

## 8. `kostiuk2026oneprompt` — arXiv:2605.22544 (21 May 2026)

Kostiuk, Enevoldsen, *One prompt is not enough: Instruction Sensitivity Undermines Embedding Model
Evaluation*.

- **Resolves:** the `embedding-model prompt selection` placeholder (§intro-known).
- **Our claim:** under adversarial prompt selection "any model can be promoted to first place" in a
  six-model, eleven-dataset embedding benchmark.
- **Source (abstract, verbatim):** "We present an empirical study of prompt sensitivity across
  **6 embedding models and 11 datasets**. ... we show that the leaderboard ranking is not robust to
  prompt selection: a developer improve their rank by favorably selecting prompts, and **under
  adversarial prompt selection any model can be promoted to first place**." (The grammatical slip in
  "a developer improve" is the source's, not a transcription error.)
- **Correction applied:** the draft said "essentially any model in a study". The abstract says *any*
  model, without hedge, so the hedge is dropped — but the study's scope (6 embedding models, 11
  datasets) is now stated, because "any model" unqualified would read as a claim about LLMs at
  large. These are instruction **embedding** models, evaluated by retrieval/classification metrics;
  no generative decoding is involved.

## 9. `sclar2024format` — arXiv:2310.11324 (ICLR 2024)

Sclar, Choi, Tsvetkov, Suhr, *Quantifying Language Models' Sensitivity to Spurious Features in
Prompt Design or: How I learned to start worrying about prompt formatting* (FormatSpread).

- **Resolves:** the `shot-count non-monotonicity` placeholder (§length), and adds one clause to
  §intro-known.
- **Our claim (§length):** the accuracy-level prompt space is reported as "largely non-monotonic",
  with graded atomic format changes giving monotone accuracy triples $32.4\%$ and $33.6\%$ of the
  time against a $33.3\%$ chance rate.
- **Source (body, §5, verbatim, via ar5iv full text):** "The space of prompt format accuracy is
  highly non-monotonic, which makes local search algorithms over the space less effective. ... We
  choose 24 tasks (13 multiple choice, 11 non-multiple choice), sample 300 $(p_1,p_2,p_3)$ triples
  for each, and the compute accuracy (using exact prefix matching) of each $p_i$ on 250 samples.
  **32.4 and 33.6% of triples were monotonic for multiple-choice and non-multiple-choice tasks
  respectively. Given that random shuffling within a triple will result in monotonicity 33.3% of the
  time**, this suggests that local search mechanisms like simulated annealing may not be effective."
  The abstract also says the space is "largely non-monotonic".
- **Our claim (§intro-known):** format performance "only weakly correlates between models".
- **Source (abstract, verbatim):** "We also show that format performance only weakly correlates
  between models, which puts into question the methodological validity of comparing models with an
  arbitrarily chosen, fixed prompt format."
- **CORRECTION — the placeholder was pointing at the wrong result.** The draft said "Prior work
  reports non-monotonicity in the number of in-context examples", and the gate's supporting quote
  for that was from Lu et al. (entry 5): "increasing the number of training samples leads to
  increases in performance. However, a high level of variance remains, even with a large number of
  samples and can even increase." That is variance-not-shrinking, **not** a non-monotone mean, so it
  does not support the sentence as written. The genuine non-monotonicity result in the literature is
  Sclar et al.'s monotonicity ratio over format triples, and the sentence now states that instead.
  The independent variable differs from ours (composed atomic format changes, not prefix token
  count) and the text says so.
- **Why this paper is also cited in §intro-known:** the F157 gate called it "THE most threatening
  paper" for our ranking claim. Citing it only for a minor non-monotonicity point while its abstract
  carries the sharpest cross-model concession would be the kind of selective citation this ledger
  exists to prevent.

## 10. `gu2025sink` — arXiv:2410.10781 (ICLR 2025 Spotlight)

Gu, Pang, Du, Liu, Zhang, Du, Wang, Lin, *When Attention Sink Emerges in Language Models: An
Empirical View*.

- **Resolves:** the `sink as no-op key bias` placeholder (§sink).
- **Our claim (§sink):** on the sink account, the sink "acts more like key biases, storing extra
  attention scores, which could be non-informative and not contribute to the value computation".
- **Source (abstract, verbatim):** "Most importantly, we find that **attention sink acts more like
  key biases, storing extra attention scores, which could be non-informative and not contribute to
  the value computation**."
- **Correction applied:** the draft said "the first token acts as a no-op key bias absorbing
  attention mass". *No-op* is our compression and the source hedges ("**could be** non-informative"),
  so the text now quotes the sentence rather than paraphrasing it harder than the authors did. The
  gate's version of the quote ("storing extra attention and meanwhile not contributing to the value
  computation") is a body paraphrase; the abstract wording above is what is cited.
- **Worth knowing, and it cuts our way and against us at once:** on this account the sink's
  attention share is largely non-informative, which is why §sink's observation — that a single BOS
  token moves a deterministic readout by ~0.7 in the fixed-point fraction, in model-specific *directions* — is hard to
  place. The mechanism named here predicts a small effect on computed values, not a sign flip.

## 11. `telephone` — arXiv:2407.04503 (ICLR 2025)

Perez, Kovač, Léger, Colas, Molinaro, Derex, Oudeyer, Moulin-Frier, *When LLMs Play the Telephone
Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings*.

- **Resolves:** the `iterated transmission chains` placeholder (§interaction).
- **Our claim (§interaction):** the nearest prior work on a structural readout uses iterated
  text-level transmission chains under stochastic sampling ($T=0.8$, top-$p$ 0.95) over six
  instruction-tuned models, and reports models drifting in opposite directions on a scalar text
  property.
- **Source (body, §4.1, verbatim, via arXiv v3 full text):** "In some cases, we also observe that
  **different models lead the distributions to be shifted in opposite directions**. For instance
  when looking at the evolution of texts length, using GPT3.5 or Llama3-8B leads text to become on
  average shorter, while using Mixtral-8x7B or GPT-4o-mini shifts the distribution towards greater
  lengths."
- **Source (body, §3.1, models):** "we run identical experiments using six different models ...
  GPT-4o-mini, GPT-3.5-turbo-0125 ('GPT3.5'), Llama3-8B-Instruct, Mistral-7B-Instruct-v0.2,
  Llama3-70B-Instruct, Mixtral-8x7B-Instruct-v0.1."
- **Source (body, hyperparameters):** "Temperature was set to 0.8 with and top_p to 0.95."
- **CORRECTION — the model count.** The draft said "two to three models". It is **six**. Watch out
  for an internal inconsistency in the paper itself: an earlier passage says "five different models"
  and lists five (omitting GPT-4o-mini), while §3.1 says six and lists six. Six is the operative
  number for the experiments; the "five" sentence looks like a leftover from an earlier version.
- **Also relevant, and it argues our way:** "the position of attractors appears to vary between
  models" (§4.3). This paper is therefore **not** a model-independence result — see entry 12 for the
  one that is. The two must not be conflated.
- **Their attractor is not ours:** a fitted linear-regression equilibrium of a scalar text property
  ($l = I/(1-s)$, convergent when $|s|<1$), estimated over 6 models × 3 tasks × 20 initial texts ×
  5 chains × 50 steps. No fixed-point census, no fixed-point fraction, no discrete class.

## 12. `paraphrase2cycle` — arXiv:2502.15208

Wang, Li, Yan, Cheng, Zhang, *Unveiling Attractor Cycles in Large Language Models: A Dynamical
Systems View of Successive Paraphrasing*, 2025.

- **Resolves:** the `model-independent attractors` placeholder (×2: §interaction and §limits) — the
  place where we say a prior result contradicts ours, so it is the entry that most had to be right.
- **Our claim (§limits):** this line reports iterated-map attractor states as model-*independent*,
  and the regime differs from ours in three named ways.
- **Source (body, §5.2, verbatim, verified by grepping the arXiv v1 HTML directly):** "Similarly, we
  introduce model variation by alternating among GPT-4o-mini, GPT-4o, Llama3-8B, and Qwen2.5-7B
  during successive paraphrasing. Although each model brings its own stylistic biases, the
  fundamental attractor cycle remains intact. Interestingly, perplexity computed by a single model
  (e.g., Llama3-8B) on paraphrases generated by other models still decreases over iterations in
  Figure 6. **This suggests that the attractor states are not confined to a single model's parameter
  space. Instead, they reflect a more general statistical optimum that multiple LLMs gravitate
  toward.**"
- **Source (abstract):** "This pattern persists with increasing generation randomness or alternating
  prompts and LLMs." (Intro: "This periodicity proves robust, remaining consistent across multiple
  models, text lengths, and prompts.")
- **Source (body, setup):** 8 English models (Mistral-7B-Instruct-v0.3, Llama-3-8B/70B-Instruct,
  Qwen2.5-7B/14B/72B-Instruct, GPT-4o-mini, GPT-4o); "we set the temperature to 0.6 and p to 0.9
  during the decoding process. We sample 10 different paraphrases at each step by setting the number
  of search beams to 10 ... We select the candidate with the highest probability for the next
  paraphrasing iteration."
- **CORRECTION — §limits was describing the wrong paper.** The draft characterised the contradicting
  regime as "an attractor defined as a fitted equilibrium of a scalar text property". That is
  `telephone`'s definition (entry 11), not this one. This paper's attractor is a **2-period cycle in
  normalised edit distance** over successive paraphrases. §limits now states the three actual
  differences: text-level iteration through an instruction-conditioned paraphrase call under
  stochastic decoding rather than a token-level deterministic argmax map; a 2-period cycle rather
  than a fixed point; and — the strongest of the three — model-independence inferred from
  **alternating** models *within* one chain plus a cross-model perplexity check, never from one
  prompt held fixed across separate models, which is the design our Table 3 runs (the
  bidirectionality table; it was Table 1 when this entry was written and the numbering has
  since moved).
- **Do not overstate the escape — REWRITTEN 21 Aug 2026, because §limits changed under it.** The
  earlier wording read: *"the contradiction is real in the regime where it was measured. §limits
  says so."* §limits no longer says that. It now argues the two measurements share no object, so
  neither dataset contradicts the other, and relocates the collision to the phrasing: their
  conclusion is stated as a property of LLMs at large, our regime is a counterexample to that
  phrasing at large, and both claims are kept scoped rather than either being declared wrong. The
  escape is still not free — what must not be overstated now is the *dissolution*. Our regime is a
  genuine counterexample to their general phrasing, and this entry's three named differences are
  what license the scoping. Recorded as a rewrite rather than edited silently, because the previous
  wording was correct about a paragraph that no longer exists.

---

## Method notes for whoever checks this next

- Four of the twelve placeholders were flagged "must not be guessed": three priority concessions
  (`model drifting` → entry 7, `MAPO` → entry 6, `prompt certification` → entry 5) and one
  contradiction (`model-independent attractors` → entry 12). All four are grounded in a verbatim
  quote from the source, three of them from the abstract.
- The candidate works were recovered from `results/prior_art_domain_journal.jsonl` (F157's gate
  journal), which records each paper's URL, extracted claims and supporting quotes. Every quote used
  above was then re-fetched from the source — arXiv abstract pages, ar5iv/arXiv HTML full text, and
  the ACL Anthology for bib details — except where this file says otherwise (the one exception is
  flagged in entry 3).
- No LaTeX toolchain is installed on this machine, so the PDF was not built. What was checked
  mechanically: every `\cite*` key in `main.tex` resolves to a `refs.bib` entry (12 of 12), and
  `refs.bib` has no duplicate keys. `refs.bib` still carries paper 1's unused entries; BibTeX
  ignores those.

## 13. `veraz2026probes` — arXiv:2608.10986 (our own companion paper)

Vera Zúñiga, *What Iterated Self-Feeding Probes of Language Models Measure, and a test that
separates the construction from the model*, 2026.

- **Cited in:** §ladder, "Relation to the companion instrument".
- **Our claim:** it established the estimator family's validity conditions, asked which readings of
  an iterated probe belong to the construction and which to the model, and gave the two-axis test
  separating them. This paper positions the prefix as a construction axis in that vocabulary.
- **Verification:** SELF-CITATION. Verified by authorship rather than by fetch — the source is
  `paper_arxiv/main.tex` in this repository, and its abstract states the discriminator directly:
  "hold the construction fixed and vary the model, or hold the model fixed and vary the
  construction, and see which readings move."
- **Why it is in this ledger anyway:** the ledger's rule is that every cited work carries a recorded
  basis, and "we wrote it" is a basis that should be stated rather than assumed. It was added when
  `tests/test_paper2_citations.py` failed on the restructure — the guard doing exactly its job on a
  citation that had been wired in without an entry.

## 14. `fu2021repetition` — arXiv:2012.14660 (AAAI 2021)

Fu, Lam, So, Shi, *A Theoretical Analysis of the Repetition Problem in Text Generation*, 2021.

- **Cited in:** §Setup, at the sentence defining the trajectory classes.
- **Our claim:** the funnel geometry — many states feeding one self-continuing token — was derived
  theoretically by them, and their inflow analysis explains why trajectories concentrate. The
  paper's classes are that geometry measured on a model's own conditional rather than on corpus
  counts.
- **Source (body, §2.2, discussion of Corollary 1.2):** "the inflow for a word is the probability
  sum of all words that take it as the subsequent word. If it is too big, the upper bound can be
  magnified extensively. This observation theoretically justifies the claim that high inflow words
  are more likely to go back to itself and cause the repetition problem." Corollary 1.2 labels the
  two halves of its denominator `outflow` and `inflow` explicitly.
- **Also supports (body, §2.1):** that greedy decoding makes the chain deterministic — "In greedy
  sampling, each word only takes a fixed subsequent word and thus $\zeta n = 1$. Therefore, ARP can
  be very large and even diverges to infinity." This is why the credit is for the geometry rather
  than for anything specific to our estimator: the deterministic argmax map is named in their §2.1,
  at a one-token window.
- **Verification:** FULL TEXT. The arXiv PDF (v4, 22 Mar 2021) was downloaded and extracted with
  `pdftotext -layout`; quotes above are transcribed from that extraction, not from an abstract and
  not from a search summary. Authors, title and the journal-ref line ("AAAI 21 Paper with Appendix")
  were re-checked against the arXiv abstract page on 21 Aug 2026.
- **What we deliberately do NOT claim from it.** Their transition matrix is built from **corpus word
  counts** in all three places it appears (Algorithm 1; §3; §4.2 "The Markov transition matrix is
  calculated by counting words in Wiki-103") — they never measure a model's own conditional. The
  paper credits the *explanation* and says its classes are the same geometry measured on the
  conditional instead. It makes no claim about their corpus-side inflow term. A separate repository
  finding (F171) tests that term directly and reports that it does not predict our endpoints once
  frequency is controlled; **that result is out of this paper's scope and is not cited here**, and
  nothing in the paper depends on it either way.
- **No page or volume numbers** are recorded in `refs.bib`: the arXiv journal-ref gives none, and
  inventing them is the failure mode this ledger exists to prevent.

## 15. `mahaut2025repetitions` — arXiv:2504.01100v2 (4 Nov 2025)

Mahaut, Franzon, *Repetitions are not all alike: distinct mechanisms sustain repetition in language
models*, 2025.

- **Cited in:** §intro-known, as adjacent to the axis this paper varies.
- **Our claim:** prompt *type* changes the mechanism of repetition itself — ICL setups that require
  copying recruit a dedicated, progressively specialising head network, while natural repetition
  emerges early without defined circuitry. And: that work measures no fixed points.
- **Source (abstract, v2, verbatim):** "ICL-induced repetition relies on a dedicated network of
  attention heads that progressively specialize over training, whereas naturally occurring
  repetition emerges early and lacks a defined circuitry. Attention inspection further shows that
  natural repetition focuses disproportionately on low-information tokens."
- **Verification:** the sentence quoted in `main.tex` is taken verbatim from the v2 **abstract**,
  re-fetched from the arXiv abstract page on 21 Aug 2026. The negative claim ("measures no fixed
  points") rests on a full-text read of v1 recorded in `results/prior_art_copy_gate.json`, which
  also records the cohort as Pythia 70M/1.4B/6.9B with no cross-model comparison.
- **A version discrepancy, recorded because it changed the draft.** The gate record was written from
  **v1**, whose framing named specific heads (L4H4, L9H9, L10H2) and "procedural copying behaviour".
  The live version is **v2**, whose abstract states the contrast differently. The sentence in the
  paper was re-drafted from v2 rather than from the gate note. Citing v1's phrasing against a v2
  paper would have been a quote the source no longer contains.
- **Not cited:** arXiv:2505.13514 (induction-head toxicity). The paper does not discuss copying, and
  that citation belongs to a mechanism write-up rather than to this manuscript.
