# Citation ledger for paper 2

Why this file exists: F157's prior-art gate **refuted 13 of 74** extracted claims for overreaching
their own sources. Its summaries are therefore not citable. Every entry in `refs.bib` must be
verified against the source itself, and the supporting quote recorded here so the next reader can
check the citation without repeating the fetch.

Status legend: **VERIFIED** = fetched the source and confirmed the claim the paper makes.
**UNRESOLVED** = placeholder still in `main.tex` as a red `\citepend{}`.

---

## VERIFIED (4)

### `cao2024worstprompt` — arXiv:2406.10248
Cao, Cai, Zhang, Zou, Lam, *On the Worst Prompt Performance of Large Language Models*, 2024.

- **Our claim:** per-prompt performance rankings agree across models only weakly, Kendall's
  $W = 0.238$.
- **Source (full text, HTML v2):** "The consistency between all models is significantly lower" at
  $W = 0.238$; Llama family $W = 0.443$, Gemma $W = 0.548$.
- **Note:** the abstract does NOT contain this number — it is in the body. The first fetch of the
  abstract alone did not support the claim, which is exactly the failure mode this ledger exists
  for. Also supports (unused): worst-prompt overlap at $k{=}1$ is 2% (Llama) / 13% (Gemma).

### `alzahrani2024benchmarks` — arXiv:2402.01781
Alzahrani et al., *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model
Leaderboards*, 2024.

- **Our claim:** semantically neutral reformatting reorders leaderboards by up to 8 positions.
- **Source (abstract, verbatim):** "minor perturbations to the benchmark, such as changing the order
  of choices or the method of answer selection, result in changes in rankings up to 8 positions."
- **CAUTION:** our draft previously said "11-model leaderboard". The abstract does **not** state the
  model count, so that number is not carried into the text. Verify before reintroducing it.

### `xiao2023streamingllm` — arXiv:2309.17453
Xiao, Tian, Chen, Han, Lewis, *Efficient Streaming Language Models with Attention Sinks*, 2023.

- **Our claim:** initial tokens dominate a scalar readout; the mechanism is positional, not semantic.
- **Source (abstract, verbatim):** "the emergence of attention sink is due to the strong attention
  scores towards initial tokens as a 'sink' even if they are not semantically important."
- **PARTIALLY UNVERIFIED:** our §sink also says "replacing the first tokens with meaningless
  linebreaks restores the readout nearly as well as the originals", with specific perplexity figures
  reported by the gate (5.60 vs 5.40). That detail is **in the body, not the abstract, and has not
  been checked**. Either verify it or cut the sentence before submission.

### `voronov2024format` — arXiv:2401.06766
Voronov, Wolf, Ryabinin, *Mind Your Format: Towards Consistent Evaluation of In-Context Learning
Improvements*, 2024.

- **Our claim:** the same template component can be best for one model and among the worst for
  another.
- **Source (abstract, verbatim):** "the best templates do not transfer between different setups and
  even between models of the same family". 21 models, 770M–70B, 4 classification datasets.
- **CAUTION:** the abstract supports **non-transfer**, which is what our sentence should say. It does
  not by itself support the stronger "best for one, among the worst for another" phrasing — that was
  the gate's wording. Soften the sentence or verify the stronger claim in the body.

---

## UNRESOLVED (12 placeholders, 8 distinct works)

**Blocked on web search**, not on judgement: this session exhausted its 200-call search budget on the
prior-art gate itself, and these works are known only by description, not by arXiv ID. Raising
`CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` unblocks them.

| placeholder | what we assert | risk |
|---|---|---|
| `model drifting` | a prompt optimised for a source LLM is suboptimal on a target; the phenomenon is *named* this | **high** — we credit a specific name |
| `MAPO, Chen et al. 2024` (×2) | earliest anchor for prompt effectiveness being model-specific | **high** — a priority claim |
| `prompt certification / non-transferability` | "cannot be certified independent of the model" appears in a prior abstract | **high** — our §interaction concedes priority to it |
| `model-independent attractors` (×2) | a prior line reports iterated-map attractor states as model-independent | **high** — we say it contradicts us; must be right |
| `iterated transmission chains` | models drift in opposite directions on a scalar text property, 2–3 models, stochastic | medium |
| `embedding-model prompt selection` | adversarial prompt selection promotes any model to rank 1 | medium |
| `sink as no-op key bias` | the sink is a no-op key bias absorbing attention mass | medium |
| `shot-count non-monotonicity` | prompt effect non-monotone in number of in-context examples | low — could be cut |

**The four marked high risk are the ones that must not be guessed.** Three of them are places where
the paper *concedes priority* (§intro-known, §interaction) and one is where it says a prior result
contradicts ours (§limits). Citing the wrong paper for a concession is worse than the concession
being vague: it attributes a claim to authors who may not have made it.
