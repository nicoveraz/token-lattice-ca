# Paper 3 — plan

**Status: private.** Branch `paper3`, unpublished. This is a plan, not a draft: no claim here has
been made in public, and several of its load-bearing analyses have not been run. It becomes a
manuscript only after the runs in §5 land and the gate in §6 clears.

**Working title (placeholder, deliberately not a verdict).** *What a cross-model fixed-point census
can and cannot arbitrate about repetition.*

**Provenance.** Written 21 Aug 2026 from F169–F171 plus a field check the same day. The census this
paper is built on (17 models) already exists; the two corpus-side tests below are counting exercises
over stored censuses, not new forward passes.

---

## 1. The question, and why it is open

Two accounts of repetition/degeneration coexist in the literature and have never been arbitrated
across a broad cohort of *pretrained* models, because each camp measures in a way that cannot see the
other:

- **Language/data side.** Repetition is caused by properties of the training text. Fu et al.
  (arXiv:2012.14660, AAAI 2021) derive it from corpus bigram structure — the *high inflow* term — and
  state the thesis outright: *"the repetition problem is caused by the language itself."* The live
  empirical champion is Li et al., *Repetition In Repetition Out* (arXiv:2310.10226, NeurIPS 2023):
  *"a strong correlation between the degeneration issue and the presence of repetitions in training
  data,"* backed by a causal intervention (attention dropout on repetitive words during training).
- **Weights side.** Repetition is a property of the trained network. *Understanding the Repeat Curse*
  (arXiv:2504.14218, ACL Findings 2025) locates SAE "repetition features"; the induction-toxicity
  line (arXiv:2505.13514) makes copying circuits the cause; *Repetitions are not all alike*
  (arXiv:2504.01100) shows prompt type selects the mechanism.

**The gap.** The data-side camp proves causation by *training* models — few models, own corpora, and
no cross-**architecture** comparison. *(Corrected 21 Aug 2026 after F175 read Li et al. in full: the
earlier wording said "no cross-model comparison", which their Figure 2(b) refutes — an off-the-shelf
OPT size ladder. What they lack is a cross-architecture or cross-corpus comparison of pretrained
models, and any readout that is not generation. They also state the architecture gap themselves:
"the two factors have not been quantitatively evaluated".)* The weights-side camp goes deep on 3–5 models with no corpus term. Nobody has
measured a corpus-independent structural readout across a *wide* cohort of off-the-shelf pretrained
models and asked which account predicts what varies.

**What this paper can be, stated honestly up front.** Observational cross-model evidence, in a
currency different from Li et al.'s controlled manipulation. It cannot refute a training intervention
and will say so in the abstract, not only in limits.

## 2. Prior-art position (as established, not as hoped)

Field check 21 Aug 2026 — three searches, four fetches, recorded here as a SCAN, not the gate:

- **Nobody has this object.** *Solve the Loop* (arXiv:2605.12466) designs fixed-point iteration into
  an architecture and solves one equilibrium per input: no census, no enumeration, no cross-model
  claim, models trained by the authors. *Internal Data Repetition Destroys LMs* (arXiv:2606.24998)
  is training-loss dynamics under document duplication and explicitly does not address generation.
  Neither collides.
- **The funnel's explanation is Fu et al.'s** (F170) and is credited as such in paper 2 already.
- **Owed and binding:** the prior-art gate at F91/F157 protocol depth. F169 was six searches by hand
  with four SNIPPET-grade items unread and two PDFs unextracted; the field check above does not
  replace it. Li et al. (2310.10226) must additionally be read in FULL before any sentence about it
  is written.

## 3. The exhibits, in the order they should be argued

**E1 — the matched-corpus pair (primary, and the one the field cannot currently deliver).**
`pythia-410m` and `pythia-410m-deduped`: same architecture, same corpus, one deduplicated — the exact
manipulation the data-side account says should matter, available off the shelf. F171 records both
landing on the same endpoint token (`'\n'`) and the same class (funnel), with identical inflow rows.
If that holds under the analysis in §5, it is a targeted null against the strongest data-side form,
at zero training cost.
**Small-n is the whole risk here: this is ONE pair. The registered claim must be scoped to it before
anything is computed.**

**E2 — the corpus term fails its author's own control** (F171, F174). **DEMOTED 21 Aug 2026 by F175, and the demotion is binding.** Li et al. §6.2 already subsumes Fu's inflow by a controlled experiment — merging only repetitive∩high-inflow pairs (8.1% of words) matches full HI-RE (31.1%), while random high-inflow pairs of the same size do nothing. **E2 is therefore CONVERGENT with the data-side camp, not a strike against it**, and must be staged as a consistency check rather than a refutation. Zihao Fu co-authors both papers. Endpoint
inflow beats frequency-matched peers on **1 of 13** models, median matched percentile **32.0** — while
the naive uncontrolled criterion reads **12 of 13 at or above the 90th percentile, median 99.87**.
The control is one Fu et al. themselves motivate (*"it is not the high-frequency words, but the high
inflow words that really lead to repetition"*). These maps funnel to COMMON tokens, not HIGH-INFLOW
ones. The trivial claim — models get stuck on frequent tokens — stands untouched and is not disputed.

**E3 — heterogeneity a uniformity thesis cannot produce.** 8 of 17 models are funnels; same-era models
trained on comparable corpora diverge in class. Language-caused predicts uniformity.

**E4 — no corpus statistic tested separates the classes** (F171 H3, TIER 2). Max corpus inflow spans
[1320.7, 4684.9] on funnels and [2651.7, 15196.2] on non-funnels — overlapping. Registered as *does
not separate*, never as proof of language-independence.

## 4. What this paper must NOT claim

- Not that repetition is weights-caused. Two corpus-side *terms* failed; the class is not exhausted.
- Not a refutation of Li et al. Their evidence is a training intervention; ours is observational
  across pretrained models. Different currencies, stated in the abstract.
- Not that Fu et al.'s **bound** was tested. F171 tested their inflow *term*. The bound is a different
  object — test it or scope it out in one loud sentence, because a referee who read them will ask.
- Not a causal claim of any kind. No p-values on 13 rows that are not 13 independent tests (§5.2).

## 5. Runs required before drafting (all zero forward passes unless noted)

**5.1 — E1 at analysis grade.** The deduped pair currently appears as two rows in an F171 table. It
needs: both censuses at full seed count with the class-stability rule applied (the rule that already
excluded LFM2 and starcoder2 for modal-endpoint disagreement across seeds); φ, class, endpoint token
and margin reported per seed; and a pre-registered statement of what "the manipulation moved nothing"
means numerically, written before the numbers are joined. Search the cohort for any SECOND
matched-corpus pair (deduped/non-deduped, or same-corpus siblings); one pair is an anecdote, two is a
pattern, and the search itself is free.

**5.2 — the non-independence prereg for E2.** F171's 13 rows are ~5 clusters: `'\n'` is the endpoint
for 6 models, `'0'` for 3, and models sharing a token share corpus statistics *exactly*; `pythia-410m`
and `-deduped` share a tokenizer and produce identical rows. The cluster-level analysis must be
registered before it is run, including whether the observed reversal (median 32 < 50) is reported at
all. F171 recorded it as an observation and declined to claim it; that decision stands until a
registered cluster analysis says otherwise.

**5.3 — a second corpus, at minimum one non-English.** Turns the F171 exclusions from embarrassments
into data: `bloom-3b` was excluded because its endpoint `' ciudad'` occurs 0 times in an English
corpus (a statement about the corpus, not the theory); `llm-jp-3-1.8b` and `polyglot-ko-1.3b` are
untested in their own languages. Corpus token-counting against stored censuses.

**5.4 — class stability as a stated boundary.** Seed-level agreement on class and modal endpoint for
every model in the cohort, reported as a table. Two models are already known unstable; the rest are
assumed stable and that assumption has never been printed.

**5.5 — the gate.** §2's owed protocol-depth gate, plus Li et al. read in full. Precondition for
drafting, not a follow-up.

## 6. Kill conditions (registered here, before the runs)

- **K1.** If a second matched-corpus pair exists and its class DIFFERS across the manipulation, E1 is
  dead and the paper loses its primary exhibit. Report it and stop.
- **K2.** If the non-English corpus reverses E2 — endpoints beating frequency-matched peers in a
  majority of models on their own-language corpus — then F171's result is an English artefact and the
  paper is about that instead.
- **K3.** If the cluster-level analysis (5.2) leaves fewer than 4 independent clusters, E2 is reported
  as descriptive only, with no rate and no comparison to 50.
- **K4.** If the protocol-depth gate finds a cross-model repetition census, this paper is a
  replication or a comment, and is re-scoped accordingly.

## 7. Sequencing

1. Paper 2 submitted and announced (in flight). **Submitted 21 Aug 2026, awaiting announcement.**
2. ~~5.1 pair search + 5.4 stability table~~ — **DONE 21 Aug 2026, F172.** K1 did not fire, but not
   reassuringly: it is conditioned on a *second* manipulation pair and there is no second one, so E1
   rests on the single `pythia-410m` / `-deduped` pair as the plan feared. Class is stable 17/17,
   modal endpoint 15/17. Two things the plan did not anticipate: the cohort holds a same-corpus
   **triple** (`gpt-neo-2.7B` is Pile-trained too) in which all three models reach the *same* endpoint
   token `'\n'` while φ spans 0.036 to 0.458 — so E3 gains a *within-corpus* instance, stronger than
   the "comparable corpora" version written in §3 — and 7 of 17 models have undisclosed corpora, so
   the pair search's null is bounded by what model cards disclose.
3. ~~5.2 cluster prereg, frozen and hashed.~~ **DONE 21 Aug 2026, F173.** Registered, then run.
   H1 not supported and **H0 stands** — E2 stays descriptive. Three gates fired: K6 (the two
   aggregation rules disagree, so neither is the answer — `OLMo-2` at 52.0 is unanimity's only
   obstacle and the glyph rule hides it), K7 (§5.2's premise is false, confirmed from data), and K5
   (the clustering did almost no work). **§5.2's premise was wrong in both directions**: the glyph
   rule over-merges, and "13 rows are ~5 clusters" is really 12 clusters, 11 of them singletons. The
   models are very nearly independent already, which also qualifies F171's non-independence caveat.
   The re-aggregation axis is now exhausted; only more models would move E2.
4. ~~5.3 second corpus.~~ **DONE 21 Aug 2026, F174.** Three languages at 20M chars each (es/ja/ko),
   size-matched to F171's English. **K2 does not fire** (1 of 3 above the null). Coverage went 1/3 →
   3/3: `bloom-3b` (es, 6.0) and `polyglot-ko-1.3b` (ko, 16.0) were unmeasurable in English — their
   endpoints occur **zero** times there — and both land far below the null, agreeing with F171.
   `llm-jp` swings +60 (36.0 en → 96.0 ja) but its endpoint's frequency also moves 11×, so the
   matched peer set changes and the swing cannot be attributed to language. **This step required
   network access**: nothing non-English of usable size was cached (302 Korean characters locally),
   and `polyglot-ko`'s tokenizer was missing, which was all of F171's `OSError`.
5. 5.5 gate — **Li et al. in full: DONE 21 Aug 2026, F175.** **Protocol-depth gate: NOT RUN**, and
   it is the larger half. K4 remains live, so **drafting is still blocked**.
6. Draft only then. Structure to follow paper 2's ladder idiom: the question, what each account
   predicts, exhibit by exhibit, then a limits section that concedes the causal currency gap first
   rather than last.

## 8. Relation to the other open thread

The mechanism thread (F165/F166 — the prefix selects a token, the model decides whether it
self-continues; the rank-1 fit at 0.790 under an unmoved 0.80 bar; copy strength twice NOT DECIDABLE
with an inverted direction) is **paper 4, not this paper**. It is blocked on widenings that have not
run, and forcing a manuscript out of pending verdicts would break the discipline that produced them.
Nothing in this plan depends on it.
