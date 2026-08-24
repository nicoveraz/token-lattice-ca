# token-lattice-ca

[![arXiv](https://img.shields.io/badge/arXiv-2608.10986-b31b1b.svg)](https://arxiv.org/abs/2608.10986)
[![arXiv](https://img.shields.io/badge/arXiv-2608.21315-b31b1b.svg)](https://arxiv.org/abs/2608.21315)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21880472.svg)](https://doi.org/10.5281/zenodo.21880472)

Turn a language model into a **cellular automaton over token space**, then use it as a black-box
measurement instrument. A ring of *N* token cells is updated by the model's own windowed conditional
`p_r(x_i | x_{i±r})`; nothing about the weights is inspected and nothing is trained.

The research record is a dated ledger, [`findings.md`](findings.md), currently **F1–F171**. It keeps
retracted and corrected findings in place, with the correction stated where the claim was made —
quoting a finding without its amendment misrepresents it.

---

## Two papers

**Paper 1 — the instrument.** *What Iterated Self-Feeding Probes of Language Models Measure, and a
test that separates the construction from the model.* **Published: [arXiv:2608.10986](https://arxiv.org/abs/2608.10986)** (cs.CL, 11 Aug 2026), 15 pages.

Its subject is the apparatus. The instrument is validated by *reproduction* — before measuring a
language model, whose true dynamical metrics are unknown, it reproduces known metrics on systems
where the answer is established (elementary CA rules, synthetic Markov sources, and the
Domany–Kinzel PCA, where the damage machinery is checked **bit-for-bit, zero mismatching cells**).
The load-bearing result is a **discriminator**: hold the construction fixed and vary the model, or
hold the model fixed and vary the construction, and see which readings move. Applied to itself it
splits the instrument in two — the construction has more dynamic range than the model does, so
`λ_ca` is a within-model developmental quantity rather than a model-comparison one, while the
attractor share passes every gate `λ_ca` fails. Source: [`paper_arxiv/`](paper_arxiv/).

**Paper 2 — the domain.** *Prompt–Model Interaction Reaches the Fixed Points: a deterministic,
task-free structural readout — and the factorizations of it that failed.* **Published:
[arXiv:2608.21315](https://arxiv.org/abs/2608.21315)** (cs.CL, 21 Aug 2026), 11 pages; source in
[`paper2_arxiv/`](paper2_arxiv/).

Its subject is what the instrument is pointed at. That a prompt's effect is not a property of the
prompt is established — but all of that evidence is about *task accuracy*, which cannot say whether
the interaction lives in the machinery of performing tasks or in the conditional distribution
itself. So it asks on a readout with no task in it: the fixed-point structure of the short-window
argmax map `x_{t+1} = argmax p(x | x_{t-1}, x_t)`, censused from 96 starts. Two results. The
interaction **reaches** that readout — nine tokens of conditioning move the fixed-point fraction
across most of its range and change a four-way structural class, while instruction tuning worth
+60.5 IFEval points moves the class by zero. And **nothing proposed carries it**. Prefix length fails:
the effect is not monotone. Five factorizations — two on the prefix's content, three on the model —
each dissolved when the sample widened, four of them within one run of being proposed. And the
nearest mechanistic account, attention-sink dominance, predicts the sign of the shift on 2 of 5
models, which is chance. What is left is the prompt–model pair.

The two are companions: paper 1 establishes what readings of an iterated probe belong to the
construction rather than the model, and paper 2 varies a construction axis — the prefix — and
reports that its effect on the readout is model-conditioned in sign.

---

## Where the detail lives

Nothing below is summarised twice. Each document is the authority for its own scope.

| Document | What it is the authority for |
|---|---|
| [`explainer.md`](explainer.md) | The whole project in plain English, no background assumed. **Start here if you are new.** |
| [`ca_constructions.md`](ca_constructions.md) | The four rules drawn side by side in ASCII — elementary CA, Domany–Kinzel, and this project's two token-lattice constructions. |
| [`findings.md`](findings.md) | The dated research record, F1–F171: every pre-registration, verdict, boundary, retraction and amendment. |
| [`what_it_measures.md`](what_it_measures.md) | What the instrument does and does not read, stated as scope rather than as caveats. |
| [`critical_analysis.md`](critical_analysis.md) | The standing adversarial read of the programme. |
| [`paper_arxiv/REVIEW.md`](paper_arxiv/REVIEW.md) | The audit that reshaped paper 1's claims. |
| [`paper2_arxiv/CITATIONS.md`](paper2_arxiv/CITATIONS.md) | Paper 2's citation ledger: every cited work verified at source with the supporting quote recorded. |
| [`paper2_arxiv/SUBMISSION.md`](paper2_arxiv/SUBMISSION.md) | Paper 2's arXiv metadata, and the record of what was decided at submission -- including two points raised and declined. |
| [`gatecheck/`](gatecheck/) | The verdict layer, published as an installable package with its own design notes. |

---

## Two results to read before using any number here

Both are boundaries the project established against itself, and both are load-bearing.

**The instrument measures the construction, not the generative process (F35).** Real autoregressive
generation does not absorb an injected token error — `P_persist = 1.000` on three models, with the
CRN null exactly zero. Free generation never resamples a token, so an error stays in context
permanently, whereas the ring CA revisits every site, which is what makes healing possible at all.
Damping length and repair length therefore characterise the **iterated-resampling construction**.

**The universality-class programme was measuring the probe (F62–F66).** A damage-spreading
transition was found, a critical point located and exponents fitted — and then the object turned out
to be the melting of an **out-of-distribution prompt degeneracy**: a second model family shows no
transition, nineteen models refute a corpus explanation, the frozen phase exists only at `r=2`
carried by one vocabulary entry, and the masked-LM construction shows none of it. The exponents are
not wrong; what they are exponents *of* is the probe. **Varying the subject is not a substitute for
varying the apparatus.**

Getting here cost eleven confident verdicts, every one caught by its own check before reaching a
paper. `findings.md` carries them with their banners.

---

## The validation ladder

Before measuring a language model — whose true dynamical metrics are unknown — the instrument
reproduces *known* metrics on systems where the answer is established: elementary CA rules (ordered
vs rest separates decisively on ignition probability; the finer three-class ordering does **not**
survive), the known transition matrices of synthetic Markov sources, and the **Domany–Kinzel**
stochastic PCA, which is the only rung that is both stochastic and discrete — the instrument's own
regime — and where the damage machinery is checked bit-for-bit against an independent prediction.

Two rungs are weaker than they look and are labelled as such rather than quietly counted: the
logistic map and the coupled-map lattice are **smooth-limit arithmetic unit tests**, not validations
of the instrument, because their estimator renormalises along the reference orbit while a token flip
is O(1) in a discrete alphabet. See F30/F31 in [`findings.md`](findings.md) and
[`what_it_measures.md`](what_it_measures.md) for the full accounting, including which earlier
headlines an adversarial audit demoted and why.

### The findings record

[`findings.md`](findings.md) is the authoritative record — **F1–F171**, each entry carrying its
pre-registration, its verdict and its boundary. It is not summarised here: an index that drifts from
the ledger is worse than no index, and this one drifted for ninety rows before it was removed.

A few landmarks, to show what kind of thing is in there:

| | |
|---|---|
| **F38** | The Domany–Kinzel rung is **exact** — the CRN damage field is provably a DK automaton, checked cell-for-cell, zero mismatches, through the same loop every model number comes from. |
| **F35** | Real generation does not absorb a token error, so healing is a property of the construction. |
| **F66** | The degeneracy is an out-of-distribution prompt artifact; the masked-LM construction shows none of it. |
| **F119** | Fifteen scripts ranked with `argsort(argsort(x))`, so a scalar whose 24 measured values were all exactly 0.000 was reported as correlating at ρ = +0.829. The data was honest; the defect was in the correlation function. |
| **F157** | Paper 2's prior-art gate refuted 13 of 74 extracted claims for overreaching their own sources — which is why `CITATIONS.md` records quotes rather than summaries. |
| **F171** | An outside theory's prediction about this data passes the obvious test at the 99.9th percentile and dies against a frozen frequency-matched control. |

## Layout

```
src/           library
  lattice.py     THE simulation loop (one Rule protocol; all backends share it)
  model.py       tiny bidirectional transformer (the CA rule family p_r)
  ca.py          toy backend (ToyRule) + metrics; run() is a shim over lattice.run
  mlm_ca.py      masked-LM backend (MLMRule, symmetric masked-centre window)
  ar_ca.py       autoregressive backend (ARRule, left-causal window)
  dk.py          Domany-Kinzel PCA (DKRule + vectorised reference + published anchors)
experiments/   runnable pipeline steps (run from repo root)
  dp_calibration.py   ONE implementation of the DP calibration gate, imported by every DP run
  dp_pipeline_validation.py  can the fitting code recover DK's known exponents? (gates the rest)
  dp_survival_scan.py        phase 1: bracket the critical region in temperature
  dp_narrow_bracket.py       phase 2: narrow it (verdict withdrawn, see F56/F57)
  dp_class_n192.py           delta/theta crossing test at a geometry that can decide (F58)
  dp_fss_z.py                finite-size scaling for the dynamic exponent z (F59)
  dp_scan_gpt2.py            second model family: bracket scan -- found NO transition (F62)
  attractor_corpus_screen.py 19-model screen of the low-T attractor (F63/F64)
  attractor_interventions.py radius sweep + token ablation, with a control (F65)
  attractor_construction.py  AR vs AR+BOS vs masked-LM -- identifies the mechanism (F66)
  mlm_transition.py          does the CLEAN construction have a transition? (#89, gated)
  vocab.py       word-level vocab builder (pilot); see bpe.py for the BPE variant (Phase 2)
  train.py       train the windowed conditional model on tinyshakespeare
  sweep.py       coarse T×r phase sweep (async + one sync row)
  census.py      attractor census + corpus recovery + melting + cycle check
  damage.py      damage spreading: CRN twin runs, damage cones
  crystal.py     run the full suite on every training checkpoint (crystallization)
  differential.py differential CRN certification (null / apparatus / model arms)
  analyze_figs.py figures 1–6 + analysis.json
  crystal_fig.py  crystallization figure

  -- later programmes; one script per pre-registered question, grouped by thread.
     Every one is resumable (saved per cell, keyed by its design tuple).

  the degeneracy artifact and what it is (F66-F72)
    context_threshold.py       is the degeneracy confined to r<=2? (F69)
    fixed_point_onset.py       when does the argmax funnel form? (F84/F85)
    argmax_census_hardened.py  96-start dual-seed census + the dedup confound pair (F90/F91)
    basin_dependence.py        does the absorbing state have a basin? (F72)
    novelty_structure.py       does the clean construction produce novelty? (F71)

  what the developmental transition IS -- four routes, all reported
    context_onset.py           route 1: context-use onset (F78, indeterminate)
    ablate_layers.py           route 2/3: per-layer and per-component ablation (F79/F80)
    conditional_sensitivity.py route 3 support: does the conditional collapse? (F82/F83)
    meanfield_lambda.py        route 4: DERIVE lambda_ca from sensitivity, on the ladder (F94)
    canalization.py            F94 follow-on: spread + sub-additivity, three regimes (F96)
    loss_collapse.py           lambda_ca vs loss (F88, NOT DECIDABLE)

  external anchors and second targets (#90/#101/#102)
    band_screen.py             the keystone: T* vs degeneration at family level (F86)
    band_family_census.py      "no attractor" is two mechanisms (F87)
    degeneration_vs_tstar.py   the static map vs the CA, on the anchor (F92)
    tstar_second_target.py     the second target rejects itself (F93)
    memorization_gate_a/_b.py  is retention even measurable at practical radii? (F89)

  assembly / complexity rung (#20 -- built, then falsified its own spec)
    assembly_calib.py          exhaustive RePair sweep on a^n (F73)
    assembly_baselines.py      compression and entropy baselines (F74/F75)
    assembly_temperature.py    the instrument-selection rung Delta fails (F76)

  paper and guards
    build_paper_manifest.py    every paper number traced to a results/ file
    precommit_guard.py         blocks a claim whose number is not in the manifest
    fig_instrument.py          Figure 1 (the instrument panel)
fingerprint/   the black-box fingerprint programme (PROGRAM.md + gate1/2/3); imports gatecheck
gatecheck/     installable verdict-layer package (DESIGN.md, 42 tests)
tests/         pytest harness regression tests (null test, determinism, sanity)
data/          shakespeare.txt, token ids, vocab.json
ckpt/          0.42M-param checkpoints (step1000..6000, final) — tracked, no retrain needed
results/       raw npz + summary.jsonl / census.json / analysis.json / differential.json
               NEVER hand-edited — when a results file is wrong, the SCRIPT is fixed and re-run
fig/           figures (png)
paper_arxiv/   paper 1 (published): main.tex + REVIEW.md + the withdrawn I4D submission
paper2_arxiv/  paper 2 (submission-ready): main.tex, CITATIONS.md, SUBMISSION.md, packaging
```

The library and scripts use **paths relative to the repo root** (`data/`,
`ckpt/`, `results/`, `fig/`). Always run from the repo root. Imports resolve via
a small `sys.path` shim at the top of each script and `conftest.py` for tests —
no install step needed.

## Install

Plain **CPU** JAX (the tiny model does not need a GPU; `jax-metal` is flaky on
M-series and is deliberately avoided).

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "jax[cpu]" numpy scipy matplotlib pytest
# Phase 3 (real pretrained MLMs) additionally:
.venv/bin/pip install torch transformers tokenizers
```

Tested with Python 3.11, jax 0.10.2, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.1.

## Tests

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest tests/ -v
```

`test_null_coupling_is_exactly_zero` is the critical regression: coupled twin
runs sharing model, init, update order, and uniforms must diverge by **exactly
zero**. If it fails, the common-random-number coupling is broken and every
damage / differential number is meaningless — fix the harness first.

The suite is **741 tests** and covers every backend, not just the toy JAX path:

- `tests/test_null_all_backends.py` — the exact-zero null over `{stub, mlm, ar}` ×
  `{async, sync}`. All three backends run through one loop (`src/lattice.py`), so a
  `StubRule` (no model load) exercises the same code path *by construction*; MLM and AR
  are also tested directly. Includes a **non-vacuity** counterpart — a perturbation must
  propagate, else the null test would pass trivially.
- `tests/test_dk_damage_identity.py` — the **exact** rung. On the Domany–Kinzel `p2=0` line
  the CRN damage field is provably a DK automaton at the same `p1`, so the damage machinery
  is predicted by an independent simulator and asserted **cell-for-cell** — through
  `lattice.run`, the loop every model number comes from. Includes an off-line control that
  must fail, so the exact test cannot pass vacuously.
- `tests/test_results_self_consistency.py` — a **results file must not contradict its own
  declared design**. Two defects lived in analysis code that generated a JSON that fed a figure
  and three documents, and no prose grep could find them: F39 (`n_pre` was 8 where the declared
  pre set gives 16) and F42 (λ averaged over runs where it is undefined). These assert the
  emitted `n` against the design, the ignition-filter asymmetry between λ and D_norm, and the
  *margin* of the `D_norm==0` fallback rather than its current value.
- `tests/test_paper_numbers.py` — **the paper must not contradict its results files.** Every
  load-bearing number in `paper.tex` is checked against the JSON it came from, plus submission
  hygiene (no unverified citations, no self-identifying strings, responsible-use section present).
  It caught a real loss on its first run: a DK measurement dropped during a page-fit trim.
- **Stale-analysis detection** (`experiments/provenance.py`): every analysis stamps the sha256 of
  its own source into the results file, and the suite recomputes it. Editing an analysis script
  while its job runs leaves the job writing its end-of-run analysis with the code imported at
  launch — that happened twice and once **inverted a conclusion**. A mismatch is now a red test
  rather than a finished-looking wrong number.
- `tests/test_refs_match_arxiv.py` — **every arXiv citation must match what arXiv says.** A hand
  audit fixed three invented titles and then missed a fourth, which survived until the entry was
  opened for an unrelated reason. `experiments/audit_refs.py` fetches the record (network); this
  test compares refs.bib against it **offline**, so the suite never depends on arXiv being up.
  24/24 verified.
- **No absolute paths in machine-written logs.** Twelve scripts printed an absolute `OUT`, so
  twelve logs carried the checkout path — a de-anonymisation leak in an artifact the submission
  mirror publishes. `provenance.rel()` is the fix, and a guard stops the thirteenth script
  reintroducing it. It checks only what *we* print: Python's own tracebacks and stdlib warnings
  carry absolute paths no source change can reach.
- **One definition of the F42 predicate.** λ is undefined without a cone, so λ statistics drop
  unignited runs while D_norm keeps them (zero damage is a true zero). That adapter was
  hand-written thirteen times and written wrongly twice — once inflating D_norm's N=192 plateau by
  14%. It now lives once, in `lyapunov.run_ignited`, and a test asserts nobody re-implements its
  *shape*, since renaming a local helper is how the fourteenth copy would slip past.
- **Visit order is shared across the batch by default, and that is a measurement hazard (F57).**
  `lattice.run` draws one `rng.permutation(N)` per sweep and *every replica follows it*. For bulk
  statistics that is harmless. For single-site damage spreading it is not: the AR rule is
  causal-left, so damage at site *j* propagates only if *j+1* or *j+2* is visited **before** *j* —
  otherwise *j* resamples against an identical context with the same uniform, heals, and the run
  is absorbed. That is **1/3 of orders**, and because the order is shared it kills the whole batch
  at once rather than a third of the replicas. 512 "replicas" then carry the statistical weight of
  one draw of the thing that decides the outcome. This is also the long-unexplained cause of F42's
  unignited runs.

  `order="per_replica"` gives each replica its own permutation. It is **opt-in** — the default is
  unchanged, because switching it would move every async number in the repo — so scripts that need
  independence ask for it. As with `u_stream`, an explicit `order_stream` can be supplied so twin
  runs share orders exactly; **CRN coupling requires the twins to be visited in the same sequence**,
  which independently drawn orders would break. The exact-zero null is asserted under the new mode
  too (`tests/test_null_all_backends.py`).
- **Run a control that should NOT show the effect** (F65). The radius sweep read as "the attractor
  survives to r=16" until the control — a model with no attractor at r=2 — acquired one there too,
  revealing a generic long-context effect rather than the phenomenon under study. The verdict logic
  now reads the treatment *minus* the control, not the treatment alone.
- **Vary the construction, not only the model** (F66). Nineteen models could not distinguish "a
  property of language models" from "a property of the probe". One change of CA — masked-centre
  infilling instead of a two-token AR prompt — settled it immediately. Every number in this repo
  before F66 came from a single rule that had never been varied.
- `tests/test_golden.py` — asserts the simulation core stays **bit-identical** against
  `tests/golden/*.npz`, which were generated *before* the Phase-1 refactor. Do not relax
  these to `allclose`; a backend that cannot be made bit-identical is a stop-and-report.

## Reproduce the pilot (Phase 1)

Run from the repo root. Checkpoints are already provided, so `train.py` is
optional (only needed to regenerate `ckpt/`).

```bash
export JAX_PLATFORMS=cpu
.venv/bin/python experiments/vocab.py        # build data/ (word-level vocab)
.venv/bin/python experiments/train.py        # OPTIONAL — ckpt/ already provided
.venv/bin/python experiments/sweep.py        # -> results/sweep_*.npz, summary.jsonl
.venv/bin/python experiments/census.py       # -> results/census.json, census_*.npz
.venv/bin/python experiments/damage.py       # -> results/damage.npz
.venv/bin/python experiments/crystal.py      # -> results/crystal.json
.venv/bin/python experiments/differential.py # -> results/differential.json
.venv/bin/python experiments/analyze_figs.py # -> fig/*.png, results/analysis.json
.venv/bin/python experiments/crystal_fig.py  # -> fig/crystallization.png
```

## Runtimes (M1 Pro, 16 GB, CPU JAX)

Phase 1 full reproduction chain: **≈ 11 min** wall (checkpoints provided, so
`train.py` is not part of it).

| Step | Command | Wall time |
|------|---------|-----------|
| vocab | `experiments/vocab.py` | ~2 s |
| train | `experiments/train.py` | not run in repro (ckpt/ provided; ~20–30 min to regenerate) |
| sweep | `experiments/sweep.py` | 219 s |
| census | `experiments/census.py` | 92 s |
| damage | `experiments/damage.py` | 166 s |
| differential | `experiments/differential.py` | 57 s |
| crystal | `experiments/crystal.py` | 132 s |
| figures | `experiments/analyze_figs.py` | 3 s |
| **chain total** | `bash experiments/_run_phase1.sh` | **≈ 669 s (11.2 min)** |

Reproduction is faithful: the order parameter matches the pilot to within
`max|Δ| = 0.034` over all (r,T); T=0.3 sweeps are bit-identical (peaked low-T
sampling is robust to BLAS drift), higher T drifts slightly. F1–F9 confirmed
qualitatively (see `results/logs_phase1/` and `experiments/_compare_phase1.py`).

### Known code↔results mismatch (resolved in Phase 2)

The provided `crystal.py` is an earlier snapshot than its committed outputs: it
emits only single-site `damage_T0.3_r1` (all-or-nothing, ≈0), while the committed
`results/crystal.json` and `crystal_fig.py` use a 3-site **block-flip + ignition**
damage probe (`bdmg_*`, F8: 0.833 at step 0 → 0.055 at step 1000). F7's
order/census/val crystallization reproduces exactly; the block-damage probe is
rebuilt as the hardened default in Phase 2 (block flips, B≥64, ignition
probability reported separately).

## Harden (Phase 2)

Fixes the pilot's known weaknesses: BPE vocab (kills the `<unk>` artifact),
≥5-seed statistics with error bars, block-flip damage with ignition probability,
and a finite-size scan. See findings **F10–F13**.

```bash
export JAX_PLATFORMS=cpu
.venv/bin/python experiments/bpe.py                          # -> data_bpe/ (4096 byte-level BPE)
.venv/bin/python experiments/train.py --data-dir data_bpe \
    --ckpt-dir ckpt_bpe --vocab 4096                         # retrain on BPE (~2.3 min)
bash experiments/_run_phase2.sh                              # census_bpe, multiseed sweep,
                                                             # block damage, finite-size, crystal
.venv/bin/python experiments/analyze_figs_phase2.py          # -> fig/*_multiseed, finite_size, ...
.venv/bin/python experiments/crystal_fig.py                  # -> fig/crystallization.png (bdmg fixed)
```

| Step | Command | Wall time |
|------|---------|-----------|
| bpe | `experiments/bpe.py` | ~3 s |
| train (BPE) | `train.py --data-dir data_bpe --ckpt-dir ckpt_bpe --vocab 4096` | 135 s |
| census (BPE) | `experiments/census_bpe.py` | 70 s |
| sweep (5-seed) | `experiments/sweep_multiseed.py` | 1871 s |
| damage (block, B=64) | `experiments/damage.py` | 362 s |
| finite-size | `experiments/finite_size.py` | 1920 s |
| crystal (block+ignition) | `experiments/crystal.py` | 385 s |
| **Phase 2 total** | | **≈ 79 min** (under the 2 h budget) |

Key Phase-2 outcomes: `<unk>` artifact removed (word-level top attractors were
11–13/15 `<unk>`-trigrams; BPE 0/15, real Shakespeare); phase curves survive
5-seed error bars (max std 0.026); **the temperature "transition" is a finite-size
crossover** (χ_peak ∝ 1/N, curves overlay — F12); block-flip ignition probability
separates cleanly from conditional spread (F13).

## Real pretrained MLMs (Phase 3)

Ports the CA rule to HuggingFace masked LMs (bert-tiny → bert-mini →
bert-base-uncased; fp16 on MPS). See findings **F14–F19**. Needs `torch`,
`transformers`, `datasets` (Phase-3 deps above). Models cache to `hf_cache/`
(gitignored). All three share the bert-base-uncased vocab, so one tokenizer serves
all; bert-tiny/mini load via the explicit `BertForMaskedLM` class (their configs
lack `model_type`).

```bash
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
.venv/bin/python experiments/wikitext_ref.py         # -> data_mlm/ (WikiText-103 proxy corpus)
.venv/bin/python experiments/mlm_smoke.py            # sanity: CA on bert-tiny + null test
caffeinate -i bash experiments/_run_phase3.sh        # tiny+mini full, base reduced, model arms
.venv/bin/python experiments/analyze_figs_phase3.py  # -> fig/mlm_*.png, analysis_phase3.json
```

`_run_phase3.sh` is idempotent (skips any probe whose json exists) — safe to
resume. Wrap it in `caffeinate -i` so idle sleep doesn't kill the long base run.

| Model | sweep | damage | census | diff | notes |
|-------|-------|--------|--------|------|-------|
| bert-tiny (2L/128) | ~400 s | 551 s | 49 s | 133 s | B=16–32, sweeps 40 |
| bert-mini (4L/256) | 795 s | 674 s | 83 s | 247 s | B=16–32, sweeps 40 |
| bert-base (12L/768)| 992 s | 1405 s | 100 s | 339 s | B=12–24, sweeps 25–30 (reduced) |
| model arms | — | — | — | 3×~60 s | tiny/mini, mini/base, tiny/base |

## Repair length + external validity (Phases B, C)

Measures the perturbation-damping (**repair**) length ξ_repair, diversity- and
velocity-controlled, and tests external validity on an autoregressive model.
Framing (per novelty check): the contribution is the **instrument** and its
measurements; edge-of-chaos/criticality is decades old (reservoir computing;
2410.02536) — we quantify it black-box, not discover it. Term is "damping length",
not "self-correction" (taken).

```bash
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
# Phase B: damping length D(r,T) with the diversity floor (D_norm = D/D0)
caffeinate -i bash experiments/_run_BC.sh              # tiny+mini repair (normalized) + AR probe
caffeinate -i bash experiments/_run_phaseB_rigor.sh   # base repair + lyapunov + N-scan
.venv/bin/python experiments/mlm_repair_analyze.py    # -> fig/repair_*.png
# Phase C: AR port (src/ar_ca.py, causal window) + calibration (Markov sources)
.venv/bin/python experiments/ar_ref.py                # data_ar/ (Pythia-tokenized WikiText proxy)
.venv/bin/python experiments/calib_markov.py --out data_markov_a  # + _b/_c, then train + calib_census.py
```

Key methods: `mlm_damage.drift_floor` (diversity floor), `lyapunov.py` (finite-size
λ), `repair_fss.py` (N-scan). Perf: on-device sampling + batched CRN twins (~3×).
Outcomes: F23 (damping length shrinks with r, capacity→sensitivity), F24 (AR
replication), C2 calibration (census recovers a known transition matrix, self-TV
0.22 vs baseline 0.91, discriminates).

## Phase 3 (real MLMs) headlines

**Phase 3 total ≈ 100 min** wall on M1 (fp16/MPS). Headlines: instrument ports and
the **null CRN coupling stays exactly 0** (F14); real MLMs are **not radius-blind**
(F15); **light cones replicate, velocity ∝ r and model-invariant** (F16); **no
strong self-healing phase** — real MLMs are far more damage-fragile than the toy,
boundary below the τ≈1.5–2 full-context crossover (F17); differential certification
holds, but the **special-token scheme is a first-class apparatus factor** (F18);
proxy census recovers WikiText's format skeleton, improving with scale (F19).

## Harden the headline (Phase A)

Closes three publication threats to F15/F16 (findings **F20–F22**): scheme
apparatus, velocity finite-size, repetition confound. Fixed scheme (cls_sep),
≥5 seeds, repetition-robust metrics (`distinct_corpus_kgrams`, coarse MI).

```bash
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
caffeinate -i bash experiments/_run_phaseA.sh          # radius (3 models) + velocity FSS
.venv/bin/python experiments/phaseA_analyze.py         # -> fig/phaseA_*.png, analysis_phaseA.json
```

| Step | Wall time |
|------|-----------|
| radius_tiny / mini / base (2 schemes × 5 seeds × 5 r) | 598 / 1150 / 2585 s |
| velocity FSS tiny (N∈{48,96,192,384}, r∈{4,8,16}) | 1834 s |
| velocity FSS mini (N∈{48,96,192}, r=8) | 404 s |
| **Phase A total** | **≈ 111 min** |

Outcomes: F15's radius profile is a model effect **only at a fixed scheme** (scheme
swap moves it ≥ a model change — F20); the F16 velocity ceiling was finite-size
wraparound, v∝r continues (F21); F15's large-r growth was repetition, the real
signal is an intermediate-radius optimum r≈4 (F22).

## Validated instrument: reproduce known metrics, cross-level, new fronts (Phases D/E)

The reframed contribution. **Validation by reproduction** (the credibility spine) +
the **cross-level boundary** (a structural negative) + a **new white-box front**
(prototype). Findings **F26–F29**; audit in `paper_arxiv/REVIEW.md`.

```bash
# --- Validation ladder: reproduce KNOWN metrics (CPU only; the credibility spine) ---
.venv/bin/python experiments/reproduce_lyapunov.py    # smooth-limit unit test (tangent, renormalized) + CML
.venv/bin/python experiments/reproduce_lyapunov.py --finite-perturbation  # the instrument's regime: no renorm, ~0.15 bias floor
.venv/bin/python experiments/logistic_epsilon_sweep.py # WHY the agreement is a limit: error is O(ε) -> fig/logistic_epsilon.png
.venv/bin/python experiments/eca_calib.py             # original ECA rung (F27; superseded by the two below)
.venv/bin/python experiments/eca_calib_hardened.py   # 19 rules x 12 seeds + bootstrap CIs (F33)
.venv/bin/python experiments/eca_calib_ignition.py   # separates ignition prob from conditional spread (F34)
.venv/bin/python experiments/lyap_fit_sensitivity.py # is the ordering an estimator artifact? (F32)
.venv/bin/python experiments/eca_ordered_vs_rest.py   # class test on ignition probability, the right statistic (F36)
.venv/bin/python experiments/cml_benettin.py          # exact Benettin ground truth for the CML rung (F37)
.venv/bin/python experiments/dk_calib.py              # Domany-Kinzel rung: exact damage identity + published p_c (F38); ~4 min
.venv/bin/python experiments/dk_calib.py --figure-only  # redraw fig/dk_ladder.png from saved results

# --- developmental transition (F39/F42), the paper's headline
caffeinate -i .venv/bin/python experiments/dev_transition_phase3.py  # 6 ckpts x 8 seeds x {N=48,96}; ~4 h
.venv/bin/python experiments/dev_transition_shape.py                 # shape, effect sizes, W9 -- applies the F42 ignition rule
.venv/bin/python experiments/fig_developmental.py                    # -> fig/developmental.png
caffeinate -i .venv/bin/python experiments/dev_transition_n192.py    # third lattice size, pre-registered 1/N vs intensive
caffeinate -i .venv/bin/python experiments/dev_transition_scale.py   # transition TIMING across 4 Pythia sizes (~9 h; resumable)
.venv/bin/python experiments/coupling_gap.py                         # how far our CRN is from the maximal coupling (F41)

# --- does the instrument measure the MODEL or the construction? (F35, the delimiting result)
.venv/bin/python experiments/real_generation_damage.py        # inject a token error into REAL AR generation
.venv/bin/python experiments/real_generation_reconvergence.py # distributional version (TV vs an independent-continuation floor)

# --- Phase 3: headline re-test at >=8 seeds, two lattice sizes, BH-FDR
caffeinate -i .venv/bin/python experiments/dev_transition_phase3.py
.venv/bin/python experiments/fig_validation_ladder.py # -> fig/validation_ladder.png

# --- Cross-level: does black-box criticality proxy white-box? (GPU/MPS) ---
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
caffeinate -im .venv/bin/python experiments/crosslevel.py        # white λ_top (depth-Lyapunov) + black D_norm, 6-model ladder + GPT-2
caffeinate -im .venv/bin/python experiments/crosslevel_lyap.py   # black λ_ca (type-matched Lyapunov)
caffeinate -im .venv/bin/python experiments/crosslevel_within.py # within-model T-sweep (finds the T-confound)
caffeinate -im .venv/bin/python experiments/crosslevel_radius.py # de-confounded r-axis (null; λ_ca is kinematic)
caffeinate -im .venv/bin/python experiments/crosslevel_dev.py    # developmental (Pythia checkpoints; λ_top is architecture-flat)
caffeinate -im .venv/bin/python experiments/masked_ladder.py     # masked BERT depth ladder (r=-0.92 but depth-mediated)
.venv/bin/python experiments/fig_crosslevel.py                   # -> fig/crosslevel.png

# --- New front (prototype): activation-lattice information-propagation cone ---
.venv/bin/python experiments/activation_cone.py       # -> fig/actcone_*.png (white-box; CRN null=0 by construction)

# --- Universality-class program (#80-#86). Run IN THIS ORDER: each gates the next. ---
# All are resumable (saved per cell, keyed by their design tuple) -- safe to interrupt.
.venv/bin/python experiments/dp_pipeline_validation.py           # free, no GPU: can the fit recover DK's known exponents?
caffeinate -dimsu .venv/bin/python -u experiments/dp_class_n192.py    # ~17 h  -> F58: delta/theta cross at a common T_c
caffeinate -dimsu .venv/bin/python -u experiments/dp_fss_z.py [hours] # ~44 h  -> F59: z from finite-size scaling
```

> **The DP scripts gate themselves and will refuse to answer.** Each one re-measures its own
> estimator on Domany–Kinzel — at its *own* lattice geometry, replica count and fit window — and
> reports `NOT DECIDABLE` rather than a number when that calibration fails. The gate is evaluated
> on DK alone, blind to the LM values, so it cannot be tuned to the answer. Three confident
> verdicts died to it (F56, F57, and F59's first pass); none of the three was visible from the
> language-model numbers themselves. `dp_fss_z.py` takes an optional hour budget as `argv[1]` and
> stops cleanly after the cell in flight, for running in overnight batches.

Outcomes: the **validation ladder** (F27 ECA classes + census transition-matrix
recovery) establishes the instrument reproduces known metrics before it measures LMs.
The **cross-level** study returns a **structural negative** (F26/F28/F29): the black-box
token-space criticality does not proxy the white-box activation-space Lyapunov, because
the latter is architectural. The **activation-cone** is a prototype for a white-box front
(novelty check: object partially anticipated — `results/deep_research_novelty_actcone.md`).
Compute: the ladder is CPU-seconds; each cross-level run is ~15–45 min on M1/MPS (cap batch
for 1B / gpt2-xl; all runs are resumable + `caffeinate`-wrapped).

## Citation

**Paper 1 (the instrument)** — [arXiv:2608.10986](https://arxiv.org/abs/2608.10986), cs.CL,
11 Aug 2026. DOI [10.48550/arXiv.2608.10986](https://doi.org/10.48550/arXiv.2608.10986).
The repository itself is archived at
[10.5281/zenodo.21880472](https://doi.org/10.5281/zenodo.21880472) — a *concept* DOI, which resolves
to the latest archived version rather than to any one of them.

**Paper 2 (the domain)** — [arXiv:2608.21315](https://arxiv.org/abs/2608.21315), cs.CL,
21 Aug 2026. DOI [10.48550/arXiv.2608.21315](https://doi.org/10.48550/arXiv.2608.21315).
Companion to paper 1; the source is in [`paper2_arxiv/`](paper2_arxiv/) and builds from this
repository.

`CITATION.cff` carries the machine-readable metadata: both papers under `identifiers`, each with its
bare arXiv ID *and* its DOI, while `preferred-citation` stays pointed at paper 1 — that field names
the citation for the **software**, not for the newest result.

When citing a finding, carry its amendment with it: `findings.md` keeps retracted and corrected
findings in place rather than deleting them, and several entries are corrections of other entries.

## Licence

Two licences, split by kind rather than by directory:

- **Code — MIT** (`LICENSE`). `src/`, `experiments/`, `tests/`, and `gatecheck/`, which
  carries its own copy since it is published as a package.
- **Prose and the research record — CC BY 4.0** (`LICENSE-docs`). `findings.md`,
  `what_it_measures.md`, `critical_analysis.md`, the other Markdown at root, and
  `paper_arxiv/`. Measurement output under `results/` is released as accompanying data
  on the same terms.

`findings.md` is a dated ledger that keeps retracted and amended findings in place
rather than deleting them, with the correction stated where the claim was made. Quoting
a finding without its amendment misrepresents the record; where an entry has been
superseded it names what supersedes it.
