# token-lattice-ca

A cellular automaton over **token space**, developed into a **validated black-box
measurement instrument** for language-model dynamics. A ring of *N* token cells is
updated by a model's windowed conditional `p_r(x_i | x_{i±r})` (radius *r*,
temperature *T*), with common-random-number (CRN) damage spreading as the probe.

The organizing principle is **validation by reproduction**: before measuring a
language model — whose "true" dynamical metrics are unknown — the instrument
reproduces *known* metrics on systems where the answer is established (the
**validation ladder**): a decisive ordered-vs-chaotic separation on elementary CA
rules (the finer 3-class ordering does **not** survive — F33/F34) and the known
transition matrices of synthetic Markov sources — the rungs that share the
instrument's regime (discrete state, finite O(1) perturbation) — plus smooth-limit
arithmetic checks on the logistic map and a coupled-map lattice.

> **Honest scope of the logistic rung (do not overread it).** The logistic-map
> agreement is a **unit test of the growth-rate arithmetic in the smooth limit**, not
> a validation of the instrument. Its estimator renormalizes the twin separation back
> to ε along the *same* orbit as the analytic reference, so
> `log(d/ε) = log|f'(x)| + O(ε)` — it is a finite-difference evaluation of the
> derivative it is compared against, which is why agreement is exact at ε=1e-9. A
> token flip is **O(1)** in a discrete alphabet, so there is no ε→0 limit in token
> space. See `experiments/logistic_epsilon_sweep.py` (error is O(ε), log-log slope
> 0.79) and `--finite-perturbation` (no renormalization: an irreducible ≈0.15
> bias floor that does *not* vanish as ε→0). The weight-bearing rungs are the ECA
> ordered-vs-chaotic separation and the census.

Only then does it report the weights-free LM measurements it yields (token-space
Lyapunov, damping length, effective interaction radius, damage light cone, attractor
census), and the **boundary** where the black-box reading provably decouples from
white-box internals.

The reframed write-up is in **[paper/paper.tex](paper/paper.tex)**; substantive
results in **[findings.md](findings.md)** (F1–F34); the adversarial audit that
reshaped the claims in **[paper/REVIEW.md](paper/REVIEW.md)**.

> **Note on earlier claims.** An adversarial audit (REVIEW.md; findings F26–F29)
> demoted/retracted several earlier headlines. In particular the F23
> "capacity→sensitivity, *p*<10⁻⁴" result was **pseudoreplicated (n=2 seeds)** and
> is retracted as a significance claim, and the λ "kinematics⊥stability"
> decomposition is **withdrawn**. The cross-level proxy hypothesis (black-box
> token-space criticality → white-box activation criticality) is a **clean
> negative** (F26/F28/F29). The table below is annotated accordingly.

### Findings at a glance

| # | Finding | Phase |
|---|---------|-------|
| F1–F9 | Pilot reproduced on M1: temperature phase structure, radius-blind statics but radius-set damage cones, partial corpus recovery, metastable churn, sync period-2 artifact, measure-not-sample, early crystallization, learned self-healing, CRN certification (null = 0) | 1 |
| F10 | BPE kills the `<unk>` artifact — word-level top attractors were 11–13/15 `<unk>`; BPE 0/15, real text | 2 |
| F11 | Phase curves survive ≥5-seed error bars (max std 0.026) | 2 |
| **F12** | **The temperature "transition" is a finite-size crossover, not a true transition** (χ_peak ∝ 1/N) | 2 |
| F13 | Block-flip ignition probability separated from conditional spread | 2 |
| F14 | Instrument ports to bert-tiny/mini/base; **null CRN coupling stays exactly 0** | 3 |
| **F15** | **Real MLMs are NOT radius-blind** (unlike the toy) — long-range structure peaks at intermediate r | 3 |
| F16 | Damage light cones replicate; **front velocity ∝ r and model-invariant** | 3 |
| **F17** | **No strong self-healing phase** on real MLMs — far more damage-fragile than the toy; boundary below the τ≈1.5–2 full-context crossover | 3 |
| F18 | Differential certification holds; the **special-token scheme is a first-class apparatus factor** | 3 |
| F19 | Proxy census recovers WikiText's format skeleton, improving with scale | 3 |
| F20 | F15 certified as a model effect **at a fixed scheme** — but CLS/SEP is a first-class apparatus (scheme swap ≥ model shift) | A |
| F21 | F16 velocity plateau was **finite-size wraparound**; velocity∝r continues (r=16: 11.5→47.5 as N grows) | A |
| F22 | F15's raw large-r growth was **repetition**; the repetition-robust signal is an intermediate-radius optimum (r≈4) | A |
| F23 | A diversity-controlled **damping length** (D_norm=D/D0) shrinks with radius. A capacity gap **tiny ≪ {mini,base}** appears but is **suggestive only** — the "p<10⁻⁴" was pseudoreplicated (n=2 seeds); **retracted** as significance (see F26/F29) | B |
| F24 | velocity∝r + damping-length-radius-trend **replicate on AR** (Pythia causal window) — not MLM artifacts; but the **capacity→sensitivity effect does NOT** (non-monotone over 4 Pythia sizes, ρ=0.17 p=0.29) — masked-specific | C1 |
| F25 | **Developmental**: across Pythia training, D_norm traces chaotic-init → order-minimum → edge-of-chaos climb; structure crystallizes before sensitivity (real-model F7) | D |
| **F26** | **Cross-level**: black-box token-space criticality does NOT robustly proxy white-box activation-space λ_top — cross-model family-dependent (Pythia +0.71 / GPT-2 −0.43) | D |
| **F27** | **Ground-truth calibration**: the ECA rung separates ordered from chaotic. *Partly superseded:* the 3-class ordering (F33) and the Rule 90 nuance (F34) are demoted/retracted | D |
| F28 | Cross-level negative confirmed: within-T confounded (uniform −0.9), within-r null (λ_ca model-invariant/kinematic); GPT-2 non-replication | D |
| **F29** | The negative is **structural**: white-box λ_top is architectural (flat across training, ≈1/L); masked ladder r=−0.92 is depth-mediated → no *useful* weights-free proxy | D/E |
| **F30** | **The logistic rung was circular** — its estimator renormalizes to the reference orbit, so it is a finite-difference derivative (error O(ε), slope 0.79); demoted to a smooth-limit arithmetic unit test. Finite/no-renorm regime has a ≈0.15 bias floor | 0 |
| F31 | Repo-wide circularity hunt: `cml_lyap` renormalizes too (same failure mode); the LM damage path is **clean** (twins never re-anchored). Also: λ_top is *tangent-space* while λ_ca is *finite* — a second cause for the cross-level negative | 0 |
| F32 | `lyap_from_cone`'s branch constants are robust (ordering holds 54/54), but a **fixed** fit window inverts edge-vs-chaotic 3/4 times → use a saturation-relative window | 0 |
| **F33** | Hardened ECA rung (19 rules × 12 seeds, rule-level bootstrap): **ordered < chaotic p<10⁻⁴ CONFIRMED**; **edge < chaotic p=0.17 NOT significant** → the 3-class ordering is demoted | 2 |
| **F34** | The ECA rung had an **ignition confound** (λ averaged over ignited + dead runs). Rule 30's negative reading explained (λ\|ignited=+0.45); **Rule 90 "marginal" nuance RETRACTED** (λ\|ignited=+0.28). The real discriminator is **ignition probability** (ordered 0.05 vs edge 0.67 vs chaotic 0.68) | 2 |

## Layout

```
src/           library
  model.py       tiny bidirectional transformer (the CA rule family p_r)
  ca.py          the automaton: ring, async/sync Glauber, CRN sampling, metrics
experiments/   runnable pipeline steps (run from repo root)
  vocab.py       word-level vocab builder (pilot); see bpe.py for the BPE variant (Phase 2)
  train.py       train the windowed conditional model on tinyshakespeare
  sweep.py       coarse T×r phase sweep (async + one sync row)
  census.py      attractor census + corpus recovery + melting + cycle check
  damage.py      damage spreading: CRN twin runs, damage cones
  crystal.py     run the full suite on every training checkpoint (crystallization)
  differential.py differential CRN certification (null / apparatus / model arms)
  analyze_figs.py figures 1–6 + analysis.json
  crystal_fig.py  crystallization figure
tests/         pytest harness regression tests (null test, determinism, sanity)
data/          shakespeare.txt, token ids, vocab.json
ckpt/          0.42M-param checkpoints (step1000..6000, final) — tracked, no retrain needed
results/       raw npz + summary.jsonl / census.json / analysis.json / differential.json
fig/           figures (png)
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
(prototype). Findings **F26–F29**; audit in `paper/REVIEW.md`.

```bash
# --- Validation ladder: reproduce KNOWN metrics (CPU only; the credibility spine) ---
.venv/bin/python experiments/reproduce_lyapunov.py    # smooth-limit unit test (tangent, renormalized) + CML
.venv/bin/python experiments/reproduce_lyapunov.py --finite-perturbation  # the instrument's regime: no renorm, ~0.15 bias floor
.venv/bin/python experiments/logistic_epsilon_sweep.py # WHY the agreement is a limit: error is O(ε) -> fig/logistic_epsilon.png
.venv/bin/python experiments/eca_calib.py             # original ECA rung (F27; superseded by the two below)
.venv/bin/python experiments/eca_calib_hardened.py   # 19 rules x 12 seeds + bootstrap CIs (F33)
.venv/bin/python experiments/eca_calib_ignition.py   # separates ignition prob from conditional spread (F34)
.venv/bin/python experiments/lyap_fit_sensitivity.py # is the ordering an estimator artifact? (F32)
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
```

Outcomes: the **validation ladder** (F27 ECA classes + census transition-matrix
recovery) establishes the instrument reproduces known metrics before it measures LMs.
The **cross-level** study returns a **structural negative** (F26/F28/F29): the black-box
token-space criticality does not proxy the white-box activation-space Lyapunov, because
the latter is architectural. The **activation-cone** is a prototype for a white-box front
(novelty check: object partially anticipated — `results/deep_research_novelty_actcone.md`).
Compute: the ladder is CPU-seconds; each cross-level run is ~15–45 min on M1/MPS (cap batch
for 1B / gpt2-xl; all runs are resumable + `caffeinate`-wrapped).
