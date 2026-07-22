# token-lattice-ca

A cellular automaton over **token space**, used as a black-box measurement
instrument for language-model structure. A ring of *N* token cells is updated by
a model's windowed conditional `p_r(x_i | x_{i±r})` (center masked, radius *r*,
temperature *T*). The instrument measures temperature×radius phase diagrams,
damage-spreading light cones (twin runs under common random numbers), attractor
censuses validated against a known corpus, probes across training checkpoints,
and a differential instrument/signal-separation protocol.

See **[findings.md](findings.md)** for the substantive results (F1–F9 from the
tiny-transformer pilot; F10+ for real pretrained MLMs).

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

**Phase 3 total ≈ 100 min** wall on M1 (fp16/MPS). Headlines: instrument ports and
the **null CRN coupling stays exactly 0** (F14); real MLMs are **not radius-blind**
(F15); **light cones replicate, velocity ∝ r and model-invariant** (F16); **no
strong self-healing phase** — real MLMs are far more damage-fragile than the toy,
boundary below the τ≈1.5–2 full-context crossover (F17); differential certification
holds, but the **special-token scheme is a first-class apparatus factor** (F18);
proxy census recovers WikiText's format skeleton, improving with scale (F19).
