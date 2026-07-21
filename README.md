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
