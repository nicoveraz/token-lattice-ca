#!/usr/bin/env bash
# Phase 3 real-MLM probe suite. Run from repo root: bash experiments/_run_phase3.sh
# Idempotent: skips any probe whose output json already exists (safe to re-run).
set -uo pipefail
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 JAX_PLATFORMS=cpu
PY=.venv/bin/python
LOG=results/logs_phase3
mkdir -p "$LOG" results/mlm
T="$LOG/timings.txt"; touch "$T"

run() {  # name out_json script...
  local name="$1" out="$2"; shift 2
  if [ -f "$out" ]; then echo "SKIP $name (exists)" | tee -a "$T"; return; fi
  echo ">>> $name : $*" | tee -a "$T"
  local start=$SECONDS
  if "$@" > "$LOG/$name.log" 2>&1; then
    printf '%-22s %6d s   OK\n' "$name" "$((SECONDS-start))" | tee -a "$T"
  else
    printf '%-22s %6d s   FAIL(rc=%d)\n' "$name" "$((SECONDS-start))" "$?" | tee -a "$T"
  fi
}

# tiny + mini: full settings
for tag in tiny mini; do
  run ${tag}_sweep  results/mlm/${tag}_sweep.json  $PY experiments/mlm_sweep.py       --model $tag --sweeps 40 --B 16
  run ${tag}_damage results/mlm/${tag}_damage.json $PY experiments/mlm_damage.py      --model $tag --sweeps 40 --B 32
  run ${tag}_census results/mlm/${tag}_census.json $PY experiments/mlm_census.py      --model $tag --sweeps 60 --B 24
  run ${tag}_diff   results/mlm/${tag}_diff.json   $PY experiments/mlm_differential.py --model $tag --sweeps 30 --B 16
done

# base: reduced settings (12-layer, ~25ms/forward) to stay tractable
run base_sweep   results/mlm/base_sweep.json   $PY experiments/mlm_sweep.py       --model base --sweeps 25 --B 12
run base_damage  results/mlm/base_damage.json  $PY experiments/mlm_damage.py      --model base --sweeps 30 --B 24
run base_census  results/mlm/base_census.json  $PY experiments/mlm_census.py      --model base --sweeps 45 --B 16
run base_diff    results/mlm/base_diff.json     $PY experiments/mlm_differential.py --model base --sweeps 24 --B 12

# model-diff arm (F9 certification across scale)
run arm_tiny_mini results/mlm/model_arm_tiny_mini.json $PY experiments/mlm_differential.py --pair tiny,mini --sweeps 30 --B 16
run arm_mini_base results/mlm/model_arm_mini_base.json $PY experiments/mlm_differential.py --pair mini,base --sweeps 24 --B 12
run arm_tiny_base results/mlm/model_arm_tiny_base.json $PY experiments/mlm_differential.py --pair tiny,base --sweeps 24 --B 12

echo "PHASE3_CHAIN_DONE" | tee -a "$T"
tail -30 "$T"
