#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 JAX_PLATFORMS=cpu
PY=.venv/bin/python; LOG=results/logs_phaseB; mkdir -p "$LOG"; T="$LOG/timings.txt"; touch "$T"
run(){ local n="$1" o="$2"; shift 2; [ -f "$o" ] && { echo "SKIP $n"|tee -a "$T"; return; }
  echo ">>> $n"|tee -a "$T"; local s=$SECONDS
  if "$@" >"$LOG/$n.log" 2>&1; then printf '%-14s %6d s OK\n' "$n" "$((SECONDS-s))"|tee -a "$T"
  else printf '%-14s %6d s FAIL\n' "$n" "$((SECONDS-s))"|tee -a "$T"; fi; }
run repair_mini results/mlm/repair_mini.json $PY experiments/mlm_repair.py --model mini --sweeps 30 --B 32
run repair_base results/mlm/repair_base.json $PY experiments/mlm_repair.py --model base --sweeps 30 --B 32
echo PHASEB_MODELS_DONE|tee -a "$T"
