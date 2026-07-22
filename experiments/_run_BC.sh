#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 JAX_PLATFORMS=cpu
PY=.venv/bin/python; LOG=results/logs_BC; mkdir -p "$LOG"; T="$LOG/timings.txt"; touch "$T"
run(){ local n="$1" o="$2"; shift 2; [ -f "$o" ] && { echo "SKIP $n"|tee -a "$T"; return; }
  echo ">>> $n"|tee -a "$T"; local s=$SECONDS
  if "$@" >"$LOG/$n.log" 2>&1; then printf '%-16s %6d s OK\n' "$n" "$((SECONDS-s))"|tee -a "$T"
  else printf '%-16s %6d s FAIL\n' "$n" "$((SECONDS-s))"|tee -a "$T"; fi; }
run repair_mini results/mlm/repair_mini.json $PY experiments/mlm_repair.py --model mini --sweeps 25 --B 32
run ar_160m     results/mlm/ar_pythia-160m.json $PY experiments/ar_probe.py --model pythia-160m --sweeps 28 --B 24
echo BC_DONE|tee -a "$T"; tail -6 "$T"
