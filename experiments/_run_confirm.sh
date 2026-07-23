#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 JAX_PLATFORMS=cpu
PY=.venv/bin/python; LOG=results/logs_confirm; mkdir -p "$LOG"; T="$LOG/timings.txt"; touch "$T"
run(){ local n="$1" o="$2"; shift 2; [ -f "$o" ] && { echo "SKIP $n"|tee -a "$T"; return; }
  echo ">>> $n"|tee -a "$T"; local s=$SECONDS
  if "$@" >"$LOG/$n.log" 2>&1; then printf '%-16s %6d s OK\n' "$n" "$((SECONDS-s))"|tee -a "$T"
  else printf '%-16s %6d s FAIL\n' "$n" "$((SECONDS-s))"|tee -a "$T"; fi; }
run capacity_nscan results/mlm/capacity_nscan.json   $PY experiments/capacity_nscan.py
run ar_70m         results/mlm/ar_pythia-70m.json     $PY experiments/ar_probe.py --model pythia-70m --sweeps 26 --B 24
echo CONFIRM_DONE|tee -a "$T"; tail -6 "$T"
