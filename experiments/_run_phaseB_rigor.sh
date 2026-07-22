#!/usr/bin/env bash
# Phase B rigor: base repair (3-point capacity trend), Lyapunov (edge-of-chaos number),
# N-scan (finite-size check of the r-trend). Idempotent. caffeinate -i bash ...
set -uo pipefail
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 JAX_PLATFORMS=cpu
PY=.venv/bin/python; LOG=results/logs_phaseB; mkdir -p "$LOG"; T="$LOG/timings.txt"; touch "$T"
run(){ local n="$1" o="$2"; shift 2; [ -f "$o" ] && { echo "SKIP $n"|tee -a "$T"; return; }
  echo ">>> $n"|tee -a "$T"; local s=$SECONDS
  if "$@" >"$LOG/$n.log" 2>&1; then printf '%-18s %6d s OK\n' "$n" "$((SECONDS-s))"|tee -a "$T"
  else printf '%-18s %6d s FAIL\n' "$n" "$((SECONDS-s))"|tee -a "$T"; fi; }
run repair_base results/mlm/repair_base.json     $PY experiments/mlm_repair.py --model base --sweeps 22 --B 24
run lyap_tiny   results/mlm/lyapunov_mlm_tiny.json $PY experiments/lyapunov.py --backend mlm --model tiny
run lyap_mini   results/mlm/lyapunov_mlm_mini.json $PY experiments/lyapunov.py --backend mlm --model mini
run lyap_base   results/mlm/lyapunov_mlm_base.json $PY experiments/lyapunov.py --backend mlm --model base
run fss_tiny    results/mlm/repair_fss_tiny.json  $PY experiments/repair_fss.py --model tiny
echo PHASEB_RIGOR_DONE|tee -a "$T"; tail -10 "$T"
