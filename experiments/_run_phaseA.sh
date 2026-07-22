#!/usr/bin/env bash
# Phase A: harden the headline. Run from repo root: caffeinate -i bash experiments/_run_phaseA.sh
# Idempotent: skips any probe whose output json exists.
set -uo pipefail
export HF_HOME=./hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 JAX_PLATFORMS=cpu
PY=.venv/bin/python
LOG=results/logs_phaseA
mkdir -p "$LOG" results/mlm
T="$LOG/timings.txt"; touch "$T"

run() {  # name out_json script...
  local name="$1" out="$2"; shift 2
  if [ -f "$out" ]; then echo "SKIP $name (exists)" | tee -a "$T"; return; fi
  echo ">>> $name : $*" | tee -a "$T"
  local start=$SECONDS
  if "$@" > "$LOG/$name.log" 2>&1; then
    printf '%-24s %6d s   OK\n' "$name" "$((SECONDS-start))" | tee -a "$T"
  else
    printf '%-24s %6d s   FAIL(rc=%d)\n' "$name" "$((SECONDS-start))" "$?" | tee -a "$T"
  fi
}

# A1 + A3: radius profiles (both schemes, 5 seeds, repetition-robust metrics)
for tag in tiny mini base; do
  run radius_$tag results/mlm/phaseA_radius_$tag.json $PY experiments/phaseA_radius.py --model $tag
done

# A2: velocity finite-size (tiny full grid; mini r=8 up to N=192 as invariance check)
run vel_tiny results/mlm/phaseA_velocity_tiny.json $PY experiments/phaseA_velocity.py --model tiny --ns 48,96,192,384 --rs 4,8,16
run vel_mini results/mlm/phaseA_velocity_mini.json $PY experiments/phaseA_velocity.py --model mini --ns 48,96,192 --rs 8

echo "PHASEA_CHAIN_DONE" | tee -a "$T"
tail -20 "$T"
