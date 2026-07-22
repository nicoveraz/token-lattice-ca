#!/usr/bin/env bash
# Phase 2 hardening compute chain. Run from repo root: bash experiments/_run_phase2.sh
set -uo pipefail
export JAX_PLATFORMS=cpu
PY=.venv/bin/python
LOG=results/logs_phase2
mkdir -p "$LOG"
T="$LOG/timings.txt"   # keep the train_bpe line already in it; append run times

run() {  # name script...
  local name="$1"; shift
  echo ">>> $name : $*" | tee -a "$T"
  local start=$SECONDS
  if "$@" > "$LOG/$name.log" 2>&1; then
    printf '%-16s %6d s   OK\n' "$name" "$((SECONDS-start))" | tee -a "$T"
  else
    printf '%-16s %6d s   FAIL(rc=%d)\n' "$name" "$((SECONDS-start))" "$?" | tee -a "$T"
  fi
}

run census_bpe      $PY experiments/census_bpe.py
run sweep_multiseed $PY experiments/sweep_multiseed.py
run damage_block    $PY experiments/damage.py
run finite_size     $PY experiments/finite_size.py
run crystal_hard    $PY experiments/crystal.py

echo "PHASE2_CHAIN_DONE" | tee -a "$T"
tail -20 "$T"
