#!/usr/bin/env bash
# Phase 1 reproduction driver. Run from repo root:  bash experiments/_run_phase1.sh
# Times each pipeline step and logs to results/logs_phase1/.
set -uo pipefail
export JAX_PLATFORMS=cpu
PY=.venv/bin/python
LOG=results/logs_phase1
mkdir -p "$LOG"
T="$LOG/timings.txt"; : > "$T"

run() {  # name script...
  local name="$1"; shift
  echo ">>> $name : $*" | tee -a "$T"
  local start=$SECONDS
  if "$@" > "$LOG/$name.log" 2>&1; then
    printf '%-14s %6d s   OK\n' "$name" "$((SECONDS-start))" | tee -a "$T"
  else
    printf '%-14s %6d s   FAIL(rc=%d)\n' "$name" "$((SECONDS-start))" "$?" | tee -a "$T"
  fi
}

# fresh summary so sweep does not append onto the committed pilot rows
rm -f results/summary.jsonl

run sweep        $PY experiments/sweep.py
run census       $PY experiments/census.py
run damage       $PY experiments/damage.py
run differential $PY experiments/differential.py
run crystal      $PY experiments/crystal.py
run analyze_figs $PY experiments/analyze_figs.py

echo "PHASE1_CHAIN_DONE" | tee -a "$T"
echo "=== timings ===" ; cat "$T"
