#!/bin/bash
# Run dp test on independent test set for all rMD17 models (uncertainty + random baseline).
set -euo pipefail

ROOT="/data/guyida/deepmd-al-hpc"
TEST_SYS="$ROOT/data/rmd17/ethanol/test"
OUTDIR="$ROOT/experiments/rmd17_ethanol_summary/independent_test_logs"
mkdir -p "$OUTDIR"

echo "=== Testing Uncertainty Models ==="
for round in 0 1 2 3; do
  for model in 0 1 2 3; do
    FROZEN="$ROOT/experiments/rmd17_ethanol_round00${round}_committee_models/model_00${model}/frozen_model.pb"
    LOG="$OUTDIR/uncertainty_round${round}_model_00${model}_test.log"
    if [ -f "$FROZEN" ] && [ ! -f "$LOG" ]; then
      dp test -m "$FROZEN" -s "$TEST_SYS" -n 0 > "$LOG" 2>&1
      echo "  uncertainty round=$round model=$model done"
    fi
  done
done

echo ""
echo "=== Testing Random Baseline Models ==="
for seed in 0 1 2; do
  for round in 1 2 3; do
    for model in 0 1 2 3; do
      FROZEN="$ROOT/experiments/baselines/random_seed${seed}_round00${round}_committee_models/model_00${model}/frozen_model.pb"
      LOG="$OUTDIR/random_seed${seed}_round00${round}_model_00${model}_test.log"
      if [ -f "$FROZEN" ] && [ ! -f "$LOG" ]; then
        dp test -m "$FROZEN" -s "$TEST_SYS" -n 0 > "$LOG" 2>&1
        echo "  random seed=$seed round=$round model=$model done"
      fi
    done
  done
done

echo ""
echo "=== All tests complete ==="
