#!/bin/bash
# Run dp test on the independent test set for all 16 rMD17 ethanol committee models.
# Usage: bash scripts/eval/test_rmd17_ethanol_models.sh
set -euo pipefail

ROOT="/data/guyida/deepmd-al-hpc"
TEST_SYSTEM="$ROOT/data/rmd17/ethanol/test"
OUTDIR="$ROOT/experiments/rmd17_ethanol_summary/independent_test_logs"
mkdir -p "$OUTDIR"

echo "=== rMD17 Ethanol Independent Test Evaluation ==="
echo "Test set: $TEST_SYSTEM"
echo "Output dir: $OUTDIR"
echo ""

for round in 0 1 2 3; do
  for model in 0 1 2 3; do
    MODEL_DIR="$ROOT/experiments/rmd17_ethanol_round00${round}_committee_models/model_00${model}"
    FROZEN="$MODEL_DIR/frozen_model.pb"
    LOGFILE="$OUTDIR/round${round}_model_00${model}_test.log"

    if [ ! -f "$FROZEN" ]; then
      echo "SKIP  round=$round model=$model (no frozen_model.pb)"
      continue
    fi

    echo -n "Testing round=$round model=$model ... "

    dp test -m "$FROZEN" -s "$TEST_SYSTEM" -n 0 > "$LOGFILE" 2>&1
    echo "done ($(wc -l < "$LOGFILE") lines)"
  done
done

echo ""
echo "=== All tests complete ==="
echo "Logs saved to: $OUTDIR"
