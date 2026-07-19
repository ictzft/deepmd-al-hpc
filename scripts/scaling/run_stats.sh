#!/bin/bash
# Run the 3 core DP-GEN-comparison configs REPEATS times each (ethanol),
# for statistical significance (mean ± std). Outputs to stat_*_rK dirs,
# then aggregates.
#
# Usage: bash scripts/scaling/run_stats.sh [repeats]
set -euo pipefail
REPEATS=${1:-3}
cd "$(dirname "$0")/../.."

for run in $(seq 1 "$REPEATS"); do
  echo "================= repeat $run / $REPEATS ================="
  # DP-GEN-style: wave N=4 (run_strong_scaling also does G=1,2; we read G=4)
  bash scripts/scaling/run_strong_scaling.sh 4 ethanol 100 "stat_wave_n4_r${run}" \
    > /dev/null 2>&1 && echo "  wave r${run} done"
  # Ours: MPS batch=8
  bash scripts/scaling/run_mps.sh 4 1 100 0   "stat_mps_b8_r${run}" \
    > /dev/null 2>&1 && echo "  mps b8 r${run} done"
  # Ours: MPS batch=256
  bash scripts/scaling/run_mps.sh 4 1 100 256 "stat_mps_b256_r${run}" \
    > /dev/null 2>&1 && echo "  mps b256 r${run} done"
done

echo "================= aggregating ================="
python3 scripts/scaling/aggregate_stats.py "$REPEATS"
