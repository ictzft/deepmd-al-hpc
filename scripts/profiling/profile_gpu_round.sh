#!/bin/bash
# Profile GPU utilization during a full active learning round.
# Usage: bash scripts/profiling/profile_gpu_round.sh <LABEL> <CONFIG_DIR> <MODEL_DIR> <VALID_DATA>
# Runs nvidia-smi dmon during training, then measures prediction GPU usage.
set -euo pipefail

LABEL="${1:-test}"
CONFIG_DIR="${2:-configs/deepmd/rmd17_ethanol_round000_committee}"
MODEL_DIR="${3:-/tmp/gpu_prof_models}"
VALID_DATA="${4:-/data/guyida/deepmd-al-hpc/data/rmd17/ethanol/valid}"
ROOT="/data/guyida/deepmd-al-hpc"
PROF_DIR="$ROOT/experiments/profiling/gpu_monitor"
mkdir -p "$PROF_DIR"

# Start GPU monitoring in background (1s interval)
echo "Starting GPU monitor for: $LABEL"
nvidia-smi dmon -s pucv -d 1 -o T > "$PROF_DIR/${LABEL}_gpu_dmon.log" 2>&1 &
DMON_PID=$!

# Run training (on 2 GPUs in parallel, 4 models)
echo "Training 4 models..."
bash "$ROOT/scripts/train/train_round_committee_models.sh" 001 "$CONFIG_DIR" "$MODEL_DIR" "$VALID_DATA" 2>&1 | tail -5

# Stop GPU monitor
kill $DMON_PID 2>/dev/null || true
sleep 1

echo "GPU monitoring data saved to: $PROF_DIR/${LABEL}_gpu_dmon.log"
echo "Lines: $(wc -l < "$PROF_DIR/${LABEL}_gpu_dmon.log")"

# Quick summary: average GPU utilization
echo ""
echo "=== GPU Utilization Summary ==="
python3 -c "
import re
with open('$PROF_DIR/${LABEL}_gpu_dmon.log') as f:
    lines = f.readlines()
g0_sm, g1_sm, g0_mem, g1_mem = [], [], [], []
for line in lines:
    if line.startswith('#') or not line.strip():
        continue
    parts = line.strip().split()
    if len(parts) >= 5:
        try:
            g0_sm.append(float(parts[1]))
            g0_mem.append(float(parts[2]))
            g1_sm.append(float(parts[3]))
            g1_mem.append(float(parts[4]))
        except ValueError:
            continue
if g0_sm:
    print(f'GPU0: SM={sum(g0_sm)/len(g0_sm):.1f}%, Mem={sum(g0_mem)/len(g0_mem):.0f}% (max SM={max(g0_sm):.0f}%)')
if g1_sm:
    print(f'GPU1: SM={sum(g1_sm)/len(g1_sm):.1f}%, Mem={sum(g1_mem)/len(g1_mem):.0f}% (max SM={max(g1_sm):.0f}%)')
print(f'Samples: {len(g0_sm)}')
"
