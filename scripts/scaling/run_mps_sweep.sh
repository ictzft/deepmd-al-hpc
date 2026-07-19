#!/bin/bash
# MPS sweep over (N, batch) — produces data for #14 red利曲线 (SM vs N/batch,
# saturation point) AND #13 energy (power + energy), in one pass.
# N in {2,4,8} x batch in {8,64,256}, MPS sharing 1 GPU, 100 steps each.
set -euo pipefail
cd "$(dirname "$0")/../.."
for N in 2 4 8; do
  for BS in 8 64 256; do
    EXP="experiments/scaling/sweep_n${N}_bs${BS}"
    mkdir -p "$EXP"
    # host monitoring: util + power (CSV, per-GPU)
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw \
      --format=csv,noheader -l 1 > "$EXP/gpu.csv" 2>&1 &
    MON=$!
    bash scripts/docker/run_in_5090.sh 0 -- \
      bash scripts/scaling/run_mps_inner.sh "$N" 1 100 "/data/zft/deepmd-al-hpc/$EXP" "$BS" \
      > /dev/null 2>&1 || true
    kill "$MON" 2>/dev/null || true
    sleep 1
    echo "done N=$N bs=$BS"
  done
done
echo "===== aggregating sweep ====="
python3 scripts/scaling/aggregate_sweep.py
