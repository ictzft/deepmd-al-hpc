#!/bin/bash
# Wave-mode training (one model per GPU), single config, with host dmon.
# Companion to run_mps.sh (which does mode=all + MPS daemon).
# Args: N G steps tag
set -euo pipefail
N=$1; G=$2; STEPS=$3; TAG=$4
cd "$(dirname "$0")/../.."
DEVS=$(seq 0 $((G-1)) | paste -sd,)
EXP="experiments/scaling/${TAG}"
mkdir -p "$EXP"
echo "=== wave: N=$N G=$G steps=$STEPS -> $EXP ==="
nvidia-smi dmon -s u -d 1 > "$EXP/gpu_dmon.log" 2>&1 &
DMON=$!
bash scripts/docker/run_in_5090.sh "$DEVS" -- python src/scheduling/concurrent_runner.py \
  --mode wave --n-models "$N" --n-gpus "$G" \
  --config-template configs/deepmd/pt_rmd17_ethanol.json \
  --train-system /data/zft/deepmd-al-hpc/data/rmd17/ethanol/train \
  --valid-system /data/zft/deepmd-al-hpc/data/rmd17/ethanol/valid \
  --exp-dir "/data/zft/deepmd-al-hpc/$EXP" --numb-steps "$STEPS"
kill "$DMON" 2>/dev/null || true
sleep 1
[ -f "$EXP/summary.json" ] && python3 -c "import json;d=json.load(open('$EXP/summary.json'));print(f'wave: wall={d[\"total_wall_s\"]}s n_ok={d[\"n_ok\"]}/{d[\"n_models\"]}')"
