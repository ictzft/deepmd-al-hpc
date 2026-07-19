#!/bin/bash
# Inner script: runs INSIDE the deepmd-5090 container (called by run_mps.sh).
# Starts the NVIDIA MPS daemon, runs concurrent_runner in --mode all (all N
# models concurrent, model i on GPU i%G), then shuts the daemon down.
# Must stay in one container session: MPS daemon does not survive --rm.
#
# Args: N G STEPS EXP [BATCH]
set -e
N=$1; G=$2; STEPS=$3; EXP=$4; BATCH=${5:-0}
cd /data/zft/deepmd-al-hpc

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "=== start MPS daemon ==="
nvidia-cuda-mps-control -d
sleep 1
echo "get_server_list" | nvidia-cuda-mps-control 2>&1 | head -3 || true

BATCH_FLAG=""
if [ "${BATCH:-0}" -gt 0 ] 2>/dev/null; then BATCH_FLAG="--batch-size $BATCH"; fi
echo "=== MPS all-concurrent: N=$N on G=$G GPU(s), $STEPS steps, batch=${BATCH:-default} ==="
python src/scheduling/concurrent_runner.py --mode all \
  --n-models "$N" --n-gpus "$G" $BATCH_FLAG \
  --config-template configs/deepmd/pt_rmd17_ethanol.json \
  --train-system /data/zft/deepmd-al-hpc/data/rmd17/ethanol/train \
  --valid-system /data/zft/deepmd-al-hpc/data/rmd17/ethanol/valid \
  --exp-dir "$EXP" --numb-steps "$STEPS"

echo "=== shutdown MPS daemon ==="
echo "quit" | nvidia-cuda-mps-control || true
