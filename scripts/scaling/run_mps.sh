#!/bin/bash
# MPS multi-model GPU sharing experiment (CCGrid 2027 Fig.6).
#
# N committee models run concurrently sharing G GPUs via NVIDIA MPS
# (model i -> GPU i%G). With BATCH>0, also enlarges per-model batch (combines
# the two launch-bound mitigations: MPS multi-model sharing + larger kernels).
#
# Usage: bash scripts/scaling/run_mps.sh [N] [G] [steps] [batch] [tag]
#   N      number of models        (default 4)
#   G      number of GPUs to share (default 1)
#   steps  training steps/model    (default 100)
#   batch  batch_size override     (default 0 = keep template's 8)
#   tag    experiment tag          (default mps_n<N>_g<G>_bs<batch>)
set -euo pipefail
N=${1:-4}
G=${2:-1}
STEPS=${3:-100}
BATCH=${4:-0}
TAG=${5:-mps_n${N}_g${G}_bs${BATCH}}

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

DEVS=$(seq 0 $((G-1)) | paste -sd,)
EXP="experiments/scaling/${TAG}"
mkdir -p "$EXP"

echo "=== MPS experiment: N=$N on G=$G GPU(s) [$DEVS], $STEPS steps, batch=${BATCH:-default} ==="
nvidia-smi dmon -s u -d 1 > "$EXP/gpu_dmon.log" 2>&1 &
DMON_PID=$!

bash scripts/docker/run_in_5090.sh "$DEVS" -- \
  bash scripts/scaling/run_mps_inner.sh "$N" "$G" "$STEPS" "/data/zft/deepmd-al-hpc/$EXP" "$BATCH"

kill "$DMON_PID" 2>/dev/null || true
sleep 1
echo "=== done: $EXP ==="
if [ -f "$EXP/summary.json" ]; then
  python3 -c "import json;d=json.load(open('$EXP/summary.json'));print(f'n_ok={d[\"n_ok\"]}/{d[\"n_models\"]}  total_wall={d[\"total_wall_s\"]}s  mean_model={d[\"mean_model_wall_s\"]}s')"
fi
echo "=== GPU SM (GPU0) ==="
awk '!/^#/ && NF>=2 && $1==0 && $2 ~ /^[0-9]+$/ {sum+=$2; n++; if($2>max)max=$2} END{if(n>0)printf "  GPU0 SM avg=%.1f%% max=%.0f%% (n=%d)\n", sum/n, max, n}' "$EXP/gpu_dmon.log"
