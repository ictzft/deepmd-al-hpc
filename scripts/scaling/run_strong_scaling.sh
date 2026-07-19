#!/bin/bash
# Strong scaling experiment for committee training (CCGrid 2027 Fig.3).
#
# Trains N committee models with a varying number of GPUs G in {1,2,4,8}.
# For each G, runs concurrent_runner (which schedules models in ceil(N/G) waves,
# one model per GPU) and records:
#   - per-model / per-wave / total wall-time  -> <exp>/summary.json
#   - GPU utilization (host nvidia-smi dmon)   -> <exp>/gpu_dmon.log
# Then collects all G points into one CSV with speedup & parallel efficiency.
#
# Usage:
#   bash scripts/scaling/run_strong_scaling.sh [N] [dataset] [numb_steps] [tag]
#   N          committee size          (default 8)
#   dataset    toy_h2 | ethanol | benzene (default ethanol)
#   numb_steps training steps/model    (default 500)
#   tag        experiment tag          (default strong_n<N>_<dataset>)
#
# Example:
#   bash scripts/scaling/run_strong_scaling.sh 8 ethanol 500
set -euo pipefail

N=${1:-8}
DATASET=${2:-ethanol}
NUMB_STEPS=${3:-500}
TAG=${4:-strong_n${N}_${DATASET}}

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_ROOT}"

case "${DATASET}" in
  toy_h2)
    TRAIN=/data/zft/deepmd-al-hpc/data/toy_h2/train
    VALID=/data/zft/deepmd-al-hpc/data/toy_h2/valid
    CFG=configs/deepmd/pt_smoke_toy_h2.json ;;
  ethanol)
    TRAIN=/data/zft/deepmd-al-hpc/data/rmd17/ethanol/train
    VALID=/data/zft/deepmd-al-hpc/data/rmd17/ethanol/valid
    CFG=configs/deepmd/pt_rmd17_ethanol.json ;;
  benzene)
    TRAIN=/data/zft/deepmd-al-hpc/data/rmd17/benzene/train
    VALID=/data/zft/deepmd-al-hpc/data/rmd17/benzene/valid
    CFG=configs/deepmd/pt_rmd17_benzene.json ;;
  *)
    echo "Unknown dataset: ${DATASET} (toy_h2|ethanol|benzene)" >&2; exit 1 ;;
esac

GPU_AVAIL="0 1 2 3 4 5 6 7"
echo "=== Strong scaling: N=${N} models, dataset=${DATASET}, numb_steps=${NUMB_STEPS}, tag=${TAG} ==="

for G in 1 2 4 8; do
  if [ "${G}" -gt "${N}" ]; then
    echo "--- G=${G} > N=${N}, skipping ---"; continue
  fi
  DEVS="$(echo ${GPU_AVAIL} | tr ' ' '\n' | head -${G} | paste -sd,)"
  EXP="experiments/scaling/${TAG}_g${G}"
  mkdir -p "${EXP}"
  echo "--- G=${G} (gpus ${DEVS}) -> ${EXP} ---"

  # background GPU utilization monitor on the HOST (all GPUs), 1s interval
  nvidia-smi dmon -s u -d 1 > "${EXP}/gpu_dmon.log" 2>&1 &
  DMON_PID=$!

  if bash scripts/docker/run_in_5090.sh "${DEVS}" -- \
       python src/scheduling/concurrent_runner.py \
         --n-models "${N}" --n-gpus "${G}" \
         --config-template "${CFG}" \
         --train-system "${TRAIN}" --valid-system "${VALID}" \
         --exp-dir "${EXP}" --numb-steps "${NUMB_STEPS}"; then
    echo "--- G=${G} OK ---"
  else
    echo "--- G=${G} FAILED (see ${EXP}/model_*/train.log) ---"
  fi

  kill "${DMON_PID}" 2>/dev/null || true
  sleep 1
done

echo "=== Scaling runs complete. Collecting... ==="
python3 scripts/scaling/collect_scaling.py "${TAG}"
