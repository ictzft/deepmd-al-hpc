#!/bin/bash
set -euo pipefail

ROUND_ID=${1:-001}
CONFIG_DIR=${2:-configs/deepmd/round_001_committee}
EXP_DIR=${3:-experiments/exp_007_round001_committee_models}
VALID_DATA=${4:-/data/zft/data/toy_h2/valid}

PROJECT_ROOT=$(pwd)
EXP_DIR_ABS="${PROJECT_ROOT}/${EXP_DIR}"
CONFIG_DIR_ABS="${PROJECT_ROOT}/${CONFIG_DIR}"

export PATH=/opt/deepmd-kit/bin:$PATH
export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:${LD_LIBRARY_PATH:-}

mkdir -p "${EXP_DIR_ABS}"

echo "========== Round ${ROUND_ID} committee training =========="
echo "project root : ${PROJECT_ROOT}"
echo "config dir   : ${CONFIG_DIR_ABS}"
echo "exp dir      : ${EXP_DIR_ABS}"
echo "valid data   : ${VALID_DATA}"
echo "start time   : $(date)"

run_model () {
  local MODEL_ID=$1
  local GPU_ID=$2
  local CONFIG_FILE=$3

  local MODEL_DIR="${EXP_DIR_ABS}/${MODEL_ID}"
  local CONFIG_ABS="${CONFIG_DIR_ABS}/${CONFIG_FILE}"

  mkdir -p "${MODEL_DIR}"

  echo "========== ${MODEL_ID} on GPU ${GPU_ID} =========="
  echo "config    : ${CONFIG_ABS}"
  echo "model dir : ${MODEL_DIR}"

  cd "${MODEL_DIR}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} dp -b tensorflow train "${CONFIG_ABS}" \
    2>&1 | tee train.log

  CUDA_VISIBLE_DEVICES=${GPU_ID} dp -b tensorflow freeze -o frozen_model.pb \
    2>&1 | tee freeze.log

  CUDA_VISIBLE_DEVICES=${GPU_ID} dp -b tensorflow test \
    -m frozen_model.pb \
    -s "${VALID_DATA}" \
    -n 50 \
    2>&1 | tee test.log
}

run_model model_000 0 "toy_h2_round${ROUND_ID}_model_000.json" &
PID0=$!

run_model model_001 1 "toy_h2_round${ROUND_ID}_model_001.json" &
PID1=$!

wait ${PID0}
wait ${PID1}

run_model model_002 0 "toy_h2_round${ROUND_ID}_model_002.json" &
PID2=$!

run_model model_003 1 "toy_h2_round${ROUND_ID}_model_003.json" &
PID3=$!

wait ${PID2}
wait ${PID3}

echo "========== Round ${ROUND_ID} committee training finished =========="
echo "end time: $(date)"
