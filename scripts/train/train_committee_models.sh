#!/bin/bash

set -e

PROJECT_ROOT=$(pwd)
EXP_ROOT="experiments/exp_004_committee_models"
CONFIG_ROOT="configs/deepmd/committee"
TEST_DATA="/data/zft/data/toy_h2/valid"

echo "========== DeePMD Committee Training =========="
echo "Project root: ${PROJECT_ROOT}"
echo "Experiment root: ${EXP_ROOT}"
echo "Config root: ${CONFIG_ROOT}"
echo "Test data: ${TEST_DATA}"
echo "Start time: $(date)"

mkdir -p "${EXP_ROOT}/logs"

run_one_model() {
    local MODEL_ID=$1
    local GPU_ID=$2
    local MODEL_DIR="${EXP_ROOT}/${MODEL_ID}"
    local INPUT_JSON="${CONFIG_ROOT}/toy_h2_${MODEL_ID}.json"
    local FROZEN_MODEL="${MODEL_DIR}/frozen_model.pb"

    echo "========== Start ${MODEL_ID} on GPU ${GPU_ID} =========="
    echo "Model directory: ${MODEL_DIR}"
    echo "Input config: ${INPUT_JSON}"

    mkdir -p "${MODEL_DIR}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} bash scripts/train/train_single_model.sh \
        "${MODEL_DIR}" \
        "${INPUT_JSON}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} bash scripts/eval/freeze_model.sh \
        "${MODEL_DIR}" \
        "frozen_model.pb"

    CUDA_VISIBLE_DEVICES=${GPU_ID} bash scripts/eval/test_single_model.sh \
        "${MODEL_DIR}" \
        "${FROZEN_MODEL}" \
        "${TEST_DATA}"

    echo "========== Finished ${MODEL_ID} on GPU ${GPU_ID} =========="
}

echo "========== Batch 1: model_000 and model_001 =========="

run_one_model model_000 0 > "${EXP_ROOT}/logs/model_000_full.log" 2>&1 &
PID0=$!

run_one_model model_001 1 > "${EXP_ROOT}/logs/model_001_full.log" 2>&1 &
PID1=$!

wait ${PID0}
wait ${PID1}

echo "========== Batch 1 finished =========="

echo "========== Batch 2: model_002 and model_003 =========="

run_one_model model_002 0 > "${EXP_ROOT}/logs/model_002_full.log" 2>&1 &
PID2=$!

run_one_model model_003 1 > "${EXP_ROOT}/logs/model_003_full.log" 2>&1 &
PID3=$!

wait ${PID2}
wait ${PID3}

echo "========== Batch 2 finished =========="

echo "========== Committee Training Finished =========="
echo "End time: $(date)"
