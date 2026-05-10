#!/bin/bash

set -e

EXP_DIR=${1:-experiments/exp_003_single_model_baseline}
MODEL=${2:-experiments/exp_003_single_model_baseline/frozen_model.pb}
TEST_DATA=${3:-/data/zft/data/toy_h2/valid}

PROJECT_ROOT=$(pwd)
EXP_DIR_ABS="${PROJECT_ROOT}/${EXP_DIR}"

mkdir -p "${EXP_DIR_ABS}"

export PATH=/opt/deepmd-kit/bin:$PATH
export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH

echo "========== Test DeePMD Model =========="
echo "Experiment directory: ${EXP_DIR_ABS}"
echo "Model: ${MODEL}"
echo "Test data: ${TEST_DATA}"
echo "Start time: $(date)"

which dp
dp --tf test -m "${MODEL}" -s "${TEST_DATA}" -n 50 2>&1 | tee "${EXP_DIR_ABS}/test.log"

echo "========== Test Finished =========="
echo "End time: $(date)"
