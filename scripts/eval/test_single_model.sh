#!/bin/bash

set -e

EXP_DIR=${1:-experiments/exp_003_single_model_baseline}
MODEL=${2:-experiments/exp_003_single_model_baseline/frozen_model.pb}
TEST_DATA=${3:-}

if [ -z "${TEST_DATA}" ]; then
  echo "Usage:"
  echo "  bash scripts/eval/test_single_model.sh EXP_DIR MODEL TEST_DATA"
  echo ""
  echo "Example:"
  echo "  bash scripts/eval/test_single_model.sh experiments/exp_003_single_model_baseline experiments/exp_003_single_model_baseline/frozen_model.pb /path/to/test_data"
  exit 1
fi

mkdir -p "${EXP_DIR}"

export PATH=/opt/deepmd-kit/bin:$PATH
export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH

echo "========== Test DeePMD Model =========="
echo "Experiment directory: ${EXP_DIR}"
echo "Model: ${MODEL}"
echo "Test data: ${TEST_DATA}"
echo "Start time: $(date)"

which dp
dp test -m "${MODEL}" -s "${TEST_DATA}" -n 100 2>&1 | tee "${EXP_DIR}/test.log"

echo "========== Test Finished =========="
echo "End time: $(date)"
