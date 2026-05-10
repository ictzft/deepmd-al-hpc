#!/bin/bash

set -e

EXP_DIR=${1:-experiments/exp_003_single_model_baseline}
INPUT_JSON=${2:-configs/deepmd/input.json}

mkdir -p "${EXP_DIR}"

echo "========== Single DeePMD Model Training =========="
echo "Experiment directory: ${EXP_DIR}"
echo "Input config: ${INPUT_JSON}"
echo "Start time: $(date)"

export PATH=/opt/deepmd-kit/bin:$PATH
export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH

echo "========== Environment =========="
which python
python --version
which dp
dp --version || true

echo "========== Start dp train =========="
dp train "${INPUT_JSON}" 2>&1 | tee "${EXP_DIR}/train.log"

echo "========== Training Finished =========="
echo "End time: $(date)"
