#!/bin/bash

set -e

EXP_DIR=${1:-experiments/exp_003_single_model_baseline}
INPUT_JSON=${2:-configs/deepmd/toy_h2_input.json}

PROJECT_ROOT=$(pwd)
EXP_DIR_ABS="${PROJECT_ROOT}/${EXP_DIR}"
INPUT_JSON_ABS="${PROJECT_ROOT}/${INPUT_JSON}"

mkdir -p "${EXP_DIR_ABS}"

echo "========== Single DeePMD Model Training =========="
echo "Project root: ${PROJECT_ROOT}"
echo "Experiment directory: ${EXP_DIR_ABS}"
echo "Input config: ${INPUT_JSON_ABS}"
echo "Start time: $(date)"

export PATH=/opt/deepmd-kit/bin:$PATH
export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH

echo "========== Environment =========="
which python
python --version
which dp
dp --version || true

cd "${EXP_DIR_ABS}"

echo "========== Start dp train =========="
dp --tf train "${INPUT_JSON_ABS}" 2>&1 | tee train.log

echo "========== Training Finished =========="
echo "End time: $(date)"
