#!/bin/bash

set -e

EXP_DIR=${1:-experiments/exp_003_single_model_baseline}
OUTPUT_MODEL=${2:-frozen_model.pb}

PROJECT_ROOT=$(pwd)
EXP_DIR_ABS="${PROJECT_ROOT}/${EXP_DIR}"

export PATH=/opt/deepmd-kit/bin:$PATH
export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH

echo "========== Freeze DeePMD Model =========="
echo "Experiment directory: ${EXP_DIR_ABS}"
echo "Output model: ${OUTPUT_MODEL}"
echo "Start time: $(date)"

cd "${EXP_DIR_ABS}"

which dp
dp --tf freeze -o "${OUTPUT_MODEL}" 2>&1 | tee freeze.log

echo "========== Freeze Finished =========="
echo "End time: $(date)"
