#!/bin/bash
# ============================================================
# V100 Profiling Recording Script
# ============================================================
# Records wall-clock time and GPU metrics for one active
# learning round (training + prediction + dataset update).
#
# Usage:
#   bash scripts/profiling/record_round_profiling.sh \
#     <ROUND_ID> <SEED_LABEL> <BRANCH>
#
# Example:
#   bash scripts/profiling/record_round_profiling.sh 002 seed0 random
#
#   This will:
#   1. Start GPU monitoring in background
#   2. Record training time (requires model training to be run separately)
#   3. Record prediction time
#   4. Record dataset update time
#   5. Stop GPU monitoring
#   6. Write a CSV row to experiments/profiling/profiling_v100_rounds.csv
#
# Note: This script records wall-clock time for the prediction
# and dataset-update phases. For training time, the user should
# record separately via train_round_committee_models.sh (which
# already prints start/end timestamps).
# ============================================================

set -euo pipefail

ROUND_ID="${1:?Usage: $0 <ROUND_ID> <SEED_LABEL> <BRANCH>}"
SEED_LABEL="${2:?}"
BRANCH="${3:?}"

ROUND_STR="$(printf '%03d' "${ROUND_ID}")"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/experiments/profiling"
GPU_LOG="${LOG_DIR}/gpu_dmon_round${ROUND_STR}_${SEED_LABEL}.log"
TIMELINE_LOG="${LOG_DIR}/timeline_round${ROUND_STR}_${SEED_LABEL}.log"

mkdir -p "${LOG_DIR}"

echo "============================================================" | tee "${TIMELINE_LOG}"
echo "V100 Profiling: Round ${ROUND_STR}  Seed: ${SEED_LABEL}  Branch: ${BRANCH}" | tee -a "${TIMELINE_LOG}"
echo "Started at: $(date -Iseconds)" | tee -a "${TIMELINE_LOG}"
echo "============================================================" | tee -a "${TIMELINE_LOG}"

# ---- Start GPU monitoring ----
echo "[$(date -Iseconds)] Starting GPU monitor (nvidia-smi dmon, 2s interval)" | tee -a "${TIMELINE_LOG}"
nvidia-smi dmon -s pucvmet -d 2 > "${GPU_LOG}" 2>&1 &
DMON_PID=$!
echo "GPU monitor PID: ${DMON_PID}" | tee -a "${TIMELINE_LOG}"

# ---- Training ----
# Training is run separately via train_round_committee_models.sh.
# That script already prints start/end timestamps.
# Extract elapsed time from train_elapsed.log if present.
TRAIN_MODELS_DIR="${PROJECT_ROOT}/experiments/baselines/random_${SEED_LABEL}_round${ROUND_STR}_committee_models"
TRAIN_ELAPSED=""
if [ -f "${TRAIN_MODELS_DIR}/train_elapsed.log" ]; then
    TRAIN_ELAPSED=$(grep "total_elapsed_s" "${TRAIN_MODELS_DIR}/train_elapsed.log" 2>/dev/null | tail -1 | grep -oP '\d+') || true
    echo "[$(date -Iseconds)] Training elapsed (from log): ${TRAIN_ELAPSED:-N/A} s" | tee -a "${TIMELINE_LOG}"
else
    echo "[$(date -Iseconds)] train_elapsed.log not found — record training time manually" | tee -a "${TIMELINE_LOG}"
fi

# ---- Prediction ----
CANDIDATE_DATA="data/toy_h2/random_${SEED_LABEL}_round_${ROUND_STR}_candidate"
PRED_DIR="${PROJECT_ROOT}/experiments/baselines/random_${SEED_LABEL}_round${ROUND_STR}_committee_prediction"
PRED_OUTPUT="${PRED_DIR}/committee_predictions.npz"
PRED_SELECTED="${PRED_DIR}/selected_topk.json"

if [ -d "${PROJECT_ROOT}/${CANDIDATE_DATA}" ]; then
    echo "[$(date -Iseconds)] Starting prediction" | tee -a "${TIMELINE_LOG}"
    PRED_START=$(date +%s)

    python "${PROJECT_ROOT}/scripts/inference/predict_committee_models.py" \
        --data "${CANDIDATE_DATA}" \
        --models \
            "${TRAIN_MODELS_DIR}/model_000/frozen_model.pb" \
            "${TRAIN_MODELS_DIR}/model_001/frozen_model.pb" \
            "${TRAIN_MODELS_DIR}/model_002/frozen_model.pb" \
            "${TRAIN_MODELS_DIR}/model_003/frozen_model.pb" \
        --output "${PRED_OUTPUT}" \
        --selected-json "${PRED_SELECTED}" \
        --top-k 10

    PRED_END=$(date +%s)
    PRED_ELAPSED=$((PRED_END - PRED_START))
    echo "[$(date -Iseconds)] Prediction done, elapsed_s=${PRED_ELAPSED}" | tee -a "${TIMELINE_LOG}"
else
    echo "[$(date -Iseconds)] Candidate data not found: ${CANDIDATE_DATA} — skipping prediction" | tee -a "${TIMELINE_LOG}"
    PRED_ELAPSED=""
fi

# ---- Dataset update (for next round) ----
NEXT_ROUND=$((ROUND_ID + 1))
if [ "${NEXT_ROUND}" -le 3 ]; then
    NEXT_ROUND_STR="$(printf '%03d' "${NEXT_ROUND}")"
    echo "[$(date -Iseconds)] Starting dataset update (round ${ROUND_STR} -> ${NEXT_ROUND_STR})" | tee -a "${TIMELINE_LOG}"
    UPDATE_START=$(date +%s)

    python "${PROJECT_ROOT}/scripts/data/merge_selected_frames.py" \
        --train "data/toy_h2/random_${SEED_LABEL}_round_${ROUND_STR}_train" \
        --candidate "${CANDIDATE_DATA}" \
        --selection "${PRED_SELECTED}" \
        --output "data/toy_h2/random_${SEED_LABEL}_round_${NEXT_ROUND_STR}_train" \
        --overwrite

    python "${PROJECT_ROOT}/scripts/data/make_remaining_candidate.py" \
        --candidate "${CANDIDATE_DATA}" \
        --selection "${PRED_SELECTED}" \
        --output "data/toy_h2/random_${SEED_LABEL}_round_${NEXT_ROUND_STR}_candidate" \
        --overwrite

    UPDATE_END=$(date +%s)
    UPDATE_ELAPSED=$((UPDATE_END - UPDATE_START))
    echo "[$(date -Iseconds)] Dataset update done, elapsed_s=${UPDATE_ELAPSED}" | tee -a "${TIMELINE_LOG}"
else
    UPDATE_ELAPSED=""
fi

# ---- Stop GPU monitoring ----
echo "[$(date -Iseconds)] Stopping GPU monitor" | tee -a "${TIMELINE_LOG}"
kill "${DMON_PID}" 2>/dev/null || true
sleep 1

# ---- Compute GPU stats from dmon log ----
GPU_UTIL_AVG=""
GPU_MEM_MAX=""
if [ -f "${GPU_LOG}" ]; then
    # Extract GPU utilization (column 3 in dmon output, skip header lines)
    GPU_UTIL_AVG=$(awk 'NR>2 && $3 != "-" {sum+=$3; n++} END {if(n>0) printf "%.1f", sum/n}' "${GPU_LOG}") || true
    GPU_MEM_MAX=$(awk 'NR>2 && $4 != "-" {if($4>max) max=$4} END {print max}' "${GPU_LOG}") || true
    echo "[$(date -Iseconds)] GPU util avg: ${GPU_UTIL_AVG:-N/A} %, GPU mem max: ${GPU_MEM_MAX:-N/A} MB" | tee -a "${TIMELINE_LOG}"
fi

# ---- Compute end-to-end ----
END_TS=$(date +%s)
echo "[$(date -Iseconds)] Profiling session ended" | tee -a "${TIMELINE_LOG}"

# ---- Print summary ----
echo "" | tee -a "${TIMELINE_LOG}"
echo "========== Profiling Summary ==========" | tee -a "${TIMELINE_LOG}"
echo "round=${ROUND_STR}  seed=${SEED_LABEL}  branch=${BRANCH}" | tee -a "${TIMELINE_LOG}"
echo "  training:       ${TRAIN_ELAPSED:-N/A} s" | tee -a "${TIMELINE_LOG}"
echo "  prediction:     ${PRED_ELAPSED:-N/A} s" | tee -a "${TIMELINE_LOG}"
echo "  dataset update: ${UPDATE_ELAPSED:-N/A} s" | tee -a "${TIMELINE_LOG}"
echo "  GPU util avg:   ${GPU_UTIL_AVG:-N/A} %" | tee -a "${TIMELINE_LOG}"
echo "  GPU mem max:    ${GPU_MEM_MAX:-N/A} MB" | tee -a "${TIMELINE_LOG}"
echo "  GPU log:        ${GPU_LOG}" | tee -a "${TIMELINE_LOG}"
echo "  Timeline log:   ${TIMELINE_LOG}" | tee -a "${TIMELINE_LOG}"

# ---- Write CSV row to profiling file ----
PROFILING_CSV="${LOG_DIR}/profiling_v100_rounds.csv"
# Build the CSV row (append to file, using the template header as reference)
CSV_LINE="${ROUND_STR},${BRANCH},${SEED_LABEL},${TRAIN_ELAPSED:-},,,""${PRED_ELAPSED:-},,,${UPDATE_ELAPSED:-},,V100-16GB,2,${GPU_UTIL_AVG:-},${GPU_MEM_MAX:-},recorded $(date -Iseconds)"
echo "${CSV_LINE}" >> "${PROFILING_CSV}"
echo "CSV row appended to: ${PROFILING_CSV}" | tee -a "${TIMELINE_LOG}"
