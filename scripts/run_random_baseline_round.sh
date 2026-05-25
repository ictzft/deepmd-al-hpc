#!/bin/bash
# ============================================================
# Random Baseline Single-Round Runner
# ============================================================
# Automates one complete random baseline round for one seed:
#   1. Prepare data + configs (for round >= 2)
#   2. Train 4 committee models (2xV100 parallel)
#   3. Predict on candidate pool
#
# Usage:
#   bash scripts/run_random_baseline_round.sh <ROUND_ID> <SEED_LABEL>
#
# Example:
#   bash scripts/run_random_baseline_round.sh 002 seed0
#   bash scripts/run_random_baseline_round.sh 002 seed1
#   bash scripts/run_random_baseline_round.sh 003 seed0
#
# Prerequisites:
#   - Must run inside DeepMD-kit Docker container
#   - Round 002 needs Round 001 prediction results
#   - Round 003 needs Round 002 prediction results
# ============================================================

set -euo pipefail

ROUND_ID="${1:?Usage: $0 <ROUND_ID> <SEED_LABEL>}"
SEED_LABEL="${2:?}"

ROUND_STR="$(printf '%03d' "${ROUND_ID}")"
PREV_ROUND=$((ROUND_ID - 1))
PREV_ROUND_STR="$(printf '%03d' "${PREV_ROUND}")"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VALID_DATA="${PROJECT_ROOT}/data/toy_h2/valid"

CONFIG_DIR="${PROJECT_ROOT}/configs/deepmd/random_${SEED_LABEL}_round_${ROUND_STR}_committee"
EXP_DIR="experiments/baselines/random_${SEED_LABEL}_round${ROUND_STR}_committee_models"
PRED_DIR="experiments/baselines/random_${SEED_LABEL}_round${ROUND_STR}_committee_prediction"
CANDIDATE_DATA="data/toy_h2/random_${SEED_LABEL}_round_${ROUND_STR}_candidate"
TRAIN_DATA="data/toy_h2/random_${SEED_LABEL}_round_${ROUND_STR}_train"

echo "============================================================"
echo "Random Baseline Round ${ROUND_STR} — ${SEED_LABEL}"
echo "============================================================"
echo "project root : ${PROJECT_ROOT}"
echo "config dir   : ${CONFIG_DIR}"
echo "exp dir      : ${EXP_DIR}"
echo "pred dir     : ${PRED_DIR}"
echo "train data   : ${TRAIN_DATA}"
echo "candidate    : ${CANDIDATE_DATA}"
echo "valid data   : ${VALID_DATA}"
echo ""

# ---- Step 1: Prepare data and configs (for round >= 2) ----
if [ "${ROUND_ID}" -ge 2 ]; then
    echo "========== Step 1/3: Prepare data + configs =========="
    python "${PROJECT_ROOT}/scripts/analysis/prepare_random_baseline_round.py" \
        --seed-label "${SEED_LABEL}" --round-id "${ROUND_ID}"
    echo ""
fi

# ---- Step 2: Train committee models ----
echo "========== Step 2/3: Train committee models =========="
bash "${PROJECT_ROOT}/scripts/train/train_round_committee_models.sh" \
    "${ROUND_STR}" \
    "configs/deepmd/random_${SEED_LABEL}_round_${ROUND_STR}_committee" \
    "${EXP_DIR}" \
    "${VALID_DATA}"
echo ""

# ---- Step 3: Committee prediction ----
echo "========== Step 3/3: Committee prediction =========="
mkdir -p "${PROJECT_ROOT}/${PRED_DIR}"
python "${PROJECT_ROOT}/scripts/inference/predict_committee_models.py" \
    --data "${CANDIDATE_DATA}" \
    --models \
        "${EXP_DIR}/model_000/frozen_model.pb" \
        "${EXP_DIR}/model_001/frozen_model.pb" \
        "${EXP_DIR}/model_002/frozen_model.pb" \
        "${EXP_DIR}/model_003/frozen_model.pb" \
    --output "${PRED_DIR}/committee_predictions.npz" \
    --selected-json "${PRED_DIR}/selected_topk.json" \
    --top-k 10
echo ""

# ---- Done ----
echo "============================================================"
echo "Round ${ROUND_STR} (${SEED_LABEL}) complete!"
echo "============================================================"
echo ""
echo "Models:     ${EXP_DIR}/model_*/frozen_model.pb"
echo "Prediction: ${PRED_DIR}/committee_predictions.npz"
echo "Selection:  ${PRED_DIR}/selected_topk.json"
echo ""
echo "To summarize this round:"
echo "  python scripts/analysis/summarize_random_round_baselines.py --round-id ${ROUND_ID}"
echo ""
if [ "${ROUND_ID}" -lt 3 ]; then
    NEXT_ROUND=$((ROUND_ID + 1))
    NEXT_ROUND_STR="$(printf '%03d' "${NEXT_ROUND}")"
    echo "To prepare Round ${NEXT_ROUND_STR}:"
    echo "  python scripts/analysis/prepare_random_baseline_round.py --seed-label ${SEED_LABEL} --round-id ${NEXT_ROUND}"
    echo "  (or use this script: bash scripts/run_random_baseline_round.sh ${NEXT_ROUND_STR} ${SEED_LABEL})"
fi
