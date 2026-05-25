#!/bin/bash
# ============================================================
# Toy H2 Four-Strategy Comparison Runner (template)
# ============================================================
# Supports: random, uncertainty, uncertainty-diversity, trust-level
#
# Usage (dry-run):
#   bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
#     --strategy uncertainty-diversity --start-round 1 --end-round 3 \
#     --top-k 10 --dry-run
#
# Usage (actual run, inside DeepMD-kit Docker):
#   bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
#     --strategy uncertainty-diversity --start-round 1 --end-round 3 \
#     --top-k 10
#
# ============================================================
set -euo pipefail

STRATEGY=""
START_ROUND=1
END_ROUND=3
TOP_K=10
SEED=0
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strategy) STRATEGY="$2"; shift 2 ;;
    --start-round) START_ROUND="$2"; shift 2 ;;
    --end-round) END_ROUND="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -z "$STRATEGY" ]; then
  echo "Usage: $0 --strategy <random|uncertainty|uncertainty-diversity|trust-level> [--start-round 1] [--end-round 3] [--top-k 10] [--dry-run]"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMP_DIR="${PROJECT_ROOT}/experiments/strategy_comparison_toy_h2"

# Map strategy name to directory label
case "$STRATEGY" in
  random) STRATEGY_DIR="random" ;;
  uncertainty|topk|top-k|uncertainty_topk) STRATEGY_DIR="uncertainty" ;;
  uncertainty-diversity|diversity|uncertainty_diversity) STRATEGY_DIR="uncertainty_diversity" ;;
  trust-level|trust|trust_level|dpgen) STRATEGY_DIR="trust_level" ;;
  *) echo "Unknown strategy: $STRATEGY"; exit 1 ;;
esac

echo "============================================================"
echo "Toy H2 Strategy Comparison Runner"
echo "============================================================"
echo "  strategy:     $STRATEGY  (dir: $STRATEGY_DIR)"
echo "  start_round:  $START_ROUND"
echo "  end_round:    $END_ROUND"
echo "  top_k:        $TOP_K"
echo "  seed:         $SEED"
echo "  dry_run:      $DRY_RUN"
echo "============================================================"

for ((ROUND=START_ROUND; ROUND<=END_ROUND; ROUND++)); do
  ROUND_STR="$(printf '%03d' "$ROUND")"
  PREV_ROUND=$((ROUND - 1))
  PREV_STR="$(printf '%03d' "$PREV_ROUND")"

  STRATEGY_DIR_PATH="${COMP_DIR}/${STRATEGY_DIR}/round_${ROUND_STR}"
  mkdir -p "${STRATEGY_DIR_PATH}"

  echo ""
  echo "===== Round ${ROUND_STR} (${STRATEGY}) ====="

  # Determine the previous round's prediction output
  if [ "$ROUND" -eq 1 ]; then
    # Round 1 uses initial committee prediction
    SRC_PRED="${PROJECT_ROOT}/experiments/exp_005_committee_prediction/committee_predictions.npz"
    SRC_TRAIN="${PROJECT_ROOT}/data/toy_h2/train"
    SRC_CAND="${PROJECT_ROOT}/data/toy_h2/valid"
  else
    PREV_DIR="${COMP_DIR}/${STRATEGY_DIR}/round_${PREV_STR}"
    SRC_PRED="${PREV_DIR}/committee_predictions.npz"
    SRC_TRAIN="${PROJECT_ROOT}/data/toy_h2/${STRATEGY_DIR}_round_${PREV_STR}_train"
    SRC_CAND="${PROJECT_ROOT}/data/toy_h2/${STRATEGY_DIR}_round_${PREV_STR}_candidate"
  fi

  TRAIN_DATA="${PROJECT_ROOT}/data/toy_h2/${STRATEGY_DIR}_round_${ROUND_STR}_train"
  CAND_DATA="${PROJECT_ROOT}/data/toy_h2/${STRATEGY_DIR}_round_${ROUND_STR}_candidate"
  CONFIG_DIR="${PROJECT_ROOT}/configs/deepmd/${STRATEGY_DIR}_round_${ROUND_STR}_committee"
  MODELS_DIR="${STRATEGY_DIR_PATH}/committee_models"
  PRED_OUTPUT="${STRATEGY_DIR_PATH}/committee_predictions.npz"
  SELECTED_JSON="${STRATEGY_DIR_PATH}/selected_topk.json"

  # --- Step 1: Selection ---
  CMD1="python scripts/selection/select_by_strategy.py --predictions ${SRC_PRED} --strategy ${STRATEGY} --top-k ${TOP_K} --output ${SELECTED_JSON}"
  echo "[Step 1] Selection: ${CMD1}"
  if [ "$DRY_RUN" = false ]; then
    $CMD1 || { echo "ERROR: Selection failed for round ${ROUND_STR}"; exit 1; }
  fi

  # --- Step 2: Data preparation ---
  CMD2A="python scripts/data/merge_selected_frames.py --train ${SRC_TRAIN} --candidate ${SRC_CAND} --selection ${SELECTED_JSON} --output ${TRAIN_DATA} --overwrite"
  CMD2B="python scripts/data/make_remaining_candidate.py --candidate ${SRC_CAND} --selection ${SELECTED_JSON} --output ${CAND_DATA} --overwrite"
  echo "[Step 2a] Merge: ${CMD2A}"
  echo "[Step 2b] Remaining: ${CMD2B}"
  if [ "$DRY_RUN" = false ]; then
    $CMD2A && $CMD2B || { echo "ERROR: Data prep failed for round ${ROUND_STR}"; exit 1; }
  fi

  # --- Step 3: Config generation ---
  CMD3="python scripts/config/make_round_committee_configs.py --base ${PROJECT_ROOT}/configs/deepmd/toy_h2_input.json --output-dir ${CONFIG_DIR} --train-system ${TRAIN_DATA} --valid-system ${PROJECT_ROOT}/data/toy_h2/valid --round-id ${ROUND} --n-models 4 --base-seed ${SEED}"
  echo "[Step 3] Configs: ${CMD3}"
  if [ "$DRY_RUN" = false ]; then
    $CMD3 || { echo "ERROR: Config generation failed for round ${ROUND_STR}"; exit 1; }
  fi

  # --- Step 4: Training (longest step) ---
  CMD4="bash scripts/train/train_round_committee_models.sh ${ROUND_STR} configs/deepmd/${STRATEGY_DIR}_round_${ROUND_STR}_committee experiments/strategy_comparison_toy_h2/${STRATEGY_DIR}/round_${ROUND_STR}/committee_models ${PROJECT_ROOT}/data/toy_h2/valid"
  echo "[Step 4] Training: ${CMD4}"
  if [ "$DRY_RUN" = false ]; then
    echo "  TODO: Run training inside DeepMD-kit Docker container."
    echo "  This step requires GPU and dp command."
  fi

  # --- Step 5: Prediction ---
  CMD5="python scripts/inference/predict_committee_models.py --data ${CAND_DATA} --models ${MODELS_DIR}/model_000/frozen_model.pb ${MODELS_DIR}/model_001/frozen_model.pb ${MODELS_DIR}/model_002/frozen_model.pb ${MODELS_DIR}/model_003/frozen_model.pb --output ${PRED_OUTPUT} --selected-json ${SELECTED_JSON} --top-k ${TOP_K}"
  echo "[Step 5] Prediction: ${CMD5}"
  if [ "$DRY_RUN" = false ]; then
    echo "  TODO: Run prediction after training completes."
  fi

  echo ""
done

echo ""
echo "============================================================"
echo "Dry-run complete. To run for real:"
echo "  1. Enter DeepMD-kit Docker container"
echo "  2. Remove --dry-run flag"
echo "  3. Ensure data/toy_h2/train and data/toy_h2/valid exist"
echo "============================================================"
