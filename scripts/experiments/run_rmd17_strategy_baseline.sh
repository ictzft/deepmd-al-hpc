#!/bin/bash
# Run rMD17 ethanol strategy baseline (diversity or trust_level).
# Usage: bash scripts/experiments/run_rmd17_strategy_baseline.sh <STRATEGY> [SEED]
#   STRATEGY: diversity | trust_level
#   Without SEED: runs all 3 seeds (0, 1, 2).
set -euo pipefail

STRATEGY="${1:?Usage: $0 <diversity|trust_level> [SEED]}"
ROOT="/data/guyida/deepmd-al-hpc"
DATA="$ROOT/data/rmd17/ethanol"
EXPS="$ROOT/experiments/baselines"
CONFIGS="$ROOT/configs/deepmd"
SCRIPTS="$ROOT/scripts"
BASE_CONFIG="$CONFIGS/rmd17_ethanol_base.json"
ROUND0_PRED="$ROOT/experiments/rmd17_ethanol_round000_committee_prediction/committee_predictions.npz"
ROUND0_TEMPLATE="$ROOT/experiments/rmd17_ethanol_round000_committee_prediction/selected_uncertainty_1000.json"
INITIAL_TRAIN="$DATA/train"
TOPK=1000

# Strategy-specific selection args
if [ "$STRATEGY" = "diversity" ]; then
  EXTRA_SEL_ARGS="--top-m 3000"
elif [ "$STRATEGY" = "trust_level" ]; then
  EXTRA_SEL_ARGS="--low-pct 50 --high-pct 90 --candidate-ratio 0.8"
else
  echo "Unknown strategy: $STRATEGY"
  exit 1
fi

run_seed() {
  local SEED=$1
  local BASE_SEED=$(( (SEED + 1) * 1000 + 500 ))
  local PRED_NPZ="$ROUND0_PRED"

  for ROUND in 1 2 3; do
    local RLABEL="${STRATEGY}_seed${SEED}_round00${ROUND}"
    local PREV_ROUND=$((ROUND - 1))

    echo ""
    echo "=== SEED=$SEED ROUND=$ROUND ($RLABEL) ==="

    # --- 1. Strategy selection ---
    local SEL_DIR="$EXPS/${RLABEL}_committee_prediction"
    local SEL_JSON="$SEL_DIR/selected_${STRATEGY}.json"
    mkdir -p "$SEL_DIR"

    echo "[1/6] ${STRATEGY} selection (top-k=$TOPK, seed=$SEED) ..."
    python3 "$SCRIPTS/active_learning/select_from_predictions.py" \
      --predictions "$PRED_NPZ" \
      --strategy "$STRATEGY" \
      --top-k $TOPK \
      --seed $SEED \
      $EXTRA_SEL_ARGS \
      --template-json "$ROUND0_TEMPLATE" \
      --output "$SEL_JSON" 2>&1 | tail -1

    # --- 2. Merge selected frames ---
    local TRAIN_DIR="$DATA/train_round00${ROUND}_${STRATEGY}_seed${SEED}"
    local PREV_TRAIN
    if [ $ROUND -eq 1 ]; then PREV_TRAIN="$INITIAL_TRAIN"
    else PREV_TRAIN="$DATA/train_round00${PREV_ROUND}_${STRATEGY}_seed${SEED}"
    fi

    local CAND_DIR
    if [ $ROUND -eq 1 ]; then CAND_DIR="$DATA/candidate"
    else CAND_DIR="$DATA/candidate_round00${PREV_ROUND}_${STRATEGY}_seed${SEED}"
    fi

    echo "[2/6] Merge -> $TRAIN_DIR"
    python3 "$SCRIPTS/data/merge_selected_frames.py" \
      --train "$PREV_TRAIN" --candidate "$CAND_DIR" \
      --selection "$SEL_JSON" --output "$TRAIN_DIR" --overwrite 2>&1 | tail -1

    # --- 3. Remaining candidate ---
    local NEW_CAND="$DATA/candidate_round00${ROUND}_${STRATEGY}_seed${SEED}"
    echo "[3/6] Remaining candidate -> $NEW_CAND"
    python3 "$SCRIPTS/data/make_remaining_candidate.py" \
      --candidate "$CAND_DIR" --selection "$SEL_JSON" \
      --output "$NEW_CAND" --overwrite 2>&1 | tail -1

    # --- 4. Configs ---
    local CONFIG_DIR="$CONFIGS/rmd17_ethanol_${STRATEGY}_seed${SEED}_round00${ROUND}_committee"
    echo "[4/6] Generate configs"
    python3 "$SCRIPTS/config/make_round_committee_configs.py" \
      --base "$BASE_CONFIG" --output-dir "$CONFIG_DIR" \
      --train-system "$TRAIN_DIR" --valid-system "$ROOT/data/rmd17/ethanol/valid" \
      --round-id "$ROUND" --n-models 4 --base-seed $((BASE_SEED + ROUND * 100)) 2>&1 | tail -1

    # --- 5. Train ---
    local MODEL_DIR="$EXPS/${RLABEL}_committee_models"
    local REL_CONFIG_DIR="configs/deepmd/rmd17_ethanol_${STRATEGY}_seed${SEED}_round00${ROUND}_committee"
    local REL_MODEL_DIR="experiments/baselines/${RLABEL}_committee_models"
    echo "[5/6] Train 4 models (2xV100 parallel)"
    docker exec -w /data/guyida/deepmd-al-hpc 6a8531d2d15a \
      bash -lc "export PATH=/opt/deepmd-kit/bin:\$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:\$LD_LIBRARY_PATH; bash scripts/train/train_round_committee_models.sh \
        $(printf '%03d' $ROUND) $REL_CONFIG_DIR $REL_MODEL_DIR /data/guyida/deepmd-al-hpc/data/rmd17/ethanol/valid" 2>&1 | grep -E "finished training|wall time|Error|==========" || true
    # Verify models were created
    for m in 0 1 2 3; do
      if [ ! -f "$MODEL_DIR/model_00${m}/frozen_model.pb" ]; then
        echo "ERROR: model_00${m} frozen_model.pb missing!"; exit 1
      fi
    done

    # --- 6. Predict ---
    local REL_NEW_CAND="data/rmd17/ethanol/candidate_round00${ROUND}_${STRATEGY}_seed${SEED}"
    local REL_PRED_DIR="experiments/baselines/${RLABEL}_committee_prediction"
    echo "[6/6] Predict on candidate pool"
    docker exec -w /data/guyida/deepmd-al-hpc 6a8531d2d15a \
      bash -lc "export PATH=/opt/deepmd-kit/bin:\$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:\$LD_LIBRARY_PATH; python3 scripts/inference/predict_committee_models.py \
        --data $REL_NEW_CAND \
        --models $REL_MODEL_DIR/model_000/frozen_model.pb $REL_MODEL_DIR/model_001/frozen_model.pb $REL_MODEL_DIR/model_002/frozen_model.pb $REL_MODEL_DIR/model_003/frozen_model.pb \
        --output $REL_PRED_DIR/committee_predictions.npz --top-k $TOPK" 2>&1 | grep "Saved" || true

    PRED_NPZ="$ROOT/experiments/baselines/${RLABEL}_committee_prediction/committee_predictions.npz"
    echo "=== $RLABEL DONE ==="
  done
}

# --- Main ---
SEEDS="${2:-0 1 2}"
echo "Strategy: $STRATEGY | Seeds: $SEEDS | Start: $(date)"
for s in $SEEDS; do
  run_seed "$s"
done
echo "Done: $(date)"
