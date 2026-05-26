#!/bin/bash
# Run rMD17 ethanol random sampling baseline (3 seeds × 3 rounds).
# Usage: bash scripts/experiments/run_rmd17_random_baseline.sh [SEED]
#   Without args: runs all 3 seeds (0, 1, 2) in sequence.
#   With SEED: runs only that seed (for parallel execution).
set -euo pipefail

ROOT="/data/guyida/deepmd-al-hpc"
DATA="$ROOT/data/rmd17/ethanol"
EXPS="$ROOT/experiments/baselines"
CONFIGS="$ROOT/configs/deepmd"
SCRIPTS="$ROOT/scripts"
BASE_CONFIG="$CONFIGS/rmd17_ethanol_base.json"

# Round 0 prediction (from uncertainty branch — shared by all seeds)
ROUND0_PRED="$ROOT/experiments/rmd17_ethanol_round000_committee_prediction/committee_predictions.npz"
ROUND0_TEMPLATE="$ROOT/experiments/rmd17_ethanol_round000_committee_prediction/selected_uncertainty_1000.json"

# Training data: initial train + valid (same as uncertainty branch)
INITIAL_TRAIN="$DATA/train"
VALID="$DATA/valid"
TOPK=1000

run_seed() {
  local SEED=$1
  local BASE_SEED=$(( (SEED + 1) * 1000 ))
  local PRED_NPZ="$ROUND0_PRED"

  for ROUND in 1 2 3; do
    local RLABEL="random_seed${SEED}_round00${ROUND}"
    local PREV_ROUND=$((ROUND - 1))

    echo ""
    echo "============================================================"
    echo "  SEED=$SEED  ROUND=$ROUND  ($RLABEL)"
    echo "============================================================"

    # --- 1. Random selection from current predictions ---
    local SEL_JSON="$EXPS/${RLABEL}_committee_prediction/selected_random.json"
    mkdir -p "$EXPS/${RLABEL}_committee_prediction"

    echo "[1/6] Random selection ..."
    python3 "$SCRIPTS/active_learning/select_from_predictions.py" \
      --predictions "$PRED_NPZ" \
      --strategy random \
      --top-k $TOPK \
      --seed $SEED \
      --template-json "$ROUND0_TEMPLATE" \
      --output "$SEL_JSON"

    # --- 2. Merge selected frames into training set ---
    local TRAIN_DIR="$DATA/train_round00${ROUND}_random_seed${SEED}"

    # Determine previous train dir
    local PREV_TRAIN
    if [ $ROUND -eq 1 ]; then
      PREV_TRAIN="$INITIAL_TRAIN"
    else
      PREV_TRAIN="$DATA/train_round00${PREV_ROUND}_random_seed${SEED}"
    fi

    # Determine which candidate pool to use
    local CAND_DIR
    if [ $ROUND -eq 1 ]; then
      CAND_DIR="$DATA/candidate"
    else
      CAND_DIR="$DATA/candidate_round00${PREV_ROUND}_random_seed${SEED}"
    fi

    echo "[2/6] Merge selected frames -> $TRAIN_DIR"
    python3 "$SCRIPTS/data/merge_selected_frames.py" \
      --train "$PREV_TRAIN" \
      --candidate "$CAND_DIR" \
      --selection "$SEL_JSON" \
      --output "$TRAIN_DIR" --overwrite

    # --- 3. Create remaining candidate pool ---
    local NEW_CAND="$DATA/candidate_round00${ROUND}_random_seed${SEED}"
    echo "[3/6] Create remaining candidate -> $NEW_CAND"
    python3 "$SCRIPTS/data/make_remaining_candidate.py" \
      --candidate "$CAND_DIR" \
      --selection "$SEL_JSON" \
      --output "$NEW_CAND" --overwrite

    # --- 4. Generate committee configs ---
    local CONFIG_DIR="$CONFIGS/rmd17_ethanol_random_seed${SEED}_round00${ROUND}_committee"
    echo "[4/6] Generate configs -> $CONFIG_DIR"
    python3 "$SCRIPTS/config/make_round_committee_configs.py" \
      --base "$BASE_CONFIG" \
      --output-dir "$CONFIG_DIR" \
      --train-system "$TRAIN_DIR" \
      --valid-system "$VALID" \
      --round-id "$ROUND" --n-models 4 --base-seed $((BASE_SEED + ROUND * 100))

    # --- 5. Train committee models (in Docker) ---
    local MODEL_DIR="$EXPS/${RLABEL}_committee_models"
    echo "[5/6] Train committee models -> $MODEL_DIR"
    # Use relative paths since train_round_committee_models.sh prepends PROJECT_ROOT
    local REL_CONFIG_DIR="configs/deepmd/rmd17_ethanol_random_seed${SEED}_round00${ROUND}_committee"
    local REL_MODEL_DIR="experiments/baselines/${RLABEL}_committee_models"
    local REL_VALID="/data/guyida/deepmd-al-hpc/data/rmd17/ethanol/valid"
    docker exec -w /data/guyida/deepmd-al-hpc 6a8531d2d15a \
      bash -lc "export PATH=/opt/deepmd-kit/bin:\$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:\$LD_LIBRARY_PATH; bash scripts/train/train_round_committee_models.sh \
        $(printf '%03d' $ROUND) $REL_CONFIG_DIR $REL_MODEL_DIR $REL_VALID"

    # --- 6. Run committee prediction (generates predictions for next round) ---
    local NEW_PRED_DIR="$EXPS/${RLABEL}_committee_prediction"
    local NEW_PRED_NPZ="$NEW_PRED_DIR/committee_predictions.npz"
    echo "[6/6] Committee prediction on new candidate pool ..."
    local REL_NEW_CAND="data/rmd17/ethanol/candidate_round00${ROUND}_random_seed${SEED}"
    local REL_MODEL_BASE="experiments/baselines/${RLABEL}_committee_models"
    local REL_PRED_DIR="experiments/baselines/${RLABEL}_committee_prediction"
    docker exec -w /data/guyida/deepmd-al-hpc 6a8531d2d15a \
      bash -lc "export PATH=/opt/deepmd-kit/bin:\$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:\$LD_LIBRARY_PATH; python3 scripts/inference/predict_committee_models.py \
        --data $REL_NEW_CAND \
        --models \
          $REL_MODEL_BASE/model_000/frozen_model.pb \
          $REL_MODEL_BASE/model_001/frozen_model.pb \
          $REL_MODEL_BASE/model_002/frozen_model.pb \
          $REL_MODEL_BASE/model_003/frozen_model.pb \
        --output $REL_PRED_DIR/committee_predictions.npz \
        --top-k $TOPK"

    # Set up for next round (absolute path for host-side scripts)
    PRED_NPZ="$ROOT/experiments/baselines/${RLABEL}_committee_prediction/committee_predictions.npz"

    echo "=== SEED=$SEED ROUND=$ROUND DONE ==="
  done
}


# --- Main ---
SEEDS="${1:-0 1 2}"
echo "Random baseline seeds: $SEEDS"
echo "Start: $(date)"
for s in $SEEDS; do
  run_seed "$s"
done
echo "Done: $(date)"
