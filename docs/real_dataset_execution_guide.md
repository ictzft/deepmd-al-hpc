# Real DFT/AIMD Dataset Execution Guide

**Status: Pipeline validated on rMD17 ethanol (2026-05-26) and rMD17 benzene uncertainty branch (2026-05-27). This guide remains as a template for future real datasets. For rMD17 results, see `docs/real_dataset_plan.md`.**

---

## Step 0: Prerequisites

- A DeePMD-format dataset directory containing `type.raw`, `type_map.raw`, and `set.*/coord.npy`, `set.*/force.npy`, `set.*/energy.npy`, `set.*/box.npy`
- 2xV100 with DeepMD-kit Docker environment
- This repository at `/data/zft/deepmd-al-hpc`

## Step 1: Four-way split

```bash
python scripts/data/prepare_real_dataset_template.py \
  --source /path/to/your/deepmd_dataset \
  --dataset-name my_system \
  --output-root data/real_datasets/my_system \
  --initial-train 100 --candidate-pool 1000 \
  --validation 100 --test 100 \
  --seed 0

# Review the metadata
cat data/real_datasets/my_system/metadata.json
```

Manually split the source data according to the generated indices into:
- `data/real_datasets/my_system/initial_train/`
- `data/real_datasets/my_system/candidate_pool/`
- `data/real_datasets/my_system/validation/`
- `data/real_datasets/my_system/independent_test/`

## Step 2: Generate committee configs for Round 0

```bash
python scripts/config/make_round_committee_configs.py \
  --base configs/deepmd/toy_h2_input.json \
  --output-dir configs/real_system/round_000_committee \
  --train-system /data/zft/deepmd-al-hpc/data/real_datasets/my_system/initial_train \
  --valid-system /data/zft/deepmd-al-hpc/data/real_datasets/my_system/validation \
  --round-id 0 --n-models 4 --base-seed 1001
```

## Step 3: Train initial committee (Round 0)

```bash
bash scripts/train/train_round_committee_models.sh \
  000 configs/real_system/round_000_committee \
  experiments/real_system/round000_committee_models \
  /data/zft/deepmd-al-hpc/data/real_datasets/my_system/validation
```

## Step 4: Initial committee prediction on candidate_pool

```bash
python scripts/inference/predict_committee_models.py \
  --data data/real_datasets/my_system/candidate_pool \
  --models \
    experiments/real_system/round000_committee_models/model_000/frozen_model.pb \
    experiments/real_system/round000_committee_models/model_001/frozen_model.pb \
    experiments/real_system/round000_committee_models/model_002/frozen_model.pb \
    experiments/real_system/round000_committee_models/model_003/frozen_model.pb \
  --output experiments/real_system/round000_committee_prediction/committee_predictions.npz \
  --selected-json experiments/real_system/round000_committee_prediction/selected_topk.json \
  --top-k 10
```

## Step 5: Run active learning rounds with all 4 strategies

For each strategy (random/uncertainty/diversity/trust_level) and each seed (0/1/2):

```bash
python scripts/selection/select_by_strategy.py \
  --predictions experiments/real_system/round000_committee_prediction/committee_predictions.npz \
  --strategy uncertainty-diversity \
  --top-k 10 --top-m 30 \
  --output experiments/real_system/diversity_seed0/round_001/selected_topk.json

# Then: merge -> configs -> train -> predict (same pattern as toy H2 pipeline)
# Repeat for Round 002, 003
```

See `scripts/experiments/run_toy_h2_strategy_comparison.sh` as template — adapt `--strategy`, data paths, and output dirs.

## Step 6: Independent test evaluation

```bash
python scripts/evaluation/evaluate_independent_test_template.py \
  --models experiments/real_system/diversity_seed0/round_003/committee_models \
  --test-data data/real_datasets/my_system/independent_test \
  --output experiments/real_system/diversity_seed0/round_003/independent_test_metrics.json
```

## Step 7: Summarize

```bash
python scripts/analysis/build_aligned_comparison.py  # adapt to real data paths
```

## Files NOT to commit

```
data/real_datasets/**/*.npy
data/real_datasets/**/set.*/
experiments/real_system/**/frozen_model.pb
experiments/real_system/**/train.log
```
