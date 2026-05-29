# 真实 DFT/AIMD 数据集执行指南

**状态：已在 rMD17 ethanol（2026-05-26）和 rMD17 benzene uncertainty branch（2026-05-27）上验证。本指南保留作为未来真实数据集的模板。rMD17 结果见 `docs/real_dataset_plan.md`。**

---

## Step 0：前提条件

- DeePMD 格式的数据目录，包含 `type.raw`、`type_map.raw` 和 `set.*/coord.npy`、`set.*/force.npy`、`set.*/energy.npy`、`set.*/box.npy`
- 具有 DeepMD-kit Docker 环境的 2×V100
- 本仓库位于 `/data/zft/deepmd-al-hpc`

## Step 1：四路划分

```bash
python scripts/data/prepare_real_dataset_template.py \
  --source /path/to/your/deepmd_dataset \
  --dataset-name my_system \
  --output-root data/real_datasets/my_system \
  --initial-train 100 --candidate-pool 1000 \
  --validation 100 --test 100 \
  --seed 0

# 检查 metadata
cat data/real_datasets/my_system/metadata.json
```

根据生成的索引手动划分源数据到：
- `data/real_datasets/my_system/initial_train/`
- `data/real_datasets/my_system/candidate_pool/`
- `data/real_datasets/my_system/validation/`
- `data/real_datasets/my_system/independent_test/`

## Step 2：为 Round 0 生成 committee 配置

```bash
python scripts/config/make_round_committee_configs.py \
  --base configs/deepmd/toy_h2_input.json \
  --output-dir configs/real_system/round_000_committee \
  --train-system /data/zft/deepmd-al-hpc/data/real_datasets/my_system/initial_train \
  --valid-system /data/zft/deepmd-al-hpc/data/real_datasets/my_system/validation \
  --round-id 0 --n-models 4 --base-seed 1001
```

## Step 3：训练初始 committee（Round 0）

```bash
bash scripts/train/train_round_committee_models.sh \
  000 configs/real_system/round_000_committee \
  experiments/real_system/round000_committee_models \
  /data/zft/deepmd-al-hpc/data/real_datasets/my_system/validation
```

## Step 4：对 candidate_pool 进行初始 committee prediction

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

## Step 5：使用所有 4 种策略运行 active learning 轮次

对每种策略（random/uncertainty/diversity/trust_level）和每个 seed（0/1/2）：

```bash
python scripts/selection/select_by_strategy.py \
  --predictions experiments/real_system/round000_committee_prediction/committee_predictions.npz \
  --strategy uncertainty-diversity \
  --top-k 10 --top-m 30 \
  --output experiments/real_system/diversity_seed0/round_001/selected_topk.json

# 然后：merge → configs → train → predict（与 toy H2 流水线相同模式）
# 对 Round 002, 003 重复
```

以 `scripts/experiments/run_toy_h2_strategy_comparison.sh` 为模板——修改 `--strategy`、数据路径和输出目录。

## Step 6：Independent test 评估

```bash
python scripts/evaluation/evaluate_independent_test_template.py \
  --models experiments/real_system/diversity_seed0/round_003/committee_models \
  --test-data data/real_datasets/my_system/independent_test \
  --output experiments/real_system/diversity_seed0/round_003/independent_test_metrics.json
```

## Step 7：汇总

```bash
python scripts/analysis/build_aligned_comparison.py  # 修改为真实数据路径
```

## 不应提交的文件

```
data/real_datasets/**/*.npy
data/real_datasets/**/set.*/
experiments/real_system/**/frozen_model.pb
experiments/real_system/**/train.log
```
