# exp_003_single_model_baseline

## 实验目的

本实验用于跑通真实 DeePMD 单模型 baseline。

## 当前目标

- 准备一个小体系 DeepMD 数据集；
- 编写 `configs/deepmd/input.json`；
- 运行 `dp train`；
- 运行 `dp freeze`；
- 运行 `dp test`；
- 保存训练日志和测试误差。

## 预期输出

```text
train.log
freeze.log
test.log
frozen_model.pb
metrics_summary.md
最小流程
bash scripts/train/train_single_model.sh \
  experiments/exp_003_single_model_baseline \
  configs/deepmd/input.json

bash scripts/eval/freeze_model.sh \
  experiments/exp_003_single_model_baseline \
  frozen_model.pb

bash scripts/eval/test_single_model.sh \
  experiments/exp_003_single_model_baseline \
  experiments/exp_003_single_model_baseline/frozen_model.pb \
  /path/to/test_data
备注

本实验暂时只训练一个模型，不进行 committee models 和主动学习。
