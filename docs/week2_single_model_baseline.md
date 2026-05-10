# 第 2 周：单模型 DeePMD Baseline

## 1. 本周目标

第 2 周的目标是跑通真实 DeePMD 单模型训练流程。

第一周已经完成了：

- GitHub 仓库初始化；
- Docker + 2×V100 环境验证；
- DeepMD-kit 环境验证；
- 主动学习框架 skeleton；
- force model deviation 与 top-K 构型筛选逻辑。

第 2 周开始，项目从“主动学习控制逻辑验证”进入“真实 DeePMD 模型训练验证”。

---

## 2. 为什么先做单模型 baseline？

后续主动学习闭环需要反复训练多个 committee models，但在此之前，必须先确认单个 DeePMD 模型可以正常完成：

```text
数据读取
  ↓
模型训练
  ↓
模型冻结
  ↓
模型测试
  ↓
误差评估

如果单模型 baseline 没有跑通，后续 committee models、model deviation 和 active learning 都无法稳定开展。

3. 本周核心任务
 找到或准备一个小体系 DeepMD 数据集；
 确认数据格式符合 DeepMD-kit 要求；
 编写 DeePMD 训练配置 input.json；
 运行 dp train input.json；
 运行 dp freeze 生成 frozen model；
 运行 dp test 评估模型误差；
 记录 Energy MAE / Force RMSE；
 保存训练日志、测试结果和实验说明；
 为第 3 周 committee models 训练做准备。
4. 最小训练流程
DeepMD 数据集
  ↓
configs/deepmd/input.json
  ↓
dp train
  ↓
dp freeze
  ↓
dp test
  ↓
Energy / Force error
5. 当前实验目录
experiments/exp_003_single_model_baseline/

本实验暂时只训练一个模型，不做主动学习，也不做 committee models。

6. 预期输出文件
experiments/exp_003_single_model_baseline/
├── README.md
├── train.log
├── freeze.log
├── test.log
├── frozen_model.pb
└── metrics_summary.md

其中：

train.log：训练日志；
freeze.log：模型冻结日志；
test.log：模型测试日志；
frozen_model.pb：冻结后的 DeePMD 模型；
metrics_summary.md：最终误差和实验记录。
7. 验收标准

第 2 周完成的最低标准是：

 dp train 可以正常启动并结束；
 dp freeze 可以生成 frozen_model.pb；
 dp test 可以输出能量和力误差；
 实验日志保存在 experiments/exp_003_single_model_baseline/；
 训练命令、数据路径、配置文件和结果都能被复现。
8. 后续衔接

第 2 周完成后，第 3 周将基于单模型 baseline 扩展到：

单模型 DeePMD
  ↓
4 个不同随机种子的 committee models
  ↓
真实 model deviation 计算
  ↓
主动学习构型筛选

