# deepmd-al-hpc

本项目是一个面向 **第一性原理机器学习势函数主动学习闭环** 的多模型并行与高性能训练原型系统。

项目不直接训练大语言模型，也不是直接复现 Megatron-LM，而是借鉴 Megatron 系列大规模训练框架中的 **多 GPU 并行、micro-batch、混合精度、批量推理、流水线调度和分布式实验管理思想**，将这些高性能训练思想迁移到 DeePMD / DeepMD-kit 机器学习势函数主动学习场景中。

当前阶段主要在 **2×V100 GPU** 平台上完成项目原型验证，包括 Docker 环境验证、DeepMD-kit 环境验证、主动学习框架 skeleton、toy H2 单模型 DeePMD baseline，以及基于模拟 committee forces 的 model deviation 和 top-K 候选构型筛选。当前项目尚未完成真实 4-model DeePMD committee 训练与真实候选构型推理，下一阶段将重点实现多个不同随机种子 DeePMD 模型的 train / freeze / test，并进一步接入真实 committee prediction 与 offline active learning 闭环。

后续计划迁移到 H100 多 GPU 平台，开展更完整的训练加速、推理加速和主动学习闭环性能优化实验。

---

## 一、研究方向

本项目属于 **AI for Science 与高性能计算交叉方向**，主要关注：

```text
第一性原理计算
  ↓
机器学习势函数
  ↓
主动学习采样
  ↓
多模型并行训练
  ↓
GPU / H100 高性能加速
```

暂定中文题目：

> 面向第一性原理机器学习势函数主动学习闭环的多模型并行训练框架研究

暂定英文题目：

> A Multi-Model Parallel Active Learning Framework for First-Principles Machine Learning Potentials

也可以进一步概括为：

> A High-Performance Active Learning Framework for First-Principles Machine Learning Potentials

---

## 二、项目目标

本项目面向第一性原理机器学习势函数的主动学习闭环训练过程，研究如何在保证势函数精度和分子动力学稳定性的前提下，提高训练、推理、不确定性评估和构型筛选的整体效率。

具体目标包括：

1. 基于 DeePMD / DeepMD-kit 跑通机器学习势函数基线训练流程；
2. 构建多个 committee models，用于模型不确定性估计；
3. 基于 force / energy / virial deviation 计算候选构型的不确定性；
4. 实现 offline active learning 闭环，模拟 DFT labeling 过程；
5. 借鉴 Megatron-style 的并行训练思想，实现多模型并行训练和批量推理；
6. 在 2×V100 平台上完成原型验证和性能 profiling；
7. 为后续 H100 多 GPU 平台上的主动学习闭环加速实验做准备。

---

## 三、为什么借鉴 Megatron-style 思想？

Megatron-LM / Megatron-Core 面向的是大语言模型的大规模分布式训练，其核心思想包括：

- 多 GPU 并行训练；
- micro-batch 切分；
- 混合精度训练；
- pipeline 调度；
- 分布式 checkpoint；
- 大规模训练任务管理。

本项目不直接使用 Megatron 训练 DeePMD，而是借鉴这些系统设计思想，解决机器学习势函数主动学习闭环中的类似问题：

```text
Megatron 面对的问题：
大模型训练太慢，需要多 GPU 并行和流水线调度。

本项目面对的问题：
主动学习势函数闭环太慢，需要多模型并行训练、批量推理和自动化调度。
```

在本项目中，Megatron-style 思想主要对应为：

| Megatron-style 思想 | 本项目中的对应设计 |
|---|---|
| Data Parallel | 多 GPU 训练 DeePMD 模型 |
| Expert / Multi-Model Parallel | committee models 并行训练 |
| Micro-batching | 候选构型池批量推理 |
| Pipeline Scheduling | 训练、推理、deviation、筛选的流水线化 |
| Mixed Precision | 后续在 A100 / H100 上探索 BF16 / FP16 / TF32 |
| Distributed Checkpoint | 管理每轮主动学习的模型、数据划分和筛选结果 |
| Profiling | 分析训练、推理、deviation 和数据处理耗时 |

---

## 四、主动学习闭环流程

本项目计划实现如下主动学习闭环：

```text
初始 DFT 数据集
  ↓
训练多个 DeePMD committee models
  ↓
在候选构型池上进行批量推理
  ↓
计算 force / energy / virial model deviation
  ↓
根据不确定性和结构多样性筛选高价值构型
  ↓
将筛选出的构型加入训练集
  ↓
重新训练 committee models
  ↓
进入下一轮主动学习
```

当前阶段优先实现 **offline active learning**：

```text
已有完整 DFT 数据集
  ↓
每轮主动学习只允许模型看到其中一部分标签
  ↓
被选中的构型才加入训练集
  ↓
用已有数据模拟真实 DFT labeling
```

这样可以避免前期大量真实 DFT 计算开销，先验证主动学习和高性能训练框架是否可行。

---

## 五、系统框架设计

本项目整体分为四层。

### 1. 模型训练层

负责调用 DeepMD-kit 训练单个 DeePMD 模型。

输入：

- DeepMD 配置文件；
- 训练集路径；
- 随机种子；
- GPU id。

输出：

- 模型 checkpoint；
- frozen model；
- 训练日志；
- loss 曲线；
- 能量、力和 virial 误差。

### 2. 多模型并行层

负责训练多个 committee models。

在 2×V100 阶段，计划采用两批训练：

```text
第一批：
GPU 0 → model_seed_001
GPU 1 → model_seed_002

第二批：
GPU 0 → model_seed_003
GPU 1 → model_seed_004
```

在后续 H100 阶段，计划扩展为：

```text
GPU 0 → model_seed_001
GPU 1 → model_seed_002
GPU 2 → model_seed_003
GPU 3 → model_seed_004
```

### 3. 批量推理与不确定性评估层

负责让多个 committee models 对候选构型池进行预测，并计算 model deviation。

候选构型池会被切分为多个 micro-batches：

```text
candidate_pool
  ↓
batch_000
batch_001
batch_002
...
```

多个模型分别预测：

- energy；
- force；
- virial。

然后计算：

- force deviation；
- energy deviation；
- virial deviation；
- combined uncertainty score。

### 4. 主动学习调度层

负责组织完整闭环：

```text
Round 0:
  构造初始训练集

Round 1:
  训练 committee models
  批量推理候选构型
  计算 model deviation
  筛选 top-K 构型
  更新训练集

Round 2:
  重新训练 committee models
  继续推理和筛选

...
```

后续希望逐步形成自动化调度框架，减少手工运行脚本的复杂度。

---

## 六、项目目录结构

当前仓库结构如下：

```text
deepmd-al-hpc/
├── README.md
├── .gitignore
├── configs/
│   ├── active_learning/       # 主动学习配置
│   └── deepmd/                # DeePMD input.json 配置
├── scripts/
│   ├── active_learning/       # 主动学习框架检查脚本
│   ├── data/                  # 数据生成与转换脚本
│   ├── docker/                # Docker 运行脚本
│   ├── eval/                  # freeze、test、误差评估脚本
│   └── train/                 # 单模型训练脚本
├── src/
│   ├── al/                    # 主动学习循环、筛选和调度
│   ├── metrics/               # model deviation 与误差指标
│   └── utils/                 # IO、日志、随机种子等工具
├── experiments/
│   ├── exp_001_env_check/
│   ├── exp_002_framework_check/
│   └── exp_003_single_model_baseline/
└── docs/
```

后续计划补充：

```text
scripts/inference/            # committee models 推理脚本
scripts/train/train_committee_models.sh
slurm/                        # Slurm 作业脚本
results/                      # 小型结果摘要
notebooks/                    # 绘图和分析 notebook
```

---

## 七、运行环境

### 1. Torch 基础开发环境

用于运行主动学习框架 skeleton。

```text
镜像：cuda-torch:cuda11.3-cudnn8-ubuntu18.04-torch2.4
用途：
- 运行 Python 主动学习框架；
- 测试 force model deviation；
- 测试 top-K 构型筛选；
- 验证 2×V100 调度逻辑。
```

进入容器：

```bash
bash scripts/docker/enter_torch_container.sh
```

### 2. DeepMD-kit 训练环境

用于真实 DeePMD 训练、测试和 LAMMPS 调用。

```text
镜像：ghcr.io/deepmodeling/deepmd-kit:master
Python：/opt/deepmd-kit/bin/python
dp：/opt/deepmd-kit/bin/dp
lmp：/opt/deepmd-kit/bin/lmp
```

进入容器：

```bash
bash scripts/docker/enter_deepmd_container.sh
```

已验证命令：

```bash
dp -h
lmp -h
python -c "import deepmd; print('deepmd import ok')"
```

环境验证日志：

```text
experiments/exp_001_env_check/deepmd_env_check.txt
```

---

## 八、Docker 目录同步方式

本项目采用宿主机代码目录挂载到 Docker 容器中的方式运行：

```bash
cd /data/zft

docker run --rm -it \
  --gpus all \
  -v /data/zft:/data/zft \
  -w /data/zft \
  <docker-image> \
  bash
```

实现效果：

```text
宿主机 /data/zft
  ↕ 实时同步
Docker 容器 /data/zft
```

因此可以在宿主机中修改代码，在 Docker 容器中运行代码，并将实验结果同步保存回宿主机目录。

为避免 Docker 生成 root 权限文件，容器启动脚本中默认使用：

```bash
--user $(id -u):$(id -g)
-e PYTHONDONTWRITEBYTECODE=1
```

---

## 九、当前已完成内容

### 1. 第 1 周：环境与仓库初始化

状态：已完成。

已完成任务：

- [x] 初始化 GitHub 仓库；
- [x] 将仓库 clone 到 shared-v100 服务器 `/data/zft`；
- [x] 配置 GitHub SSH 推送；
- [x] 创建项目基础目录结构；
- [x] 编写中文 README；
- [x] 创建 `.gitignore`，避免提交缓存、日志、模型和数据文件；
- [x] 清理并移除已被 Git 跟踪的 `__pycache__` 和 `.pyc` 文件；
- [x] 验证 Docker 可以调用 2×V100 GPU；
- [x] 打通宿主机 `/data/zft` 与 Docker 容器 `/data/zft` 的目录同步；
- [x] 跑通主动学习 framework check；
- [x] 基于模拟 committee forces 验证 force model deviation 与 top-K 构型筛选逻辑；
- [x] 验证 DeepMD-kit Docker 环境；
- [x] 验证 `dp -h`、`lmp -h` 和 `python import deepmd`；
- [x] 完成代码 commit 并 push 到 GitHub。

### 2. 第 2 周：单模型 DeePMD baseline

状态：最小闭环已跑通。

当前已经完成：

- [x] 生成 toy H2 DeepMD 数据集；
- [x] 编写 DeePMD 训练配置 `configs/deepmd/toy_h2_input.json`；
- [x] 运行 `dp train`；
- [x] 运行 `dp freeze`；
- [x] 运行 `dp test`；
- [x] 输出 Energy / Force 误差；
- [x] 保存单模型 baseline 实验结果。

当前 toy H2 数据集只用于流程验证，不用于真实科学结论。

---

## 十、主动学习 skeleton 验证结果

当前已经跑通最小主动学习框架：

```text
模拟 4 个 committee models 的 force prediction
  ↓
计算 force model deviation
  ↓
按照不确定性分数排序
  ↓
选择 top-K 高不确定性构型
  ↓
保存筛选结果
```

运行命令：

```bash
PYTHONPATH=. python scripts/active_learning/run_framework_check.py \
  --config configs/active_learning/framework_check.json
```

输出文件：

```text
experiments/exp_002_framework_check/result.json
```

当前 2×V100 上的 committee model 调度方式为：

```text
第一批：
GPU 0 → model_id 0
GPU 1 → model_id 1

第二批：
GPU 0 → model_id 2
GPU 1 → model_id 3
```

需要说明的是：当前 skeleton 中的 committee forces 来自随机数模拟，主要用于验证主动学习流程、deviation 计算和调度逻辑是否打通。下一阶段将接入真实 DeePMD frozen models 的预测结果。

---

## 十一、单模型 DeePMD baseline 结果

当前使用一个最小 toy H2 数据集进行流程验证。

```text
体系：H2 toy system
元素类型：H
原子数：2
训练集帧数：200
验证集帧数：50

训练集路径：
/data/zft/data/toy_h2/train

验证集路径：
/data/zft/data/toy_h2/valid
```

数据生成脚本：

```text
scripts/data/make_toy_h2_deepmd.py

训练配置：

configs/deepmd/toy_h2_input.json
```

实验目录：

```text
experiments/exp_003_single_model_baseline/
```

当前已经跑通以下流程：

```text
DeepMD 数据集
  ↓
dp train
  ↓
dp freeze
  ↓
dp test
  ↓
Energy / Force error
```

训练结果摘要：

```text
finished training
wall time: 9.411 s
average training time: 0.0075 s/batch
```

测试结果如下：

```text
Energy MAE         : 3.815557e-01 eV
Energy RMSE        : 3.815592e-01 eV
Energy MAE/Natoms  : 1.907779e-01 eV
Energy RMSE/Natoms : 1.907796e-01 eV
Force  MAE         : 2.702034e-02 eV/Å
Force  RMSE        : 7.977260e-02 eV/Å
```

说明：

```text
当前结果只用于验证 DeePMD 训练流程是否打通，不用于评价真实材料或分子体系上的模型精度。
```

---

## 十二、当前已知限制

当前项目仍处于原型验证阶段，主要限制包括：

1. 当前 active learning skeleton 使用随机数模拟 committee forces，尚未接入真实 DeePMD frozen models；
2. 当前只跑通了单模型 toy H2 baseline，尚未完成 4 个不同随机种子的真实 committee models 训练；
3. 当前 toy H2 数据集仅用于流程验证，不能代表真实材料或分子体系上的模型精度；
4. 当前尚未完成真实 candidate pool 的批量推理脚本；
5. 当前尚未完成 offline active learning 多轮闭环；
6. 当前尚未进行 H100 多 GPU 加速实验。

---

## 十三、项目不上传的内容

以下内容不应提交到 GitHub：

- 原始数据集；
- 处理后的大规模数据；
- DeePMD 训练 checkpoint；
- `.pb` 模型文件；
- 大型日志文件；
- 大规模实验结果；
- 轨迹文件；
- LAMMPS 输出文件；
- 大型图片和中间结果；
- Python 缓存文件；
- Docker 临时文件；
- DeePMD 自动生成的 `lcurve.out`、`out.json`、`input_v2_compat.json`。

这些内容应保存在服务器本地或单独的数据目录中。

---

## 十四、阶段性计划

### 第 1 周：环境与仓库初始化

状态：已完成。

### 第 2 周：单模型 DeePMD baseline

状态：最小闭环已跑通。

后续仍需继续完善：

- [ ] 记录训练日志摘要；
- [ ] 整理 `metrics_summary.md`；
- [ ] 检查 `.gitignore` 是否覆盖所有 DeePMD 中间文件；
- [ ] 固化 toy H2 baseline 的运行命令；
- [ ] 补充 toy H2 数据生成脚本或说明数据生成方式。

### 第 3 周：committee models 与 model deviation

下一阶段重点从单模型 baseline 推进到真实 committee models。

计划内容：

- [ ] 新建 `experiments/exp_004_committee_models/`；
- [ ] 新增 `scripts/train/train_committee_models.sh`；
- [ ] 基于 4 个不同随机种子生成 4 份 DeePMD input 配置；
- [ ] 在 2×V100 上分两批训练 4 个 DeePMD 模型；
- [ ] 分别执行 `dp freeze` 得到 4 个 frozen models；
- [ ] 分别执行 `dp test`，汇总 Energy / Force MAE 和 RMSE；
- [ ] 输出 `experiments/exp_004_committee_models/metrics_summary.md`；
- [ ] 为后续真实 committee prediction 和 model deviation 计算做准备。

### 第 4 周：简化 offline active learning 闭环

计划内容：

- [ ] 实现 committee models 对 candidate pool 的批量推理；
- [ ] 将预测结果整理为统一张量格式；
- [ ] 计算真实 force / energy model deviation；
- [ ] 比较 random sampling 和 model deviation sampling；
- [ ] 输出初步 learning curve；
- [ ] 完成第一版技术报告。

---

## 十五、代码与配置检查

在每次重要修改后，建议执行以下检查：

```bash
wc -l \
  scripts/active_learning/run_framework_check.py \
  src/metrics/deviation.py \
  src/al/scheduler.py \
  scripts/train/train_single_model.sh \
  scripts/eval/freeze_model.sh \
  scripts/eval/test_single_model.sh \
  .gitignore \
  configs/deepmd/toy_h2_input.json

python -m py_compile \
  scripts/active_learning/run_framework_check.py \
  src/metrics/deviation.py \
  src/al/scheduler.py \
  src/al/loop.py

bash -n \
  scripts/train/train_single_model.sh \
  scripts/eval/freeze_model.sh \
  scripts/eval/test_single_model.sh

python -m json.tool configs/deepmd/toy_h2_input.json > /tmp/check_toy_h2_input.json
```

---

## 十六、预期研究贡献

本项目后续希望形成以下几个方面的贡献：

1. **面向机器学习势函数主动学习闭环的多模型并行训练框架**  
   将 committee models 作为并行训练单元，提高多轮主动学习中模型训练效率。

2. **面向候选构型池的批量推理与 model deviation 评估方法**  
   通过 micro-batch 和多 GPU 推理，提高大规模候选构型不确定性评估效率。

3. **面向 DFT 标注节省的主动学习采样策略**  
   结合模型不确定性和结构多样性，减少冗余构型标注。

4. **面向 H100 平台的主动学习闭环加速实验**  
   从训练时间、推理吞吐、model deviation 计算时间、多 GPU 加速比和端到端 wall-clock time 等方面评估性能提升。

---

## 十七、最终研究主线

本项目最终希望形成如下研究主线：

```text
DFT / AIMD 精度高但计算昂贵
  ↓
DeePMD / DeepMD-kit 学习第一性原理势能面
  ↓
主动学习减少 DFT 标注冗余
  ↓
committee models 和 model deviation 评估构型不确定性
  ↓
借鉴 Megatron-style 思想进行多模型并行训练与批量推理
  ↓
GPU / H100 加速训练、推理和主动学习闭环
  ↓
形成面向 AI for Science 的高性能主动学习势函数训练框架
```

---

## 十八、当前阶段总结

当前项目已经完成：

```text
环境验证
  ↓
主动学习 skeleton
  ↓
toy H2 单模型 DeePMD train / freeze / test
  ↓
基于模拟 committee forces 的 deviation 和 top-K 筛选
```

下一阶段核心目标是：

```text
4 个真实 DeePMD committee models
  ↓
train / freeze / test
  ↓
真实 candidate pool 推理
  ↓
真实 model deviation
  ↓
offline active learning 最小闭环
```

当前仓库可以视为 **第一阶段工程原型**，但还不能视为完整主动学习框架。项目下一步的关键是从“单模型 baseline + 模拟 skeleton”推进到“真实 4-model committee baseline”。
