# deepmd-al-hpc

本项目是一个面向 **第一性原理机器学习势函数主动学习闭环** 的多模型并行与高性能训练原型系统。

项目不直接训练大语言模型，也不是直接复现 Megatron-LM，而是借鉴 Megatron 系列大规模训练框架中的 **多 GPU 并行、micro-batch、混合精度、批量推理、流水线调度和分布式实验管理思想**，将这些高性能训练思想迁移到 DeePMD / DeepMD-kit 机器学习势函数主动学习场景中。

当前阶段主要在 **2×V100 GPU** 平台上跑通原型流程，包括单模型训练、committee models 并行训练、model deviation 计算、候选构型筛选和简化主动学习闭环；后续计划迁移到 H100 多 GPU 平台，开展更完整的训练加速、推理加速和主动学习闭环性能优化实验。

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

其中，当前阶段优先实现 offline active learning：

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

本项目整体分为四层：

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

---

### 2. 多模型并行层

负责训练多个 committee models。

在 2×V100 阶段，采用两批训练：

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

---

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

---

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

## 六、当前阶段

当前处于：

```text
第 1 周：环境与仓库初始化
```

本周目标：

- [ ] 初始化 GitHub 仓库；
- [ ] 创建项目目录结构；
- [ ] 编写中文 README；
- [ ] 编写 `.gitignore`；
- [ ] 搭建 DeepMD-kit 环境；
- [ ] 检查 2×V100 是否可用；
- [ ] 验证 `dp`、`lmp` 和 Python 环境；
- [ ] 保存第一次环境检查日志；
- [ ] 完成首次 commit 并 push 到 GitHub。

---

## 七、项目目录结构

```text
deepmd-al-hpc/
├── README.md
├── .gitignore
├── environment/              # 环境安装与配置说明
├── configs/                  # DeepMD、主动学习和体系配置文件
│   ├── deepmd/
│   ├── active_learning/
│   └── systems/
├── scripts/                  # 数据处理、训练、推理、主动学习和性能分析脚本
│   ├── data/
│   ├── train/
│   ├── inference/
│   ├── active_learning/
│   └── profiling/
├── slurm/                    # Slurm 作业提交脚本
├── src/                      # 自己实现的 Python 源代码
│   ├── al/
│   ├── metrics/
│   └── utils/
├── experiments/              # 实验记录
├── notebooks/                # 绘图和分析 notebook
├── docs/                     # 项目计划、实验日志、论文想法
├── results/                  # 小型结果摘要
└── logs/                     # 本地日志，不上传 GitHub
```

---

## 八、初始硬件平台

当前前期开发平台：

```text
GPU：2×V100
用途：
- DeePMD baseline 训练；
- committee model 原型验证；
- offline active learning；
- model deviation 计算；
- 初步 profiling。
```

后续正式实验计划迁移到：

```text
4×A100 / 4×H100 或 8×H100
```

用于：

- 多模型并行训练；
- 大规模候选构型批量推理；
- 多 GPU scaling；
- 主动学习闭环端到端加速；
- CCF-B 标准实验验证。

---

## 九、项目不上传的内容

以下内容不应提交到 GitHub：

- 原始数据集；
- 处理后的大规模数据；
- DeePMD 训练 checkpoint；
- `.pb` 模型文件；
- 大型日志文件；
- 大规模实验结果；
- 轨迹文件；
- LAMMPS 输出文件；
- 大型图片和中间结果。

这些内容应保存在服务器本地或单独的数据目录中。

---

## 十、阶段性计划

### 第 1 周：环境与仓库初始化

- 建立 GitHub 仓库；
- 搭建项目目录；
- 安装 DeepMD-kit；
- 检查 2×V100；
- 保存环境检查日志。

### 第 2 周：单模型 DeePMD baseline

- 选择一个小体系数据集；
- 准备 DeepMD 数据格式；
- 训练单个 DeePMD 模型；
- 评估 Energy MAE 和 Force RMSE；
- 记录训练时间和 GPU 利用率。

### 第 3 周：committee models 与 model deviation

- 训练 4 个不同随机种子的模型；
- 实现 committee prediction；
- 实现 force model deviation 计算；
- 绘制 model deviation 分布。

### 第 4 周：简化主动学习闭环

- 实现 offline active learning；
- 比较 random sampling 和 model deviation sampling；
- 输出初步 learning curve；
- 完成第一版技术报告。

---

## 十一、预期研究贡献

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

## 十二、最终研究主线

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

## 十三、当前进展记录

截至当前阶段，本项目已经完成了主动学习基础框架的最小原型验证。

### 1. Docker 运行环境验证

当前项目采用宿主机代码目录挂载到 Docker 容器中的方式运行：

```bash
cd /data/zft

docker run --rm -it \
  --gpus all \
  -v /data/zft:/data/zft \
  -w /data/zft \
  cuda-torch:cuda11.3-cudnn8-ubuntu18.04-torch2.4 \
  bash
