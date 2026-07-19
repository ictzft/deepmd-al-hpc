# IPDPS 2027 投稿改造路线图

> ⚠️ **venue（2026-07-19 用户确定）**：投 **IPDPS 2027**（CCF-B，submit ~2026-10 / notify ~2027-01-02 / 会议 2027-06 Seattle；HPC 应用+系统对口）。时间窗口 **3 个月（紧）**，大体系（S12）须尽快补。备选 HPDC（CCF-B，notify 3-31，7 月窗口更从容）。CCGrid 是 CCF-C（曾误判，已弃）。下方早期 "CCGrid" 字段为历史，实验规划与 venue 无关仍有效。

> **目标**：CCGrid 2027（CCF-B，Cluster/Cloud/Internet Computing），预计 2026-12 投稿 / 2027-02 录用。
> **本文件**：从"方法学验证仓库"改造为"系统性能论文仓库"的 5 个月执行蓝图。
> **创建**：2026-07-19　**作者**：项目维护者 + Claude

---

## 进度速览与剩余下一步（2026-07-19 更新）

**已完成**：P1（scaling）+ P2（根因+优化）+ P3（DP-GEN 对比）+ P4（出图）+ 补强 #11（统计）/ #12（长训练）/ #13（能耗）/ #14（红利曲线）。论文 4 figure + Table 2 定稿，核心论点（MPS 同 wall、省 ~4× GPU）有统计 + 长训练双验证。

**剩余下一步（按优先级）**：

| 优先级 | 任务 | 怎么做 | 阻塞 |
|---|---|---|---|
| 🔴 P0 | **S12 大体系验证** | 用户补大体系数据集（蛋白质片段 / 水盒子 / 周期性材料）→ 验证 launch-bound 是否仍存在（Nsight SM）+ MPS 是否仍有效（wave vs MPS）→ 出 Fig。**堵"体系小"致命弱点**。 | **等数据集** |
| 🟠 P1 | **S15 论文正文** | abstract / intro / background / method / evaluation 骨架，数据齐全，可直接起草。 | 无 |
| 🟡 P2 | DDP 对照点 | 单模型多卡 DDP 实验，堵"并行维度单一"质疑（论证或实测）。 | 无 |
| 🟡 P2 | 统计加强 | n=3 → 5，补 benzene 体系统计。 | 无 |
| ⚪ future | S13 真跑 dpgen | 需 VASP 环境，论文标 future work。 | VASP |
| ⚪ future | S14 跨架构 | V100/H100 验证，论文标 future work。 | 硬件 |

**投稿前 checklist**：① S12 大体系 ✅ → ② 论文正文 → ③ DDP / 统计加强（可选）→ ④ figure 定稿。

**下一步具体行动**：用户补大体系数据集后 → 我跑 launch-bound + MPS 验证、出 Fig；或用户先让我起草论文正文（S15）。

---

## 0. 一句话定位

把仓库从 **"DeePMD committee 主动学习的方法学验证"** 重新定位为：

> **"委员会式主动学习在多 GPU 系统上的性能表征与优化"**
> *(Performance Characterization & Optimization of Committee-Based Active Learning for ML Potentials on Multi-GPU Systems)*

**主语从「active learning 方法」变成「系统性能」**。方法是背景与实验载体，系统贡献是卖点。这是能否被 CCGrid 接收的关键转换。

---

## 1. 论文新定位与叙事

### 1.1 核心 Problem（motivation）

Committee-based active learning（DP-GEN 范式）是机器学习势函数（MLP）数据生成的**主流方法**：训练 N 个模型 → 对大候选池推理 → 按模型分歧选样 → 重训，如此多轮。但这个闭环在 GPU 上的性能**长期被默认为"并行就高效"，缺乏系统性的性能表征**。

我们已有的初步实测表明，该闭环 **GPU 利用率严重不足**：
- 训练阶段 SM 利用率仅 **23–35%**
- 推理阶段 GPU0 SM avg **51%** / max **74%**，GPU1 大量空闲
- 端到端一轮中训练 47% + 推理 50%，二者合计占 ~97%，但 GPU 都没吃满

**这是一个真实的、未被研究的系统问题**，且我们恰好已经站在"瓶颈已识别、优化未做"的起点上。

### 1.2 Contributions（系统向，3–4 条）

| # | 贡献 | 对应论文位置 |
|---|---|---|
| **C1** | **端到端性能表征**：首个对 committee-based AL 闭环在多 GPU 上的系统性 profiling，量化 train/predict/select 各阶段占比与 GPU 利用率 | §Characterization |
| **C2** | **瓶颈根因分析**：用 Nsight Systems/Compute 定位 SM 利用率低的根因（kernel launch 开销、小 batch、Python/TF runtime、多模型串行占用单卡） | §Root-cause |
| **C3** | **系统优化**：提出并实现若干优化（候选：多模型 GPU 共享/MPS、train–predict 流水线 overlap、bf16），量化端到端吞吐提升 | §Optimization |
| **C4** | **大规模评估**：8×GPU 强/弱 scaling、跨架构（V100/5090）、与 **DP-GEN** 的 head-to-head 系统对比 | §Evaluation |

### 1.3 目标 Track

CCGrid：**HPC / Distributed Computing / Scientific Applications**。本工作天然落在"异构/分布式系统上的科学计算应用优化"，历史上 CCGrid 接收大量此类 AI4Science + 系统论文，**比 IPDPS（偏并行算法/理论）对本工作更对口**。

### 1.4 候选 Title（写作时定稿）

- A. *Performance Characterization and Optimization of Committee-Based Active Learning for Machine-Learning Potentials on Multi-GPU Systems*
- B. *Why Is My GPU Idle? Understanding and Optimizing Committee-Based Active Learning for ML Potentials*

---

## 2. 当前仓库 Gap 分析（系统视角）

对照一篇 CCGrid 系统论文需要的要素：

| 论文要素 | 论文需要 | 当前仓库现状 | Gap |
|---|---|---|---|
| 端到端 profiling | 各阶段耗时 + GPU util | V100 wall-time ✓、部分 util ✓ | 🟡 需补多卡 util 曲线 |
| **多 GPU scaling** | 1/2/4/8 强+弱 scaling | ❌ 仅 2-GPU model-parallel（1.97×，trivial） | 🔴 **核心缺失** |
| 瓶颈根因 | Nsight trace + roofline | ❌ 无 | 🔴 完全缺失 |
| 系统优化 | ≥1 优化 + 量化提升 | ❌ 无 | 🔴 完全缺失 |
| Baseline 对比 | vs DP-GEN | ❌ 无 | 🔴 完全缺失 |
| 跨架构 | V100 / 5090 /（H100） | V100 ✓、5090 仅单卡 smoke test | 🟡 需 5090 多卡 |
| 大规模应用 | 更大体系 | toy H2 + 9–12 原子小分子 | 🟡 需 1–2 个大体系 |
| 统计显著性 | 多 seed + CI | 3-seed 有，但随机方差大 | 🟡 需加强 |
| 可复现 artifact | 脚本+配置+数据+文档 | ✅ 很好 | 🟢 **优势，保留** |

**一句话结论**：方法学与工程基础扎实（🟢），但**系统贡献的四块核心 —— scaling / 根因 / 优化 / baseline —— 全部缺失（🔴）**。这四块就是接下来 5 个月的主战场。

---

## 3. 目标仓库结构规划

### 3.1 当前结构（简）

```
src/al/        loop, selector(4策略), scheduler(2-GPU 分卡轮询)
src/metrics/   deviation (force_dev_max 等)
src/utils/     io, logger, seed
scripts/       data, train, inference, selection, analysis, eval, profiling, docker, ...
configs/       deepmd, datasets, active_learning
experiments/   toy_h2 + rmd17 各 round 产物 + profiling/
data/          toy_h2 + rmd17/{ethanol,benzene}
```

### 3.2 需要新增的模块（建议路径）

```
src/scheduling/
  concurrent_runner.py    # 8 卡可配置并发调度（替换 scheduler.py 的 2 卡轮询）
  pipeline_overlap.py     # train/predict/select 阶段流水线重叠
  mps_manager.py          # 多模型 GPU 共享（MPS / CUDA stream 时间片）

scripts/scaling/
  run_strong_scaling.sh   # 1/2/4/8 GPU 强 scaling
  run_weak_scaling.sh     # 弱 scaling
  collect_scaling.py      # 汇总 wall-time / util → CSV

scripts/profiling/        # 扩展现有目录
  run_nsight_train.sh     # nsys / ncu trace 采集
  roofline_analysis.py

scripts/baselines/
  dpgen/                  # DP-GEN 集成 + 相同任务 head-to-head
  run_dpgen_comparison.sh

experiments/
  scaling/                # 强弱 scaling 产物
  optimization/           # 优化前后对比、ablation
  dpgen_comparison/       # vs DP-GEN
  cross_arch/             # V100 vs 5090（vs H100）

docs/
  ccgrid_2027_roadmap.md  # 本文件
  paper/                  # 论文草稿、related_work.md、figures 规划
```

**改动原则**：**扩展而非重写**。现有 `selector.py`（4 策略）、`predict_committee_models.py`、data pipeline 全部保留，作为"被优化的对象"和实验载体。

---

## 4. 实验计划（论文 Figure / Table 映射）

每个论文产出 → 对应实验 → 依赖 → 现状。

| 论文产出 | 实验 | 依赖（代码/数据） | 现状 |
|---|---|---|---|
| **Fig.1** 端到端 round 时间分解 | profile train/predict/select 占比 | 现有 profiling + 新 collect | 🟡 V100 有，需 5090 + 优化后 |
| **Fig.2** GPU 利用率曲线（**motivation 钩子**） | nvidia-smi dmon 连续记录 | 现有 + 多卡扩展 | 🟡 单卡有，需多卡 + 前后对比 |
| **Fig.3** 强 scaling（1→8 GPU） | 并发训 1/2/4/8 卡，测 throughput | `concurrent_runner.py` | 🔴 新建 |
| **Fig.4** 弱 scaling | 固定每卡负载，增卡 | 同上 | 🔴 新建 |
| **Fig.5** 瓶颈分解（Nsight/roofline） | nsys/ncu trace | `run_nsight_*.sh` | 🔴 新建 |
| **Fig.6** 优化前后吞吐 + ablation | 各优化 on/off | `pipeline_overlap` / `mps_manager` | 🔴 新建 |
| **Tab.1** 跨架构对比 | V100/5090 同配置 | `cross_arch/` | 🟡 部分有 |
| **Tab.2** vs DP-GEN 系统对比 | 同任务跑 DP-GEN | `baselines/dpgen/` | 🔴 新建 |
| **Fig.7** 大体系端到端 AL 效率 | 更大分子/材料 | 新数据 | 🟡 需新体系 |
| **Tab.3** 方法侧（4 策略 RMSE，作 appendix） | 现有实验 | 已完成 | 🟢 已有 |

> **Fig.2 是这篇论文的"钩子"**：GPU 利用率 23–51% 的实测曲线，审稿人第一眼就明白"问题真实存在"。务必把它做成一张有冲击力的图。

---

## 5. 5 个月时间线（2026-07 → 2026-12 投稿）

对齐 CCGrid 2027 预计 deadline（abstract/paper ~2026-12 中，需官网核实）。

| 阶段 | 时间 | 里程碑 | 产出 |
|---|---|---|---|
| **P1 基础设施 + 强 scaling** | 07 下 – 09 上 | `concurrent_runner.py` 上线；1/2/4/8 GPU 强+弱 scaling | Fig.3, Fig.4 |
| **P2 瓶颈 + 优化** | 09 下 – 10 上 | Nsight 根因；≥1 优化实现 + 量化 | Fig.5, Fig.6（C2/C3） |
| **P3 baseline + 大体系** | 10 下 – 11 上 | DP-GEN 对比；1–2 个更大体系 | Tab.2, Fig.7（C4） |
| **P4 写作 + 统计** | 11 下 – 12 上 | 论文初稿；多 seed + CI；跨架构表 | 全文 + Tab.1 |
| **投稿** | ~12 中 | CCGrid 2027 abstract + paper | 提交 |

**关键依赖**：P1 的 `concurrent_runner.py` 是一切的基础，**第一周就启动**（见第 7 节）。

---

## 6. 风险与退路

| 风险 | 应对 |
|---|---|
| CCGrid 2027 CFP 日期变动 / 比 12 月更早 | 密切关注官网；**HPDC 2027**（submit ~2027-02，notify ~2027-03-31）作退路 |
| 8 卡 scaling 效率掉得厉害 | **反而是 motivation 强化**（说明问题严重），但要能把"优化后效率恢复"讲成正面贡献 |
| 优化效果不显著（<10%） | 至少保证 1 个有明显提升的优化 —— **优先试 MPS 多模型共享**（潜力最大：当前一卡只跑一个模型，闲置严重） |
| DP-GEN 集成工作量大 | P3 才正式做，但 **P1 同期就调研 DP-GEN 部署**，避免后期卡壳 |
| 5090 Blackwell 兼容性（TF 后端不可用） | 已知只能 PyTorch 后端，镜像 `deepmd-5090` 已就绪，纳入 Tab.1 说明 |
| random 方差大削弱方法侧结论 | 方法侧降级为 appendix（Tab.3），主轴是系统，规避此风险 |

---

## 7. 现有可复用资产清单（不要重造轮子）

**直接复用**：
- `src/al/selector.py`（4 策略）→ 方法学 appendix + 选样步骤
- `scripts/inference/predict_committee_models.py` → 推理基准的"被优化对象"
- `scripts/data/` 全套 → 数据准备（rMD17 转换、合并、剩余候选池）
- `experiments/profiling/` 现有 V100 数据 → Fig.1 / Tab.1 的 V100 列
- `data/rmd17/{ethanol,benzene}` → 实验载体（已就位）
- 全部文档体系 + 诚实的 claim boundary → 可复现 artifact 优势（CCGrid 加分项）

**需要改造**：
- `src/al/scheduler.py`（2 卡轮询 `model_id % n_gpus`）→ 升级为 `src/scheduling/concurrent_runner.py`（可配置并发 + 真正异步）
- `scripts/train/train_committee_models.sh`（硬编码 2 GPU 两批）→ 参数化 1/2/4/8

---

## 8. 立即行动（第一周）

1. **改 `scheduler.py` → `concurrent_runner.py`**：支持 `--n-gpus {1,2,4,8}` 可配置并发，8 个 committee 模型可同时铺到 8 卡（或 4 模型 × 数据并行）。
2. **写 `scripts/scaling/run_strong_scaling.sh` + `collect_scaling.py`**：自动跑 1/2/4/8 并产出统一 wall-time / GPU-util CSV。
3. **在 8×5090 上跑第一组强 scaling**，拿到 Fig.3 的初始数据。
4. **并行调研 DP-GEN**：部署方式、如何在相同 rMD17 任务上跑、需要记录哪些系统指标。

> 这一步验证"系统方向是否走得通"。如果 8 卡效率掉得厉害 —— 恭喜，那就是你 motivation 的第一个证据。

---

## 附录 A：CCGrid 2027 关键日期（预测，须官网核实）

基于 CCGrid 2026 实测（submit 2025-12-15 / notify 2026-02-10 / 会议 5 月）推算：

- Abstract / Paper deadline：~2026-12 中
- Notification：~2027-02
- Conference：~2027-05

退路：HPDC 2027（submit ~2027-02 / notify ~2027-03-31，CCF-B）。

## 附录 B：CCF 等级与 Track 参考

- CCGrid：CCF-B（体系结构/并行与分布/存储系统）
- HPDC：CCF-B（同上）
- ICDCS：CCF-B（网络/系统）
- ICPP：CCF-B（体系结构/并行与分布/存储系统）

<!-- ccgrid_2027_roadmap.md created 2026-07-19 -->
