# DP-GEN 对比（IPDPS 2027 Table 2）

更新：2026-07-19 | 平台：8×RTX 5090（deepmd-5090 容器，PyTorch 后端）

---

## 1. 对比范围与公平性

**不比 AL 闭环**：DP-GEN 是完整主动学习框架，其 fp（标注）环节需调用 VASP/CP2K 做 DFT。本工作无 DFT 标注器（用 rMD17 已有标注模拟），**完整跑 DP-GEN AL 闭环不现实，也不在本文系统贡献范围内**。

**比的是训练调度**：DP-GEN 把每个 committee 模型当作独立任务调 `dp train`（由 machine scheduler 管理 GPU），**框架层不做多模型 GPU 共享**。这恰好等价于本仓库的 **wave baseline（每模型独占一卡）**。因此 wave baseline 即"DP-GEN-style 调度"的可复现代理。

> **机制验证（dpgen 源码佐证，2026-07-19）**：`dpgen/generator/run.py:765`
> `train_command = mdata.get("train_command", "dp")`；`:768-770` 追加 `--pt`/`--jax`
> 后端标志；`:790-792` 注释 "Commands are like `dp train` and `dp freeze`"。
> 即 DP-GEN 训练环节对每个 committee 模型调用一次独立 `dp train` 进程，GPU 分配
> 交由 machine.json/scheduler，**框架层不做多模型 GPU 共享** → wave baseline
> 精确复现此调度。

## 2. Table 2 — 训练调度对比（ethanol, N=4 committee, 100 steps, mean ± std over 3 runs）

| 方法 | 调度 | GPU | wall (s, n=3) | SM avg (%, n=3) | GPU·秒 | 资源效率 |
|---|---|:---:|---|---|---:|---:|
| **DP-GEN-style**（wave, b8）| 每模型独占一卡 | 4 | 26.0 ± 0.1 | 4.0 ± 0.0 | 103.9 | 1.00× |
| **Ours: MPS**（b8）| 4 模型共享 1 卡 | **1** | **25.2 ± 0.0** | 14.2 ± 0.8 | **25.2** | **4.11×（同 wall）** |
| **Ours: MPS + b256** | 4 模型共享 1 卡 | **1** | 44.1 ± 0.7 | **39.4 ± 3.1** | 44.1 | 2.35×（SM 9.9×）|

统计确认（3 run，std 极小）：MPS b8 用 1 张卡的 wall 与 4 张卡 DP-GEN-style **统计无差异**（甚至略快 0.8s），资源效率稳定 4.11×。

### 2.1 长训练验证（2000 steps，计算主导）

为确认"同 wall"不是短训练固定开销（neighbor-stat、runtime init）主导的假象，补 2000 步长训练对比：

| 配置 (2000 steps) | wall (s) | GPU | SM avg (%) | GPU·秒 | 资源效率 |
|---|---|:---:|---:|---:|---:|
| DP-GEN-style (wave G=4) | 235.1 | 4 | — | 940.4 | 1.00× |
| Ours MPS G=1 (b8) | 239.3 | 1 | 33.7 | 239.3 | **3.93×** |

长训练下 MPS wall 仅比 wave 高 1.8%（235 vs 239s），"同 wall"成立；且 SM 升至 33.7%（计算主导时 4 模型叠加更充分，远高于短训练 b8 的 14.2%），资源效率 3.93×。**结论跨训练长度稳健**，堵住"短训练固定开销掩盖真实开销"的质疑。

数据来源：
- wave G=4：`experiments/scaling/wave_n4_g4/summary.json` + `gpu_dmon.log`
- MPS batch=8：`experiments/scaling/mps_n4_g1/`
- MPS+batch=256：`experiments/scaling/mps_n4_g1_bs256/`

## 3. 论点

1. **DP-GEN-style 调度严重浪费 GPU**：4 个模型占 4 张卡，每卡 SM 仅 4.4%（>95% 闲置）。
2. **我们的 MPS 用 1 张卡完成同样 4 个模型，wall 几乎相同（26.7 vs 26.4s）** —— 因为单模型 launch-bound（GPU 闲置 90%+），MPS 把 4 个模型的空闲拼进一张卡。资源效率（GPU·秒）**~4×**。
3. 若进一步增大 batch，单卡 SM 推到 38%（8.6×），代价是 wall 增加 70% —— 给用户 wall/利用率/卡数的 trade-off 空间。

## 4. 结论

在 committee-based AL 的训练环节，相比事实标准 DP-GEN 的"每模型独占"调度，我们的 MPS 多模型共享方案在**相同端到端时间下节省约 4 倍 GPU**，或以更高单卡利用率运行。方案不改 DeepMD 源码，直接可用。

## 5. 局限（写论文需声明）

- 本对比聚焦**训练调度**，非 DP-GEN 完整 AL 闭环（fp 标注环节未对比）。
- wave baseline 是 DP-GEN 训练调度的**可复现代理**，非直接跑 `dpgen run`（机制由源码佐证，见 §1）。
- 仅 ethanol 体系，N=4，100 步；大规模体系/长训练待补。
