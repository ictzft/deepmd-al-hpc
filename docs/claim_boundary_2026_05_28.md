# Claim Boundary 与待完成工作

更新日期：2026-05-28

## 当前已支持的结论

1. 本仓库实现了一个面向 DeePMD committee models 的 offline active learning 原型系统。
2. 该工作流已在 toy H2 和两个 rMD17 分子体系（ethanol 和 benzene）上完成验证。
3. rMD17 benzene 上已完成 uncertainty selection Round 000–003 和 random baseline seed0/1/2 Round 001–003，并完成 independent test 评估。
4. rMD17 ethanol 上已完成 uncertainty 和 random baseline，以及 independent test 和短时 MD stability 验证。
5. V100 实验表明该流水线能够完成 committee training、prediction、selection、dataset update 和 retraining 闭环。

## 不应过度声称的结论

1. 当前工作流是 offline active learning，不是完整的 online DFT/AIMD labeling 系统。
2. H100 和多 GPU scaling 实验尚未完成。
3. 10K NVE stability 仅支持短时稳定性验证。100K+ 解离表明当前势函数尚不足以支撑长时间 MD。
4. 当前证据不能证明 uncertainty 在所有情况下统计显著优于 diversity 或 trust-level。
5. 策略级别的结论应谨慎表述：uncertainty 通常优于 random，但 active 策略之间的排名仍不确定。
6. V100 profiling 当前提供 baseline 耗时数据；GPU 利用率、throughput、端到端加速比和 scaling 分析仍需补充。

## 论文安全表述

当前系统应描述为面向 DeePMD committee-model training 和 selection 的 offline active learning 原型。使用已有 rMD17 标注数据模拟昂贵的 DFT labeling 步骤。系统已在 toy H2 和两个 rMD17 分子体系（ethanol 和 benzene）上完成评估。当前结果支持闭环工作流的可行性，并表明 uncertainty-based selection 在多个 independent test 设置中优于 random sampling。但 online DFT labeling、H100 scaling 和长时间 MD stability 仍属于后续工作。
