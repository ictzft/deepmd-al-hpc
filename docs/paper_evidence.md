# Paper Evidence Checklist

This document tracks the current evidence and pending experiments for the `deepmd-al-hpc` project. Last updated: 2026-05-25.

---

## 1. Claims currently supported (已支持的结论)

1. The repository implements a reproducible toy H2 offline active learning prototype on 2×V100.
2. 4-model committee training, prediction, and model deviation (`force_dev_max`, `force_dev_mean`, `energy_dev`) are implemented end-to-end.
3. Uncertainty-based top-K selection consistently selects higher-uncertainty configurations than random sampling.
4. In the uncertainty branch, selected top-K `force_dev_max_mean` decreases monotonically across Round 0–3 (0.441 → 0.170).
5. Random sampling baseline has been completed at scale: 3 seeds × 3 rounds = 9 independent committee training runs + predictions.
6. Multi-seed random mean ± std is available for Round 001/002/003.
7. Uncertainty vs random comparison figures (Force RMSE, Energy RMSE, candidate force_dev, dataset size) are generated across all rounds.
8. 2×V100 model-level parallel training achieves ~1.97× speedup (near-linear) over serial execution.
9. Per-model training wall time is ~10.9s (1000 steps, 8.7ms/batch) for the toy H2 model.
10. All experiments are reproducible via documented scripts and configs (see `docs/random_baseline_execution_checklist.md`).

---

## 2. Claims partially supported, requiring further validation (部分支持但仍需验证的结论)

1. **Uncertainty sampling reduces remaining candidate-pool uncertainty more effectively than random sampling.**
   - *Evidence:* In Round 001 remaining candidate-pool comparison, uncertainty_round001 (0.126) < all three random seeds (0.355, 0.488, 0.332). However, the comparison table mixes "selected top-K uncertainty" for the uncertainty branch with "remaining candidate-pool uncertainty" for the random branch.
   - *Gap:* Fair comparison requires both branches to use the same metric (e.g., remaining candidate-pool force_dev_max). The current data supports the trend but needs metric alignment.

2. **Multi-round active learning reduces validation Force RMSE.**
   - *Evidence:* Uncertainty branch Force RMSE fluctuates (0.182 → 0.162 → 0.194 → 0.174). Random mean Force RMSE: 0.211 → 0.196 → 0.189.
   - *Gap:* No strictly monotonic decrease across rounds. The toy H2 dataset is too small for robust statistical conclusions.

3. **The committee model approach yields reliable model deviation estimates.**
   - *Evidence:* Committee models exhibit high variance in Energy RMSE (e.g., seed0 R002: 0.638–3.649 eV across 4 models). This variance is expected for small datasets but also limits the reliability of deviation-based selection.
   - *Gap:* Need analysis of how committee size and model variance affect selection quality.

---

## 3. Claims not yet supported (尚不支持的结论)

1. The method significantly outperforms random sampling across multiple datasets or multiple rounds.
2. The method is validated on real DFT/AIMD datasets.
3. The framework has demonstrated H100 or multi-node scaling.
4. The framework has been validated through MD stability tests.
5. Uncertainty-diversity selection provides benefit over pure uncertainty top-K.
6. The framework is production-ready or meets CCF-B submission standards.
7. The active learning workflow reduces total DFT labeling cost compared to random or systematic sampling on realistic systems.

---

## 4. Current limitations (当前限制)

1. Toy H2 dataset (2 atoms, 250 frames) — cannot represent realistic material systems.
2. Valid set also serves as candidate pool — no independent test set.
3. No real DFT/AIMD dataset has been tested.
4. No structural diversity constraint in selection (pure uncertainty top-K).
5. No H100 or multi-node scaling experiments.
6. No MD stability verification.
7. GPU utilization and memory profiling not systematically recorded.
8. Prediction and I/O profiling are estimates, not precise measurements.

---

## 5. Next experiments (下一步实验)

1. **Align comparison metrics** — Use the same metric (remaining candidate-pool force_dev_max) for both uncertainty and random branches in all comparison tables and figures.
2. **Add GPU monitoring** — Run nvidia-smi dmon during one complete round to record GPU utilization and memory.
3. **Add uncertainty-diversity selection** — Implement structural diversity constraints.
4. **Move to real DFT/AIMD dataset** — Convert and validate on realistic first-principles data.
5. **Run H100 / multi-GPU scaling** — Benchmark training throughput and end-to-end round time.
6. **MD stability tests** — Validate committee model quality through MD trajectory stability.
