# Paper Evidence Checklist

This document tracks the current evidence and pending experiments for the `deepmd-al-hpc` project. Last updated: 2026-05-25.

---

## 1. Claims currently supported (已支持的结论)

1. The repository implements a reproducible toy H2 offline active learning prototype on 2×V100.
2. 4-model committee training, prediction, and model deviation are implemented end-to-end.
3. Uncertainty-based top-K selection consistently selects higher-uncertainty configurations than random sampling.
4. In the uncertainty branch, selected top-K `force_dev_max_mean` decreases monotonically across Round 0–3 (0.441 → 0.170).
5. Random sampling baseline has been completed at scale: 3 seeds × 3 rounds = 9 independent runs.
6. Multi-seed random mean ± std is available for Round 001/002/003.
7. Uncertainty vs random comparison figures are generated across all rounds.
8. 2×V100 model-level parallel training achieves ~1.97× speedup.
9. Per-model training wall time is ~10.9s (1000 steps) for the toy H2 model.
10. Four selection strategies are implemented with full multi-seed multi-round comparison: all four (uncertainty, random, diversity, trust_level) completed seed0/seed1/seed2 Round 001–003 (2026-05-25, 2×V100).
11. Aligned four-strategy comparison table with cross-seed mean ± std uses consistent "remaining candidate-pool" metrics across all strategies.
12. All experiments are reproducible via documented scripts and configs.

---

## 2. Claims partially supported, requiring further validation (部分支持但仍需验证的结论)

1. **Uncertainty sampling reduces remaining candidate-pool uncertainty more effectively than random sampling.**
   - *Evidence:* In Round 001 remaining candidate-pool comparison, uncertainty_round001 (0.126) < all three random seeds (0.355, 0.488, 0.332). Trust_level Round 001 also shows lower remaining uncertainty (0.160).
   - *Gap:* Comparison table currently mixes selected-K uncertainty with remaining candidate-pool uncertainty; needs metric alignment.

2. **Uncertainty-diversity sampling improves structural coverage without severely degrading model quality.**
   - *Evidence:* Multi-seed Round 001–003 shows diversity F_RMSE (2.05e-01, 1.74e-01, 1.76e-01) comparable to random baseline. Selection-level comparison confirms wider structural coverage.
   - *Gap:* Quantitative diversity metrics not yet computed; toy H2 only.

3. **DP-GEN-style trust-level sampling is feasible in the committee model framework.**
   - *Evidence:* Trust-level correctly separates 50-frame pool into 25 accurate / 20 candidate / 5 failed. Multi-seed Round 001–003 F_RMSE (1.35e-01, 1.49e-01, 1.78e-01) is competitive.
   - *Gap:* Only uses force_dev_max; full DP-GEN uses both force and energy deviation.

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
