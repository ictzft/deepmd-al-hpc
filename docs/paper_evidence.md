# Paper Evidence Checklist

This document tracks the current evidence and pending experiments for the `deepmd-al-hpc` project. Last updated: 2026-05-26.

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
12. V100 training wall-time profiling: 132 models, mean=11.0s, 2×V100 parallel ~22s/round.
13. Structural diversity analysis: diversity (FPS) achieves 3.1x greater structural spread vs uncertainty top-K in toy H2.
14. rMD17 ethanol uncertainty branch Round 0–3 active learning loop completed; Force RMSE decreases monotonically on both validation (0.374→0.354) and independent test (0.344→0.327 eV/Å).
15. rMD17 ethanol random baseline (3 seeds × 3 rounds): uncertainty Force RMSE monotonically improves while random shows large variance (Round 3: 0.354 vs 0.607 ± 0.385 eV/Å). In this single-system experiment, uncertainty shows a more stable improvement trend than random.
16. rMD17 ethanol four-strategy comparison: all three active strategies (uncertainty/diversity/trust_level) show similar Force RMSE (0.354-0.362 eV/Å, within 1σ), all significantly better than random (0.607 eV/Å). Consistent with toy H2 finding.
17. MD stability test (NVE 10K): all models stable with drift ~0.035 eV/ps; 100K+ dissociation indicates current Force RMSE ~0.35 eV/Å is insufficient for high-T MD.
18. All experiments are reproducible via documented scripts and configs.

---

## 2. Claims partially supported, requiring further validation (部分支持但仍需验证的结论)

1. **Uncertainty sampling reduces remaining candidate-pool uncertainty more effectively than random sampling.**
   - *Evidence:* In Round 001 remaining candidate-pool comparison, uncertainty_round001 (0.126) < all three random seeds (0.355, 0.488, 0.332). Trust_level Round 001 also shows lower remaining uncertainty (0.160).
   - *Gap:* Aligned comparison now uses consistent metric (remaining candidate-pool) for all four strategies. Differences in Force RMSE are within 1σ on toy H2; real datasets needed for statistical significance.

2. **Uncertainty-diversity sampling improves structural coverage without severely degrading model quality.**
   - *Evidence:* Multi-seed Round 001–003 shows diversity F_RMSE (2.05e-01, 1.74e-01, 1.76e-01) comparable to random baseline. Selection-level comparison confirms wider structural coverage.
   - *Gap:* Selection-level analysis shows 3.1x structural spread improvement; multi-round diversity analysis not yet done; toy H2 only (H-H distance).

3. **DP-GEN-style trust-level sampling is feasible in the committee model framework.**
   - *Evidence:* Trust-level correctly separates 50-frame pool into 25 accurate / 20 candidate / 5 failed. Multi-seed Round 001–003 F_RMSE (1.35e-01, 1.49e-01, 1.78e-01) is competitive.
   - *Gap:* Only uses force_dev_max; full DP-GEN uses both force and energy deviation.

---

## 3. Claims not yet supported (尚不支持的结论)

1. The method significantly outperforms random sampling across multiple datasets or multiple rounds (single-system rMD17 ethanol evidence available, multi-system needed).
2. The method is validated on multiple real DFT/AIMD systems (only rMD17 ethanol tested).
3. The framework has demonstrated H100 or multi-node scaling.
4. The framework has been validated through high-temperature MD stability tests (10K NVE stable; 100K+ dissociation).
5. The framework is production-ready or meets CCF-B submission standards.
6. The active learning workflow reduces total DFT labeling cost compared to random or systematic sampling on realistic systems (offline AL simulates labeling; online validation needed).

---

## 4. Current limitations (当前限制)

1. Toy H2 dataset (2 atoms, 250 frames) — cannot represent realistic material systems.
2. Valid set also serves as candidate pool — no independent test set.
3. rMD17 ethanol four-strategy comparison completed; all three active strategies outperform random (differences within 1σ).
4. Uncertainty-diversity (FPS) and trust-level (DP-GEN-style) implemented and validated on both toy H2 and rMD17 ethanol.
5. No H100 or multi-node scaling experiments.
6. No MD stability verification.
7. Full GPU utilization curves not yet recorded (representative sample available).
8. Prediction and I/O profiling measured for representative cases; not recorded for all 36 rounds.

---

## 5. Next experiments (下一步实验)

1. **Align comparison metrics** — DONE (aligned_comparison.csv uses consistent remaining candidate-pool metric).
2. **Add GPU monitoring curves** — Run nvidia-smi dmon during one complete round (representative sample done, full curves pending).
3. **Add uncertainty-diversity selection** — DONE (FPS + pairwise-distance descriptor, 3.1x structural spread).
4. **Move to real DFT/AIMD dataset** — rMD17 ethanol four-strategy multi-seed multi-round completed; independent test done; MD stability done. Next: multi-system validation.
5. **Run H100 / multi-GPU scaling** — Benchmark training throughput and end-to-end round time.
6. **MD stability tests** — Validate committee model quality through MD trajectory stability.
