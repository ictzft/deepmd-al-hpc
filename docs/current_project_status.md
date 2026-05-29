# Current Project Status

Date: 2026-05-28 | Host: shared-v100 (2×Tesla V100-16GB) | Branch: main

---

## 1. One-sentence Summary

toy H2 四策略 multi-seed multi-round 已完成；V100 profiling baseline 已完成。rMD17 ethanol 四策略（uncertainty / random / diversity / trust_level）multi-seed multi-round 已完成，independent test evaluation 已完成，10K NVE MD stability 已完成。rMD17 benzene uncertainty branch 和 random baseline 已完成，independent test 已完成（diversity / trust_level baseline 待补充）。H100 scaling 未开始。

---

## 2. Completed Components

| # | Component | Rounds / Seeds | Status |
|---|---|---|---|
| 1 | DeepMD-kit Docker environment | — | done |
| 2 | toy H2 data generation | 250 frames | done |
| 3 | Single-model DeePMD baseline | — | done |
| 4 | 4-model committee training + prediction | — | done |
| 5 | force_dev_max / force_dev_mean / energy_dev | — | done |
| 6 | Uncertainty top-K offline active learning | Round 0–3 | done |
| 7 | Random sampling selection-level baseline | Round 0–1, seed0/1/2 | done |
| 8 | Random retraining baseline | Round 001–003, seed0/1/2 | done |
| 9 | Uncertainty multi-seed retraining | Round 001–003, seed0/1/2 | done |
| 10 | Diversity multi-seed retraining | Round 001–003, seed0/1/2 | done |
| 11 | Trust-level multi-seed retraining | Round 001–003, seed0/1/2 | done |
| 12 | Multi-seed mean ± std | all 4 strategies | done |
| 13 | Aligned four-strategy comparison | consistent metric | done |
| 14 | Comparison learning curve figures | 4 SVG | done |
| 15 | V100 training wall-time profiling | 132 train.log | done |
| 16 | V100 GPU utilization sample | representative | done |
| 17 | Diversity descriptor analysis | H-H bond length, 3 strategies | done |
| 18 | Profiling scripts + summary | profile_command, parse, summarize | done |
| 19 | rMD17 ethanol uncertainty branch | Round 0–3 training + prediction + summary + independent test done | done |
| 20 | rMD17 ethanol independent test | 10000-frame test set, Force RMSE 0.344→0.327 eV/Å (monotonic) | done |
| 21 | rMD17 ethanol random baseline | 3 seeds × 3 rounds (36 models), uncertainty clearly better | done |
| 22 | rMD17 ethanol MD stability | NVE 10K stable (drift ~0.035 eV/ps), 100K+ dissociates | done |
| 23 | rMD17 ethanol diversity baseline | 3 seeds × 3 rounds (36 models), F_RMSE=0.3555 | done |
| 24 | rMD17 ethanol trust_level baseline | 3 seeds × 3 rounds (36 models), F_RMSE=0.3616 | done |
| 25 | rMD17 benzene uncertainty branch | Round 000–003, 4 rounds × 4 models, top-1000 per round | done |
| 26 | rMD17 benzene random baseline | seed0/1/2 Round 001–003 | done |
| 27 | rMD17 benzene independent test | 10000-frame test set | done |
| 28 | rMD17 benzene diversity baseline | — | pending |
| 29 | rMD17 benzene trust_level baseline | — | pending |
| 30 | rMD17 benzene MD stability | — | pending |

---

## 3. Available Data and Results

### 3.1 Uncertainty branch (al_rounds_summary.csv)

| Round | Train | Candidate | Force RMSE | force_dev_max (selected top-K) |
|---:|---:|---:|---:|---:|
| 0 | 200 | 50 | 0.182 | 0.441 |
| 1 | 210 | 40 | 0.162 | 0.269 |
| 2 | 220 | 30 | 0.194 | 0.187 |
| 3 | 230 | 20 | 0.174 | 0.170 |

### 3.2 Four-Strategy Aligned Force RMSE (3-seed mean ± std)

| Strategy | R1 | R2 | R3 |
|---|---:|---:|---:|
| uncertainty | 1.51e-01 ± 0.025 | 2.13e-01 ± 0.024 | 1.96e-01 ± 0.024 |
| random | 2.11e-01 ± 0.055 | 1.96e-01 ± 0.016 | 1.89e-01 ± 0.048 |
| diversity | 2.05e-01 ± 0.058 | 1.74e-01 ± 0.009 | 1.76e-01 ± 0.041 |
| trust_level | 1.35e-01 ± 0.028 | 1.49e-01 ± 0.023 | 1.78e-01 ± 0.006 |

See `experiments/baselines/aligned_comparison.md` for full aligned comparison.

### 3.2b rMD17 Ethanol Four-Strategy Force RMSE (3-seed mean ± std, Round 3)

| Strategy | Force RMSE | Std |
|---|---:|---:|
| uncertainty | 3.537e-01 | 2.472e-02 |
| diversity | 3.555e-01 | 1.434e-02 |
| trust_level | 3.616e-01 | 1.660e-02 |
| random | 6.067e-01 | 6.826e-01 |

See `experiments/rmd17_ethanol_summary/four_strategy_comparison.csv`.

### 3.2c rMD17 Benzene Uncertainty Branch (Round 000–003)

| Round | Train | Candidate | Selection | Status |
|---:|---:|---:|---|---|
| 000 | 1000 | 60000 | initial | done |
| 001 | 2000 | 59000 | uncertainty top-1000 | done |
| 002 | 3000 | 58000 | uncertainty top-1000 | done |
| 003 | 4000 | 57000 | uncertainty top-1000 | done |

- Dataset: rMD17 benzene (C₆H₆, 12 atoms, ~99,995 frames total)
- Data split: train 1000 / candidate 60000 / valid 5000 / test 10000
- 4 committee models per round, `DP_INFER_BATCH_SIZE=1800`
- random baseline (seed0/1/2 Round 001–003) done
- independent test done
- diversity / trust_level baselines pending
- Detailed results: `docs/results/rmd17_benzene_active_learning.md`

### 3.3 Key result files

**toy H2:**
- `experiments/al_rounds_summary.csv` — uncertainty Round 0–3
- `experiments/baselines/random_round001_baseline_summary.csv`
- `experiments/baselines/random_round002_baseline_summary.csv`
- `experiments/baselines/random_round003_baseline_summary.csv`
- `experiments/baselines/aligned_comparison.csv` — 统一 remaining-candidate-pool 指标的四策略对比（authoritative）

**rMD17 ethanol:**
- `experiments/rmd17_ethanol_summary/al_rounds_summary.csv`
- `experiments/rmd17_ethanol_summary/independent_test_all_summary.csv`
- `experiments/rmd17_ethanol_summary/random_baseline_round_summary.csv`
- `experiments/rmd17_ethanol_summary/profiling_unified.csv`
- `experiments/rmd17_ethanol_summary/md_stability/md_summary.json`

**rMD17 benzene:**
- `experiments/rmd17_benzene_round000_committee_prediction/selected_topk.json`
- `experiments/rmd17_benzene_round001_committee_prediction/selected_topk.json`
- `experiments/rmd17_benzene_round002_committee_prediction/selected_topk.json`
- `experiments/rmd17_benzene_round003_committee_prediction/selected_topk.json`
- `docs/results/rmd17_benzene_active_learning.md`

---

## 4. Available Scripts

| Category | Key Scripts |
|---|---|
| Data | `make_toy_h2_deepmd.py`, `merge_selected_frames.py`, `make_remaining_candidate.py`, `convert_rmd17_to_deepmd.py`, `split_rmd17_to_deepmd.py` |
| Training | `train_single_model.sh`, `train_committee_models.sh`, `train_round_committee_models.sh` |
| Inference | `predict_committee_models.py` |
| Selection | `select_from_predictions.py`, `select_by_strategy.py` |
| Analysis | `summarize_al_rounds.py`, `summarize_rmd17_random_baseline.py`, `plot_rmd17_ethanol.py`, `plot_rmd17_final_comparison.py` |
| Evaluation | `quick_md_test.py`, `run_md_stability.py`, `test_rmd17_all_independent.sh` |
| Profiling | `record_round_profiling.sh` |
| Runner | `run_random_baseline_round.sh`, `run_rmd17_random_baseline.sh` |

---

## 5. Available Figures

**toy H2:**
```
experiments/figures/
  dataset_size_rounds.svg
  force_model_deviation_rounds.svg
  validation_rmse_rounds.svg
  random_vs_uncertainty_force_rmse.svg
  random_vs_uncertainty_energy_rmse.svg
  random_vs_uncertainty_candidate_force_dev.svg
  random_vs_uncertainty_dataset_size.svg
```

**rMD17 ethanol:**
```
experiments/rmd17_ethanol_summary/
  rmd17_ethanol_force_model_deviation_rounds.svg
  rmd17_ethanol_validation_rmse_rounds.svg
  rmd17_ethanol_dataset_size_rounds.svg
  rmd17_ethanol_independent_test_force_rmse.svg
  rmd17_ethanol_independent_test_energy_rmse.svg
  rmd17_ethanol_final_comparison.svg
  rmd17_ethanol_valid_vs_test.svg
```

---

## 6. V100 Profiling Progress

**Done (toy H2):**
- Training wall time: 132 models, mean=11.0s ± 0.5s, 8.7ms/batch (1000 steps)
- 2×V100 parallel speedup: 1.97× (near-linear), ~22s/round
- Estimated end-to-end: ~32s/round (train + predict + I/O)

**Done (rMD17 ethanol):**
- Training wall time: 52 models, uncertainty mean=50.4s, random mean=56.7s (2000 steps, 1000-4000 frames)
- Prediction: 57k-60k frames × 4 models = ~176s/round
- Per-round end-to-end: uncertainty ~5 min, random (3 seeds) ~10 min
- Full uncertainty branch (4 rounds): ~20 min; random baseline (3 seeds × 3 rounds): ~29 min
- Unified per-stage CSV: `experiments/rmd17_ethanol_summary/profiling_unified.csv`
- Per-model CSV: `experiments/rmd17_ethanol_summary/profiling_all_models.csv`

**Not done:**
- Full GPU utilization curves across entire rounds

---

## 7. Remaining V100 Tasks

1. (done) Systematic end-to-end profiling (training + prediction + I/O, 52 models, unified CSV)
2. GPU utilization/memory curves for a full training round
3. (done) Uncertainty-diversity multi-round (Round 002–003)
4. (done) Trust-level multi-round (Round 002–003)
5. (done) Full 4-strategy comparison (uncertainty / random / diversity / trust-level)
6. (done) rMD17 benzene uncertainty branch Round 000–003
7. rMD17 benzene: diversity / trust_level baselines
8. rMD17 benzene: MD stability

---

## 8. Tasks Deferred to H100

1. Multi-GPU scaling (4/8 GPU)
2. Bfloat16 / mixed-precision training
3. Large-batch committee training throughput
4. H100 vs V100 speedup comparison

---

## 9. Claim Boundary

**Can claim:**
- Reproducible active learning pipeline on toy H2, rMD17 ethanol, and rMD17 benzene (2×V100)
- Four-strategy multi-seed multi-round comparison completed on both toy H2 and rMD17 ethanol
- All three active strategies (uncertainty/diversity/trust_level) have clearly lower mean Force RMSE than random on rMD17 ethanol (Round 3: 0.354-0.362 vs 0.607 eV/Å); however, random cross-seed variance is large (std=0.683), so strict statistical significance cannot be claimed from current data alone
- Uncertainty branch shows monotonically decreasing Force RMSE on both validation (0.374→0.354) and independent test (0.344→0.327 eV/Å) for rMD17 ethanol
- Uncertainty branch and random baseline completed on rMD17 benzene (Round 000–003 + random seed0/1/2 Round 001–003 + independent test)
- 2×V100 model-level parallel training is near-linear (1.97× speedup)
- NVE MD at 10K stable with drift ~0.035 eV/ps (ethanol)
- Pipeline profiling complete (124 models: 16 unc + 36 rnd + 36 div + 36 trust, unified CSV)

**Cannot claim (yet):**
- Results generalize broadly (only 2 rMD17 systems tested; benzene diversity/trust_level pending)
- One active strategy consistently outperforms others (differences within 1σ on both toy H2 and ethanol)
- High-temperature MD stability (100K+ dissociation)
- H100 multi-GPU scaling results
- Full CCF-B paper-level evidence

---

## 10. Recommended Next Commit / Experiment Order

1. (done) Sync documentation
2. (done) Create local data/artifact inventory
3. (done) Systematic end-to-end V100 profiling (124 models, unified CSV)
4. (done) Uncertainty-diversity Round 002–003
5. (done) Trust-level Round 002–003
6. (done) Full 4-strategy comparison
7. (done) rMD17 ethanol: uncertainty + random baseline + independent test + MD + profiling
8. (done) rMD17 ethanol: diversity + trust_level baselines
9. (done) rMD17 benzene: uncertainty branch Round 000–003
10. rMD17 benzene: diversity / trust_level baselines + MD stability
11. GPU utilization/memory curves for full training round (nvidia-smi dmon)
12. I/O breakdown and prediction batch-size / candidate-size scaling
13. H100 scaling
