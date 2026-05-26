# Diversity and Trust-Level Baseline Plan

---

## 1. Status

Diversity and trust-level strategies are **fully implemented and validated** (multi-seed Round 001–003, 2026-05-25, 2×V100). This document is retained as a reference for the original plan and implementation details.

- **Uncertainty-diversity**: reduces structural redundancy in top-K uncertainty selection
- **Trust-level (DP-GEN-style)**: splits candidate pool by model deviation into accurate / candidate / failed regions

---

## 2. Current Available Components

| Component | Location | Status |
|---|---|---|
| Uncertainty-diversity selector | `src/al/selector.py:select_uncertainty_diversity` | implemented |
| Trust-level selector | `src/al/selector.py:select_trust_level` | implemented |
| Strategy CLI | `scripts/selection/select_by_strategy.py` | implemented |
| Selector with diversity/trust support | `scripts/active_learning/select_from_predictions.py` | updated |
| Diversity multi-seed Round 001–003 | `experiments/baselines/diversity_*` | done (2026-05-25) |
| Trust-level multi-seed Round 001–003 | `experiments/baselines/trust_level_*` | done (2026-05-25) |
| Selection-level comparison | `experiments/baselines/strategy_comparison_round000.csv` | done |
| Aligned four-strategy comparison | `experiments/baselines/aligned_comparison.csv` | done (2026-05-25) |
| Diversity descriptor analysis | `experiments/baselines/diversity_analysis.md` | done (3.1× spread) |

---

## 3. Uncertainty-Diversity Selection

**Algorithm**: Top-M by force_dev_max → pairwise-distance descriptor → Farthest Point Sampling to select K diverse frames.

**Default params**: top_k=10, top_m=30, descriptor=pairwise-distance.

**Usage**:
```bash
python scripts/selection/select_by_strategy.py \
  --predictions <committee_predictions.npz> \
  --strategy uncertainty-diversity \
  --top-k 10 --top-m 30 \
  --output <output.json>
```

**Multi-seed Round 001–003 results (3-seed mean ± std)**:

| Round | F_RMSE | F_RMSE std |
|---:|---:|---:|
| 1 | 2.052e-01 | 5.789e-02 |
| 2 | 1.738e-01 | 9.290e-03 |
| 3 | 1.759e-01 | 4.082e-02 |

See `experiments/baselines/aligned_comparison.md` for full comparison with other strategies.

---

## 4. Trust-Level / DP-GEN-style Selection

**Algorithm**: Force_dev_max percentile splitting:
- accurate (<P50): skip
- candidate (P50-P90): select from here (80%)
- failed (>P90): keep small fraction (20%)

**Default params**: low_pct=50, high_pct=90, candidate_ratio=0.8.

**Usage**:
```bash
python scripts/selection/select_by_strategy.py \
  --predictions <committee_predictions.npz> \
  --strategy trust-level \
  --top-k 10 --low-pct 50 --high-pct 90 \
  --output <output.json>
```

**Multi-seed Round 001–003 results (3-seed mean ± std)**:

| Round | F_RMSE | F_RMSE std |
|---:|---:|---:|
| 1 | 1.353e-01 | 2.761e-02 |
| 2 | 1.491e-01 | 2.256e-02 |
| 3 | 1.782e-01 | 6.470e-03 |

Accurate/Candidate/Failed split (initial 50-frame pool): 25/20/5.

See `experiments/baselines/aligned_comparison.md` for full comparison with other strategies.

---

## 5. Round 002–003 Execution (Completed)

Round 002–003 for both diversity and trust-level strategies were completed on 2026-05-25 (2×V100). The execution followed the same pattern as `scripts/run_random_baseline_round.sh`:
1. Prepare data using `merge_selected_frames.py` + `make_remaining_candidate.py`
2. Generate configs using `make_round_committee_configs.py`
3. Train using `train_round_committee_models.sh`
4. Predict using `predict_committee_models.py`
5. Select next round frames using `select_by_strategy.py`

See `docs/random_baseline_execution_checklist.md` for the command template.

---

## 6. Generated Output Files

```
experiments/baselines/diversity_round002_*/ (trained models + prediction) — done
experiments/baselines/diversity_round003_*/ — done
experiments/baselines/trust_level_round002_*/ — done
experiments/baselines/trust_level_round003_*/ — done
experiments/baselines/aligned_comparison.csv (4 strategies × 3 rounds) — done
```

---

## 7. Comparison Table Design

| Strategy | Round | E_RMSE | F_RMSE | Cand force_dev_max | Selected force_dev_max |
|---|---:|---:|---:|---:|---:|
| uncertainty | 1 | ... | ... | ... | ... |
| random (mean) | 1 | ... | ... | ... | ... |
| diversity | 1 | ... | ... | ... | ... |
| trust_level | 1 | ... | ... | ... | ... |

---

## 8. Limitations

1. Toy H2 only (2 atoms) — structural diversity descriptor is H-H bond length only. Real systems need richer descriptors (SOAP, ACSF).
2. Trust-level uses only force_dev_max; full DP-GEN uses both force and energy deviation.
3. Cross-model variance limits per-run comparison reliability.
4. All conclusions still need validation on real DFT/AIMD datasets.
