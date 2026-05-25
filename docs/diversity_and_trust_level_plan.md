# Diversity and Trust-Level Baseline Plan

---

## 1. Motivation

Random baseline (3 seeds × 3 rounds) is complete on toy H2. The next V100 work should move beyond random to two stronger baseline variants:

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
| Diversity Round 001 (trained) | `experiments/baselines/diversity_round001_*` | done |
| Trust-level Round 001 (trained) | `experiments/baselines/trust_level_round001_*` | done |
| Selection-level comparison | `experiments/baselines/strategy_comparison_round000.csv` | done |

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

**Round 001 result**: E_RMSE=0.877 eV, F_RMSE=0.269 eV/Å, cand_dev=0.218.

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

**Round 001 result**: E_RMSE=0.549 eV, F_RMSE=0.136 eV/Å, cand_dev=0.160.

---

## 5. Recommended V100 Execution Commands

### Round 002–003 for each strategy

Follow the same pattern as `scripts/run_random_baseline_round.sh`:
1. Prepare data using `merge_selected_frames.py` + `make_remaining_candidate.py`
2. Generate configs using `make_round_committee_configs.py`
3. Train using `train_round_committee_models.sh`
4. Predict using `predict_committee_models.py`
5. Select next round frames using `select_by_strategy.py`

See `docs/random_baseline_execution_checklist.md` for the command template.

---

## 6. Expected Output Files

```
experiments/baselines/diversity_round002_*/ (trained models + prediction)
experiments/baselines/diversity_round003_*/
experiments/baselines/trust_level_round002_*/
experiments/baselines/trust_level_round003_*/
experiments/baselines/strategy_comparison_full.csv (4 strategies × 3 rounds)
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
3. Single-run results (no multi-seed for diversity/trust-level yet).
4. Cross-model variance limits per-run comparison reliability.
