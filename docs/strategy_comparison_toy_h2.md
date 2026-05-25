# Toy H2 Four-Strategy Comparison

---

## 1. Why four-strategy comparison

Random baseline alone only tells us "uncertainty is better than random." A stronger paper needs to compare uncertainty against other plausible selection strategies:

1. **Random** — simplest baseline, verifies that selection strategy matters at all
2. **Uncertainty top-K** — standard committee-based active learning
3. **Uncertainty-diversity** — reduces structural redundancy in high-uncertainty regions
4. **Trust-level (DP-GEN-style)** — splits candidate pool by model deviation thresholds

---

## 2. Strategy descriptions

| Strategy | Algorithm | Parameters |
|---|---|---|
| random | Uniform random sampling | `seed` |
| uncertainty | Sort by `force_dev_max` descending, take top-K | `top_k` |
| uncertainty-diversity | Top-M by uncertainty → pairwise-distance descriptor → Farthest Point Sampling | `top_k`, `top_m`, `descriptor` |
| trust-level | Percentile-based accurate/candidate/failed region split | `top_k`, `low_pct`, `high_pct`, `candidate_ratio` |

See `docs/selection_strategies.md` for detailed descriptions.

---

## 3. How to dry-run

```bash
for strategy in random uncertainty uncertainty-diversity trust-level; do
  bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
    --strategy $strategy \
    --start-round 1 --end-round 3 \
    --top-k 10 \
    --dry-run
done
```

The dry-run prints all commands without executing them. Use this to verify paths and parameters before running on V100.

---

## 4. How to run on V100

1. Enter the DeepMD-kit Docker container
2. Verify data paths exist (`data/toy_h2/train`, `data/toy_h2/valid`)
3. Remove `--dry-run` and execute:

```bash
# Inside Docker container:
bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
  --strategy uncertainty-diversity \
  --start-round 1 --end-round 3 \
  --top-k 10

bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
  --strategy trust-level \
  --start-round 1 --end-round 3 \
  --top-k 10
```

Note: The random and uncertainty strategies already have results in `experiments/baselines/` and `experiments/exp_*/`. The comparison framework can reference those existing results.

---

## 5. How to summarize results

```bash
python scripts/analysis/summarize_strategy_comparison.py
```

Outputs:
- `experiments/strategy_comparison_toy_h2/strategy_summary.csv`
- `experiments/strategy_comparison_toy_h2/strategy_summary.md`

---

## 6. Toy H2 limitations

- 2-atom system: structural diversity descriptor is just H-H bond length
- Small dataset (250 frames): high cross-model variance
- Valid set doubles as candidate pool: no independent test set
- Results indicate workflow feasibility, not general material-system performance
- Comparison should be treated as a prototype framework, not a final paper table

---

## 7. Real DFT/AIMD migration

See `docs/real_dataset_plan.md` for the plan to migrate this comparison framework to a real DFT/AIMD dataset with proper train/candidate/validation/test splits.
