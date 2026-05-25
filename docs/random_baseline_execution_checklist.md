# Random Baseline Execution Checklist

**Status: ALL COMPLETED (2026-05-25, 2×V100).** This file documents what was done and how to reproduce it.

---

## 1. Completed Items

| Seed | Round 001 | Round 002 | Round 003 |
|---|---|---|---|
| seed0 | done | done | done |
| seed1 | done | done | done |
| seed2 | done | done | done |

Each (seed, round) run included: data prep → 4-model committee training → committee prediction → metrics summary.

---

## 2. Generated Summary Files

| File | Content |
|---|---|
| `experiments/baselines/random_round001_baseline_summary.csv` | seed0/1/2 Round 001 metrics + candidate uncertainty |
| `experiments/baselines/random_round002_baseline_summary.csv` | seed0/1/2 Round 002 metrics |
| `experiments/baselines/random_round003_baseline_summary.csv` | seed0/1/2 Round 003 metrics |
| `experiments/baselines/random_vs_uncertainty_summary.csv` | 16 rows: uncertainty R0-3 + random R1-3 (all seeds + mean) |

---

## 3. Generated Figures

```
experiments/figures/
  random_vs_uncertainty_force_rmse.svg
  random_vs_uncertainty_energy_rmse.svg
  random_vs_uncertainty_candidate_force_dev.svg
  random_vs_uncertainty_dataset_size.svg
```

---

## 4. Reproduction Commands

### Quick mode (one command per seed/round)

```bash
# Inside DeepMD-kit Docker container:
bash scripts/run_random_baseline_round.sh 002 seed0
bash scripts/run_random_baseline_round.sh 002 seed1
bash scripts/run_random_baseline_round.sh 002 seed2
bash scripts/run_random_baseline_round.sh 003 seed0
bash scripts/run_random_baseline_round.sh 003 seed1
bash scripts/run_random_baseline_round.sh 003 seed2
```

### Summarize results

```bash
python scripts/analysis/summarize_random_round_baselines.py --round-id 2
python scripts/analysis/summarize_random_round_baselines.py --round-id 3
python scripts/analysis/summarize_random_vs_uncertainty.py
python scripts/analysis/plot_random_vs_uncertainty.py
```

---

## 5. Files NOT Tracked by Git

- `data/toy_h2/random_seed*_round_*_train/` — training data (.npy arrays)
- `data/toy_h2/random_seed*_round_*_candidate/` — candidate data
- `experiments/baselines/random_seed*_round*_committee_models/` — frozen models, logs, checkpoints
- `experiments/baselines/random_seed*_round*_committee_prediction/committee_predictions.npz` — prediction arrays

---

## 6. Follow-up Tasks

See `docs/random_baseline_next_steps.md` for post-random-baseline work.
