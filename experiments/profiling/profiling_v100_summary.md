# V100 Profiling Summary (待实测)

This file will be populated as profiling data is collected from actual V100 runs.

## Current Status

Profiling infrastructure is ready. No systematic profiling data has been recorded yet for Round 001–003.

Template files:
- `experiments/profiling/profiling_v100_rounds.csv` — per-round CSV with column headers
- `scripts/profiling/record_round_profiling.sh` — automated recording script

## How to start

```bash
# 1. Train models first (train_round_committee_models.sh)
bash scripts/train/train_round_committee_models.sh 002 \
  configs/deepmd/random_seed0_round_002_committee \
  experiments/baselines/random_seed0_round002_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid

# 2. Run profiling for prediction + dataset update
bash scripts/profiling/record_round_profiling.sh 002 seed0 random
```

## Per-round end-to-end time (to be filled)

| Round | Branch | Seed | Train / s | Prediction / s | Dataset Update / s | End-to-end / s | GPU Util Avg / % | GPU Mem Max / MB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| — | — | — | — | — | — | — | — | — |

## Breakdown by phase (to be filled after data collection)

| Phase | Typical Time / s | % of Total |
|---|---|---|
| Training (4 models, 2×V100) | — | — |
| Prediction (40 frames, 4 models) | — | — |
| Dataset update (merge + remaining) | — | — |

## GPU observations (to be filled)

- GPU model: Tesla V100-16GB
- GPU count: 2
- Training GPU utilization: —
- Training GPU memory: —
- Prediction GPU utilization: —
- Prediction GPU memory: —

## Notes

- Profiling target: establish reproducible V100 baseline, not performance optimization.
- All data should come from actual runs on the 2×V100 platform.
- This baseline will be compared against future H100 scaling results.
