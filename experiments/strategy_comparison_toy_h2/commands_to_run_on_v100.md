# Commands to Run on V100

Current status (2026-05-25): diversity and trust_level seed0 Round 001-003 are complete. Seed1/seed2 remain for multi-seed comparison.

## Prerequisites

```bash
cd /data/zft/deepmd-al-hpc
# Enter DeepMD-kit Docker container
bash scripts/docker/enter_deepmd_container.sh
# Or use: docker run --rm --gpus all -v /data/zft:/data/zft -w /data/zft/deepmd-al-hpc ghcr.io/deepmodeling/deepmd-kit:master bash
```

## P1: diversity seed1 Round 001-003

```bash
bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
  --strategy uncertainty-diversity --seed 1 --start-round 1 --end-round 3 --top-k 10
```

## P2: diversity seed2 Round 001-003

```bash
bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
  --strategy uncertainty-diversity --seed 2 --start-round 1 --end-round 3 --top-k 10
```

## P3: trust_level seed1 Round 001-003

```bash
bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
  --strategy trust-level --seed 1 --start-round 1 --end-round 3 --top-k 10
```

## P4: trust_level seed2 Round 001-003

```bash
bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
  --strategy trust-level --seed 2 --start-round 1 --end-round 3 --top-k 10
```

## After all experiments complete

```bash
# Generate full comparison summary
python scripts/analysis/summarize_strategy_comparison.py

# Profile the runs
python scripts/profiling/parse_profile_logs.py \
  --input-dir experiments/strategy_comparison_toy_h2 \
  --output-csv experiments/profiling/v100/strategy_profiling_summary.csv \
  --output-md experiments/profiling/v100/strategy_profiling_summary.md
```

## Dry-run (verify commands without executing)

```bash
bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
  --strategy all --seed 0 --start-round 1 --end-round 3 --top-k 10 --dry-run
```

## Estimated time

~2 min per seed per round (2xV100, 4 models). Total for 4 missing seeds × 3 rounds = ~24 min.
