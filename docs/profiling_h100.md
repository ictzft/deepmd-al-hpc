# H100 Profiling Plan

This document is reserved for future H100 / multi-GPU scaling experiments.
**No H100 result is reported in the current repository.**

Current status (2026-05-25):
- V100 wall-clock profiling has been completed (see `docs/profiling_v100.md`).
- H100 experiments have NOT been conducted.

## Planned work

1. Port the validated V100 pipeline to H100.
2. Compare V100 vs H100 per-model training wall-clock time.
3. Evaluate 1/2/4/8 GPU scaling if resources are available.
4. Measure committee training throughput (models/hour).
5. Measure candidate prediction throughput (frames/second).
6. Report GPU utilization and memory usage via nvidia-smi dmon.
7. Compare end-to-end active learning round time (V100 vs H100).

## Prerequisites

- H100 node access with DeepMD-kit Docker environment.
- Identical training configs (use existing `configs/deepmd/` JSON files) for fair comparison.
- Identical toy H2 dataset for controlled benchmarking.

## Relationship to V100 profiling

See `docs/profiling_v100.md` for the current V100 baseline data, including:
- Per-model training wall time: mean=10.9s (1000 steps, 8.7ms/batch)
- 2×V100 parallel speedup: 1.97×
- Representative GPU utilization and memory data

The H100 profiling should use the same methodology and report the same metrics for direct comparison.
