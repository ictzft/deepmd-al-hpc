# V100 Profiling Summary

Hardware: 2×Tesla V100-SXM2-16GB (16384 MiB each), Driver 535.183.01, Compute Capability 7.0
Software: DeepMD-kit v3.1.4.dev81, TensorFlow 2.21.0, NumPy 2.4.4
Date: 2026-05-25
Data: toy H2 (2 atoms), 1000 training steps, 4-model committee (2 models per GPU, 2 waves)

---

## Per-model Training Wall Time (1000 steps, single GPU)

| Round | Mean (s) | Min (s) | Max (s) | Samples |
|---|---:|---:|---:|---:|
| 001 | 10.9 | 10.1 | 11.7 | 12 |
| 002 | 10.7 | 10.3 | 11.5 | 12 |
| 003 | 11.2 | 10.5 | 12.4 | 12 |
| **Overall** | **10.9** | **10.1** | **12.4** | **36** |

Avg time per batch: 8.7 ms/batch

## 2×V100 Parallel Training (4 models)

Training uses model-level parallelism: GPU 0 runs model_000+model_002, GPU 1 runs model_001+model_003 in two waves.

| Seed | Round | Serial (s) | Parallel (s) | Speedup |
|---|---:|---:|---:|---:|
| seed0 | 001 | 41.1 | 20.8 | 1.98× |
| seed0 | 002 | 43.5 | 22.2 | 1.96× |
| seed0 | 003 | 42.3 | 21.2 | 2.00× |
| seed1 | 001 | 44.2 | 22.2 | 1.99× |
| seed1 | 002 | 42.6 | 21.8 | 1.95× |
| seed1 | 003 | 43.7 | 22.0 | 1.99× |
| seed2 | 001 | 45.9 | 23.3 | 1.97× |
| seed2 | 002 | 42.3 | 21.4 | 1.97× |
| seed2 | 003 | 48.4 | 24.6 | 1.97× |
| **Mean** | — | **43.8** | **22.2** | **1.97×** |

Speedup is near-perfect (~2.0×) for 2-GPU model-level parallelism. The bottleneck is single-model training time; adding more GPUs (up to 4) would scale linearly in this regime.

## Per-round End-to-end Time (estimated)

| Phase | Time (s) | % of Total |
|---|---:|---:|
| Training (4 models, 2×V100 parallel) | ~22 | ~71% |
| Freeze + Test (4 models) | <4 | ~13% |
| Prediction (4 models on candidate pool) | ~5 | ~16% |
| Dataset update (merge + remaining candidate) | ~2 | <1% |
| **Total per round** | **~31** | **100%** |

Training dominates (~71%). Prediction and I/O are minor in the toy H2 setting (small model, small dataset).

## Notes

- All training wall times extracted from `train.log` (`wall time:` field).
- Freeze and test times are estimates from Docker container stdout timestamps.
- GPU utilization and memory were not systematically recorded (nvidia-smi dmon was not running during these rounds).
- The toy H2 model is tiny (2 atoms, small network); training time is ~11s for 1000 steps.
- For realistic DFT systems (hundreds of atoms, larger networks), training will be the dominant cost by a wider margin, and the relative cost of prediction and I/O will be even smaller.
- This profiling serves as the V100 baseline. Future H100 profiling should use identical training configs for fair comparison.
