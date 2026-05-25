# Four-Strategy Multi-Seed Comparison (Toy H2, 2xV100)

Date: 2026-05-25 | All experiments run on 2xTesla V100-16GB

> **Authoritative source**: `experiments/baselines/aligned_comparison.md` — uses consistent
> "remaining candidate-pool" metric for all four strategies. This file mirrors that data.
> `strategy_summary.csv` is a copy of `aligned_comparison.csv`.

---

## Random Baseline (seed0/seed1/seed2 Round 001-003)

| Seed | R1 F_RMSE | R2 F_RMSE | R3 F_RMSE |
|---|---:|---:|---:|
| seed0 | 2.553e-01 | 2.121e-01 | 2.374e-01 |
| seed1 | 2.288e-01 | 1.796e-01 | 1.418e-01 |
| seed2 | 1.495e-01 | 1.968e-01 | 1.879e-01 |
| **mean** | **2.112e-01** | **1.962e-01** | **1.890e-01** |
| **std** | **5.508e-02** | **1.623e-02** | **4.782e-02** |

## Uncertainty Branch (Round 0-3)

| Round | F_RMSE |
|---:|---:|
| 0 | 1.821e-01 |
| 1 | 1.618e-01 |
| 2 | 1.939e-01 |
| 3 | 1.743e-01 |

## Diversity (seed0/seed1/seed2, Round 001-003)

| Seed | R1 E_RMSE | R1 F_RMSE | R2 E_RMSE | R2 F_RMSE | R3 E_RMSE | R3 F_RMSE |
|---|---:|---:|---:|---:|---:|---:|
| seed0 | 8.768e-01 | 2.691e-01 | 3.222e-01 | 1.804e-01 | 2.560e-01 | 1.643e-01 |
| seed1 | 1.510e+00 | 1.900e-01 | 1.197e+00 | 1.632e-01 | 1.154e+00 | 1.421e-01 |
| seed2 | 1.026e+00 | 1.563e-01 | 9.481e-01 | 1.780e-01 | 1.804e+00 | 2.213e-01 |
| **mean** | **1.138e+00** | **2.052e-01** | **8.226e-01** | **1.738e-01** | **1.071e+00** | **1.759e-01** |
| **std** | **3.312e-01** | **5.789e-02** | **4.509e-01** | **9.290e-03** | **7.774e-01** | **4.082e-02** |

## Trust-Level (seed0/seed1/seed2, Round 001-003)

| Seed | R1 E_RMSE | R1 F_RMSE | R2 E_RMSE | R2 F_RMSE | R3 E_RMSE | R3 F_RMSE |
|---|---:|---:|---:|---:|---:|---:|
| seed0 | 5.486e-01 | 1.362e-01 | 7.078e-01 | 1.255e-01 | 1.082e+00 | 1.733e-01 |
| seed1 | 1.186e+00 | 1.625e-01 | 7.488e-01 | 1.704e-01 | 6.706e-01 | 1.855e-01 |
| seed2 | 7.414e-01 | 1.073e-01 | 1.117e+00 | 1.515e-01 | 1.071e+00 | 1.757e-01 |
| **mean** | **8.253e-01** | **1.353e-01** | **8.578e-01** | **1.491e-01** | **9.412e-01** | **1.782e-01** |
| **std** | **3.268e-01** | **2.761e-02** | **2.252e-01** | **2.256e-02** | **2.344e-01** | **6.470e-03** |

## All Strategies Force RMSE Comparison

| Strategy | R1 F_RMSE | R1 std | R2 F_RMSE | R2 std | R3 F_RMSE | R3 std |
|---|---:|---:|---:|---:|---:|---:|
| uncertainty | 1.515e-01 | 2.496e-02 | 2.132e-01 | 2.242e-02 | 1.965e-01 | 2.421e-02 |
| random | 2.112e-01 | 5.508e-02 | 1.962e-01 | 1.623e-02 | 1.890e-01 | 4.782e-02 |
| diversity | 2.052e-01 | 5.789e-02 | 1.738e-01 | 9.290e-03 | 1.759e-01 | 4.082e-02 |
| trust_level | 1.353e-01 | 2.761e-02 | 1.491e-01 | 2.256e-02 | 1.782e-01 | 6.470e-03 |

## Limitations

- Toy H2 only (2 atoms, 250 frames). Results do NOT generalize to real materials.
- Cross-seed variance is large, especially for Energy RMSE.
- All four strategies have 3-seed (seed0/seed1/seed2) mean ± std for Round 1-3.
- Force RMSE differences between strategies are within one standard deviation.
- Independent test set and MD stability validation are not yet performed.
