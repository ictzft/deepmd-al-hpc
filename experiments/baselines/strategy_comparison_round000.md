# Round 0 Strategy Comparison (Selection-level)

Selection from the initial 50-frame candidate pool using different strategies.
This is a **selection-level** comparison — it shows what each strategy selects,
not the downstream retraining effect.

## Selected force_dev_max statistics

| Strategy | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| diversity | 2.541727e-01 | 2.391906e-01 | 7.109544e-02 | 7.891949e-01 |
| random_seed0 | 9.757178e-02 | 1.084898e-01 | 2.627537e-02 | 3.985356e-01 |
| random_seed1 | 1.503437e-01 | 1.510344e-01 | 2.888289e-02 | 4.490663e-01 |
| random_seed2 | 1.811068e-01 | 1.503667e-01 | 4.970875e-02 | 4.544573e-01 |
| trust_level | 3.511015e-01 | 1.951509e-01 | 1.384963e-01 | 7.891949e-01 |
| uncertainty | 4.409891e-01 | 1.304900e-01 | 3.228673e-01 | 7.891949e-01 |

## Strategy descriptions

- **uncertainty**: pure top-K by `force_dev_max`. Selects the 10 highest-uncertainty frames.
- **random_seed{0,1,2}**: random sampling with different random seeds (for baseline).
- **diversity**: top-M=30 by `force_dev_max`, then Farthest Point Sampling on pairwise-distance
  descriptor to select 10 structurally diverse frames. Reduces mean selected uncertainty
  compared to pure top-K but improves structural coverage.
- **trust_level**: DP-GEN-style prototype. Splits candidate pool into accurate (<p50),
  candidate (p50-p90), and failed (>p90) regions. Selects 80% from candidate + 20% from failed.

**Important**: This is a toy H2 prototype. Full multi-round active learning retraining
has not yet been run for diversity or trust_level strategies.