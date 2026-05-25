# Uncertainty vs Random Baseline Comparison

## Per-round summary (all methods)

| Round | Method | Seed | Train | Candidate | Energy RMSE | Force RMSE | force_dev_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | uncertainty | — | 200 | 50 | 7.357575e-01 | 1.821392e-01 | 4.409891e-01 |
| 1 | uncertainty | — | 210 | 40 | 7.292945e-01 | 1.617669e-01 | 2.693333e-01 |
| 2 | uncertainty | — | 220 | 30 | 1.206733e+00 | 1.938590e-01 | 1.874125e-01 |
| 3 | uncertainty | — | 230 | 20 | 9.990814e-01 | 1.742648e-01 | 1.701889e-01 |
| 1 | random | seed0 | 210 | 40 | 6.908853e-01 | 2.553366e-01 | 3.554204e-01 |
| 1 | random | seed1 | 210 | 40 | 2.488029e-01 | 2.288370e-01 | 4.877953e-01 |
| 1 | random | seed2 | 210 | 40 | 4.284121e-01 | 1.494923e-01 | 3.321379e-01 |
| 1 | random | mean | 210 | 40 | 4.560334e-01 | 2.112220e-01 | 3.917845e-01 |

## Round-level comparison (uncertainty vs random mean ± std)

| Round | Method | Energy RMSE mean | Energy RMSE std | Force RMSE mean | Force RMSE std | force_dev_max mean | force_dev_max std |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | uncertainty | 7.357575e-01 | — | 1.821392e-01 | — | 4.409891e-01 | — |
| 1 | uncertainty | 7.292945e-01 | — | 1.617669e-01 | — | 2.693333e-01 | — |
| 1 | random | 4.560334e-01 | 2.223318e-01 | 2.112220e-01 | 5.507696e-02 | 3.917845e-01 | 8.395874e-02 |
| 2 | uncertainty | 1.206733e+00 | — | 1.938590e-01 | — | 1.874125e-01 | — |
| 3 | uncertainty | 9.990814e-01 | — | 1.742648e-01 | — | 1.701889e-01 | — |

## Notes

- Uncertainty branch uses `force_dev_max` top-K selection across Round 0–3.
- Random baseline currently has Round 001 multi-seed data (seed0/seed1/seed2).
- Random Round 002/003 data is pending — scripts and configs are prepared for reproducibility.
- This is a toy H2 workflow validation. Real DFT/AIMD datasets and H100 scaling are not yet included.

**Field meaning note:** `force_dev_max_mean` has different semantics per branch:
- For **uncertainty**: mean `force_dev_max` of the top-K **selected** frames from committee prediction.
- For **random**: mean `force_dev_max` of the **remaining candidate pool** after retraining.
- These are not directly comparable; for a fair remaining candidate-pool comparison,
  see the per-seed `random_seed*_round001_prediction_summary.csv` files (which include
  the uncertainty_round001 remaining candidate-pool row).
