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
| 2 | random | seed0 | 220 | 30 | 1.977364e+00 | 2.120793e-01 | 1.362240e-01 |
| 2 | random | seed1 | 220 | 30 | 1.847642e+00 | 1.796397e-01 | 1.219434e-01 |
| 2 | random | seed2 | 220 | 30 | 9.067673e-01 | 1.968315e-01 | 1.611978e-01 |
| 3 | random | seed0 | 230 | 20 | 1.369364e+00 | 2.373825e-01 | 1.578821e-01 |
| 3 | random | seed1 | 230 | 20 | 1.707502e+00 | 1.417639e-01 | 1.352771e-01 |
| 3 | random | seed2 | 230 | 20 | 1.310121e+00 | 1.879388e-01 | 1.239594e-01 |
| 1 | random | mean | 210 | 40 | 4.560334e-01 | 2.112220e-01 | 3.917845e-01 |
| 2 | random | mean | 220 | 30 | 1.577258e+00 | 1.961835e-01 | 1.397884e-01 |
| 3 | random | mean | 230 | 20 | 1.462329e+00 | 1.890284e-01 | 1.390395e-01 |

## Round-level comparison (uncertainty vs random mean ± std)

| Round | Method | Energy RMSE mean | Energy RMSE std | Force RMSE mean | Force RMSE std | force_dev_max mean | force_dev_max std |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | uncertainty | 7.357575e-01 | — | 1.821392e-01 | — | 4.409891e-01 | — |
| 1 | uncertainty | 7.292945e-01 | — | 1.617669e-01 | — | 2.693333e-01 | — |
| 1 | random | 4.560334e-01 | 2.223318e-01 | 2.112220e-01 | 5.507696e-02 | 3.917845e-01 | 8.395874e-02 |
| 2 | uncertainty | 1.206733e+00 | — | 1.938590e-01 | — | 1.874125e-01 | — |
| 2 | random | 1.577258e+00 | 5.842731e-01 | 1.961835e-01 | 1.622951e-02 | 1.397884e-01 | 1.986846e-02 |
| 3 | uncertainty | 9.990814e-01 | — | 1.742648e-01 | — | 1.701889e-01 | — |
| 3 | random | 1.462329e+00 | 2.143823e-01 | 1.890284e-01 | 4.781861e-02 | 1.390395e-01 | 1.727149e-02 |

## Notes

- Uncertainty branch: `force_dev_max` top-K selection across Round 0–3.
- Random baseline: multi-seed (seed0/seed1/seed2) Round 001/002/003 retraining completed (2026-05-25, 2×V100).
- This is a toy H2 workflow validation. Real DFT/AIMD datasets and H100 scaling are not yet included.

**Field meaning note:** `force_dev_max_mean` has different semantics per branch:
- For **uncertainty**: mean `force_dev_max` of the top-K **selected** frames from committee prediction.
- For **random**: mean `force_dev_max` of the **remaining candidate pool** after retraining.
- These are not directly comparable; for a fair remaining candidate-pool comparison,
  see the per-seed `random_seed*_round001_prediction_summary.csv` files (which include
  the uncertainty_round001 remaining candidate-pool row).
