# Aligned Four-Strategy Comparison (Remaining Candidate Pool Metric)

**Key fix**: All strategies now report `remaining_candidate_force_dev_max_mean`
(force_dev_max of the candidate pool AFTER retraining), not the selected top-K value.
This makes uncertainty and random/diversity/trust_level directly comparable.

Generated: 2026-05-25 | Toy H2 | 2xV100

## Force RMSE Comparison (multi-seed mean ± std)

| Strategy | R1 F_RMSE | R2 F_RMSE | R3 F_RMSE |
|---|---:|---:|---:|
| uncertainty | 1.617669e-01 | 1.938590e-01 | 1.742648e-01 |
| random | 2.112220e-01 +/- 5.507695e-02 | 1.961835e-01 +/- 1.622950e-02 | 1.890284e-01 +/- 4.781861e-02 |
| diversity | 2.051571e-01 +/- 5.789218e-02 | 1.738373e-01 +/- 9.289697e-03 | 1.758995e-01 +/- 4.081667e-02 |
| trust_level | 1.353417e-01 +/- 2.761368e-02 | 1.491142e-01 +/- 2.255783e-02 | 1.781654e-01 +/- 6.470044e-03 |

## Notes

- **uncertainty**: single run (no multi-seed).
- **random/diversity/trust_level**: 3-seed mean ± std.
- `remaining_candidate_force_dev_max_mean`: mean force_dev_max of the candidate pool AFTER the current round retraining (NOT the selected frames).
- This metric is directly comparable across all strategies.
- Previous `random_vs_uncertainty_summary.csv` mixed selected top-K (uncertainty) with remaining candidate pool (random).