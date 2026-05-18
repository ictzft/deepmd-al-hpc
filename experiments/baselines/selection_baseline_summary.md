# Selection-level Baseline Summary

This table summarizes selection-level behavior for uncertainty sampling and random sampling.

| Round | Strategy | Runs | Top-K | Avg selected force_dev_max | Std | Avg selected energy_dev | Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| round_000 | random | 3 | 10 | 0.14300743 | 0.042247961 | 0.82693809 | 0.00056140414 |
| round_000 | uncertainty | 1 | 10 | 0.4409891 |  | 0.84075573 |  |
| round_001 | random | 3 | 10 | 0.14573127 | 0.013063767 | 0.44808055 | 0.0002812579 |
| round_001 | uncertainty | 1 | 10 | 0.26933326 |  | 0.44775544 |  |

Note: This is a selection-level baseline summary. It does not yet represent retrained model accuracy.
The next step is to merge selected frames, retrain committee models, and compare Energy/Force RMSE curves.
