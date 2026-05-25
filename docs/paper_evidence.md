# Paper Evidence Checklist

This document tracks the current evidence and pending experiments for the `deepmd-al-hpc` project. It is intended to provide an honest assessment of what the current prototype can and cannot support.

## Current evidence

- Toy H2 DeePMD workflow has been validated on 2×V100 GPUs.
- 4-model committee training is implemented and produces frozen models for inference.
- Offline active learning Round 0–3 (uncertainty branch) has been completed.
- Per-round committee prediction and `force_dev_max` / `force_dev_mean` / `energy_dev` calculation are implemented.
- Candidate-pool uncertainty (`force_dev_max_mean`) decreases monotonically across uncertainty rounds: 0.441 (Round 0) → 0.170 (Round 3).
- Random baseline Round 001 has been extended from seed0 to seed0/seed1/seed2.
- Multi-seed random mean ± std is available for Round 001.
- Uncertainty branch yields lower remaining candidate-pool `force_dev_max_mean` than all three random seeds in Round 001 (0.126 vs random mean 0.392).
- Data preparation scripts and config generation for random Round 002/003 are ready.
- Uncertainty vs random comparison figures (Force RMSE, Energy RMSE, candidate deviation, dataset size) are generated.

## Claims currently supported

1. The repository implements a reproducible toy H2 offline active learning prototype.
2. Committee-based model deviation (`force_dev_max`) can be used to select high-uncertainty candidate structures.
3. In the toy H2 setting, uncertainty-based selection reduces remaining candidate-pool uncertainty across four active learning rounds.
4. In Round 001, the uncertainty branch achieves lower remaining candidate-pool `force_dev_max_mean` than all three random seeds (seed0/seed1/seed2).
5. The random baseline multi-seed framework (seed0/seed1/seed2) is in place and can be extended to additional rounds using provided scripts.

## Claims not yet supported

1. The method significantly outperforms random sampling across multiple datasets or multiple rounds.
2. The method is validated on real DFT/AIMD datasets.
3. The framework has demonstrated H100 or multi-node scaling.
4. The framework has been validated through MD stability tests.
5. Uncertainty-diversity selection provides benefit over pure uncertainty top-K.
6. The framework is production-ready or meets CCF-B submission standards.

## Next experiments

1. **Complete random Round 002/003 retraining** — run training and inference for seed0/seed1/seed2 across Round 002 and Round 003 using the prepared scripts.
2. **Generate full learning curves** — update summary and figures with Round 002/003 data once available.
3. **Add uncertainty-diversity selection** — implement structural diversity constraints in candidate selection.
4. **Move to real DFT/AIMD dataset** — convert real first-principles data to DeepMD npy format and re-run pipeline.
5. **Add systematic profiling** — record per-round training time, prediction time, and end-to-end wall-clock time.
6. **Run H100 / multi-GPU scaling** — benchmark committee training and prediction throughput on H100 nodes.

## Scripts reference

| Script | Purpose |
|---|---|
| `scripts/analysis/prepare_random_baseline_round.py` | Generate data + configs for a random baseline round |
| `scripts/analysis/summarize_random_vs_uncertainty.py` | Aggregate uncertainty vs random across all rounds |
| `scripts/analysis/plot_random_vs_uncertainty.py` | Generate comparison SVG figures |
| `scripts/analysis/summarize_random_round001_baselines.py` | Summarize random Round 001 across seed0/1/2 |
| `scripts/analysis/summarize_selection_baselines.py` | Summarize selection-level baseline |
