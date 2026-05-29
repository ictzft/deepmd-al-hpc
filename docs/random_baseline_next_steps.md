# Next Steps After Random Baseline (Updated 2026-05-28)

> Random baseline and all four-strategy work on toy H2 and rMD17 ethanol are complete. This document now describes the remaining work.

---

## Completed (toy H2)

- Random baseline seed0/seed1/seed2 Round 001–003 (done 2026-05-25)
- Multi-seed random mean ± std (Round 001/002/003)
- Uncertainty-diversity Round 002–003 (done)
- Trust-level Round 002–003 (done)
- Full 4-strategy comparison with aligned metrics (done)
- Systematic end-to-end V100 profiling (132 models, unified CSV) (done)

## Completed (rMD17 ethanol)

- Uncertainty branch Round 0–3 (done)
- Random baseline 3 seeds × 3 rounds (done)
- Diversity / trust_level baselines (done)
- Four-strategy comparison (done)
- Independent test evaluation (done)
- MD stability 10K/100K NVE (done)
- Pipeline profiling (52 models, all stages) (done)

## Remaining

1. rMD17 benzene: diversity / trust_level baselines + MD stability
2. GPU utilization/memory curves for full training round (nvidia-smi dmon)
3. H100 / multi-GPU scaling experiments
4. Additional molecular or material systems beyond rMD17
