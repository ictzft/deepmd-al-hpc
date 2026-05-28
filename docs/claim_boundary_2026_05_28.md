# Claim Boundary and Remaining Work

Updated: 2026-05-28

## What is currently supported

1. The repository implements an offline active-learning prototype for DeePMD committee models.
2. The workflow has been validated on toy H2 and two rMD17 molecular systems: ethanol and benzene.
3. For rMD17 benzene, uncertainty selection Round000-003 and random seed0/1/2 Round001-003 have been completed with independent-test evaluation.
4. For rMD17 ethanol, uncertainty and random baselines have been completed, with independent-test and short-horizon MD stability sanity checks.
5. V100 experiments demonstrate that the pipeline can run committee training, prediction, selection, dataset update, and retraining loops.

## What should not be overclaimed

1. The current workflow is offline active learning, not a fully online DFT/AIMD labeling system.
2. H100 and multi-GPU scaling experiments have not yet been completed.
3. 10K NVE stability only supports short-horizon stability. 100K+ dissociation indicates that the current potential is not yet sufficient for long-time MD.
4. Current evidence does not prove that uncertainty is statistically significantly better than diversity or trust-level across all cases.
5. Strategy-level conclusions should be phrased carefully: uncertainty is competitive and often better than random, but active-strategy ranking remains inconclusive.
6. V100 profiling currently provides baseline timing evidence; full utilization, throughput, end-to-end speedup, and scaling analysis still need to be added.

## Paper-safe wording

The current system should be described as an offline active-learning prototype for DeePMD committee-model training and selection. It uses existing labeled rMD17 data to emulate the expensive DFT labeling step. The system has been evaluated on toy H2 and two rMD17 molecular systems, ethanol and benzene. Current results support the feasibility of the closed-loop workflow and show that uncertainty-based selection can outperform random sampling in several independent-test settings. However, online DFT labeling, H100 scaling, and long-horizon MD stability remain future work.
