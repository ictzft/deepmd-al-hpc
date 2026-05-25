# Next Steps After Random Baseline

> 2026-05-25: Random baseline (seed0/seed1/seed2 Round 001–003) is complete. This document describes the remaining V100 work and recommended execution order.

---

## 1. Why random baseline is sufficient for the toy H2 stage

The random baseline now covers 3 seeds × 3 rounds = 9 independent committee training runs, plus multi-seed mean ± std and full uncertainty vs random comparison. Further random rounds on toy H2 add diminishing returns. The next V100 work should focus on profiling and stronger baseline variants.

---

## 2. Remaining profiling work

1. End-to-end round wall-clock time (training + freeze + test + prediction + dataset update)
2. Per-stage breakdown: prediction time, deviation calculation time, selection time, dataset update time
3. GPU utilization and memory curves via `nvidia-smi dmon` during full training
4. See `docs/profiling_v100.md` for detailed commands.

---

## 3. Uncertainty-diversity sampling

**Status**: Selector implemented in `src/al/selector.py`, Round 001 validated.

**Next**:
1. Run Round 002–003 for uncertainty-diversity (same pipeline as random baseline)
2. Compare selected force_dev_max distribution vs pure top-K vs random
3. Quantify structural diversity improvement

Commands:
```bash
# Selection-level (no training needed)
python scripts/selection/select_by_strategy.py \
  --predictions experiments/.../committee_predictions.npz \
  --strategy uncertainty-diversity \
  --top-k 10 --top-m 30 \
  --output experiments/baselines/diversity_roundXXX/selected_topk.json

# Full round (data prep + train + predict)
# Follow pattern in scripts/run_random_baseline_round.sh
```

---

## 4. Trust-level / DP-GEN-style baseline

**Status**: Selector implemented in `src/al/selector.py`, Round 001 validated.

**Next**:
1. Run Round 002–003 for trust-level
2. Tune low_pct / high_pct thresholds based on candidate pool statistics
3. Compare accurate/candidate/failed region distributions across rounds

---

## 5. Four-strategy comparison

When diversity and trust-level both have multi-round results:
- Generate `strategy_comparison_summary.csv` with all 4 strategies
- Generate comparison figures via `plot_random_vs_uncertainty.py` (extend to 4 strategies)

---

## 6. Real DFT/AIMD migration

After toy H2 is fully characterized:
1. Acquire or generate a small real DFT/AIMD dataset
2. Convert to DeepMD npy format
3. Re-run the same pipeline (uncertainty + random + diversity + trust-level)
4. Compare toy H2 results vs real dataset results

---

## 7. Recommended V100 execution order

```text
1. [done] Sync docs and summary scripts
2. [done] Random baseline Round 001–003
3. [now] Check local data and artifact inventory
4. Complete measured profiling (prediction, I/O, end-to-end)
5. Uncertainty-diversity Round 002–003
6. Trust-level Round 002–003
7. Full 4-strategy comparison
8. Prepare small real DFT/AIMD dataset
```
