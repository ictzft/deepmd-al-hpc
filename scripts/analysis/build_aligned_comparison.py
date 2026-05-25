#!/usr/bin/env python3
"""Build aligned comparison table with consistent metrics across all strategies.

Key fix: use REMAINING CANDIDATE POOL force_dev_max_mean for all branches.
Previous version mixed selected top-K (uncertainty) with remaining candidate (random).
"""
import re, csv, statistics
from pathlib import Path
import numpy as np

PROJ = Path("/data/zft/deepmd-al-hpc")
BASELINES = PROJ / "experiments" / "baselines"
OUT_CSV = BASELINES / "aligned_comparison.csv"
OUT_MD = BASELINES / "aligned_comparison.md"


def extract_rmse_from_test_log(model_dir: Path) -> tuple:
    """Return (energy_rmse_mean, force_rmse_mean) across 4 models."""
    ev, fv = [], []
    for m in ["model_000", "model_001", "model_002", "model_003"]:
        log = model_dir / m / "test.log"
        if not log.exists():
            continue
        t = log.read_text()
        em = re.findall(r"Energy RMSE\s+:\s+(\S+) eV", t)
        fm = re.findall(r"Force  RMSE\s+:\s+(\S+) eV", t)
        if em and fm:
            ev.append(float(em[-1]))
            fv.append(float(fm[-1]))
    if ev:
        return statistics.mean(ev), statistics.mean(fv)
    return None, None


def extract_remaining_candidate_dev(pred_dir: Path) -> tuple:
    """Return (force_dev_max_mean, force_dev_max_std) from prediction NPZ."""
    npz = pred_dir / "committee_predictions.npz"
    if not npz.exists():
        return None, None
    data = np.load(npz, allow_pickle=True)
    fdm = data["force_dev_max"]
    ed = data["energy_dev"]
    return (float(np.mean(fdm)), float(np.std(fdm)), float(np.mean(ed)))


def main():
    rows = []

    # ---- Uncertainty branch (Round 0-3, now with multi-seed seed0/1/2) ----
    for r in [0, 1, 2, 3]:
        rs = f"{r:03d}" if r > 0 else "000"
        train_frames = 200 + r * 10
        cand_frames = 50 - r * 10

        if r == 0:
            # Round 0 has only one set of models (no multi-seed)
            e_val = None
            f_val = None
            # Read from al_rounds_summary.csv for Round 0
            al_csv = PROJ / "experiments" / "al_rounds_summary.csv"
            if al_csv.exists():
                with al_csv.open() as f:
                    for row in csv.DictReader(f):
                        if int(row["round"]) == r:
                            e_val = float(row["energy_rmse_mean"])
                            f_val = float(row["force_rmse_mean"])
            rows.append({
                "strategy": "uncertainty", "seed": "—", "round": r,
                "train_frames": train_frames, "candidate_frames": cand_frames,
                "energy_rmse": f"{e_val:.6e}" if e_val else "",
                "force_rmse": f"{f_val:.6e}" if f_val else "",
                "remaining_candidate_force_dev_max_mean": "",
                "remaining_candidate_force_dev_max_std": "",
                "remaining_candidate_energy_dev_mean": "",
                "metric_note": "Round 0 (initial committee, no multi-seed)",
            })
        else:
            # Rounds 1-3: multi-seed via seed0/seed1/seed2
            seed_e = []; seed_f = []
            for si, seed in enumerate(["seed0", "seed1", "seed2"]):
                model_dir = BASELINES / f"uncertainty_{seed}_round{rs}_committee_models"
                pred_dir = BASELINES / f"uncertainty_{seed}_round{rs}_committee_prediction"
                e_rmse, f_rmse = extract_rmse_from_test_log(model_dir)
                fdm_mean, fdm_std, ed_mean = extract_remaining_candidate_dev(pred_dir)
                if e_rmse: seed_e.append(e_rmse); seed_f.append(f_rmse)
                rows.append({
                    "strategy": "uncertainty", "seed": seed, "round": r,
                    "train_frames": train_frames, "candidate_frames": cand_frames,
                    "energy_rmse": f"{e_rmse:.6e}" if e_rmse else "",
                    "force_rmse": f"{f_rmse:.6e}" if f_rmse else "",
                    "remaining_candidate_force_dev_max_mean": f"{fdm_mean:.6e}" if fdm_mean else "",
                    "remaining_candidate_force_dev_max_std": "",
                    "remaining_candidate_energy_dev_mean": f"{ed_mean:.6e}" if ed_mean else "",
                    "metric_note": "remaining candidate pool",
                })
            if len(seed_f) >= 2:
                rows.append({"strategy": "uncertainty", "seed": "mean", "round": r,
                    "train_frames": train_frames, "candidate_frames": cand_frames,
                    "energy_rmse": f"{statistics.mean(seed_e):.6e}", "force_rmse": f"{statistics.mean(seed_f):.6e}",
                    "remaining_candidate_force_dev_max_mean": "", "remaining_candidate_force_dev_max_std": "",
                    "remaining_candidate_energy_dev_mean": "", "metric_note": "cross-seed mean (3 seeds)"})
                rows.append({"strategy": "uncertainty", "seed": "std", "round": r,
                    "train_frames": train_frames, "candidate_frames": cand_frames,
                    "energy_rmse": f"{statistics.stdev(seed_e):.6e}", "force_rmse": f"{statistics.stdev(seed_f):.6e}",
                    "remaining_candidate_force_dev_max_mean": "", "remaining_candidate_force_dev_max_std": "",
                    "remaining_candidate_energy_dev_mean": "", "metric_note": "cross-seed std (3 seeds)"})

    # ---- Random / Diversity / Trust-level (Round 1-3, seed0/1/2) ----
    for strategy, dir_prefix in [("random", "random"), ("diversity", "diversity"), ("trust_level", "trust_level")]:
        for r in range(1, 4):
            rs = f"{r:03d}"
            train_frames = 200 + r * 10
            cand_frames = 50 - r * 10

            seed_e = []
            seed_f = []
            seed_fdm = []
            seed_ed = []

            for si, seed in enumerate(["seed0", "seed1", "seed2"]):
                model_dir = BASELINES / f"{dir_prefix}_{seed}_round{rs}_committee_models"
                pred_dir = BASELINES / f"{dir_prefix}_{seed}_round{rs}_committee_prediction"

                e_rmse, f_rmse = extract_rmse_from_test_log(model_dir)
                fdm_mean, fdm_std, ed_mean = extract_remaining_candidate_dev(pred_dir)

                if e_rmse:
                    seed_e.append(e_rmse)
                    seed_f.append(f_rmse)

                rows.append({
                    "strategy": strategy, "seed": seed, "round": r,
                    "train_frames": train_frames, "candidate_frames": cand_frames,
                    "energy_rmse": f"{e_rmse:.6e}" if e_rmse else "",
                    "force_rmse": f"{f_rmse:.6e}" if f_rmse else "",
                    "remaining_candidate_force_dev_max_mean": f"{fdm_mean:.6e}" if fdm_mean else "",
                    "remaining_candidate_force_dev_max_std": f"{fdm_std:.6e}" if fdm_std else "",
                    "remaining_candidate_energy_dev_mean": f"{ed_mean:.6e}" if ed_mean else "",
                    "metric_note": "remaining candidate pool",
                })

            # Cross-seed mean row
            if len(seed_e) >= 2:
                rows.append({
                    "strategy": strategy, "seed": "mean", "round": r,
                    "train_frames": train_frames, "candidate_frames": cand_frames,
                    "energy_rmse": f"{statistics.mean(seed_e):.6e}",
                    "force_rmse": f"{statistics.mean(seed_f):.6e}",
                    "remaining_candidate_force_dev_max_mean": "",
                    "remaining_candidate_force_dev_max_std": "",
                    "remaining_candidate_energy_dev_mean": "",
                    "metric_note": "cross-seed mean (3 seeds)",
                })
                rows.append({
                    "strategy": strategy, "seed": "std", "round": r,
                    "train_frames": train_frames, "candidate_frames": cand_frames,
                    "energy_rmse": f"{statistics.stdev(seed_e):.6e}",
                    "force_rmse": f"{statistics.stdev(seed_f):.6e}",
                    "remaining_candidate_force_dev_max_mean": "",
                    "remaining_candidate_force_dev_max_std": "",
                    "remaining_candidate_energy_dev_mean": "",
                    "metric_note": "cross-seed std (3 seeds)",
                })

    # Write CSV
    fieldnames = [
        "strategy", "seed", "round", "train_frames", "candidate_frames",
        "energy_rmse", "force_rmse",
        "remaining_candidate_force_dev_max_mean", "remaining_candidate_force_dev_max_std",
        "remaining_candidate_energy_dev_mean", "metric_note",
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Write MD
    md = [
        "# Aligned Four-Strategy Comparison (Remaining Candidate Pool Metric)",
        "",
        "**Key fix**: All strategies now report `remaining_candidate_force_dev_max_mean`",
        "(force_dev_max of the candidate pool AFTER retraining), not the selected top-K value.",
        "This makes uncertainty and random/diversity/trust_level directly comparable.",
        "",
        "Generated: 2026-05-25 | Toy H2 | 2xV100",
        "",
        "## Force RMSE Comparison (multi-seed mean ± std)",
        "",
        "| Strategy | R1 F_RMSE | R2 F_RMSE | R3 F_RMSE |",
        "|---|---:|---:|---:|",
    ]

    strategies = ["uncertainty", "random", "diversity", "trust_level"]
    for s in strategies:
        vals = {}
        for r in [1, 2, 3]:
            mean_rows = [x for x in rows if x["strategy"] == s and x["seed"] == "mean" and x["round"] == r]
            std_rows = [x for x in rows if x["strategy"] == s and x["seed"] == "std" and x["round"] == r]
            if mean_rows:
                v = mean_rows[0]["force_rmse"]
                if std_rows and std_rows[0]["force_rmse"] != "":
                    v += f" +/- {std_rows[0]['force_rmse']}"
                vals[r] = v
            else:
                # fallback: non-seed row (Round 0)
                u_rows = [x for x in rows if x["strategy"] == s and x["seed"] == "—" and x["round"] == r]
                if u_rows:
                    vals[r] = u_rows[0]["force_rmse"]

        if vals:
            parts = [s]
            for r in [1, 2, 3]:
                parts.append(vals.get(r, "—"))
            md.append("| " + " | ".join(parts) + " |")

    md.extend([
        "",
        "## Notes",
        "",
        "- **All four strategies**: 3-seed (seed0/seed1/seed2) mean ± std for Round 1-3.",
        "- **Round 0** (uncertainty only): single run (initial committee).",
        "- `remaining_candidate_force_dev_max_mean`: mean force_dev_max of the candidate pool AFTER the current round retraining (NOT the selected frames).",
        "- This metric is directly comparable across all strategies.",
        "- Previous `random_vs_uncertainty_summary.csv` mixed selected top-K (uncertainty) with remaining candidate pool (random).",
    ])
    OUT_MD.write_text("\n".join(md))

    print(f"Wrote: {OUT_CSV}  ({len(rows)} rows)")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
