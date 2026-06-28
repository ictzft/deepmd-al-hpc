#!/usr/bin/env python3
"""Extract and summarize all four-strategy results for rMD17 benzene."""
import json, re, csv
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("/data/guyida/deepmd-al-hpc")


def parse_test_log(path: Path) -> dict:
    """Parse DeePMD test.log for RMSE values."""
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    result = {}
    for key, pattern in [
        ("energy_rmse", r"Energy RMSE\s*:\s*([0-9.eE+-]+)\s*eV"),
        ("force_rmse", r"Force\s+RMSE\s*:\s*([0-9.eE+-]+)\s*eV/Å"),
        ("energy_rmse_natoms", r"Energy RMSE/Natoms\s*:\s*([0-9.eE+-]+)\s*eV"),
    ]:
        vals = re.findall(pattern, text)
        if vals:
            result[key] = float(vals[-1])
    return result


def parse_selection_json(path: Path) -> dict:
    """Parse selected_*.json for deviation statistics."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(errors="replace"))
    selected = data.get("selected_frames", [])
    force_vals = [float(x["force_dev_max"]) for x in selected if "force_dev_max" in x]
    energy_vals = [float(x["energy_dev"]) for x in selected if "energy_dev" in x]
    return {
        "n_frames": data.get("n_frames"),
        "top_k": data.get("top_k", len(selected)),
        "force_dev_max_mean": mean(force_vals) if force_vals else None,
        "force_dev_max_max": max(force_vals) if force_vals else None,
        "force_dev_max_min": min(force_vals) if force_vals else None,
        "energy_dev_mean": mean(energy_vals) if energy_vals else None,
    }


def get_strategy_results(strategy: str, model_dir_template: str,
                         pred_dir_template: str, sel_file_template: str,
                         seeds: list, rounds: list):
    """Extract results for one strategy across all seeds and rounds."""
    rows = []
    for seed in seeds:
        for rd in rounds:
            model_dir = ROOT / model_dir_template.format(seed=seed, round=rd)
            pred_dir = ROOT / pred_dir_template.format(seed=seed, round=rd)
            sel_json = pred_dir / sel_file_template.format(seed=seed, round=rd)

            models = {}
            for mp in sorted(model_dir.glob("model_*")) if model_dir.exists() else []:
                if not mp.is_dir():
                    continue
                metrics = parse_test_log(mp / "test.log")
                if metrics:
                    models[mp.name] = metrics

            pred = parse_selection_json(sel_json)

            force_vals = [m["force_rmse"] for m in models.values() if "force_rmse" in m]
            energy_vals = [m["energy_rmse"] for m in models.values() if "energy_rmse" in m]

            rows.append({
                "strategy": strategy,
                "seed": seed,
                "round": rd,
                "n_models": len(models),
                "force_rmse_mean": mean(force_vals) if force_vals else None,
                "force_rmse_std": stdev(force_vals) if len(force_vals) >= 2 else None,
                "force_rmse_min": min(force_vals) if force_vals else None,
                "force_rmse_max": max(force_vals) if force_vals else None,
                "energy_rmse_mean": mean(energy_vals) if energy_vals else None,
                "force_dev_max_mean": pred.get("force_dev_max_mean"),
                "force_dev_max_max": pred.get("force_dev_max_max"),
                "energy_dev_mean": pred.get("energy_dev_mean"),
                "pred_n_frames": pred.get("n_frames"),
            })
    return rows


def uncertainty_rows():
    """Uncertainty branch uses different naming convention."""
    rows = []
    for rd in [0, 1, 2, 3]:
        model_dir = ROOT / f"experiments/rmd17_benzene_round{rd:03d}_committee_models"
        pred_dir = ROOT / f"experiments/rmd17_benzene_round{rd:03d}_committee_prediction"
        sel_json = pred_dir / "selected_uncertainty_1000.json"

        models = {}
        for mp in sorted(model_dir.glob("model_*")) if model_dir.exists() else []:
            if not mp.is_dir():
                continue
            metrics = parse_test_log(mp / "test.log")
            if metrics:
                models[mp.name] = metrics

        pred = parse_selection_json(sel_json)
        force_vals = [m["force_rmse"] for m in models.values() if "force_rmse" in m]
        energy_vals = [m["energy_rmse"] for m in models.values() if "energy_rmse" in m]

        rows.append({
            "strategy": "uncertainty",
            "seed": "N/A",
            "round": rd,
            "n_models": len(models),
            "force_rmse_mean": mean(force_vals) if force_vals else None,
            "force_rmse_std": stdev(force_vals) if len(force_vals) >= 2 else None,
            "force_rmse_min": min(force_vals) if force_vals else None,
            "force_rmse_max": max(force_vals) if force_vals else None,
            "energy_rmse_mean": mean(energy_vals) if energy_vals else None,
            "force_dev_max_mean": pred.get("force_dev_max_mean"),
            "force_dev_max_max": pred.get("force_dev_max_max"),
            "energy_dev_mean": pred.get("energy_dev_mean"),
            "pred_n_frames": pred.get("n_frames"),
        })
    return rows


def main():
    OUT_DIR = ROOT / "experiments/rmd17_benzene_summary"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    # 1. Uncertainty branch
    print("Collecting uncertainty branch...")
    all_rows.extend(uncertainty_rows())

    # 2. Random baseline
    print("Collecting random baseline...")
    all_rows.extend(get_strategy_results(
        "random",
        "experiments/rmd17_benzene_random_seed{seed}_round{round:03d}_committee_models",
        "experiments/rmd17_benzene_random_seed{seed}_round{round:03d}_selection",
        "selected_random_1000.json",
        seeds=[0, 1, 2], rounds=[1, 2, 3],
    ))

    # 3. Diversity baseline (benzene-specific naming)
    print("Collecting diversity baseline...")
    all_rows.extend(get_strategy_results(
        "diversity",
        "experiments/baselines/rmd17_benzene_diversity_seed{seed}_round00{round}_committee_models",
        "experiments/baselines/rmd17_benzene_diversity_seed{seed}_round00{round}_committee_prediction",
        "selected_diversity.json",
        seeds=[0, 1, 2], rounds=[1, 2, 3],
    ))

    # 4. Trust-level baseline (benzene-specific naming)
    print("Collecting trust_level baseline...")
    all_rows.extend(get_strategy_results(
        "trust_level",
        "experiments/baselines/rmd17_benzene_trust_level_seed{seed}_round00{round}_committee_models",
        "experiments/baselines/rmd17_benzene_trust_level_seed{seed}_round00{round}_committee_prediction",
        "selected_trust_level.json",
        seeds=[0, 1, 2], rounds=[1, 2, 3],
    ))

    # Write detailed CSV
    fields = ["strategy", "seed", "round", "n_models",
              "force_rmse_mean", "force_rmse_std", "force_rmse_min", "force_rmse_max",
              "energy_rmse_mean", "force_dev_max_mean", "force_dev_max_max",
              "energy_dev_mean", "pred_n_frames"]

    csv_path = OUT_DIR / "all_strategies_detail.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {csv_path} ({len(all_rows)} rows)")

    # Generate cross-seed summary per strategy per round
    print("\n=== Four-Strategy Cross-Seed Mean ± Std (Round 3, Force RMSE) ===")
    for rd in [1, 2, 3]:
        print(f"\n--- Round {rd} ---")
        for strategy in ["uncertainty", "random", "diversity", "trust_level"]:
            strat_rows = [r for r in all_rows if r["strategy"] == strategy and r["round"] == rd]
            if not strat_rows:
                print(f"  {strategy}: NO DATA")
                continue
            force_vals = [r["force_rmse_mean"] for r in strat_rows if r["force_rmse_mean"] is not None]
            if force_vals:
                if len(force_vals) >= 2:
                    print(f"  {strategy:15s}: {mean(force_vals):.4e} ± {stdev(force_vals):.4e} (n={len(force_vals)})")
                else:
                    print(f"  {strategy:15s}: {mean(force_vals):.4e} (n={len(force_vals)})")

    # Generate comparison table (cross-seed mean per strategy per round)
    compare_path = OUT_DIR / "four_strategy_comparison.csv"
    with compare_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "round", "n_seeds", "force_rmse_mean", "force_rmse_std",
                     "energy_rmse_mean", "force_dev_max_mean"])
        for strategy in ["uncertainty", "random", "diversity", "trust_level"]:
            for rd in [1, 2, 3]:
                strat_rows = [r for r in all_rows if r["strategy"] == strategy and r["round"] == rd]
                force_vals = [r["force_rmse_mean"] for r in strat_rows if r["force_rmse_mean"] is not None]
                energy_vals = [r["energy_rmse_mean"] for r in strat_rows if r["energy_rmse_mean"] is not None]
                dev_vals = [r["force_dev_max_mean"] for r in strat_rows if r["force_dev_max_mean"] is not None]
                if force_vals:
                    w.writerow([
                        strategy, rd, len(force_vals),
                        f"{mean(force_vals):.6e}",
                        f"{stdev(force_vals):.6e}" if len(force_vals) >= 2 else "N/A",
                        f"{mean(energy_vals):.6e}" if energy_vals else "N/A",
                        f"{mean(dev_vals):.6e}" if dev_vals else "N/A",
                    ])
    print(f"\nWrote {compare_path}")

    # Also check for outlier models (Force RMSE > 1.0)
    print("\n=== Outlier Models (Force RMSE > 1.0 eV/Å) ===")
    for row in all_rows:
        if row["force_rmse_max"] and row["force_rmse_max"] > 1.0:
            print(f"  {row['strategy']} seed={row['seed']} round={row['round']}: "
                  f"F_RMSE max={row['force_rmse_max']:.4e}, mean={row['force_rmse_mean']:.4e}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
