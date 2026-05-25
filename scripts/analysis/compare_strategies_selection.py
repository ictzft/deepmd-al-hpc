#!/usr/bin/env python3
"""Compare all selection strategies at the selection level (no retraining)."""
import json
import csv
import statistics
from pathlib import Path

PROJ = Path("/data/zft/deepmd-al-hpc")
baselines = PROJ / "experiments" / "baselines"

strategy_files = {
    "uncertainty": baselines / "uncertainty_topk_round000" / "selected_topk.json",
    "random_seed0": PROJ / "experiments" / "exp_005_committee_prediction" / "selected_random_seed0.json",
    "random_seed1": PROJ / "experiments" / "exp_005_committee_prediction" / "selected_random_seed1.json",
    "random_seed2": PROJ / "experiments" / "exp_005_committee_prediction" / "selected_random_seed2.json",
    "diversity": baselines / "diversity_round000" / "selected_topk.json",
    "trust_level": baselines / "trust_level_round000" / "selected_topk.json",
}


def main() -> None:
    print("===== Round 0 Strategy Comparison (selection-level) =====")
    print()

    rows = []
    for name, path in sorted(strategy_files.items()):
        if not path.exists():
            print(f"  {name}: MISSING ({path})")
            continue
        d = json.loads(path.read_text())
        fdm = [x["force_dev_max"] for x in d["selected_frames"]]
        m = statistics.mean(fdm)
        s = statistics.stdev(fdm) if len(fdm) > 1 else 0.0
        rmin = min(fdm)
        rmax = max(fdm)
        print(f"  {name:18s}: mean={m:.6e}  std={s:.6e}  min={rmin:.6e}  max={rmax:.6e}")
        rows.append({
            "strategy": name,
            "selected_force_dev_max_mean": m,
            "selected_force_dev_max_std": s,
            "selected_force_dev_max_min": rmin,
            "selected_force_dev_max_max": rmax,
            "n_selected": len(fdm),
        })

    # CSV
    csv_path = baselines / "strategy_comparison_round000.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.6e}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"\nWrote: {csv_path}")

    # MD
    md_lines = [
        "# Round 0 Strategy Comparison (Selection-level)",
        "",
        "Selection from the initial 50-frame candidate pool using different strategies.",
        "This is a **selection-level** comparison — it shows what each strategy selects,",
        "not the downstream retraining effect.",
        "",
        "## Selected force_dev_max statistics",
        "",
        "| Strategy | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['strategy']} | {r['selected_force_dev_max_mean']:.6e} | "
            f"{r['selected_force_dev_max_std']:.6e} | "
            f"{r['selected_force_dev_max_min']:.6e} | "
            f"{r['selected_force_dev_max_max']:.6e} |"
        )
    md_lines.extend([
        "",
        "## Strategy descriptions",
        "",
        "- **uncertainty**: pure top-K by `force_dev_max`. Selects the 10 highest-uncertainty frames.",
        "- **random_seed{0,1,2}**: random sampling with different random seeds (for baseline).",
        "- **diversity**: top-M=30 by `force_dev_max`, then Farthest Point Sampling on pairwise-distance",
        "  descriptor to select 10 structurally diverse frames. Reduces mean selected uncertainty",
        "  compared to pure top-K but improves structural coverage.",
        "- **trust_level**: DP-GEN-style prototype. Splits candidate pool into accurate (<p50),",
        "  candidate (p50-p90), and failed (>p90) regions. Selects 80% from candidate + 20% from failed.",
        "",
        "**Important**: This is a toy H2 prototype. Full multi-round active learning retraining",
        "has not yet been run for diversity or trust_level strategies.",
    ])
    md_path = baselines / "strategy_comparison_round000.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
