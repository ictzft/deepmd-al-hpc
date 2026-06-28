#!/usr/bin/env python3
"""Generate four-strategy comparison learning curve for rMD17 benzene."""
import csv
from pathlib import Path
from statistics import mean

OUT_DIR = Path("/data/guyida/deepmd-al-hpc/experiments/rmd17_benzene_summary")
DATA = OUT_DIR / "all_strategies_detail.csv"

# Read data
unc_rows, rnd_rows, div_rows, trust_rows = {}, {}, {}, {}
with DATA.open() as f:
    for r in csv.DictReader(f):
        strategy = r["strategy"]
        rd = int(r["round"])
        fm = float(r["force_rmse_mean"]) if r["force_rmse_mean"] else None
        if strategy == "uncertainty":
            unc_rows[rd] = fm
        elif strategy == "random":
            rnd_rows.setdefault(rd, []).append(fm)
        elif strategy == "diversity":
            div_rows.setdefault(rd, []).append(fm)
        elif strategy == "trust_level":
            trust_rows.setdefault(rd, []).append(fm)


def scale(v, vmin, vmax, lo, hi):
    if vmax == vmin:
        return (lo + hi) / 2
    return lo + (v - vmin) / (vmax - vmin) * (hi - lo)


def make_svg(path, title, xlabel, ylabel, x_vals, series):
    w, h = 760, 480
    L, R, T, B = 90, 40, 60, 70
    pw, ph = w - L - R, h - T - B

    all_y = [v for _, vals in series for v in vals if v is not None]
    if not all_y:
        print(f"  WARNING: no data for {title}")
        return
    ymin, ymax = min(all_y), max(all_y)
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.05
    ymin -= pad
    ymax += pad

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="32" text-anchor="middle" font-size="18" font-family="Arial" font-weight="bold">{title}</text>',
        f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="black" stroke-width="1.5"/>',
        f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="black" stroke-width="1.5"/>',
    ]

    xmin, xmax = min(x_vals), max(x_vals)
    for x in x_vals:
        xp = scale(x, xmin, xmax, L, L + pw)
        parts.append(f'<line x1="{xp}" y1="{T+ph}" x2="{xp}" y2="{T+ph+6}" stroke="black"/>')
        parts.append(f'<text x="{xp}" y="{T+ph+22}" text-anchor="middle" font-size="13" font-family="Arial">{int(x)}</text>')

    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        yp = scale(yv, ymin, ymax, T + ph, T)
        parts.append(f'<line x1="{L-6}" y1="{yp}" x2="{L}" y2="{yp}" stroke="black"/>')
        parts.append(f'<text x="{L-10}" y="{yp+4}" text-anchor="end" font-size="12" font-family="Arial">{yv:.4f}</text>')
        parts.append(f'<line x1="{L}" y1="{yp}" x2="{L+pw}" y2="{yp}" stroke="#ddd" stroke-width="0.5"/>')

    parts.append(f'<text x="{L+pw/2}" y="{h-22}" text-anchor="middle" font-size="15" font-family="Arial">{xlabel}</text>')
    parts.append(f'<text x="22" y="{T+ph/2}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 22 {T+ph/2})">{ylabel}</text>')

    lx, ly = L + pw - 250, T + 10
    for idx, (name, values) in enumerate(series):
        color = colors[idx % len(colors)]
        pts = [(scale(x, xmin, xmax, L, L+pw), scale(y, ymin, ymax, T+ph, T))
               for x, y in zip(x_vals, values) if y is not None]
        if len(pts) >= 2:
            poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for xp, yp in pts:
            parts.append(f'<circle cx="{xp:.2f}" cy="{yp:.2f}" r="5" fill="{color}"/>')
        parts.append(f'<line x1="{lx}" y1="{ly+idx*22}" x2="{lx+25}" y2="{ly+idx*22}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{lx+32}" y="{ly+idx*22+4}" font-size="13" font-family="Arial">{name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))
    print(f"  Saved: {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rounds = [1, 2, 3]  # Cross-seed comparison rounds

    # 1. Four-strategy Force RMSE comparison (cross-seed mean)
    make_svg(
        OUT_DIR / "rmd17_benzene_four_strategy_force_rmse.svg",
        "rMD17 Benzene — Four-Strategy Force RMSE (Validation, 3-seed mean)",
        "Round",
        "Force RMSE (eV/Å)",
        rounds,
        [
            ("uncertainty", [unc_rows.get(r) for r in rounds]),
            ("random", [mean(rnd_rows.get(r, [])) if rnd_rows.get(r) else None for r in rounds]),
            ("diversity", [mean(div_rows.get(r, [])) if div_rows.get(r) else None for r in rounds]),
            ("trust_level", [mean(trust_rows.get(r, [])) if trust_rows.get(r) else None for r in rounds]),
        ],
    )

    # 2. Uncertainty-only learning curve (Round 0-3)
    all_rounds = [0, 1, 2, 3]
    make_svg(
        OUT_DIR / "rmd17_benzene_uncertainty_learning_curve.svg",
        "rMD17 Benzene — Uncertainty Branch Force RMSE (Round 0-3)",
        "Round",
        "Force RMSE (eV/Å)",
        all_rounds,
        [("uncertainty", [unc_rows.get(r) for r in all_rounds])],
    )

    print("\nDone. Figures saved to experiments/rmd17_benzene_summary/")


if __name__ == "__main__":
    main()
