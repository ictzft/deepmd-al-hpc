"""Generate final uncertainty vs random comparison learning curve for rMD17 ethanol."""
import csv
from pathlib import Path

OUT_DIR = Path("/data/guyida/deepmd-al-hpc/experiments/rmd17_ethanol_summary")

# Read data
test_rows = {}
with (OUT_DIR / "independent_test_all_summary.csv").open() as f:
    for r in csv.DictReader(f):
        key = (int(r["round"]), r["strategy"])
        test_rows[key] = float(r["force_rmse_mean"])

def scale(v, vmin, vmax, lo, hi):
    if vmax == vmin: return (lo + hi) / 2
    return lo + (v - vmin) / (vmax - vmin) * (hi - lo)

def make_svg(path, title, xlabel, ylabel, x_vals, series):
    w, h = 760, 480
    L, R, T, B = 90, 40, 60, 70
    pw, ph = w - L - R, h - T - B

    all_y = [v for _, vals in series for v in vals if v is not None]
    ymin, ymax = min(all_y), max(all_y)
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 1.0
    ymin -= pad; ymax += pad

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{w/2}" y="32" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>',
             f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="black" stroke-width="1.5"/>',
             f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="black" stroke-width="1.5"/>']

    xmin, xmax = min(x_vals), max(x_vals)
    for x in x_vals:
        xp = scale(x, xmin, xmax, L, L + pw)
        parts.append(f'<line x1="{xp}" y1="{T+ph}" x2="{xp}" y2="{T+ph+6}" stroke="black"/>')
        parts.append(f'<text x="{xp}" y="{T+ph+25}" text-anchor="middle" font-size="13" font-family="Arial">{int(x)}</text>')

    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        yp = scale(yv, ymin, ymax, T + ph, T)
        parts.append(f'<line x1="{L-6}" y1="{yp}" x2="{L}" y2="{yp}" stroke="black"/>')
        parts.append(f'<text x="{L-10}" y="{yp+4}" text-anchor="end" font-size="12" font-family="Arial">{yv:.4f}</text>')
        parts.append(f'<line x1="{L}" y1="{yp}" x2="{L+pw}" y2="{yp}" stroke="#dddddd" stroke-width="0.5"/>')

    parts.append(f'<text x="{L+pw/2}" y="{h-20}" text-anchor="middle" font-size="15" font-family="Arial">{xlabel}</text>')
    parts.append(f'<text x="22" y="{T+ph/2}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 22 {T+ph/2})">{ylabel}</text>')

    lx, ly = L + pw - 220, T + 10
    for idx, (name, values) in enumerate(series):
        color = colors[idx % len(colors)]
        pts = [(scale(x, xmin, xmax, L, L+pw), scale(y, ymin, ymax, T+ph, T))
               for x, y in zip(x_vals, values) if y is not None]
        if len(pts) >= 2:
            poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for xp, yp in pts:
            parts.append(f'<circle cx="{xp:.2f}" cy="{yp:.2f}" r="5" fill="{color}"/>')
        ly_pos = ly + idx * 22
        parts.append(f'<line x1="{lx}" y1="{ly_pos}" x2="{lx+25}" y2="{ly_pos}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{lx+32}" y="{ly_pos+4}" font-size="13" font-family="Arial">{name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))

# Build data
rounds = [0, 1, 2, 3]
unc_test = [test_rows.get((r, "uncertainty")) for r in rounds]
rnd_test = [None] + [test_rows.get((r, "random")) for r in [1, 2, 3]]

make_svg(
    OUT_DIR / "rmd17_ethanol_final_comparison.svg",
    "rMD17 Ethanol — Independent Test Force RMSE: Uncertainty vs Random",
    "Active learning round",
    "Force RMSE (eV/Å)",
    rounds,
    [("Uncertainty", unc_test), ("Random (3-seed mean)", rnd_test)],
)

# Also read valid data for 2-panel comparison
valid_rows = {}
with (OUT_DIR / "al_rounds_summary.csv").open() as f:
    for r in csv.DictReader(f):
        valid_rows[int(r["round"])] = float(r["force_rmse_mean"])

make_svg(
    OUT_DIR / "rmd17_ethanol_valid_vs_test.svg",
    "rMD17 Ethanol — Force RMSE: Validation vs Independent Test (Uncertainty)",
    "Active learning round",
    "Force RMSE (eV/Å)",
    rounds,
    [("Validation (5000 frames)", [valid_rows.get(r) for r in rounds]),
     ("Independent Test (10000 frames)", unc_test)],
)

print("Saved figures:")
for p in sorted(OUT_DIR.glob("rmd17_ethanol_final*")):
    print(p)
for p in sorted(OUT_DIR.glob("rmd17_ethanol_valid_vs*")):
    print(p)
