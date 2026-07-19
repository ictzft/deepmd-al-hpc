#!/bin/bash
# #12 Long-training comparison: does the "MPS ≈ wave wall" result hold when
# compute dominates (2000 steps) instead of fixed overhead (100 steps)?
# Runs wave G=4 and MPS G=1 (batch=8), ethanol N=4, 2000 steps, sequentially.
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "===== LONG TRAINING (2000 steps): wave G=4 vs MPS G=1, ethanol N=4 ====="
bash scripts/scaling/run_wave.sh 4 4 2000 long_wave_n4_g4
echo ""
bash scripts/scaling/run_mps.sh 4 1 2000 0 long_mps_n4_g1_b8
echo ""
echo "===== long-training wall comparison ====="
python3 -c "
import json
rows=[]
for tag,ngpu in [('long_wave_n4_g4',4),('long_mps_n4_g1_b8',1)]:
    d=json.load(open(f'experiments/scaling/{tag}/summary.json'))
    rows.append((tag,d['total_wall_s'],ngpu,d['n_ok']))
    print(f'{tag}: wall={d[\"total_wall_s\"]}s n_ok={d[\"n_ok\"]}/4 GPU={ngpu}')
if len(rows)==2:
    wwave=rows[0][1]; wmps=rows[1][1]
    print(f'wall ratio MPS/wave = {wmps/wwave:.2f}  (wave 4 GPUs, MPS 1 GPU)')
"
