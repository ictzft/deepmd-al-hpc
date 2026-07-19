#!/bin/bash
# Profile a single ethanol model training with NVIDIA Nsight Systems.
# Goal: diagnose why GPU SM utilization is only ~6-9% during committee training
# (CCGrid 2027 roadmap Fig.5 root-cause analysis).
#
# Runs INSIDE the deepmd-5090 container. The .nsys-rep it produces is analysed
# afterwards by run_in_5090.sh + nsys stats (see follow-up step).
#
# Usage:
#   bash scripts/docker/run_in_5090.sh 0 -- bash scripts/profiling/run_nsight.sh
set -e
cd /data/zft/deepmd-al-hpc
mkdir -p experiments/nsight_prof

# 1. generate a short (30-step) training config from the ethanol template
python - <<'PY'
import json
p = "configs/deepmd/pt_rmd17_ethanol.json"
c = json.load(open(p))
c["training"]["numb_steps"] = 30
c["training"]["save_freq"] = 30
json.dump(c, open("experiments/nsight_prof/ethanol_30.json", "w"), indent=2)
print("wrote experiments/nsight_prof/ethanol_30.json (30 steps)")
PY

# 2. check nsys availability
echo "=== nsys ==="
nsys --version

# 3. profile 30 training steps (cuda kernels + nvtx + OS/runtime threads)
echo "=== nsys profile (30 steps) ==="
cd experiments/nsight_prof
nsys profile -t cuda,nvtx,osrt -o ethanol_30 -f true -- \
  dp -b pytorch train /data/zft/deepmd-al-hpc/experiments/nsight_prof/ethanol_30.json \
  2>&1 | tail -15

echo "=== generated trace ==="
ls -la ethanol_30.nsys-rep 2>/dev/null && echo "trace OK" || echo "trace MISSING"
