#!/bin/bash
# Record GPU utilization, memory, power via nvidia-smi dmon.
# Usage:
#   bash scripts/profiling/run_nvidia_smi_dmon.sh <output_csv> <interval_sec>
#   bash scripts/profiling/run_nvidia_smi_dmon.sh experiments/profiling/v100/gpu_logs/gpu.csv 1
#
# To run in background and capture PID:
#   bash scripts/profiling/run_nvidia_smi_dmon.sh out.csv 1 &
#   DMON_PID=$!
#   ... do work ...
#   kill $DMON_PID

set -euo pipefail

OUTPUT_CSV="${1:-experiments/profiling/v100/gpu_logs/gpu_dmon.csv}"
INTERVAL="${2:-2}"

OUTPUT_DIR="$(dirname "${OUTPUT_CSV}")"
mkdir -p "${OUTPUT_DIR}"

if ! command -v nvidia-smi &> /dev/null; then
    echo "[dmon] nvidia-smi not found — running in no-op mode." | tee "${OUTPUT_CSV}"
    echo "timestamp,util_gpu_pct,mem_used_mib,mem_total_mib,power_w" >> "${OUTPUT_CSV}"
    echo "$(date -Iseconds),0,0,0,0" >> "${OUTPUT_CSV}"
    echo "[dmon] (no-op mode: wrote placeholder row)"
    exit 0
fi

echo "[dmon] Starting nvidia-smi dmon (interval=${INTERVAL}s, output=${OUTPUT_CSV})"

# Write CSV header
echo "timestamp,util_gpu_pct,mem_used_mib,mem_total_mib,power_w" > "${OUTPUT_CSV}"

# Run dmon: p=power, u=utilization, m=memory, t=temperature
# Parse into simple CSV format
nvidia-smi dmon -s pum -d "${INTERVAL}" -o DT 2>/dev/null | \
  awk -v OFS=',' '
    BEGIN { skip=2 }
    NR>skip && $1 !~ /^#/ && NF>3 {
      ts=strftime("%Y-%m-%dT%H:%M:%S%z")
      for(i=1;i<=NF;i++) { gsub(/[^0-9.-]/,"",$i) }
      # GPU 0: cols 2(sm), 3(mem), 4(pwr)
      printf "%s,%.1f,%.0f,%.0f,%.1f\n", ts, $2, $3*16384/100, 16384, $4
    }
  ' >> "${OUTPUT_CSV}"

echo "[dmon] Stopped."
