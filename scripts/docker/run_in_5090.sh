#!/bin/bash
# Run a command inside the deepmd-5090 (PyTorch backend) container, non-interactive.
# All GPU / training / inference operations should go through this wrapper so the
# host stays free of deepmd-kit / torch.
#
# Usage:
#   bash scripts/docker/run_in_5090.sh [gpu_ids] -- <command...>
#   gpu_ids: "all" (default) | "0,1,2,3" | "0" ...
#
# Examples:
#   bash scripts/docker/run_in_5090.sh all  -- dp --version
#   bash scripts/docker/run_in_5090.sh 0    -- dp train config.json
#   bash scripts/docker/run_in_5090.sh 0,1,2,3 -- python myscript.py
set -euo pipefail

GPU_DEVICES="${1:-all}"

# consume first positional arg unless it is the "--" separator
if [ "${1:-}" != "--" ] && [ "${1:-}" != "" ]; then
  shift
fi
# consume the optional "--" separator
if [ "${1:-}" = "--" ]; then
  shift
fi

if [ $# -eq 0 ]; then
  echo "Usage: $0 [gpu_ids] -- <command...>" >&2
  echo "  e.g. $0 all -- dp --version" >&2
  exit 1
fi

if [ "${GPU_DEVICES}" = "all" ]; then
  GPU_ARG=(--gpus all)
else
  GPU_ARG=(--gpus "\"device=${GPU_DEVICES}\"")
fi

docker run --rm \
  "${GPU_ARG[@]}" \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e DEEPMD_BACKEND=pytorch \
  -v /home/zhufantao/deepmd-work/deepmd-al-hpc:/data/zft/deepmd-al-hpc \
  -w /data/zft/deepmd-al-hpc \
  deepmd-5090:latest \
  bash -lc "$*"
