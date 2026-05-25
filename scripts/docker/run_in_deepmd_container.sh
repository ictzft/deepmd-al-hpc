#!/bin/bash
# Run a command inside the DeepMD-kit Docker container (non-interactive).
# Usage: bash scripts/docker/run_in_deepmd_container.sh <command...>
set -euo pipefail

PROJECT_ROOT="/data/zft/deepmd-al-hpc"

docker run --rm \
  --gpus all \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -v /data/zft:/data/zft \
  -w "${PROJECT_ROOT}" \
  ghcr.io/deepmodeling/deepmd-kit:master \
  bash -lc "export PATH=/opt/deepmd-kit/bin:\$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:\$LD_LIBRARY_PATH; $*"
