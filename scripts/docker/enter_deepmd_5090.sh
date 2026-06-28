#!/bin/bash
# Enter deepmd-5090 Docker container
# Usage: bash scripts/docker/enter_deepmd_5090.sh [device_ids]
#   device_ids: optional, e.g., "0,1,2,3" (default: all GPUs)

GPU_DEVICES="${1:-all}"

docker run --rm -it \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -e DEEPMD_BACKEND=pytorch \
  -v /home/zhufantao/deepmd-work/deepmd-al-hpc:/data/zft/deepmd-al-hpc \
  -w /data/zft/deepmd-al-hpc \
  deepmd-5090:latest \
  bash
