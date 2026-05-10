#!/bin/bash

cd /data/zft

docker run --rm -it \
  --gpus all \
  --user $(id -u):$(id -g) \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -v /data/zft:/data/zft \
  -w /data/zft \
  ghcr.io/deepmodeling/deepmd-kit:master \
  bash -lc 'export PATH=/opt/deepmd-kit/bin:$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH; export PS1="deepmd-container:\w\$ "; exec bash --noprofile --norc'
