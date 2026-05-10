#!/bin/bash

cd /data/zft

docker run --rm -it \
  --gpus all \
  --user $(id -u):$(id -g) \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /data/zft:/data/zft \
  -w /data/zft \
  cuda-torch:cuda11.3-cudnn8-ubuntu18.04-torch2.4 \
  bash
