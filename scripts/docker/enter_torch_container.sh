#!/bin/bash

cd /data/zft

docker run --rm -it \
  --gpus all \
  -v /data/zft:/data/zft \
  -w /data/zft \
  cuda-torch:cuda11.3-cudnn8-ubuntu18.04-torch2.4 \
  bash
