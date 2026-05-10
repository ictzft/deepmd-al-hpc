#!/bin/bash

docker run --rm \
  --gpus all \
  -v /data/zft:/data/zft \
  -w /data/zft \
  ghcr.io/deepmodeling/deepmd-kit:master \
  bash -lc '
    export PATH=/opt/deepmd-kit/bin:$PATH
    export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH

    echo "========== GPU =========="
    nvidia-smi

    echo "========== Python =========="
    which python
    python --version

    echo "========== DeePMD-kit dp =========="
    which dp
    dp -h | head -n 30

    echo "========== LAMMPS lmp =========="
    which lmp
    lmp -h | head -n 30

    echo "========== Python import deepmd =========="
    python -c "import deepmd; print(\"deepmd import ok\")"
  '
