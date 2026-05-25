# V100 Environment Check

Date: 2026-05-25
Host: shared-v100
Workdir: /data/zft/deepmd-al-hpc

## GPU

- 2×Tesla V100-SXM2-16GB (16384 MiB each)
- Driver: 535.183.01

## Software

- Python: 3.10.12 (host)
- DeepMD-kit: v3.1.4.dev81 (in Docker container: ghcr.io/deepmodeling/deepmd-kit:master)
- TensorFlow: 2.21.0 (in container)
- CUDA: available via nvidia-docker

## Notes

- `dp` and `deepmd` are NOT installed on the host; all DeePMD operations must run inside the Docker container.
- GPU is accessible from Docker via `--gpus all`.
- Data directory `/data/zft/deepmd-al-hpc/data/` exists with toy H2 datasets.
