# 环境配置与 Docker 使用

本文档用于记录 `deepmd-al-hpc` 的运行环境、Docker 容器、GPU 检查和路径约定。

当前项目主要采用：

```text
宿主机 /data/zft
  ↕ 实时同步
Docker 容器 /data/zft
```

其中：

```text
DeepMD Docker 容器：
  用于 DeePMD-kit 训练、冻结、测试、推理和 DeepPot prediction。

Torch Docker 容器：
  用于主动学习 skeleton、PyTorch 调度逻辑和部分纯 Python 工程脚本。

宿主机：
  用于 Git 管理、文档编辑和简单文件检查。
```

---

## 1. 项目目录

宿主机上的项目目录为：

```bash
cd /data/zft
```

进入目录后，建议先检查当前 Git 状态：

```bash
git status -sb
git log --oneline -5
```

---

## 2. 推荐运行方式

建议按照任务类型选择运行环境。

### 2.1 宿主机

宿主机主要用于：

```text
git status
git add
git commit
git push
README / docs 编辑
简单文件检查
```

例如：

```bash
git status --short
git log --oneline -5
```

---

### 2.2 DeepMD Docker 容器

DeepMD Docker 容器主要用于：

```text
dp train
dp freeze
dp test
DeepPot prediction
numpy / DeepMD 相关 Python 实验脚本
toy H2 数据生成
committee prediction
```

凡是涉及以下内容的命令，建议都在 DeepMD Docker 容器中运行：

```text
numpy
DeepMD-kit
DeepPot
dp train
dp freeze
dp test
frozen_model.pb
committee prediction
```

---

### 2.3 Torch Docker 容器

Torch Docker 容器主要用于：

```text
主动学习 skeleton 检查
PyTorch / 调度逻辑验证
force model deviation 测试
top-K selection 测试
部分纯 Python 工程脚本
```

---

## 3. DeepMD-kit Docker 环境

本项目真实 DeePMD 训练、冻结、测试和推理依赖 DeepMD-kit Docker 环境。

推荐使用仓库脚本进入容器：

```bash
cd /data/zft
bash scripts/docker/enter_deepmd_container.sh
```

进入容器后检查当前路径：

```bash
pwd
```

预期输出：

```text
/data/zft
```

---

## 4. 手动启动 DeepMD Docker 容器

如果脚本不可用，可以手动启动容器：

```bash
cd /data/zft

docker run --rm -it \
  --gpus all \
  --user $(id -u):$(id -g) \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -v /data/zft:/data/zft \
  -w /data/zft \
  ghcr.io/deepmodeling/deepmd-kit:master \
  bash
```

该命令含义：

```text
--gpus all
  将 GPU 暴露给容器

-v /data/zft:/data/zft
  将宿主机项目目录挂载到容器中

-w /data/zft
  设置容器内工作目录

--user $(id -u):$(id -g)
  使用当前用户身份运行，避免生成 root 权限文件
```

---

## 5. 已验证环境信息

当前已验证环境：

```text
镜像：ghcr.io/deepmodeling/deepmd-kit:master
DeepMD-kit：v3.1.3-81-geab34197
Python：/opt/deepmd-kit/bin/python
dp：/opt/deepmd-kit/bin/dp
lmp：/opt/deepmd-kit/bin/lmp
numpy：2.4.4
GPU：2×Tesla V100-SXM2-16GB
```

说明：

```text
该环境主要用于当前 toy H2 原型验证。
后续迁移到 H100/5090 时，需要重新记录镜像、驱动、CUDA、DeepMD-kit 版本和 GPU 信息。
```

---

### 5.1 RTX 5090 环境（2026-06-28 适配）

当前 RTX 5090 已验证环境：

```text
自定义镜像：deepmd-5090:latest（基于 nvcr.io/nvidia/pytorch:25.06-py3）
DeepMD-kit：v3.1.3（PyTorch 后端，从源码编译以匹配容器 PyTorch 2.8.0）
Python：/usr/local/bin/python3（Python 3.12）
PyTorch：2.8.0a0+cu129 (NVIDIA 定制版)
numpy：<2（锁定以兼容容器 PyTorch）
mpi4py：已安装
GPU：8×NVIDIA GeForce RTX 5090 (32GB, Blackwell sm_120)
驱动：570.124.06，CUDA 12.8（容器内 CUDA 12.9，Minor Version Compat 模式）
```

关键适配说明：

```text
1. TensorFlow 后端不可用：TF 编译时仅支持到 sm_89/compute_90，不含 Blackwell sm_120，
   因此使用 PyTorch 后端（dp -b pytorch train ...）。

2. 镜像构建脚本：scripts/docker/Dockerfile.deepmd-5090
   进入容器：bash scripts/docker/enter_deepmd_5090.sh

3. 必须使用 root 运行容器：--user 模式下 MPS 会导致 CUDA 初始化挂起。

4. deepmd-kit 从源码编译：PyPI wheel 使用 PyTorch 2.10.0 编译（ABI 不兼容），
   需设置 DP_ENABLE_PYTORCH=1 pip install --no-binary deepmd-kit。

5. apt mpich 不可安装：会引入 Ubuntu UCX 与 HPC-X UCX 冲突，
   仅需 pip install mpich（Python 包元数据）。
```

---

## 6. DeepMD 环境检查

进入 DeepMD Docker 容器后，建议依次检查：

```bash
which python
which python3
```

检查 Python 和 numpy：

```bash
python - <<'PY'
import sys
import numpy as np

print(sys.executable)
print(sys.version)
print("numpy:", np.__version__)
PY
```

检查 DeepMD-kit 命令：

```bash
dp -h
lmp -h
```

检查 DeepMD Python import：

```bash
python -c "import deepmd; print('deepmd import ok')"
python -c "from deepmd.infer import DeepPot; print('DeepPot import ok')"
```

检查 GPU：

```bash
nvidia-smi
```

如果以上命令均正常，说明 DeepMD-kit Docker 环境可用于后续实验。

---

## 7. Torch 基础开发环境

Torch 环境主要用于运行主动学习框架 skeleton、model deviation 计算和部分 Python 工程脚本。

进入 Torch 容器：

```bash
cd /data/zft
bash scripts/docker/enter_torch_container.sh
```

当前 Torch 容器用途：

```text
运行 Python 主动学习框架
测试 force model deviation
测试 top-K 构型筛选
验证 2×V100 调度逻辑
```

当前记录的 Torch 镜像：

```text
cuda-torch:cuda11.3-cudnn8-ubuntu18.04-torch2.4
```

---

## 8. Python 命令说明

如果在宿主机中没有 `python` 命令，而只有 `python3`，可以使用：

```bash
python3
```

但在 DeepMD Docker 容器中通常可以直接使用：

```bash
python
```

因此文档中的命令默认以 DeepMD Docker 容器为主，通常写作：

```bash
python xxx.py
```

如果在宿主机运行检查命令时失败，可改为：

```bash
python3 xxx.py
```

---

## 9. 常见问题

### 9.1 容器内路径不对

如果进入容器后不是 `/data/zft`，先运行：

```bash
pwd
cd /data/zft
```

### 9.2 DeepPot import 失败

如果出现：

```text
ModuleNotFoundError: No module named deepmd
```

说明当前不在 DeepMD Docker 容器中，或者使用了错误的 Python 环境。

应重新进入：

```bash
bash scripts/docker/enter_deepmd_container.sh
```

### 9.3 GPU 不可见

如果容器内 `nvidia-smi` 失败，需要检查：

```text
Docker 是否支持 GPU
NVIDIA Container Toolkit 是否可用
启动容器时是否使用 --gpus all
```

### 9.4 生成 root 权限文件

如果容器里生成了 root 权限文件，说明启动容器时可能没有使用：

```bash
--user $(id -u):$(id -g)
```

建议使用仓库提供的进入脚本，避免权限问题。

---

## 10. 下一步

环境检查通过后，可以继续阅读：

```text
docs/toy_h2_pipeline.md
```

用于复现：

```text
toy H2 数据生成
单模型 DeePMD baseline
初始 4-model committee training
初始 committee prediction
```