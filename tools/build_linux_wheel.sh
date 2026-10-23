#!/usr/bin/env bash
# Run inside the CUDA 12.8 build environment, from the repository root.
set -euo pipefail
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
python -m build --wheel --no-isolation --outdir build/wheels
# CUDA runtime dependencies are supplied by the declared NVIDIA Python wheels.
auditwheel repair build/wheels/coil_eval-*.whl --plat manylinux_2_28_x86_64 --only-plat \
  --exclude libcudart.so.12 --exclude libcublas.so.12 --exclude libcublasLt.so.12 \
  --exclude libcusolver.so.11 --exclude libcusparse.so.12 --exclude libnvJitLink.so.12 \
  --wheel-dir dist
python -m build --sdist --no-isolation --outdir dist
