#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dp3-rmbench"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="$ROOT_DIR/environments/dp3_rmbench_requirements.txt"
CUROBO_DIR="$ROOT_DIR/third_party/sim/RMBench/envs/curobo"
CUROBO_REF="v0.7.8"
CUDA_CHANNEL="nvidia/label/cuda-12.8.0"
CUDA_NVCC_VERSION="12.8.61"
CUDA_CUDART_DEV_VERSION="12.8.57"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
WARP_VERSION="1.6.2"
TORCH_CUDA_ARCH_LIST="12.0"
PYTORCH3D_REF="33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"
DRY_RUN=0

usage() { printf '%s\n' "Usage: $0 [--dry-run]"; }
run_env() { env PYTHONNOUSERSITE=1 conda run --no-capture-output -n "$ENV_NAME" "$@"; }

snapshot() {
  local dir="$1"
  mkdir -p "$dir"
  run_env python --version > "$dir/python-version.txt"
  conda list -n "$ENV_NAME" > "$dir/conda-list.txt"
  conda list -n "$ENV_NAME" --explicit > "$dir/conda-list-explicit.txt"
  conda env export -n "$ENV_NAME" > "$dir/conda-export.yml"
  conda env export -n "$ENV_NAME" --from-history > "$dir/conda-export-from-history.yml"
  run_env python -m pip freeze > "$dir/pip-freeze.txt"
  run_env python -m pip check > "$dir/pip-check.txt" 2>&1 || true
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${CONDA_DEFAULT_ENV:-}" == "dp3" || "$(basename "${CONDA_PREFIX:-/not-an-env}")" == "dp3" ]]; then
  printf '%s\n' "error: bootstrap_env.sh refuses to run from the existing dp3 environment" >&2
  exit 2
fi
command -v conda >/dev/null || { printf '%s\n' 'error: conda is required' >&2; exit 2; }
[[ -f "$REQ_FILE" ]] || { printf 'error: missing requirements: %s\n' "$REQ_FILE" >&2; exit 2; }

if ! conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
  if ((DRY_RUN)); then
    printf '%s\n' "would create conda environment $ENV_NAME with python=3.10"
  else
    conda create -n "$ENV_NAME" python=3.10 -y
  fi
fi

if ((DRY_RUN)); then
  printf '%s\n' "environment: $ENV_NAME"
  printf '%s\n' "would install torch==$TORCH_VERSION and torchvision==$TORCHVISION_VERSION from $PYTORCH_INDEX"
  printf '%s\n' "would install pinned RMBench requirements from $REQ_FILE"
  printf '%s\n' "would install CUDA NVCC $CUDA_NVCC_VERSION from NVIDIA's CUDA 12.8 channel"
  printf '%s\n' "would install CUDA Runtime development headers $CUDA_CUDART_DEV_VERSION"
  printf '%s\n' "would pin warp-lang==$WARP_VERSION for CuRobo 0.7.8's warp.torch API"
  printf '%s\n' "would install PyTorch3D at $PYTORCH3D_REF"
  printf '%s\n' "would clone/install CuRobo $CUROBO_REF at $CUROBO_DIR"
  printf '%s\n' "would run content-checked site-packages patches and editable-install $ROOT_DIR/3D-Diffusion-Policy"
  exit 0
fi

if [[ "$(run_env python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
  printf '%s\n' "error: $ENV_NAME is not Python 3.10; refusing to mutate it" >&2
  exit 1
fi

ENV_PREFIX="$(conda run -n "$ENV_NAME" bash -c 'printf %s "$CONDA_PREFIX"')"
CURRENT_NVCC="$(run_env nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)"
if [[ "$CURRENT_NVCC" != 12.8* ]] || ! run_env bash -c 'test -f "$CONDA_PREFIX/include/cuda_runtime.h" || test -f "$CONDA_PREFIX/targets/x86_64-linux/include/cuda_runtime.h"'; then
  printf '%s\n' 'Installing the CUDA 12.8 NVCC toolchain required by CuRobo ...'
  conda install -n "$ENV_NAME" --override-channels -c "$CUDA_CHANNEL" -c nvidia -c defaults \
    "cuda-nvcc=$CUDA_NVCC_VERSION" "cuda-cudart-dev=$CUDA_CUDART_DEV_VERSION" -y
fi

BEFORE_DIR="$ROOT_DIR/environments/snapshots/dp3_rmbench_before_install"
AFTER_DIR="$ROOT_DIR/environments/snapshots/dp3_rmbench_after_install"
snapshot "$BEFORE_DIR"

printf '%s\n' 'Installing the pinned PyTorch CUDA 12.8 wheel set ...'
run_env python -m pip install --upgrade --prefer-binary --index-url "$PYTORCH_INDEX" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

printf '%s\n' 'Installing pinned RMBench Stage 0 requirements ...'
run_env python -m pip install --prefer-binary -r "$REQ_FILE"

printf '%s\n' 'Installing the pinned PyTorch3D source build ...'
P3D_INCLUDE_PATH="$ENV_PREFIX/targets/x86_64-linux/include:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/include:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cublas/include:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/include:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/curand/include:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/include:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cufft/include"
P3D_LIBRARY_PATH="$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/curand/lib:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cufft/lib"
if run_env python -c 'import pytorch3d.ops' >/dev/null 2>&1; then
  printf '%s\n' 'PyTorch3D is already importable; keeping the verified pinned installation.'
else
  run_env env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS=1 \
    CUB_HOME="$ENV_PREFIX/targets/x86_64-linux" CPATH="$P3D_INCLUDE_PATH" LIBRARY_PATH="$P3D_LIBRARY_PATH" \
    python -m pip install --no-build-isolation --prefer-binary \
    "git+https://github.com/facebookresearch/pytorch3d.git@$PYTORCH3D_REF"
fi

if [[ ! -e "$CUROBO_DIR" ]]; then
  mkdir -p "$(dirname "$CUROBO_DIR")"
  git clone --branch "$CUROBO_REF" --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
elif [[ ! -d "$CUROBO_DIR/.git" ]]; then
  printf '%s\n' "error: existing CuRobo path is not a Git checkout: $CUROBO_DIR" >&2
  exit 1
fi
CUROBO_HEAD="$(git -C "$CUROBO_DIR" rev-parse HEAD)"
CUROBO_EXPECTED="$(git -C "$CUROBO_DIR" rev-parse "$CUROBO_REF^{commit}")"
if [[ "$CUROBO_HEAD" != "$CUROBO_EXPECTED" ]]; then
  printf 'error: CuRobo is %s, expected %s (%s)\n' "$CUROBO_HEAD" "$CUROBO_EXPECTED" "$CUROBO_REF" >&2
  exit 1
fi
if run_env python -c 'import torch; import warp as wp; assert hasattr(wp, "torch"); from curobo.curobolib import geom_cu, kinematics_fused_cu, lbfgs_step_cu, line_search_cu, tensor_step_cu' >/dev/null 2>&1; then
  printf '%s\n' 'CuRobo is already importable; keeping the verified pinned installation.'
else
  run_env env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS=8 \
    python -m pip install -e "$CUROBO_DIR" --no-build-isolation
fi

printf '%s\n' 'Reapplying Stage 0 compatibility pins after CuRobo dependency resolution ...'
run_env python -m pip install --prefer-binary \
  setuptools==75.1.0 scipy==1.10.1 scikit-image==0.22.0

printf '%s\n' 'Applying guarded SAPIEN/MPLib compatibility patches ...'
run_env python "$ROOT_DIR/scripts/rmbench/patch_site_packages.py"

printf '%s\n' 'Installing the current repository DP3 package in editable mode ...'
run_env python -m pip install -e "$ROOT_DIR/3D-Diffusion-Policy" \
  --config-settings editable_mode=compat

snapshot "$AFTER_DIR"
diff -u "$BEFORE_DIR/pip-freeze.txt" "$AFTER_DIR/pip-freeze.txt" > "$ROOT_DIR/environments/snapshots/dp3_rmbench_package-diff.txt" || true
printf '%s\n' "bootstrap complete; snapshots are under environments/snapshots/ and CuRobo is at $CUROBO_DIR"
