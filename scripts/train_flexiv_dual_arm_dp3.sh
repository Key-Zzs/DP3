#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/train_flexiv_dual_arm_dp3.sh

Configure the task, zarr path, algorithm, physical GPU, output directory,
WandB, dataloaders, optimizer, training, and checkpoint behavior in:
  3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_train_config.yaml

launcher.overwrite=false rejects an existing run directory unless
training.resume=true and checkpoints/latest.ckpt exists. Set overwrite=true
only when the entire existing run directory should be deleted before training.
USAGE
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "This launcher no longer accepts training arguments; edit dp3_train_config.yaml instead." >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HYDRA_BASE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_train_config.yaml"

CONFIG_VALUES="$(python - "${CONFIG_PATH}" <<'PY'
import pathlib
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

config_path = pathlib.Path(sys.argv[1])
with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
    cfg = compose(config_name=config_path.stem)

required = (
    "launcher.gpu_id",
    "launcher.overwrite",
    "training.resume",
    "algorithm",
    "task.dataset.zarr_path",
    "run_dir",
)
missing = [key for key in required if OmegaConf.select(cfg, key) is None]
if missing:
    raise SystemExit(f"Missing required training config values: {', '.join(missing)}")

print(cfg.launcher.gpu_id)
print(str(cfg.launcher.overwrite).lower())
print(str(cfg.training.resume).lower())
print(cfg.algorithm)
print(cfg.task.dataset.zarr_path)
print(cfg.run_dir)
PY
)"
mapfile -t CONFIG_LINES <<<"${CONFIG_VALUES}"

GPU_ID="${CONFIG_LINES[0]}"
OVERWRITE="${CONFIG_LINES[1]}"
RESUME="${CONFIG_LINES[2]}"
ALG_NAME="${CONFIG_LINES[3]}"
ZARR_PATH="${CONFIG_LINES[4]}"
RUN_DIR="${CONFIG_LINES[5]}"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "launcher.gpu_id must be a non-negative physical GPU index, got: ${GPU_ID}" >&2
  exit 2
fi
if [[ "${OVERWRITE}" != "true" && "${OVERWRITE}" != "false" ]]; then
  echo "launcher.overwrite must be true or false, got: ${OVERWRITE}" >&2
  exit 2
fi
if [[ "${RESUME}" != "true" && "${RESUME}" != "false" ]]; then
  echo "training.resume must be true or false, got: ${RESUME}" >&2
  exit 2
fi
if [[ "${OVERWRITE}" == "true" && "${RESUME}" == "true" ]]; then
  echo "launcher.overwrite and training.resume cannot both be true." >&2
  exit 2
fi
if [[ "${ALG_NAME}" != "simple_dp3" && "${ALG_NAME}" != "dp3" ]]; then
  echo "algorithm must be simple_dp3 or dp3, got: ${ALG_NAME}" >&2
  exit 2
fi

if [[ "${ZARR_PATH}" = /* ]]; then
  ZARR_PATH_ABS="$(realpath -m -- "${ZARR_PATH}")"
else
  # train.py changes cwd to HYDRA_BASE_DIR before constructing the dataset.
  ZARR_PATH_ABS="$(realpath -m -- "${HYDRA_BASE_DIR}/${ZARR_PATH}")"
fi
if [[ ! -d "${ZARR_PATH_ABS}" ]]; then
  echo "Configured zarr dataset does not exist: ${ZARR_PATH_ABS}" >&2
  echo "Update task.dataset.zarr_path in ${CONFIG_PATH}" >&2
  exit 2
fi

if [[ "${RUN_DIR}" = /* ]]; then
  RUN_DIR_ABS="$(realpath -m -- "${RUN_DIR}")"
else
  RUN_DIR_ABS="$(realpath -m -- "${HYDRA_BASE_DIR}/${RUN_DIR}")"
fi

case "${RUN_DIR_ABS}" in
  /|"${HYDRA_BASE_DIR}"|"${REPO_ROOT}"|"${REPO_ROOT}/3D-Diffusion-Policy")
    echo "Refusing unsafe RUN_DIR: ${RUN_DIR_ABS}" >&2
    exit 2
    ;;
esac

if [[ -e "${RUN_DIR_ABS}" ]]; then
  if [[ ! -d "${RUN_DIR_ABS}" ]]; then
    echo "RUN_DIR is not a directory: ${RUN_DIR_ABS}" >&2
    exit 3
  fi
  if [[ "${RESUME}" == "true" ]]; then
    if [[ ! -f "${RUN_DIR_ABS}/checkpoints/latest.ckpt" ]]; then
      echo "Cannot resume without checkpoints/latest.ckpt: ${RUN_DIR_ABS}" >&2
      exit 3
    fi
    echo "Resuming existing output directory: ${RUN_DIR_ABS}" >&2
  elif [[ "${OVERWRITE}" != "true" ]]; then
    echo "Output directory already exists: ${RUN_DIR_ABS}" >&2
    echo "Change run_dir/exp_name, set training.resume=true, or set launcher.overwrite=true." >&2
    exit 3
  else
    echo "Overwriting existing output directory: ${RUN_DIR_ABS}" >&2
    rm -rf -- "${RUN_DIR_ABS}"
  fi
elif [[ "${RESUME}" == "true" ]]; then
  echo "Cannot resume because output directory does not exist: ${RUN_DIR_ABS}" >&2
  exit 3
fi

cd "${REPO_ROOT}/3D-Diffusion-Policy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Starting ${ALG_NAME} training on physical GPU ${GPU_ID}"
echo "Dataset: ${ZARR_PATH_ABS}"
echo "Output: ${RUN_DIR_ABS}"

python train.py --config-name="dp3_train_config.yaml"
