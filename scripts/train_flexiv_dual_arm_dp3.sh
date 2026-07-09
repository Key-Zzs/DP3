#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/train_flexiv_dual_arm_dp3.sh <xyz|xyzrgb> <zarr_path> [simple_dp3|dp3] [gpu_id] [seed] [hydra_overrides...] [--overwrite]

Options:
  --overwrite
      Delete the whole target Hydra output directory before training if it already exists.
      Without this flag, an existing output directory is treated as an error.

Environment overrides:
  DEBUG=False|True
  SAVE_CKPT=True|False
  WANDB_MODE=offline|online|disabled
  DEVICE=cuda:0|cpu
  MAX_TRAIN_EPISODES=90|null
  BATCH_SIZE=128
  NUM_WORKERS=8
  EXP_NAME=<name>
  RUN_DIR=<hydra output dir>

Default checkpoint directory, relative to this repository:
  outputs/<exp_name>_seed<seed>/checkpoints/
USAGE
}

OVERWRITE=false
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}"

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

MODE="$1"
ZARR_PATH="$2"
ALG_NAME="${3:-simple_dp3}"
GPU_ID="${4:-0}"
SEED="${5:-42}"
EXTRA_ARGS=()
if [[ $# -gt 5 ]]; then
  EXTRA_ARGS=("${@:6}")
fi

case "${MODE}" in
  xyz)
    TASK_CONFIG="real/flexiv_dual_arm_head_xyz"
    USE_PC_COLOR=false
    PC_IN_CHANNELS=3
    ;;
  xyzrgb)
    TASK_CONFIG="real/flexiv_dual_arm_head_xyzrgb"
    USE_PC_COLOR=true
    PC_IN_CHANNELS=6
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ "${ALG_NAME}" != "simple_dp3" && "${ALG_NAME}" != "dp3" ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HYDRA_BASE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"

DEBUG="${DEBUG:-False}"
SAVE_CKPT="${SAVE_CKPT:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
DEVICE="${DEVICE:-cuda:0}"
MAX_TRAIN_EPISODES="${MAX_TRAIN_EPISODES:-90}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXP_NAME="${EXP_NAME:-flexiv_dual_arm_head_${MODE}-${ALG_NAME}}"
# train.py changes cwd to the parent workspace before Hydra resolves run.dir.
# Keep this relative path pointed at this repository's outputs directory.
RUN_DIR="${RUN_DIR:-3D-Diffusion-Policy/outputs/${EXP_NAME}_seed${SEED}}"
for extra_arg in "${EXTRA_ARGS[@]}"; do
  case "${extra_arg}" in
    hydra.run.dir=*)
      RUN_DIR="${extra_arg#hydra.run.dir=}"
      ;;
  esac
done

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
  if [[ "${OVERWRITE}" != "true" ]]; then
    echo "Output directory already exists: ${RUN_DIR_ABS}" >&2
    echo "Use a different RUN_DIR/EXP_NAME, or pass --overwrite to delete this whole directory first." >&2
    exit 3
  fi
  if [[ ! -d "${RUN_DIR_ABS}" ]]; then
    echo "Cannot overwrite non-directory RUN_DIR: ${RUN_DIR_ABS}" >&2
    exit 3
  fi
  echo "Overwriting existing output directory: ${RUN_DIR_ABS}" >&2
  rm -rf -- "${RUN_DIR_ABS}"
fi

cd "${REPO_ROOT}/3D-Diffusion-Policy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python train.py --config-name="${ALG_NAME}.yaml" \
  task="${TASK_CONFIG}" \
  task.dataset.zarr_path="${ZARR_PATH}" \
  task.dataset.max_train_episodes="${MAX_TRAIN_EPISODES}" \
  hydra.run.dir="${RUN_DIR}" \
  training.debug="${DEBUG}" \
  training.seed="${SEED}" \
  training.device="${DEVICE}" \
  exp_name="${EXP_NAME}" \
  logging.mode="${WANDB_MODE}" \
  checkpoint.save_ckpt="${SAVE_CKPT}" \
  dataloader.batch_size="${BATCH_SIZE}" \
  val_dataloader.batch_size="${BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  val_dataloader.num_workers="${NUM_WORKERS}" \
  policy.use_pc_color="${USE_PC_COLOR}" \
  policy.pointcloud_encoder_cfg.in_channels="${PC_IN_CHANNELS}" \
  "${EXTRA_ARGS[@]}"
