#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/train_flexiv_dual_arm_dp3.sh <xyz|xyzrgb> <zarr_path> [simple_dp3|dp3] [gpu_id] [seed] [hydra_overrides...]

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
