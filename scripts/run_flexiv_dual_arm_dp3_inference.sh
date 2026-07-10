#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_inference_config.yaml"
CONFIG_PATH="${FLEXIV_DP3_INFERENCE_CONFIG:-${DEFAULT_CONFIG}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Inference config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

mapfile -t CONFIG_VALUES < <(
  python - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1]).expanduser()
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
inference = cfg["inference"]
print(int(inference["gpu_id"]))
print(Path(str(inference["stop_file"])).expanduser())
PY
)

GPU_ID="${CONFIG_VALUES[0]}"
STOP_FILE="${CONFIG_VALUES[1]}"

cd "${REPO_ROOT}"
rm -f -- "${STOP_FILE}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONNOUSERSITE=1

echo "[inference] config: ${CONFIG_PATH}"
echo "[inference] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[inference] stop from another terminal: touch ${STOP_FILE}"

exec python scripts/run_flexiv_dual_arm_dp3_inference.py \
  --config "${CONFIG_PATH}" \
  "$@"
