#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_inference_config.yaml"
CONFIG_PATH="${FLEXIV_DP3_INFERENCE_CONFIG:-${DEFAULT_CONFIG}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Perception config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

GPU_ID="$(python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    cfg = yaml.safe_load(stream)
print(int(cfg["inference"]["gpu_id"]))
PY
)"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONNOUSERSITE=1

echo "[perception-only] config: ${CONFIG_PATH}"
echo "[perception-only] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[perception-only] Flexiv RDK and robot arms will not be connected"

exec python tools/run_flexiv_dp3_perception_only.py \
  --config "${CONFIG_PATH}" \
  "$@"
