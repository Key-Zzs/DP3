#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dp3-rmbench"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RMBENCH_DIR="$ROOT_DIR/third_party/sim/RMBench"
ASSETS_DIR="$RMBENCH_DIR/assets"
MIN_AVAILABLE_KIB=$((5 * 1024 * 1024))
HF_REPO="TianxingChen/RMBench"
HF_REVISION="${RMBENCH_HF_REVISION:-d899d72b53270a89f71d216c08ecbd4d9a7004fd}"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) printf '%s\n' "Usage: $0 [--dry-run]"; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -d "$RMBENCH_DIR" ]] || { printf 'error: missing RMBench subtree: %s\n' "$RMBENCH_DIR" >&2; exit 1; }
[[ -f "$ASSETS_DIR/_download.py" ]] || { printf '%s\n' 'error: upstream asset downloader is missing' >&2; exit 1; }

AVAILABLE_KIB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
if ((AVAILABLE_KIB < MIN_AVAILABLE_KIB)); then
  printf 'error: only %s KiB is available; at least %s KiB is required\n' "$AVAILABLE_KIB" "$MIN_AVAILABLE_KIB" >&2
  exit 1
fi
printf 'available disk: %s KiB\n' "$AVAILABLE_KIB"

if ((DRY_RUN)); then
  printf '%s\n' "would verify Hugging Face authentication in $ENV_NAME"
  printf '%s\n' "would download only $HF_REPO@$HF_REVISION patterns embodiments/** and objects/** into $ASSETS_DIR"
  printf '%s\n' 'would update embodiment paths and verify aloha-agilex, franka-panda, and 005_button'
  exit 0
fi

if ! env PYTHONNOUSERSITE=1 conda run --no-capture-output -n "$ENV_NAME" python -c 'from huggingface_hub import HfApi; HfApi().whoami()' >/dev/null 2>&1; then
  printf '%s\n' 'error: Hugging Face authentication is unavailable in dp3-rmbench; run huggingface-cli login in that environment' >&2
  exit 1
fi

(cd "$ASSETS_DIR" && env PYTHONNOUSERSITE=1 RMBENCH_HF_REPO="$HF_REPO" RMBENCH_HF_REVISION="$HF_REVISION" conda run --no-capture-output -n "$ENV_NAME" python -c '
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["RMBENCH_HF_REPO"],
    revision=os.environ["RMBENCH_HF_REVISION"],
    allow_patterns=["embodiments/**", "objects/**"],
    local_dir=".",
    repo_type="dataset",
    resume_download=True,
)
')
(cd "$RMBENCH_DIR" && env PYTHONNOUSERSITE=1 conda run --no-capture-output -n "$ENV_NAME" python script/update_embodiment_config_path.py)

missing_assets=()
for required in \
  "$ASSETS_DIR/embodiments/aloha-agilex" \
  "$ASSETS_DIR/embodiments/franka-panda" \
  "$ASSETS_DIR/objects/005_button"; do
  [[ -e "$required" ]] || missing_assets+=("$required")
done
if ((${#missing_assets[@]})); then
  printf '%s\n' 'error: required scoped assets are missing:' >&2
  printf '  %s\n' "${missing_assets[@]}" >&2
  exit 1
fi

du -sh "$ASSETS_DIR/embodiments" "$ASSETS_DIR/objects"
printf '%s\n' 'RMBench Stage 0 assets are ready; complete dataset download was not requested.'
