#!/usr/bin/env bash
set -euo pipefail

# Reproducible, non-Docker entrypoint for the pinned official description.
# The generator itself performs the environment, branch, and submodule checks.
repo_root="$(git rev-parse --show-toplevel)"
generator="${repo_root}/scripts/rmbench/flexiv/generate_embodiment.py"

if [[ "${CONDA_DEFAULT_ENV:-}" != "dp3-rmbench" ]]; then
  echo "bootstrap_description.sh requires CONDA_DEFAULT_ENV=dp3-rmbench" >&2
  echo "Activate dp3-rmbench first; the existing dp3 environment is out of scope." >&2
  exit 2
fi

exec python "${generator}" "$@"
