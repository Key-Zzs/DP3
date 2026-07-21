#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_NAME="rmbench-upstream"
REMOTE_URL="https://github.com/RoboTwin-Platform/RMBench.git"
PREFIX="third_party/sim/RMBench"
PIN="87e0498891073d483d330195c0f160709bd92ff5"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --commit) PIN="${2:?missing commit after --commit}"; shift 2 ;;
    -h|--help) printf '%s\n' "Usage: $0 [--dry-run] [--commit SHA]"; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

cd "$ROOT_DIR"
[[ "$(git branch --show-current)" == "develop/RMBench" ]] || { printf '%s\n' 'error: refusing to run outside develop/RMBench' >&2; exit 1; }
configured="$(git config --get "remote.$REMOTE_NAME.url" || true)"
if [[ -z "$configured" ]]; then
  if ((DRY_RUN)); then
    printf '%s\n' "would add remote $REMOTE_NAME $REMOTE_URL"
  else
    git remote add "$REMOTE_NAME" "$REMOTE_URL"
  fi
elif [[ "$configured" != "$REMOTE_URL" ]]; then
  printf 'error: remote %s has unexpected URL %s\n' "$REMOTE_NAME" "$configured" >&2
  exit 1
fi

if ((DRY_RUN)); then
  if [[ -e "$PREFIX" ]]; then
    printf '%s\n' "would fetch and subtree-pull $PREFIX at $PIN"
  else
    printf '%s\n' "would fetch and subtree-add $PREFIX at $PIN"
  fi
  exit 0
fi

git fetch "$REMOTE_NAME" "$PIN"
git cat-file -e "$PIN^{commit}"
if [[ -e "$PREFIX" ]]; then
  git subtree pull --prefix="$PREFIX" "$REMOTE_NAME" "$PIN" --squash
else
  git subtree add --prefix="$PREFIX" "$REMOTE_NAME" "$PIN" --squash
fi
