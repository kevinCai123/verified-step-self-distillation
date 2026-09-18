#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .tools/uv ]]; then
  SEARCH_UV_BIN="$PWD/.tools/uv"
elif command -v uv >/dev/null 2>&1; then
  SEARCH_UV_BIN="$(command -v uv)"
else
  echo 'Install uv or place its executable at .tools/uv before running this script.' >&2
  exit 1
fi
export UV_CACHE_DIR="$PWD/.cache/uv"
SEARCH_PYTHON="${SEARCH_PYTHON:-python3.12}"
for environment in data rollout train; do
  "$SEARCH_UV_BIN" venv ".venv-$environment" --python "$SEARCH_PYTHON"
  "$SEARCH_UV_BIN" pip install --python ".venv-$environment/bin/python" -r "requirements-$environment.lock"
done
