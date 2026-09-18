#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime_env.sh"
cd "$SEARCH_ROOT"
export HF_HUB_OFFLINE=1
.venv-data/bin/python scripts/freeze_sources.py
exec .venv-data/bin/python -u scripts/run_experiment.py
