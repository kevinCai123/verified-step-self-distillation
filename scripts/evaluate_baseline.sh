#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime_env.sh"
cd "$SEARCH_ROOT"
export HF_HUB_OFFLINE=1
MODEL_PATH="$(.venv-data/bin/python -c 'import json; print(json.load(open("config/local.json"))["model_path"])')"
.venv-data/bin/python scripts/wait_server.py --model Qwen/Qwen3.5-9B
.venv-rollout/bin/python -u scripts/evaluate.py --questions data/splits/dev_monitor.jsonl --index data/index/wiki.sqlite --model-path "$MODEL_PATH" --policy-id base:c202236235762e1c871ad0ccb60c8ee5ba337b9a --output runs/round1-seed42/evaluations/step-000 --limit 500
