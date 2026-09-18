#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime_env.sh"
cd "$SEARCH_ROOT"
export HF_HUB_OFFLINE=1
MODEL_PATH="$(.venv-data/bin/python -c 'import json; print(json.load(open("config/local.json"))["model_path"])')"
trap '.venv-data/bin/python scripts/job_status.py "training retry failed; inspect runs/training-retry.log"' ERR
.venv-data/bin/python -c 'import json; from pathlib import Path; p=Path("runs/opsd-smoke/status.json"); assert not p.exists() or not json.loads(p.read_text()).get("complete"), "Completed compatibility checkpoint already exists; refusing overwrite"'
.venv-data/bin/python scripts/job_status.py 'pilot complete; resuming one OPSD update after tokenizer fix'
.venv-train/bin/python -u scripts/train_smoke.py --records runs/pilot500/verified_steps.jsonl --model-path "$MODEL_PATH" --output runs/opsd-smoke --batch-size 8
.venv-data/bin/python scripts/job_status.py 'local pilot and one-update compatibility test complete; main training pending'
