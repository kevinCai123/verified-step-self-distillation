#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime_env.sh"
cd "$SEARCH_ROOT"
export HF_HUB_OFFLINE=1
MODEL_PATH="$(.venv-data/bin/python -c 'import json; print(json.load(open("config/local.json"))["model_path"])')"
trap '.venv-data/bin/python scripts/job_status.py "failed; inspect runs/local-run.log"' ERR
.venv-data/bin/python scripts/job_status.py 'checking supporting-sentence mapping'
.venv-data/bin/python scripts/check_coverage.py --questions data/splits/pilot.jsonl --index data/index/wiki.sqlite --output runs/coverage.json
.venv-data/bin/python -c 'import json; c=json.load(open("runs/coverage.json")); assert c["coverage"]==1., "Resolve evidence mapping before running the pilot"'
.venv-data/bin/python scripts/job_status.py 'running 500-question trajectory-repair pilot'
.venv-rollout/bin/python -u scripts/pilot.py --questions data/splits/pilot.jsonl --index data/index/wiki.sqlite --model-path "$MODEL_PATH" --output runs/pilot500 --limit 500 --repair
.venv-data/bin/python scripts/job_status.py 'freeing GPU for one OPSD compatibility update'
.venv-data/bin/python scripts/stop_server.py
.venv-data/bin/python scripts/job_status.py 'running one OPSD compatibility update'
.venv-train/bin/python -u scripts/train_smoke.py --records runs/pilot500/verified_steps.jsonl --model-path "$MODEL_PATH" --output runs/opsd-smoke --batch-size 8
.venv-data/bin/python -c 'import json; s=json.load(open("runs/opsd-smoke/status.json")); assert s["complete"], s.get("error", "Compatibility test incomplete")'
.venv-data/bin/python scripts/job_status.py 'local pilot and one-update compatibility test complete; main training pending'
