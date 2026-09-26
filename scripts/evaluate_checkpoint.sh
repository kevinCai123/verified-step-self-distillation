#!/usr/bin/env bash
# Evaluate one saved adapter checkpoint of a run on the internal development split and compare it with
# that run's step-000 baseline, outside the experiment driver. Used when a run was stopped by a guard
# (e.g. the premature-finish tripwire) and the driver will therefore never reach its final evaluation.
#   bash scripts/evaluate_checkpoint.sh <run-dir> <step>        e.g. runs/armB-seed42 19
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime_env.sh
export HF_HUB_OFFLINE=1
RUN=$1; STEP=$2; STEP3=$(printf '%03d' "$STEP")
CHECKPOINT="$RUN/checkpoints/step-$STEP3"; OUT="$RUN/evaluations/step-$STEP3"
[[ -f "$CHECKPOINT/status.json" ]] || { echo "no checkpoint $CHECKPOINT"; exit 1; }
POLICY="$(.venv-data/bin/python -c "import json,sys; print(json.load(open('$CHECKPOINT/status.json'))['policy_id'])")"
ALIAS="step-$STEP-${POLICY#adapter:}"; ALIAS="${ALIAS:0:$((5+${#STEP}+1+12))}"
MODEL_PATH="$(.venv-data/bin/python -c 'import json; print(json.load(open("config/local.json"))["model_path"])')"
read -r EVAL_FILE EVAL_N < <(.venv-data/bin/python -c "import json; c=json.load(open('$RUN/config.json')); print(c['evaluation_file'], c['evaluation_questions'])")
mkdir -p "$OUT"
if [[ "$(.venv-data/bin/python -c "import json,os; p='$OUT/summary.json'; print(json.load(open(p)).get('complete') if os.path.exists(p) else False)")" != "True" ]]; then
  .venv-data/bin/python scripts/stop_server.py >> "$RUN/server-control.log" 2>&1 || true
  bash scripts/serve.sh "$CHECKPOINT/adapter" "$ALIAS" >> "$RUN/server-eval-step-$STEP3.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 180); do
    sleep 5
    if ! kill -0 "$SERVER" 2>/dev/null; then echo "server exited while loading"; exit 1; fi
    if .venv-data/bin/python - "$ALIAS" "$CHECKPOINT/adapter" <<'EOF'
import json, sys
from pathlib import Path
from urllib.request import urlopen
try:
    models = json.load(urlopen('http://127.0.0.1:8093/v1/models', timeout=5))['data']
except OSError: sys.exit(1)
m = next((m for m in models if m['id'] == sys.argv[1]), None)
sys.exit(0 if m and Path(m['root']).resolve() == Path(sys.argv[2]).resolve() else 1)
EOF
    then break; fi
  done
  .venv-rollout/bin/python scripts/evaluate.py --questions "$EVAL_FILE" --index data/index/wiki.sqlite --model-path "$MODEL_PATH" \
      --model "$ALIAS" --policy-id "$POLICY" --output "$OUT" --limit "$EVAL_N" >> "$OUT/run.log" 2>&1
  STATUS=$?
  .venv-data/bin/python scripts/stop_server.py >> "$RUN/server-control.log" 2>&1 || true
  [[ $STATUS -eq 0 ]] || { echo "evaluation failed (see $OUT/run.log)"; exit 1; }
fi
[[ -f "$OUT/comparison.json" ]] || PYTHONPATH=src .venv-data/bin/python scripts/compare_evaluation.py --baseline "$RUN/evaluations/step-000" --updated "$OUT" --output "$OUT/comparison.json" > "$OUT/comparison.log" 2>&1
echo "evaluated $RUN step $STEP ($ALIAS)"
