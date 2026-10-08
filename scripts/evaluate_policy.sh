#!/usr/bin/env bash
# Evaluate one policy (the base model, or a saved adapter checkpoint of a run) on a question file, outside
# the experiment driver, and compare it with a baseline evaluation folder when one is given.
#   bash scripts/evaluate_policy.sh base   <questions> <limit> <output-dir>
#   bash scripts/evaluate_policy.sh <run-dir> <step> <questions> <limit> <output-dir> [baseline-dir]
# Used for the locked final benchmark (data/splits/final_eval.jsonl, 7,405 questions): the base model
# once into runs/final-eval/base, then each trained policy against it.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime_env.sh
export HF_HUB_OFFLINE=1
MODEL_PATH="$(.venv-data/bin/python -c 'import json; print(json.load(open("config/local.json"))["model_path"])')"
if [[ "$1" == "base" ]]; then
  QUESTIONS=$2; LIMIT=$3; OUT=$4; BASELINE=""
  ADAPTER=""; ALIAS="Qwen/Qwen3.5-9B"
  POLICY="$(PYTHONPATH=src .venv-data/bin/python -c 'from self_evolve_search.persistence import BASE_POLICY; print(BASE_POLICY)')"
else
  RUN=$1; STEP=$2; QUESTIONS=$3; LIMIT=$4; OUT=$5; BASELINE=${6:-}
  STEP3=$(printf '%03d' "$STEP"); CHECKPOINT="$RUN/checkpoints/step-$STEP3"
  [[ -f "$CHECKPOINT/status.json" ]] || { echo "no checkpoint $CHECKPOINT"; exit 1; }
  POLICY="$(.venv-data/bin/python -c "import json; print(json.load(open('$CHECKPOINT/status.json'))['policy_id'])")"
  ADAPTER="$CHECKPOINT/adapter"; ALIAS="step-$STEP-${POLICY#adapter:}"; ALIAS="${ALIAS:0:$((5+${#STEP}+1+12))}"
fi
mkdir -p "$OUT"
complete() { .venv-data/bin/python -c "import json,os; p='$OUT/summary.json'; s=json.load(open(p)) if os.path.exists(p) else {}; raise SystemExit(0 if s.get('complete') and s.get('policy_id')=='$POLICY' else 1)"; }
if ! complete; then
  .venv-data/bin/python scripts/stop_server.py >> runs/server-control.log 2>&1 || true
  if [[ -n "$ADAPTER" ]]; then bash scripts/serve.sh "$ADAPTER" "$ALIAS" >> "$OUT/server.log" 2>&1 &
  else bash scripts/serve.sh >> "$OUT/server.log" 2>&1 & fi
  SERVER=$!
  for _ in $(seq 1 180); do
    sleep 5
    if ! kill -0 "$SERVER" 2>/dev/null; then echo "server exited while loading"; exit 1; fi
    if .venv-data/bin/python - "$ALIAS" "${ADAPTER:-$MODEL_PATH}" <<'EOF'
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
  .venv-rollout/bin/python scripts/evaluate.py --questions "$QUESTIONS" --index data/index/wiki.sqlite --model-path "$MODEL_PATH" \
      --model "$ALIAS" --policy-id "$POLICY" --output "$OUT" --limit "$LIMIT" >> "$OUT/run.log" 2>&1
  STATUS=$?
  .venv-data/bin/python scripts/stop_server.py >> runs/server-control.log 2>&1 || true
  [[ $STATUS -eq 0 ]] || { echo "evaluation failed (see $OUT/run.log)"; exit 1; }
fi
if [[ -n "$BASELINE" && ! -f "$OUT/comparison.json" ]]; then
  PYTHONPATH=src .venv-data/bin/python scripts/compare_evaluation.py --baseline "$BASELINE" --updated "$OUT" --output "$OUT/comparison.json" > "$OUT/comparison.log" 2>&1 || { echo "comparison failed (see $OUT/comparison.log)"; exit 1; }
fi
echo "evaluated $POLICY ($ALIAS) on $QUESTIONS -> $OUT"
