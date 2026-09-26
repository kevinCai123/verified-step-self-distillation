#!/usr/bin/env bash
# Runs the whole post-round-1 cycle unattended, in order, with durable resume points:
#   1. tests (data and training environments)
#   2. memorization gate with correction-only guidance (picks 2e-5 or 5e-5; stops the cycle if neither learns)
#   3. arm A  verified-step OPSD           (config/round1b.json)
#   4. arm D  random-step control          (config/armD-random-step.json)
#   5. arm C  DPO objective                (config/armC-dpo.json)
#   6. arm B  plain privileged-context OPSD (config/armB-evidence-opsd.json; max_updates = arm A's count)
# Every step is skipped when runs/cycle/<step>.done exists; each arm resumes from its own state.
# Launch from WSL:  setsid nohup bash scripts/run_cycle.sh > runs/cycle-main.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime_env.sh
export HF_HUB_OFFLINE=1
mkdir -p runs/cycle
MODEL_PATH="$(.venv-data/bin/python -c 'import json; print(json.load(open("config/local.json"))["model_path"])')"
log() { echo "$(date -Is) $*" | tee -a runs/cycle/cycle.log; }
step() {  # step <name> <command...>
  local name=$1; shift
  if [[ -f "runs/cycle/$name.done" ]]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  if "$@" >> "runs/cycle/$name.log" 2>&1; then touch "runs/cycle/$name.done"; log "done $name"; return 0
  else log "FAILED $name (see runs/cycle/$name.log)"; return 1; fi
}
echo $$ > runs/cycle/cycle.pid
log "cycle started (pid $$)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /' | tee -a runs/cycle/cycle.log

step tests-data .venv-data/bin/python -m pytest tests/test_protocol.py tests/test_continuation.py -q || exit 1
step tests-train .venv-train/bin/python -m pytest tests/test_opsd.py tests/test_tokenization.py -q || exit 1
.venv-data/bin/python scripts/stop_server.py >> runs/cycle/server-control.log 2>&1 || true

# --- memorization gate -------------------------------------------------------------------------
# 21 September: with the teacher shown the retrieved evidence ("full" guidance) the check FAILED at
# 2e-5 and 5e-5: the student drifted to premature finishing (runs/memorization-check*, kept as
# evidence). The protocol now shows the teacher the verified correction only; the gate below must
# pass with that guidance before any arm starts, otherwise the cycle stops here.
gate() {  # gate <lr>  -> 0 when the greedy actions moved toward the corrections
  step "memorization-correction-$1" timeout --signal=TERM --kill-after=60 5400 .venv-train/bin/python scripts/memorization_check.py --model-path "$MODEL_PATH" \
      --output "runs/memorization-correction-$1" --steps 30 --learning-rate "$1" --loss-tokens content --guidance correction || return 1
  .venv-data/bin/python -c "import json,sys; s=json.load(open('runs/memorization-correction-$1/status.json')); print('gate $1:', s['verdict'], 'types after:', s['after'].get('action_types')); sys.exit(0 if s['verdict'].startswith('learns') else 1)" | tee -a runs/cycle/cycle.log
  return "${PIPESTATUS[0]}"
}
if [[ ! -f runs/cycle/gate-learning-rate.txt ]]; then
  if gate 2e-5; then echo 2e-5 > runs/cycle/gate-learning-rate.txt
  elif gate 5e-5; then echo 5e-5 > runs/cycle/gate-learning-rate.txt
  else log "memorization gate FAILED with correction-only guidance at 2e-5 and 5e-5; not starting the arms"; exit 1
  fi
  LR="$(cat runs/cycle/gate-learning-rate.txt)"; log "memorization gate passed at $LR"
  .venv-data/bin/python - "$LR" <<'EOF2'
import json, sys
from pathlib import Path
lr = float(sys.argv[1])
for name in ('round1b', 'armB-evidence-opsd', 'armC-dpo', 'armD-random-step'):
    path = Path('config') / (name + '.json'); config = json.loads(path.read_text())
    config['train_learning_rate'] = lr; path.write_text(json.dumps(config, indent=2) + '\n')
print('arm configurations set to learning rate', lr)
EOF2
fi
LR="$(cat runs/cycle/gate-learning-rate.txt)"
step memorization-correction-legal-only timeout --signal=TERM --kill-after=60 5400 .venv-train/bin/python scripts/memorization_check.py --model-path "$MODEL_PATH" \
    --output runs/memorization-correction-legal --steps 30 --learning-rate "$LR" --loss-tokens content --guidance correction --only-informationally-legal || true

# --- arms --------------------------------------------------------------------------------------
arm() {  # arm <run-dir-name> <config>
  step "run-$1" env SELF_EVOLVE_RUN="runs/$1" SELF_EVOLVE_CONFIG="$2" bash scripts/run_experiment.sh
}
arm round1b-seed42 config/round1b.json || log "arm A stopped early; continuing with the controls"
arm armD-seed42 config/armD-random-step.json || log "arm D stopped early; continuing"
arm armC-seed42 config/armC-dpo.json || log "arm C stopped early; continuing"
if [[ -f runs/round1b-seed42/state.json && ! -f runs/armB-seed42/config.json ]]; then
  .venv-data/bin/python - <<'EOF'
import json
from pathlib import Path
updates = json.loads(Path('runs/round1b-seed42/state.json').read_text())['updates']
path = Path('config/armB-evidence-opsd.json'); config = json.loads(path.read_text())
config['max_updates'] = max(1, updates); path.write_text(json.dumps(config, indent=2) + '\n')
print('arm B max_updates matched to arm A:', updates)
EOF
fi
arm armB-seed42 config/armB-evidence-opsd.json || log "arm B stopped early"
# 25 September: arm B was stopped by the premature-finish tripwire at update 19 (serving check: 75% of probed
# states answered `finish`, base 0%). The driver never reaches a final evaluation for a guarded stop, so the
# last checkpoint is evaluated here to quantify the drift on the development split.
if [[ -f runs/armB-seed42/state.json ]] && ! .venv-data/bin/python -c "import json,sys; sys.exit(0 if json.load(open('runs/armB-seed42/state.json')).get('complete') else 1)"; then
  LAST="$(ls -d runs/armB-seed42/checkpoints/step-* | sort | tail -1)"; LAST="${LAST##*step-}"; LAST=$((10#$LAST))
  step "armB-checkpoint-eval-$LAST" bash scripts/evaluate_checkpoint.sh runs/armB-seed42 "$LAST" || log "arm B checkpoint evaluation failed"
fi
.venv-data/bin/python scripts/stop_server.py >> runs/cycle/server-control.log 2>&1 || true
.venv-data/bin/python scripts/cycle_status.py >> runs/cycle/cycle.log 2>&1 || true
log "cycle finished"
