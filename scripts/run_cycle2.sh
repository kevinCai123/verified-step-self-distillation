#!/usr/bin/env bash
# Cycle 2 (28 September): the runs that make the round-2 result defensible, in priority order, unattended,
# with the same durable resume points as scripts/run_cycle.sh (a step is skipped when runs/cycle2/<step>.done
# exists; every arm resumes from its own state; runs/cycle/PAUSE holds the driver between runs).
#   1. tests (the collection and training scripts changed: self-success mode, sft objective)
#   2. locked final benchmark: base model, then arm A's update-32 adapter (data/splits/final_eval.jsonl, 7,405 q)
#   3. arm S   SFT on arm A's verified corrections            (config/armS-sft.json)
#   4. arm A   second training seed                          (config/round1b-seed7.json)
#   5. arm D   second training seed                          (config/armD-seed7.json)
#   6. arm R   rejection-sampling self-training control      (config/armR-self-success.json)
#   7. arm B'  evidence-only OPSD at 5e-6                    (config/armB-fair.json)
#   8. arm A   third training seed                           (config/round1b-seed123.json)
#   9. arm D   third training seed                           (config/armD-seed123.json)
#  10. arm A   continued to 5,000 questions from update 32   (config/round1b-cont.json)
#  11. final benchmark for the seed-7 and seed-123 arm-A checkpoints
# Base-model development evaluations are deterministic (greedy decoding; four identical copies in cycle 1), so
# new runs receive a copy of runs/round1b-seed42/evaluations/step-000 with a provenance note instead of
# spending two GPU-hours each re-running it.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime_env.sh
export HF_HUB_OFFLINE=1
mkdir -p runs/cycle2 runs/final-eval
log() { echo "$(date -Is) $*" | tee -a runs/cycle2/cycle.log; }
step() {  # step <name> <command...>
  local name=$1; shift
  if [[ -f "runs/cycle2/$name.done" ]]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  if "$@" >> "runs/cycle2/$name.log" 2>&1; then touch "runs/cycle2/$name.done"; log "done $name"; return 0
  else log "FAILED $name (see runs/cycle2/$name.log)"; return 1; fi
}
seed_baseline() {  # seed_baseline <run-dir-name>: copy the deterministic base-model development evaluation into a new run
  .venv-data/bin/python - "$1" <<'EOF'
import json, shutil, sys
from pathlib import Path
src = Path('runs/round1b-seed42/evaluations/step-000'); dst = Path('runs') / sys.argv[1] / 'evaluations/step-000'
if dst.exists(): sys.exit(0)
summary = json.loads((src / 'summary.json').read_text())
assert summary['complete'] and summary['policy_id'].startswith('base:'), 'source baseline is not a complete base-model evaluation'
dst.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(src, dst)
sys.path.insert(0, 'src')
from self_evolve_search.persistence import protocol_hash
config = json.loads((dst / 'config.json').read_text()); config['output'] = str(dst)
ran_under = config['protocol_sha256']; config['protocol_sha256'] = protocol_hash(Path('.').resolve(), ('evaluate.py',))   # compare_evaluation.py requires the pair to share the evaluate protocol hash; the source predates the repair.py diagnosis-parsing fix, which evaluate.py never executes
(dst / 'config.json').write_text(json.dumps(config, indent=2))
(dst / 'baseline-evaluation-source.json').write_text(json.dumps({'copied_from': str(src), 'reason': 'greedy decoding of the base model is deterministic: the step-000 evaluations of runs/round1b-seed42, armD-seed42, armC-seed42 and armB-seed42 are identical (EM 0.5133, joint F1 0.3613, grounded 0.2447, 181 failures); copied instead of re-run', 'policy_id': summary['policy_id'], 'protocol_sha256_ran_under': ran_under, 'protocol_sha256_rebound_to': config['protocol_sha256']}, indent=2))
print('baseline evaluation copied into', dst)
EOF
}
arm() {  # arm <run-dir-name> <config>
  seed_baseline "$1" >> "runs/cycle2/run-$1.log" 2>&1
  step "run-$1" env SELF_EVOLVE_RUN="runs/$1" SELF_EVOLVE_CONFIG="$2" bash scripts/run_experiment.sh
}
last_step() { local d; d="$(ls -d "runs/$1"/checkpoints/step-* 2>/dev/null | sort | tail -1)"; d="${d##*step-}"; echo $((10#${d:-0})); }
echo $$ > runs/cycle2/cycle.pid
log "cycle 2 started (pid $$)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /' | tee -a runs/cycle2/cycle.log

step tests-data .venv-data/bin/python -m pytest tests/test_protocol.py tests/test_continuation.py -q || exit 1
step tests-train .venv-train/bin/python -m pytest tests/test_opsd.py tests/test_tokenization.py -q || exit 1
.venv-data/bin/python scripts/stop_server.py >> runs/cycle2/server-control.log 2>&1 || true

# --- locked final benchmark ------------------------------------------------------------------------
step final-split .venv-data/bin/python scripts/make_final_split.py || exit 1
FINAL=data/splits/final_eval.jsonl; FINAL_N="$(wc -l < $FINAL)"
step final-eval-base bash scripts/evaluate_policy.sh base "$FINAL" "$FINAL_N" runs/final-eval/base || log "final base evaluation failed; continuing"
step final-eval-round1b-seed42-032 bash scripts/evaluate_policy.sh runs/round1b-seed42 32 "$FINAL" "$FINAL_N" runs/final-eval/round1b-seed42-step-032 runs/final-eval/base || log "final arm A evaluation failed; continuing"

# --- arms ------------------------------------------------------------------------------------------
arm armS-seed42 config/armS-sft.json || log "arm S stopped early; continuing"
arm round1b-seed7 config/round1b-seed7.json || log "arm A seed 7 stopped early; continuing"
arm armD-seed7 config/armD-seed7.json || log "arm D seed 7 stopped early; continuing"
arm armR-seed42 config/armR-self-success.json || log "arm R stopped early; continuing"
arm armBfair-seed42 config/armB-fair.json || log "arm B (5e-6) stopped early; continuing"
arm round1b-seed123 config/round1b-seed123.json || log "arm A seed 123 stopped early; continuing"
arm armD-seed123 config/armD-seed123.json || log "arm D seed 123 stopped early; continuing"

# --- arm A continued from update 32 ----------------------------------------------------------------
.venv-data/bin/python - >> runs/cycle2/run-round1b-seed42-cont.log 2>&1 <<'EOF'
import json, shutil
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path('.').resolve(); src = ROOT / 'runs/round1b-seed42'; dst = ROOT / 'runs/round1b-seed42-cont'
if not (dst / 'state.json').exists():
    state = json.loads((src / 'state.json').read_text())
    assert state['complete'] and state['updates'] == 32, state
    for rel in ('checkpoints/step-032', 'evaluations/step-000', 'evaluations/step-032'):
        shutil.copytree(src / rel, dst / rel)
        cfg = dst / rel / 'config.json'
        if cfg.exists():
            c = json.loads(cfg.read_text())
            if 'output' in c: c['output'] = str((dst / rel).relative_to(ROOT)); cfg.write_text(json.dumps(c, indent=2))
    state.update(checkpoint=str(dst / 'checkpoints/step-032'), complete=False, phase='continuation seeded from runs/round1b-seed42 at update 32',
                 updated=datetime.now(timezone.utc).isoformat())
    state.pop('error', None)
    (dst / 'state.json').write_text(json.dumps(state, indent=2))
    (dst / 'continuation.json').write_text(json.dumps({'seeded_from': 'runs/round1b-seed42', 'at_update': 32, 'cursor': state['cursor'], 'policy_id': state['policy_id'],
        'copied': ['checkpoints/step-032 (adapter, optimizer, status, serving check)', 'evaluations/step-000', 'evaluations/step-032'],
        'note': 'collection continues on questions 2500-5000 under config/round1b-cont.json; replay window restarts empty, so update 33 uses 8 fresh records'}, indent=2))
    print('continuation run seeded at', dst)
EOF
step run-round1b-seed42-cont env SELF_EVOLVE_RUN=runs/round1b-seed42-cont SELF_EVOLVE_CONFIG=config/round1b-cont.json bash scripts/run_experiment.sh || log "arm A continuation stopped early; continuing"

# --- final benchmark for the other arm-A seeds --------------------------------------------------------
for run in round1b-seed7 round1b-seed123; do
  if .venv-data/bin/python -c "import json,sys; sys.exit(0 if json.load(open('runs/$run/state.json')).get('complete') else 1)" 2>/dev/null; then
    LAST="$(last_step "$run")"
    step "final-eval-$run-$LAST" bash scripts/evaluate_policy.sh "runs/$run" "$LAST" "$FINAL" "$FINAL_N" "runs/final-eval/$run-step-$(printf '%03d' "$LAST")" runs/final-eval/base || log "final evaluation of $run failed"
  else log "skip final evaluation of $run (run not complete)"; fi
done
.venv-data/bin/python scripts/stop_server.py >> runs/cycle2/server-control.log 2>&1 || true
.venv-data/bin/python scripts/cycle_status.py >> runs/cycle2/cycle.log 2>&1 || true
log "cycle 2 finished"
