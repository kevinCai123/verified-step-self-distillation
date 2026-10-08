"""One-screen status of the post-round-1 cycle (standard library only; safe to run anywhere)."""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    path = Path(path)
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError): return None

def main():
    out = {'now': datetime.now(timezone.utc).isoformat()}
    cycle = ROOT / 'runs/cycle'
    out['steps_done'] = sorted(p.stem for p in cycle.glob('*.done')) if cycle.exists() else []
    log = cycle / 'cycle.log'
    out['last_log_lines'] = log.read_text(encoding='utf-8').splitlines()[-6:] if log.exists() else []
    lr = cycle / 'learning-rate.txt'
    out['learning_rate'] = lr.read_text().strip() if lr.exists() else None
    for name in ('memorization-check', 'memorization-check-5e-5', 'memorization-check-legal', 'memorization-correction-2e-5', 'memorization-correction-5e-5', 'memorization-correction-legal'):
        status = read(ROOT / 'runs' / name / 'status.json')
        if status:
            out[name] = {k: status.get(k) for k in ('complete', 'stage', 'records', 'leaky_records', 'verdict', 'moved_toward_replacement', 'moved_away_from_replacement', 'error')}
            for phase in ('before', 'after'):
                if status.get(phase): out[name][phase] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in status[phase].items()}
            if status.get('losses'): out[name]['last_loss'] = status['losses'][-1]
    arms = {}
    for run in sorted((ROOT / 'runs').glob('*-seed*')):
        if run.name == 'round1-seed42': continue
        state = read(run / 'state.json') or {}
        arm = {'phase': state.get('phase'), 'updates': state.get('updates'), 'cursor': state.get('cursor'), 'complete': state.get('complete'), 'updated': state.get('updated'), 'error': state.get('error')}
        pending = run / f"batches/step-{(state.get('updates') or 0) + 1:03}"
        fresh = read(pending / 'fresh/summary.json') or read(pending / 'summary.json')
        if fresh: arm['current_batch'] = {k: fresh.get(k) for k in ('attempted', 'verified', 'next_cursor', 'batch_ready')}
        last = run / f"checkpoints/step-{state.get('updates') or 0:03}/status.json"
        status = read(last)
        if status:
            arm['last_checkpoint'] = {k: status.get(k) for k in ('mean_loss', 'mean_jsd_content', 'mean_structural_share_of_jsd', 'sample_action_types', 'sample_type_matches_replacement', 'mean_margin', 'reward_accuracy', 'mean_nll_content', 'replacement_types', 'gradient_norm', 'elapsed_seconds')}
        serving = read(run / f"checkpoints/step-{state.get('updates') or 0:03}/serving-check.json")
        if serving: arm['last_serving_check'] = {k: serving.get(k) for k in ('different', 'premature_finish_share', 'base_premature_finish_share')}
        evaluations = {}
        for folder in sorted((run / 'evaluations').glob('step-*')):
            summary = read(folder / 'summary.json')
            if not summary: continue
            entry = {k: (round(summary[k], 4) if isinstance(summary.get(k), float) else summary.get(k)) for k in ('questions', 'complete', 'answer_em', 'joint_f1', 'grounded_success', 'failures')}
            comparison = read(folder / 'comparison.json')
            if comparison:
                entry['delta_joint_f1_pp'] = round(100 * comparison['metrics']['joint_f1']['delta'], 2)
                entry['ci95_pp'] = [round(100 * x, 2) for x in comparison['metrics']['joint_f1']['paired_ci95']]
                entry['delta_em_pp'] = round(100 * comparison['metrics']['answer_em']['delta'], 2)
            evaluations[folder.name] = entry
        arm['evaluations'] = evaluations
        # yield so far across completed batches
        attempted = verified = 0
        for batch in (run / 'batches').glob('step-*'):
            s = read(batch / 'fresh/summary.json')
            if s: attempted += s.get('attempted', 0); verified += s.get('verified', 0)
        arm['fresh_yield'] = {'attempted': attempted, 'verified': verified, 'rate': round(verified / attempted, 4) if attempted else None}
        arms[run.name] = arm
    out['arms'] = arms
    final = {}
    for folder in sorted((ROOT / 'runs/final-eval').glob('*')):
        summary = read(folder / 'summary.json')
        if not summary: continue
        entry = {k: (round(summary[k], 4) if isinstance(summary.get(k), float) else summary.get(k)) for k in ('questions', 'complete', 'answer_em', 'joint_f1', 'grounded_success', 'failures')}
        comparison = read(folder / 'comparison.json')
        if comparison:
            entry['delta_joint_f1_pp'] = round(100 * comparison['metrics']['joint_f1']['delta'], 2)
            entry['ci95_pp'] = [round(100 * x, 2) for x in comparison['metrics']['joint_f1']['paired_ci95']]
        final[folder.name] = entry
    out['final_benchmark'] = final
    cycle2 = ROOT / 'runs/cycle2'
    if cycle2.exists():
        out['cycle2_steps_done'] = sorted(p.stem for p in cycle2.glob('*.done'))
        log2 = cycle2 / 'cycle.log'
        out['cycle2_last_log_lines'] = log2.read_text(encoding='utf-8').splitlines()[-6:] if log2.exists() else []
    print(json.dumps(out, indent=2))

if __name__ == '__main__': main()
