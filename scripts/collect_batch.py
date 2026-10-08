"""Collect at most eight training records (replay-verified steps, or control-arm records) under one immutable checkpoint."""
import argparse
import json
import time
from pathlib import Path
from transformers import AutoTokenizer
from self_evolve_search.agent import Client, offline_guard, run
from self_evolve_search.persistence import atomic_json, bind_config, file_hash, protocol_hash
from self_evolve_search.repair import random_steps, record, repair
from self_evolve_search.retrieval import Library

def self_success(row, student, client, seed):
    """Control (arm R, rejection-sampling self-training): a grounded-successful student trajectory yields one
    record whose target is the student's own action at a seeded random step. No teacher trajectory, no
    diagnosis, no replay; failed trajectories yield nothing. Same record layout as a verified repair so the
    batch, replay-window and training code are unchanged (guidance carries the target action itself)."""
    evidence = {'task_id': row['id'], 'attempts': [], 'confirmed': None, 'diagnosis_error': None,
                'ambiguous_answer_excluded': False, 'mode': 'self-success', 'step_selection': 'random'}
    if not student['metrics']['grounded_success']: return evidence
    if student['metrics']['answer_alias_ambiguous']:
        evidence['ambiguous_answer_excluded'] = True; return evidence
    chosen = random_steps(student, seed, 1)
    if not chosen: return evidence
    step = student['trace'][chosen[0]]
    evidence['confirmed'] = record(row, student, None, client, chosen[0], None, step['action'],
                                   "self-success: the student's own action at a seeded random step of a grounded-successful trajectory", 'self-success')
    return evidence

def main():
    parser = argparse.ArgumentParser()
    for option in ('questions', 'index', 'model-path', 'model', 'policy-id', 'output'):
        parser.add_argument('--'+option, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8093')
    parser.add_argument('--repair-mode', choices=('verified', 'unverified', 'evidence-only', 'self-success'), default='verified',
                        help='self-success: rejection-sampling self-training control (arm R); no teacher run, no diagnosis, no replay')
    parser.add_argument('--step-selection', choices=('ranked', 'random'), default='ranked')
    parser.add_argument('--seed', type=int, default=42, help='seed for random step selection')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'questions_sha256': file_hash(args.questions), 'protocol_sha256': protocol_hash(root, ('collect_batch.py',))}
    bind_config(out / 'config.json', config)
    rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines()]
    if not 0 <= args.start < args.end <= len(rows): raise ValueError('Invalid collection bounds')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    client = Client(args.endpoint, args.model, tokenizer, args.policy_id); client.health()
    library = Library(args.index); offline_guard()
    results = []; records = []
    for position in range(args.start, args.end):
        row = rows[position]; path = out / (row['id'] + '.json')
        if path.exists():
            item = json.loads(path.read_text())
        else:
            start = time.perf_counter(); call_start = len(client.calls); tool_start = len(library.calls)
            student = run(row, client, library, 2)
            if args.repair_mode == 'self-success':
                item = {'position': position, 'policy_id': args.policy_id, 'student': student, 'teacher': None,
                        'repair': self_success(row, student, client, args.seed)}
            else:
                teacher = run(row, client, library, 8)
                item = {'position': position, 'policy_id': args.policy_id, 'student': student, 'teacher': teacher,
                        'repair': repair(row, student, teacher, client, library, args.repair_mode, args.step_selection, args.seed)}
            item.update(calls=client.calls[call_start:], tool_calls=library.calls[tool_start:], elapsed_seconds=time.perf_counter()-start)
            atomic_json(path, item)
        if item['position'] != position or item['policy_id'] != args.policy_id: raise ValueError('Stale collection result')
        results.append(item)
        if item['repair']['confirmed']: records.append(item['repair']['confirmed'])
        atomic_json(out / 'records.json', records)
        summary = {'policy_id': args.policy_id, 'attempted': len(results), 'start': args.start, 'next_cursor': position+1,
                   'verified': len(records), 'batch_ready': len(records)==args.batch_size, 'exhausted': position+1==args.end,
                   'model_calls': sum(len(r['calls']) for r in results), 'tool_calls': sum(len(r['tool_calls']) for r in results),
                   'elapsed_seconds': sum(r['elapsed_seconds'] for r in results),
                   'input_tokens': sum(c['usage']['prompt_tokens'] for r in results for c in r['calls']),
                   'output_tokens': sum(c['usage']['completion_tokens'] for r in results for c in r['calls'])}
        atomic_json(out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
        if summary['batch_ready']: break

if __name__ == '__main__':
    main()
