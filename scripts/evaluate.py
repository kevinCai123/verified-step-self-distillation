"""Restricted-student evaluation on a fixed internal-development partition."""
import argparse
import json
import time
from pathlib import Path
from transformers import AutoTokenizer
from self_evolve_search.agent import Client, offline_guard, run
from self_evolve_search.persistence import atomic_json, bind_config, file_hash, protocol_hash
from self_evolve_search.retrieval import Library

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--policy-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8093')
    parser.add_argument('--limit', type=int, default=500)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'questions_sha256': file_hash(args.questions), 'protocol_sha256': protocol_hash(root, ('evaluate.py',))}
    bind_config(out / 'config.json', config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    client = Client(args.endpoint, args.model, tokenizer); client.health()
    library = Library(args.index); offline_guard()
    rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines()][:args.limit]
    results = []
    for number, row in enumerate(rows, 1):
        path = out / (row['id'] + '.json')
        if path.exists():
            result = json.loads(path.read_text())
        else:
            start = time.perf_counter(); call_start = len(client.calls)
            result = run(row, client, library, 2)
            result.update(policy_id=args.policy_id, calls=client.calls[call_start:], elapsed_seconds=time.perf_counter()-start)
            atomic_json(path, result)
        if result['policy_id'] != args.policy_id:
            raise ValueError('Evaluation checkpoint mismatch')
        results.append(result)
        summary = {'questions': number, 'expected_questions': len(rows), 'complete': number == len(rows), 'policy_id': args.policy_id}
        for metric in ('answer_em', 'answer_f1', 'support_f1', 'joint_f1', 'grounded_success'):
            summary[metric] = sum(r['metrics'][metric] for r in results) / number
        summary.update(failures=sum(bool(r['error']) for r in results), elapsed_seconds=sum(r['elapsed_seconds'] for r in results), model_calls=sum(len(r['calls']) for r in results), tool_calls=sum(r['state']['tool_calls'] for r in results))
        atomic_json(out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
