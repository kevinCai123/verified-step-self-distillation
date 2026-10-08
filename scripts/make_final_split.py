"""Write the locked final task benchmark: all official HotpotQA development questions, in the same row
layout as the training-side splits (`data/splits/final_eval.jsonl`). Idempotent; records the row count
and file hash in `data/splits/final_eval.manifest.json` and checks the IDs against the `final_ids`
that `prepare_splits.py` locked in `data/splits/manifest.json`. No label is read here beyond writing the
rows out unchanged."""
import hashlib, json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / 'data/raw', ROOT / 'data/splits'

def main():
    target = OUT / 'final_eval.jsonl'
    locked = json.loads((OUT / 'manifest.json').read_text())['final_ids']
    if not target.exists():
        rows = pq.read_table(RAW / 'validation-00000-of-00001.parquet').to_pylist()
        if [r['id'] for r in rows] != locked: raise ValueError('Final question IDs differ from the locked manifest')
        tmp = target.with_suffix('.jsonl.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        tmp.replace(target)
    ids = [json.loads(line)['id'] for line in target.read_text(encoding='utf-8').splitlines()]
    if ids != locked: raise ValueError('final_eval.jsonl does not match the locked final IDs')
    manifest = {'count': len(ids), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'source': json.loads((RAW / 'dataset_revision.json').read_text()),
                'note': 'official HotpotQA development questions; the final research evaluation set of docs/EXPERIMENT_PLAN.md, opened once per policy, never for tuning'}
    (OUT / 'final_eval.manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))

if __name__ == '__main__': main()
