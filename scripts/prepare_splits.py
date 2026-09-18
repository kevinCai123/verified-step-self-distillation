import hashlib, json, random, re
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / 'data/raw', ROOT / 'data/splits'
OUT.mkdir(parents=True, exist_ok=True)

def key(q): return ' '.join(re.findall(r'\w+', q.casefold()))

def main():
    excluded = set(json.loads((ROOT / 'config/excluded_pilot.json').read_text())['ids'])
    # Only question IDs/text are used to exclude cross-split duplicates. No final labels are opened.
    final = pq.read_table(RAW / 'validation-00000-of-00001.parquet', columns=['id', 'question']).to_pylist()
    final_q = {key(r['question']) for r in final}
    rows, seen, rejected = [], set(), []
    for part in sorted(RAW.glob('train-*.parquet')):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=1024):
            for r in batch.to_pylist():
                q = key(r['question'])
                if r['id'] in excluded or q in seen or q in final_q:
                    rejected.append(r['id']); continue
                seen.add(q)
                rows.append(r)
    rows.sort(key=lambda x: x['id'])
    random.Random(42).shuffle(rows)
    splits = {'internal_dev': rows[:1500], 'repair_benchmark': rows[1500:2000], 'round1': rows[2000:7000], 'round2': rows[7000:12000], 'pilot': rows[2000:2500], 'dev_monitor': rows[:500]}
    manifest = {'seed': 42, 'deduplication': 'casefold alphanumeric question text; exact normalized duplicates only', 'excluded_ids': rejected, 'source': json.loads((RAW / 'dataset_revision.json').read_text()), 'final_ids': [r['id'] for r in final], 'splits': {}}
    for name, rr in splits.items():
        target = OUT / (name + '.jsonl')
        with target.open('w', encoding='utf-8') as f:
            for r in rr: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        manifest['splits'][name] = {'count': len(rr), 'ids': [r['id'] for r in rr], 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    for a in ('internal_dev', 'repair_benchmark', 'round1', 'round2'):
        for b in ('internal_dev', 'repair_benchmark', 'round1', 'round2'):
            if a != b: assert set(manifest['splits'][a]['ids']).isdisjoint(manifest['splits'][b]['ids'])
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'available_train': len(rows), 'excluded': len(rejected), 'splits': {k: len(v) for k,v in splits.items()}, 'final_questions_locked': len(final)}, indent=2))

if __name__ == '__main__': main()
