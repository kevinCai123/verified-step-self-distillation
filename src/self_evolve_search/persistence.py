"""Durable experiment records and checkpoint identities."""
import hashlib
import json
import os
from pathlib import Path

BASE_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
BASE_POLICY = 'base:' + BASE_REVISION

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def adapter_policy(path):
    path = Path(path)
    digest = hashlib.sha256()
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        digest.update(name.encode() + file_hash(path / name).encode())
    return 'adapter:' + digest.hexdigest()

def protocol_hash(root, scripts=()):
    root = Path(root)
    names = ('agent.py', 'metrics.py', 'retrieval.py', 'repair.py', 'tokenization.py', 'persistence.py')
    paths = [root / 'src/self_evolve_search' / name for name in names]
    paths += [root / 'scripts' / name for name in scripts]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + path.read_bytes())
    return digest.hexdigest()

def bind_config(path, config):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Saved configuration differs: ' + str(path))
    else:
        atomic_json(path, config)

def validate_training_batch(records, policy_id, batch_size=8):
    if len(records) != batch_size:
        raise ValueError(f'Need {batch_size} fresh verified records, got {len(records)}')
    if len({r['task_id'] for r in records}) != batch_size:
        raise ValueError('Duplicate training question')
    for record in records:
        if record.get('policy_id') != policy_id:
            raise ValueError('Stale or unidentified repair policy')
        if record['edited_wins'] < 2 or record['original_wins'] > 1:
            raise ValueError('Unverified repair')

