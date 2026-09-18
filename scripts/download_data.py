"""Acquire public data separately from offline experiments. No evaluation calls."""
import argparse, concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
RAW.mkdir(parents=True, exist_ok=True)

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def fetch(url, name, expected_md5=None):
    dest = RAW / name
    part = dest.with_suffix(dest.suffix + '.part')
    if not dest.exists():
        for attempt in range(8):
            try:
                offset = part.stat().st_size if part.exists() else 0
                request = urllib.request.Request(url, headers={'User-Agent': 'self-evolve-search/0.1', **({'Range': f'bytes={offset}-'} if offset else {})})
                with urllib.request.urlopen(request, timeout=90) as response:
                    resume = offset and response.status == 206
                    with part.open('ab' if resume else 'wb') as out:
                        for block in iter(lambda: response.read(1024 * 1024), b''): out.write(block)
                part.replace(dest)
                break
            except Exception as e:
                print(json.dumps({'download': name, 'attempt': attempt + 1, 'error': str(e)}), flush=True)
                if attempt == 7: raise
                time.sleep(min(2 ** attempt, 20))
    if expected_md5 and digest(dest, 'md5') != expected_md5: raise ValueError('MD5 mismatch: ' + name)
    result = {'file': name, 'url': url, 'bytes': dest.stat().st_size, 'sha256': digest(dest)}
    (RAW / (name + '.manifest.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps({'download_complete': name, 'bytes': result['bytes']}), flush=True)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', default='1908d6afbbead072334abe2965f91bd2709910ab', help='HotpotQA dataset revision; defaults to the recorded experiment')
    revision = parser.parse_args().revision
    (RAW / 'dataset_revision.json').write_text(json.dumps({'dataset': 'hotpotqa/hotpot_qa', 'revision': revision}, indent=2))
    tasks = [('https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/' + revision + '/fullwiki/' + name, name, None) for name in ['train-00000-of-00002.parquet', 'train-00001-of-00002.parquet', 'validation-00000-of-00001.parquet']]
    tasks += [('https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2', 'wiki-abstracts.tar.bz2', '01edf64cd120ecc03a2745352779514c'), ('https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py', 'hotpot_evaluate_v1.py', None)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda t: fetch(*t), tasks): pass
