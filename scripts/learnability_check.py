"""Can the gold evidence be reached under student permissions at all? (BM25 only, no model.)

For a seeded sample of failed round-1 trajectories this asks whether queries built purely from
what the restricted student can see would put both gold supporting documents in the top 2:

  hop 1  queries from the question: capitalised entity phrases, then the whole question
         (built exactly as retrieval.Library.search builds its FTS query);
  hop 2  queries from capitalised phrases in the text of a gold document reached in hop 1
         (what a student would see after reading it), alone and paired with a question entity.

It also records what the student actually retrieved and read, so failures split into
"never retrieved the evidence" versus "had it and mis-read / mis-answered".

Self-contained (standard library only) and resumable: progress is appended to
<output>.progress.jsonl and the summary is rewritten on every run. Use --time-budget to
stop cleanly after N seconds and call again to continue.

    python3 scripts/learnability_check.py --sample 250 --time-budget 165
"""
import argparse, json, random, re, sqlite3, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = set('a an the is was were are in on at of to for with and or which what who when where how did does do by from as it its this that has have had'.split())
QUESTION_WORDS = STOP | set('whom whose whats were what which who when where why how many much'.split())

def fts_terms(query):
    return [t for t in re.findall(r'\w+', str(query).casefold()) if t not in STOP][:12]

def search(db, query, top_k=2):
    """Round-1 retrieval: BM25 only (what the student faced in round 1)."""
    terms = fts_terms(query)
    if not terms: return []
    match = ' OR '.join('"' + t + '"' for t in terms)
    return [row[0] for row in db.execute('SELECT rowid FROM search WHERE search MATCH ? ORDER BY bm25(search,3,1),rowid LIMIT ?', (match, top_k)).fetchall()]

def library_search(index_path):
    """Current retrieval: self_evolve_search.retrieval.Library (title tiers, then BM25)."""
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from self_evolve_search.retrieval import Library
    library = Library(index_path)
    return lambda db, query, top_k=2: [int(d['doc_id']) for d in library.search(query, top_k)]

def entity_phrases(text, limit):
    """Maximal runs of capitalised or numeric tokens (allowing 'of/the/and/de' inside), most specific first."""
    tokens = re.findall(r"[A-Za-z0-9][\w'’\-\.]*", text)
    phrases, current = [], []
    for i, token in enumerate(tokens):
        word = token.strip('.')
        capital = word[:1].isupper() or word[:1].isdigit()
        connector = word.casefold() in ('of', 'the', 'de', 'la', 'du', 'von', 'van') and current
        if capital and not (i == 0 and word.casefold() in QUESTION_WORDS):
            current.append(word)
        elif connector:
            current.append(word)
        else:
            if current: phrases.append(' '.join(current))
            current = []
    if current: phrases.append(' '.join(current))
    cleaned = []
    for phrase in phrases:
        words = phrase.split()
        while words and words[0].casefold() in ('of', 'the', 'de', 'la', 'du', 'von', 'van'): words.pop(0)
        while words and words[-1].casefold() in ('of', 'the', 'de', 'la', 'du', 'von', 'van'): words.pop()
        phrase = ' '.join(words)
        if phrase and phrase.casefold() not in QUESTION_WORDS and phrase not in cleaned: cleaned.append(phrase)
    cleaned.sort(key=lambda p: -len(p.split()))
    return cleaned[:limit]

def question_files(folder):
    return sorted(p for p in folder.glob('*.json') if re.fullmatch('[a-f0-9]{24}', p.stem))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=str(ROOT / 'data/index/wiki.sqlite'))
    parser.add_argument('--questions', default=str(ROOT / 'data/splits/round1.jsonl'))
    parser.add_argument('--folders', nargs='*')
    parser.add_argument('--output', default=str(ROOT / 'runs/learnability-check.json'))
    parser.add_argument('--sample', type=int, default=250)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hop1-phrases', type=int, default=4)
    parser.add_argument('--hop2-phrases', type=int, default=6)
    parser.add_argument('--time-budget', type=float, default=1e9)
    parser.add_argument('--top-k', type=int, default=2, help='retrieval width to simulate (2 = student permissions)')
    parser.add_argument('--engine', choices=('fts', 'library'), default='fts', help='fts: round-1 BM25 query; library: the current Library.search (title tiers)')
    args = parser.parse_args()
    start = time.time()
    folders = [Path(f) for f in args.folders] if args.folders else [ROOT / 'runs/pilot500'] + [p for p in sorted((ROOT / 'runs/round1-seed42/batches').glob('step-*')) if p.name != 'step-001']
    rows = {}
    for line in Path(args.questions).read_text(encoding='utf-8').splitlines():
        row = json.loads(line); rows[row['id']] = row
    output = Path(args.output); progress = output.with_suffix('.progress.jsonl')
    done = {}
    if progress.exists():
        for line in progress.read_text(encoding='utf-8').splitlines():
            if line.strip(): item = json.loads(line); done[item['task_id']] = item
    # Deterministic sample of failed trajectories (files are only opened for the sampled IDs).
    failed_ids = []
    index_path = ROOT / 'runs/learnability-failed-ids.json'
    if index_path.exists():
        failed_ids = json.loads(index_path.read_text())
    else:
        for folder in folders:
            for path in question_files(folder):
                item = json.loads(path.read_text(encoding='utf-8'))
                if not item['student']['metrics']['grounded_success']:
                    failed_ids.append({'task_id': path.stem, 'path': str(path.relative_to(ROOT))})
        index_path.write_text(json.dumps(failed_ids))
    sample = random.Random(args.seed).sample(failed_ids, min(args.sample, len(failed_ids)))
    db = sqlite3.connect('file:' + args.index + '?mode=ro', uri=True); db.execute('PRAGMA cache_size=-262144')
    engine = library_search(args.index) if args.engine == 'library' else search
    for entry in sample:
        if entry['task_id'] in done: continue
        if time.time() - start > args.time_budget: break
        item = json.loads((ROOT / entry['path']).read_text(encoding='utf-8')); row = rows[entry['task_id']]
        gold_titles = list(dict.fromkeys(row['supporting_facts']['title']))
        gold = {}
        for title in gold_titles:
            found = db.execute('SELECT id, sentences FROM docs WHERE title=?', (title,)).fetchall()
            if found: gold[title] = {'ids': {int(f[0]) for f in found}, 'text': ' '.join(json.loads(found[0][1]))}
        student = item['student']['state']
        retrieved = {int(i) for i in student['retrieved']}
        read = {doc['title'] for doc in student['read_docs']}
        result = {'task_id': entry['task_id'], 'type': row.get('type'), 'gold_titles': gold_titles, 'gold_in_index': len(gold),
                  'student_retrieved_gold': sum(bool(g['ids'] & retrieved) for g in gold.values()),
                  'student_read_gold': sum(t in read for t in gold), 'student_error': item['student']['error'],
                  'student_answer_em': item['student']['metrics']['answer_em'], 'queries': 0, 'hop1': {}, 'hop2': {}}
        reached = {}
        phrases = entity_phrases(row['question'], args.hop1_phrases)
        candidates = phrases + ([' '.join(phrases[:2])] if len(phrases) > 1 else []) + [row['question']]
        for query in candidates:
            if len(reached) == len(gold): break
            top = engine(db, query, args.top_k); result['queries'] += 1
            for title, g in gold.items():
                if title not in reached and g['ids'] & set(top): reached[title] = query
        result['hop1'] = dict(reached)
        if 0 < len(reached) < len(gold):
            question_entities = entity_phrases(row['question'], 2)
            for source in list(reached):
                phrases = [p for p in entity_phrases(gold[source]['text'], args.hop2_phrases * 2) if p.casefold() != source.casefold()][:args.hop2_phrases]
                for phrase in phrases:
                    if len(reached) == len(gold): break
                    for query in [phrase] + [phrase + ' ' + e for e in question_entities[:1]]:
                        top = engine(db, query, args.top_k); result['queries'] += 1
                        for title, g in gold.items():
                            if title not in reached and g['ids'] & set(top): reached[title] = query; result['hop2'][title] = query
                        if len(reached) == len(gold): break
        result['all_gold_reachable'] = len(gold) > 0 and len(reached) == len(gold)
        result['reached'] = len(reached)
        with progress.open('a', encoding='utf-8') as handle: handle.write(json.dumps(result, ensure_ascii=False) + '\n')
        done[entry['task_id']] = result
        print(json.dumps({'done': len(done), 'of': len(sample), 'task': entry['task_id'], 'reachable': result['all_gold_reachable'], 'queries': result['queries']}), flush=True)
    sampled_done = [done[e['task_id']] for e in sample if e['task_id'] in done]
    n = len(sampled_done)
    def rate(items): return sum(1 for x in items if x['all_gold_reachable']) / len(items) if items else None
    summary = {'scope': f'reachability of gold supporting documents from student-visible terms only at top-{args.top_k} (2 = student permissions) with engine {args.engine}; no model calls', 'top_k': args.top_k, 'engine': args.engine,
               'failed_trajectories_total': len(failed_ids), 'sample_size': len(sample), 'completed': n, 'complete': n == len(sample),
               'all_gold_reachable_rate': rate(sampled_done),
               'by_question_type': {t: {'count': len([x for x in sampled_done if x['type'] == t]), 'reachable_rate': rate([x for x in sampled_done if x['type'] == t])} for t in sorted({x['type'] for x in sampled_done})},
               'by_student_retrieval': {f'retrieved_{k}_gold': {'count': len([x for x in sampled_done if x['student_retrieved_gold'] == k]), 'reachable_rate': rate([x for x in sampled_done if x['student_retrieved_gold'] == k])} for k in (0, 1, 2)},
               'student_retrieved_all_gold_rate': sum(x['student_retrieved_gold'] == x['gold_in_index'] and x['gold_in_index'] > 0 for x in sampled_done) / n if n else None,
               'student_read_all_gold_rate': sum(x['student_read_gold'] == x['gold_in_index'] and x['gold_in_index'] > 0 for x in sampled_done) / n if n else None,
               'hop1_only_rate': sum(1 for x in sampled_done if x['all_gold_reachable'] and not x['hop2']) / n if n else None,
               'needed_hop2_rate': sum(1 for x in sampled_done if x['all_gold_reachable'] and x['hop2']) / n if n else None,
               'gold_missing_from_index': sum(1 for x in sampled_done if x['gold_in_index'] < len(x['gold_titles'])),
               'mean_queries_per_question': sum(x['queries'] for x in sampled_done) / n if n else None,
               'student_errors': dict(Counter(x['student_error'] for x in sampled_done if x['student_error'])),
               'elapsed_seconds_this_run': time.time() - start}
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__': main()
