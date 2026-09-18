"""Read-only audit of saved pilot completeness, replays and actual server usage."""
import collections, hashlib, json, re
from pathlib import Path

root=Path(__file__).resolve().parents[1]; folder=root/'runs/pilot500'
files=sorted(p for p in folder.glob('*.json') if re.fullmatch('[a-f0-9]{24}',p.stem))
expected={json.loads(line)['id'] for line in (root/'data/splits/pilot.jsonl').read_text().splitlines()}
results=[json.loads(p.read_text()) for p in files]
verified=[json.loads(line) for line in (folder/'verified_steps.jsonl').read_text().splitlines() if line.strip()]
confirmed=[r['repair']['confirmed'] for r in results if r.get('repair',{}).get('confirmed')]
assert {p.stem for p in files}==expected and len(results)==500
assert sorted(verified,key=lambda r:r['task_id'])==sorted(confirmed,key=lambda r:r['task_id'])
assert all(r['edited_wins']>=2 and r['original_wins']<=1 for r in confirmed)
repair_types=collections.Counter()
for result in results:
    record=result.get('repair',{}).get('confirmed')
    if not record: continue
    selected=result['repair']['selected']; pairs=selected['confirmation']
    assert [pair['seed'] for pair in pairs]==[1701,1702,1703]
    assert sum(pair['edited']['metrics']['grounded_success'] for pair in pairs)==record['edited_wins']
    assert sum(pair['original']['metrics']['grounded_success'] for pair in pairs)==record['original_wins']
    for pair in pairs:
        for role in ('original','edited'):
            replay=pair[role]
            assert replay['top_k']==2 and replay['state']['tool_calls']<=8 and replay['state']['generated_tokens']<=2048
            assert replay['trace'][0]['pre_state']==record['student_state']
    repair_types[record['replacement']['action']]+=1
calls=[c for r in results for c in r['calls']]
ordinary=[]
def visit(node):
    if isinstance(node,dict):
        if 'pre_state' in node and 'reply' in node and node['reply'].get('finish_reason')!='forced': ordinary.append(node['reply']['usage']['prompt_tokens'])
        for value in node.values(): visit(value)
    elif isinstance(node,list):
        for value in node: visit(value)
for result in results: visit(result)
report={'saved_questions':len(results),'verified_repairs':len(verified),'all_expected_ids_present':True,'verified_records_match_replays':True,'temporary_question_files':len(list(folder.glob('*.tmp'))),'actual_max_prompt_tokens':max(c['usage']['prompt_tokens'] for c in calls),'actual_max_total_tokens':max(c['usage']['total_tokens'] for c in calls),'ordinary_trace_prompts_above_6144':sum(n>6144 for n in ordinary),'diagnoses_above_7552':sum(r.get('repair',{}).get('diagnosis_reply',{}).get('usage',{}).get('prompt_tokens',0)>7552 for r in results),'partial_overlap_excluded':sum(r.get('repair',{}).get('ambiguous_answer_excluded',False) for r in results),'student_errors':dict(collections.Counter(r['student']['error'] for r in results if r['student']['error'])),'teacher_errors':dict(collections.Counter(r['teacher']['error'] for r in results if r['teacher']['error'])),'saved_files_sha256':hashlib.sha256(b''.join(p.name.encode()+hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest(),'pilot_protocol_sha256':json.loads((folder/'config.json').read_text())['protocol_sha256'],'tokenizer_bug':'The old chat-template default returned BatchEncoding; len counted fields rather than tokens. Actual server token usage is audited here; fixed code requests return_dict=False.'}
report['confirmed_replay_states_budgets_and_seeds_checked']=True
report['verified_replacement_action_types']=dict(repair_types)
(root/'runs/recovery-audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
