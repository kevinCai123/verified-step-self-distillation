import argparse, hashlib, json, time
from pathlib import Path
from transformers import AutoTokenizer
from self_evolve_search.agent import Client, offline_guard, run
from self_evolve_search.retrieval import Library
from self_evolve_search.repair import repair
from self_evolve_search.reporting import write_report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--questions',required=True); p.add_argument('--index',required=True); p.add_argument('--model-path',required=True); p.add_argument('--endpoint',default='http://127.0.0.1:8093'); p.add_argument('--output',required=True); p.add_argument('--limit',type=int,default=500); p.add_argument('--repair',action='store_true'); p.add_argument('--pause-after',type=int,help='Pause cleanly after this many completed questions; resume without changing the protocol'); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    config=vars(a).copy(); config['questions_sha256']=hashlib.sha256(Path(a.questions).read_bytes()).hexdigest()
    config.pop('pause_after')
    root=Path(__file__).resolve().parents[1]
    sources=[root/'src/self_evolve_search'/name for name in ('agent.py','retrieval.py','metrics.py','repair.py','tokenization.py')]+[Path(__file__)]
    config['protocol_sha256']=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    config['index_manifest']=json.loads(Path(a.index).with_suffix('.manifest.json').read_text())
    cfg=out/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config: raise ValueError('Output directory contains a different experiment configuration')
    cfg.write_text(json.dumps(config,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
    client=Client(a.endpoint,'Qwen/Qwen3.5-9B',tokenizer)
    client.health()
    lib=Library(a.index); offline_guard()
    rows=[json.loads(x) for x in Path(a.questions).read_text().splitlines()][:a.limit]
    results=[]; start=time.time()
    for n,row in enumerate(rows,1):
        target=out/(row['id']+'.json')
        if target.exists(): item=json.loads(target.read_text())
        else:
            call_start=len(client.calls); tool_start=len(lib.calls); question_start=time.perf_counter()
            student=run(row,client,lib,2); teacher=run(row,client,lib,8)
            item={'student':student,'teacher':teacher}
            if a.repair: item['repair']=repair(row,student,teacher,client,lib)
            item['calls']=client.calls[call_start:]
            item['tool_calls']=lib.calls[tool_start:]
            item['elapsed_seconds']=time.perf_counter()-question_start
            temp=target.with_suffix('.tmp'); temp.write_text(json.dumps(item,ensure_ascii=False)); temp.replace(target)
        results.append(item)
        summary=summarize(results)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        with (out/'verified_steps.jsonl').open('w') as f:
            for x in results:
                verified=x.get('repair',{}).get('confirmed')
                if verified: f.write(json.dumps(verified,ensure_ascii=False)+'\n')
        write_report(Path(__file__).resolve().parents[1],out)
        print(json.dumps({'completed':n,'total':len(rows),'student_em':summary['student']['answer_em'],'teacher_em':summary['teacher']['answer_em'],'confirmed':summary['confirmed_repairs'],'seconds':round(time.time()-start)}),flush=True)
        if a.pause_after and n>=a.pause_after: break

def summarize(results):
    n=len(results); metrics=('answer_em','answer_f1','support_f1','joint_f1','grounded_success')
    result={'questions':n}
    for role in ('student','teacher'):
        result[role]={m:sum(r[role]['metrics'][m] for r in results)/n for m in metrics}
        result[role]['failures']=sum(bool(r[role]['error']) for r in results)
    result['failed_student_trajectories']=sum(not r['student']['metrics']['grounded_success'] for r in results)
    result['confirmed_repairs']=sum(bool(r.get('repair',{}).get('confirmed')) for r in results)
    result['training_yield']=result['confirmed_repairs']/n
    result['repair_at_3']=result['confirmed_repairs']/max(1,result['failed_student_trajectories'])
    attempts=[a for r in results for a in r.get('repair',{}).get('attempts',[])]
    result['candidate_proposals']=len(attempts)
    result['legal_proposals']=sum(a['legal'] for a in attempts)
    result['legal_rate']=result['legal_proposals']/len(attempts) if attempts else None
    result['model_calls']=sum(len(r['calls']) for r in results)
    result['input_tokens']=sum(c['usage']['prompt_tokens'] for r in results for c in r['calls'])
    result['output_tokens']=sum(c['usage']['completion_tokens'] for r in results for c in r['calls'])
    result['tool_calls']=sum(len(r['tool_calls']) for r in results)
    result['elapsed_seconds']=sum(r['elapsed_seconds'] for r in results)
    result['seconds_per_confirmed_repair']=result['elapsed_seconds']/result['confirmed_repairs'] if result['confirmed_repairs'] else None
    confirmations=[r['repair']['selected'] for r in results if r.get('repair',{}).get('selected')]
    result['confirmation_candidates']=len(confirmations)
    result['paired_replay_gain']=sum((c['edited_wins']-c['original_wins'])/3 for c in confirmations)/len(confirmations) if confirmations else None
    return result

if __name__=='__main__': main()
