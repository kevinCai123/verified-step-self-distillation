"""Check a served adapter changes token probabilities on fixed training prompts."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
from self_evolve_search.persistence import atomic_json

parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
rows=[json.loads(line) for line in (root/'runs/pilot500/verified_steps.jsonl').read_text().splitlines()][:8]
checks=[]
for row in rows:
    replies=[]
    for model in ('Qwen/Qwen3.5-9B',args.model):
        body={'model':model,'messages':row['student_messages'],'temperature':0,'seed':42,'max_tokens':48,
              'logprobs':True,'top_logprobs':3,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_object'}}
        request=Request('http://127.0.0.1:8093/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=180) as handle: replies.append(json.load(handle))
    before,after=[reply['choices'][0] for reply in replies]
    changed=before['message']['content']!=after['message']['content'] or before['logprobs']!=after['logprobs']
    checks.append({'task_id':row['task_id'],'changed_output_or_probabilities':changed,'base':replies[0],'adapter':replies[1]})
    atomic_json(args.output,{'model':args.model,'checked':len(checks),'different':sum(c['changed_output_or_probabilities'] for c in checks),'checks':checks})
    print(json.dumps({'checked':len(checks),'different':sum(c['changed_output_or_probabilities'] for c in checks)}),flush=True)
if not any(c['changed_output_or_probabilities'] for c in checks): raise RuntimeError('No adapter effect detected; do not start updated-model collection')
