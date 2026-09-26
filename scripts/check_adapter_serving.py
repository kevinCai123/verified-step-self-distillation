"""Check a served adapter changes token probabilities on fixed training prompts."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
from self_evolve_search.opsd import premature_finish_share
from self_evolve_search.persistence import atomic_json

parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--records',help='JSON list or JSONL of verified records whose student prefixes are probed (default: the pilot records)')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
if args.records and args.records.endswith('.json'):
    rows=json.loads(Path(args.records).read_text())[:8]
else:
    source=Path(args.records) if args.records else root/'runs/pilot500/verified_steps.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()][:8]
checks=[]
for row in rows:
    replies=[]
    for model in ('Qwen/Qwen3.5-9B',args.model):
        body={'model':model,'messages':row['student_messages'],'temperature':0,'seed':42,'max_tokens':96,
              'logprobs':True,'top_logprobs':3,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_object'}}
        request=Request('http://127.0.0.1:8093/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=180) as handle: replies.append(json.load(handle))
    before,after=[reply['choices'][0] for reply in replies]
    changed=before['message']['content']!=after['message']['content'] or before['logprobs']!=after['logprobs']
    checks.append({'task_id':row['task_id'],'changed_output_or_probabilities':changed,'base':replies[0],'adapter':replies[1]})
    atomic_json(args.output,{'model':args.model,'checked':len(checks),'different':sum(c['changed_output_or_probabilities'] for c in checks),'checks':checks})
    print(json.dumps({'checked':len(checks),'different':sum(c['changed_output_or_probabilities'] for c in checks)}),flush=True)
if not any(c['changed_output_or_probabilities'] for c in checks): raise RuntimeError('No adapter effect detected; do not start updated-model collection')
share = premature_finish_share(rows, [c['adapter']['choices'][0]['message']['content'] for c in checks])
base_share = premature_finish_share(rows, [c['base']['choices'][0]['message']['content'] for c in checks])
report = json.loads(Path(args.output).read_text()); report.update(premature_finish_share=share, base_premature_finish_share=base_share); atomic_json(args.output, report)
print(json.dumps({'premature_finish_share': share, 'base_premature_finish_share': base_share}), flush=True)
if share >= .75 and share > base_share + .5:
    raise RuntimeError(f'Adapter drifted to premature finishing on {share:.0%} of probed training states (base {base_share:.0%}); stopping before collection')
