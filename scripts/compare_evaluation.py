"""Paired internal-development deltas and question-bootstrap intervals."""
import argparse
import json
import random
import re
from pathlib import Path
from self_evolve_search.persistence import atomic_json

parser=argparse.ArgumentParser()
parser.add_argument('--baseline',required=True)
parser.add_argument('--updated',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
folders=[Path(args.baseline),Path(args.updated)]
summaries=[json.loads((folder/'summary.json').read_text()) for folder in folders]
assert all(s['complete'] for s in summaries)
configs=[json.loads((folder/'config.json').read_text()) for folder in folders]
assert configs[0]['questions_sha256']==configs[1]['questions_sha256']
assert configs[0]['protocol_sha256']==configs[1]['protocol_sha256']
rows=[]
for folder in folders:
    rows.append({path.stem:json.loads(path.read_text()) for path in folder.glob('*.json') if re.fullmatch('[a-f0-9]{24}',path.stem)})
assert rows[0].keys()==rows[1].keys() and len(rows[0])==summaries[0]['questions']
ids=sorted(rows[0]); metrics=('answer_em','answer_f1','support_f1','joint_f1','grounded_success')
rng=random.Random(42)
indices=[[rng.randrange(len(ids)) for _ in ids] for _ in range(2000)]
report={'questions':len(ids),'baseline_policy':summaries[0]['policy_id'],'updated_policy':summaries[1]['policy_id'],
        'bootstrap_samples':2000,'bootstrap_seed':42,'scope':'internal development, not final evaluation','metrics':{}}
for metric in metrics:
    deltas=[rows[1][i]['metrics'][metric]-rows[0][i]['metrics'][metric] for i in ids]
    draws=sorted(sum(deltas[j] for j in draw)/len(ids) for draw in indices)
    report['metrics'][metric]={'baseline':summaries[0][metric],'updated':summaries[1][metric],
        'delta':sum(deltas)/len(ids),'paired_ci95':[draws[49],draws[1949]]}
atomic_json(args.output,report); print(json.dumps(report,indent=2))
