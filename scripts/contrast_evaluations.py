"""Paired contrast between two trained policies' internal-development evaluations (same 1,500 questions):
question-bootstrap intervals for updated − baseline, exactly as scripts/compare_evaluation.py computes a policy's
delta against the base model, but for arm-vs-arm comparisons (A vs D, A vs S, ...). Unlike compare_evaluation.py it
does not require equal evaluate protocol hashes: the arms were evaluated under the pre- and post-repair.py-fix hashes
(a diagnosis-parsing change evaluate.py never executes), so both hashes are recorded in the output instead.
    python scripts/contrast_evaluations.py --baseline runs/armD-seed42/evaluations/step-023 \
        --updated runs/round1b-seed42/evaluations/step-032 --output results/contrasts/A42-vs-D42.json"""
import argparse, json, random, re
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--updated',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); folders=[Path(a.baseline),Path(a.updated)]
summaries=[json.loads((f/'summary.json').read_text()) for f in folders]; assert all(s['complete'] for s in summaries)
configs=[json.loads((f/'config.json').read_text()) for f in folders]; assert configs[0]['questions_sha256']==configs[1]['questions_sha256']
rows=[{x.stem:json.loads(x.read_text()) for x in f.glob('*.json') if re.fullmatch('[a-f0-9]{24}',x.stem)} for f in folders]
assert rows[0].keys()==rows[1].keys() and len(rows[0])==summaries[0]['questions']
ids=sorted(rows[0]); metrics=('answer_em','answer_f1','support_f1','joint_f1','grounded_success'); rng=random.Random(42)
idx=[[rng.randrange(len(ids)) for _ in ids] for _ in range(2000)]
rep={'questions':len(ids),'baseline':str(folders[0]),'updated':str(folders[1]),'baseline_policy':summaries[0]['policy_id'],'updated_policy':summaries[1]['policy_id'],
     'protocol_sha256':[c['protocol_sha256'] for c in configs],'bootstrap_samples':2000,'bootstrap_seed':42,'scope':'internal development, arm-vs-arm paired contrast','metrics':{}}
for m in metrics:
    d=[rows[1][i]['metrics'][m]-rows[0][i]['metrics'][m] for i in ids]; draws=sorted(sum(d[j] for j in dr)/len(ids) for dr in idx)
    rep['metrics'][m]={'baseline':summaries[0][m],'updated':summaries[1][m],'delta':sum(d)/len(ids),'paired_ci95':[draws[49],draws[1949]]}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(rep,indent=2)); print(a.output, {m:round(v['delta'],4) for m,v in rep['metrics'].items()})
