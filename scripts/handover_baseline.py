"""Stop only the standalone baseline worker so the resumable controller can own it."""
import json
import os
import signal
from pathlib import Path
from self_evolve_search.persistence import atomic_json

root=Path(__file__).resolve().parents[1]
targets=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit(): continue
    try:
        command=(path/'cmdline').read_bytes().split(b'\0')
        if b'scripts/evaluate.py' in command and b'runs/round1-seed42/evaluations/step-000' in command:
            if (path/'cwd').resolve()!=root: raise RuntimeError('Evaluation worker belongs to a different directory')
            targets.append(int(path.name))
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        pass
if len(targets)>1: raise RuntimeError('More than one matching baseline worker; refusing ambiguous handover')
for pid in targets: os.kill(pid,signal.SIGINT)
atomic_json(root/'runs/baseline-handover.json',{'interrupted_worker_pids':targets,'reason':'Transfer the existing baseline to the resumable main controller; completed atomic question files are retained. Any unfinished question is rerun.'})
print(json.dumps({'interrupted_worker_pids':targets}))
