"""Stop only this repository's validated inference process and descendants."""
import json, os, signal, time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
pid=int((root/'runs/model-server.pid').read_text())
cmd=Path(f'/proc/{pid}/cmdline')
if not cmd.exists(): print('Server already exited'); raise SystemExit(0)
command=cmd.read_bytes().replace(b'\0',b' ').decode(errors='replace')
if str(root/'scripts/serve.sh') not in command and (str(root/'scripts') not in command and '.venv-rollout/bin/vllm' not in command):
    raise RuntimeError('PID does not identify this rollout server: '+command)
if '.venv-rollout/bin/vllm' in command:
    cwd=Path(f'/proc/{pid}/cwd').resolve()
    if cwd!=root: raise RuntimeError('Server belongs to a different working directory')
parents={}; starts={}
for path in Path('/proc').iterdir():
    if not path.name.isdigit(): continue
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        parents[int(path.name)]=int(fields[1]); starts[int(path.name)]=fields[19]
    except (OSError,IndexError): pass
targets={pid}
while True:
    next_targets=targets|{p for p,parent in parents.items() if parent in targets}
    if next_targets==targets: break
    targets=next_targets
for target in sorted(targets,reverse=True):
    try: os.kill(target,signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
remaining=[]
for target in targets:
    try:
        fields=Path(f'/proc/{target}/stat').read_text().rsplit(')',1)[1].split()
        if fields[19]==starts.get(target) and fields[0]!='Z': os.kill(target,signal.SIGKILL); remaining.append(target)
    except (OSError,IndexError): pass
print(json.dumps({'validated_server_pid':pid,'signaled_pids':sorted(targets),'needed_kill':remaining}))
