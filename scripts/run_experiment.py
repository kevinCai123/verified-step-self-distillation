"""Single-GPU sequential collection / OPSD loop with durable resume points."""
import fcntl
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from self_evolve_search.persistence import BASE_POLICY, adapter_policy, atomic_json, bind_config, file_hash, protocol_hash
from self_evolve_search.reporting import write_report

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/round1-seed42'
DATA=ROOT/'.venv-data/bin/python'
ROLLOUT=ROOT/'.venv-rollout/bin/python'
TRAIN=ROOT/'.venv-train/bin/python'
LOCAL=json.loads((ROOT/'config/local.json').read_text())
BASE_MODEL='Qwen/Qwen3.5-9B'

def relative(path):
    return str(Path(path).relative_to(ROOT))

class Experiment:
    def __init__(self):
        RUN.mkdir(parents=True,exist_ok=True)
        self.lock=(RUN/'run.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.chdir(ROOT)
        self.config={'seed':42,'max_updates':200,'max_questions':5000,'batch_size':8,'bootstrap_questions_charged':500,
            'round1_sha256':file_hash(ROOT/'data/splits/round1.jsonl'),'dev_sha256':file_hash(ROOT/'data/splits/dev_monitor.jsonl'),
            'pilot_verified_sha256':file_hash(ROOT/'runs/pilot500/verified_steps.jsonl'),
            'protocol_sha256':protocol_hash(ROOT,('collect_batch.py','evaluate.py','train_update.py','run_experiment.py')),
            'serve_sha256':file_hash(ROOT/'scripts/serve.sh'),'learning_rate':5e-6,'lora_rank':16,'base_policy':BASE_POLICY,
            'evaluation_steps':'0, 1, every 50 updates, final','bootstrap':'Reconstruct optimizer on the same eight pilot records used by the compatibility check; keep that check separate.'}
        bind_config(RUN/'config.json',self.config)
        self.state=json.loads((RUN/'state.json').read_text()) if (RUN/'state.json').exists() else {
            'updates':0,'cursor':500,'policy_id':BASE_POLICY,'checkpoint':None,'phase':'initializing','complete':False,
            'pilot_records_available':14,'pilot_records_used':8,'pilot_records_discarded_after_update':6}
        self.server=None

    def status(self,phase,**fields):
        self.state.update(phase=phase,updated=datetime.now(timezone.utc).isoformat(),**fields)
        atomic_json(RUN/'state.json',self.state)
        write_report(ROOT)
        print(json.dumps(self.state),flush=True)

    def command(self,arguments,log):
        log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as output:
            process=subprocess.Popen([str(x) for x in arguments],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    time.sleep(10); write_report(ROOT)
            except BaseException:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise
            if process.returncode: raise RuntimeError(f'Command exited {process.returncode}: {arguments}; see {log}')

    def healthy(self,alias,adapter):
        try:
            with urlopen('http://127.0.0.1:8093/v1/models',timeout=5) as handle: models=json.load(handle)['data']
            match=next((m for m in models if m['id']==alias),None)
            if match is None: return False
            expected=Path(adapter).resolve() if adapter else Path(LOCAL['model_path']).resolve()
            return Path(match['root']).resolve()==expected
        except OSError: return False

    def stop_server(self):
        self.command([DATA,'scripts/stop_server.py'],RUN/'server-control.log')
        if self.server:
            self.server.wait(timeout=30); self.server=None

    def ensure_server(self,adapter=None,alias=BASE_MODEL):
        if self.healthy(alias,adapter): return
        self.stop_server()
        command=['bash','scripts/serve.sh']
        if adapter: command += [str(adapter),alias]
        log=(RUN/f'server-step-{self.state["updates"]:03}.log').open('a')
        self.server=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        log.close()
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if self.server.poll() is not None: raise RuntimeError('Inference server exited while loading checkpoint')
            if self.healthy(alias,adapter): return
            time.sleep(5); write_report(ROOT)
        raise TimeoutError('Inference server startup timed out')

    def evaluate(self,step=None):
        step=self.state['updates'] if step is None else step
        folder=RUN/f'evaluations/step-{step:03}'
        policy=BASE_POLICY if step==0 else json.loads((RUN/f'checkpoints/step-{step:03}/status.json').read_text())['policy_id']
        summary=folder/'summary.json'
        saved=json.loads(summary.read_text()) if summary.exists() else {}
        if saved and saved['policy_id']!=policy: raise ValueError('Wrong saved evaluation policy')
        if not saved.get('complete'):
            alias=BASE_MODEL if step==0 else 'step-'+str(step)+'-'+policy.split(':')[1][:12]
            adapter=RUN/f'checkpoints/step-{step:03}/adapter' if step else None
            self.status(f'evaluating internal development at update {step}')
            self.ensure_server(adapter,alias)
            self.command([ROLLOUT,'scripts/evaluate.py','--questions','data/splits/dev_monitor.jsonl','--index','data/index/wiki.sqlite',
                '--model-path',LOCAL['model_path'],'--model',alias,'--policy-id',policy,'--output',relative(folder),'--limit','500'],folder/'run.log')
        if step and not (folder/'comparison.json').exists():
            self.command([DATA,'scripts/compare_evaluation.py','--baseline',relative(RUN/'evaluations/step-000'),
                '--updated',relative(folder),'--output',relative(folder/'comparison.json')],folder/'comparison.log')

    def bootstrap(self):
        folder=RUN/'batches/step-001'; folder.mkdir(parents=True,exist_ok=True)
        records={r['task_id']:r for r in [json.loads(line) for line in (ROOT/'runs/pilot500/verified_steps.jsonl').read_text().splitlines()]}
        ids=json.loads((ROOT/'runs/opsd-smoke/status.json').read_text())['record_ids']
        chosen=[records[i]|{'policy_id':BASE_POLICY} for i in ids]
        assert len(chosen)==8
        if (folder/'records.json').exists():
            if json.loads((folder/'records.json').read_text())!=chosen: raise ValueError('Bootstrap records changed')
        else:
            atomic_json(folder/'records.json',chosen)
            atomic_json(folder/'summary.json',{'policy_id':BASE_POLICY,'start':0,'next_cursor':500,'attempted':500,
                'verified':8,'batch_ready':True,'exhausted':False,'source':'completed pilot; eight compatibility-test decision points'})

    def run(self):
        if self.state['complete']: print('This experiment already completed.'); return
        self.bootstrap()
        if self.state['updates']:
            self.evaluate(0)
            if self.state['updates']==1 or self.state['updates']%50==0: self.evaluate()
        while self.state['updates']<self.config['max_updates'] and self.state['cursor']<self.config['max_questions']:
            next_step=self.state['updates']+1
            batch=RUN/f'batches/step-{next_step:03}'
            checkpoint=RUN/f'checkpoints/step-{next_step:03}'
            checkpoint_status=checkpoint/'status.json'
            trained=json.loads(checkpoint_status.read_text()) if checkpoint_status.exists() else {}
            if not trained.get('complete'):
                batch_summary=json.loads((batch/'summary.json').read_text()) if (batch/'summary.json').exists() else {}
                if not batch_summary.get('batch_ready'):
                    self.status(f'collecting fresh verified repairs for update {next_step}')
                    alias='step-'+str(self.state['updates'])+'-'+self.state['policy_id'].split(':')[1][:12]
                    adapter=Path(self.state['checkpoint'])/'adapter'
                    self.ensure_server(adapter,alias)
                    self.command([ROLLOUT,'scripts/collect_batch.py','--questions','data/splits/round1.jsonl','--index','data/index/wiki.sqlite',
                        '--model-path',LOCAL['model_path'],'--model',alias,'--policy-id',self.state['policy_id'],
                        '--start',str(self.state['cursor']),'--end','5000','--output',relative(batch)],batch/'run.log')
                    batch_summary=json.loads((batch/'summary.json').read_text())
                if not batch_summary['batch_ready']:
                    self.status('question budget exhausted without a full final batch',cursor=batch_summary['next_cursor'],unused_final_records=batch_summary['verified'])
                    break
                self.status(f'training OPSD update {next_step}')
                self.stop_server()
                command=[TRAIN,'scripts/train_update.py','--records',relative(batch/'records.json'),'--model-path',LOCAL['model_path'],
                    '--source-policy',self.state['policy_id'],'--step',str(next_step),'--seed','42','--output',relative(checkpoint)]
                if self.state['checkpoint']: command += ['--previous',self.state['checkpoint']]
                self.command(command,checkpoint/'run.log')
                trained=json.loads(checkpoint_status.read_text())
            if not trained['complete'] or trained['source_policy']!=self.state['policy_id']: raise ValueError('Invalid completed checkpoint chain')
            if trained['records_sha256']!=file_hash(batch/'records.json'): raise ValueError('Training batch changed after update')
            if trained['policy_id']!=adapter_policy(checkpoint/'adapter'): raise ValueError('Saved checkpoint changed')
            batch_summary=json.loads((batch/'summary.json').read_text())
            self.status(f'completed update {next_step}',updates=next_step,checkpoint=str(checkpoint),policy_id=trained['policy_id'],cursor=batch_summary['next_cursor'])
            alias='step-'+str(next_step)+'-'+trained['policy_id'].split(':')[1][:12]
            self.ensure_server(checkpoint/'adapter',alias)
            self.command([DATA,'scripts/check_adapter_serving.py','--model',alias,'--output',relative(checkpoint/'serving-check.json')],checkpoint/'serving-check.log')
            if next_step==1: self.evaluate(0)
            if next_step==1 or next_step%50==0: self.evaluate()
        self.evaluate()
        self.stop_server()
        self.status('round 1 complete; review internal-development results before further rounds',complete=True)

if __name__=='__main__':
    experiment=None
    try:
        experiment=Experiment(); experiment.run()
    except BaseException as error:
        if experiment:
            experiment.status('stopped on error; safe to resume after resolving it',error=repr(error))
            (RUN/'traceback.txt').write_text(traceback.format_exc())
        raise
