"""Single-GPU sequential collection / OPSD loop with durable resume points.

Run directory and experiment configuration are selected with environment variables so
that a completed run is never resumed under changed settings:

    SELF_EVOLVE_RUN=runs/round1b-seed42 SELF_EVOLVE_CONFIG=config/round1b.json bash scripts/run_experiment.sh

Defaults (runs/round1-seed42, config/experiment.json) reproduce the round-1 behaviour.
Configuration keys read from the experiment file (all optional except the round-1 ones):
  train_learning_rate, effective_batch_size, fresh_records_per_update, replay_window_updates,
  max_record_reuse, loss_tokens, evaluation_file, evaluation_questions, evaluation_interval,
  bootstrap ('pilot' or 'none'), start_cursor, max_updates, max_questions,
  repair_mode ('verified' | 'unverified' | 'evidence-only' | 'self-success'), step_selection ('ranked' | 'random'),
  objective ('opsd' | 'dpo' | 'sft'), dpo_beta.
A run directory may be pre-seeded with a state.json, checkpoints/step-NNN and evaluations from another run
to continue training from that checkpoint (see scripts/run_cycle2.sh, arm A continued).
"""
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
RUN=ROOT/os.environ.get('SELF_EVOLVE_RUN','runs/round1-seed42')
CONFIG_PATH=ROOT/os.environ.get('SELF_EVOLVE_CONFIG','config/experiment.json')
DATA=ROOT/'.venv-data/bin/python'
ROLLOUT=ROOT/'.venv-rollout/bin/python'
TRAIN=ROOT/'.venv-train/bin/python'
LOCAL=json.loads((ROOT/'config/local.json').read_text())
BASE_MODEL='Qwen/Qwen3.5-9B'

def relative(path):
    return str(Path(path).relative_to(ROOT))

def read_json(path, default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default

class Experiment:
    def __init__(self):
        RUN.mkdir(parents=True,exist_ok=True)
        self.lock=(RUN/'run.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.chdir(ROOT)
        experiment=json.loads(CONFIG_PATH.read_text())
        bootstrap=experiment.get('bootstrap','pilot')
        self.config={'seed':experiment.get('seed',42),'max_updates':experiment.get('max_updates',200),'max_questions':experiment.get('max_questions',5000),
            'batch_size':experiment.get('effective_batch_size',8),
            'fresh_records_per_update':experiment.get('fresh_records_per_update',experiment.get('effective_batch_size',8)),
            'replay_window_updates':experiment.get('replay_window_updates',0),'max_record_reuse':experiment.get('max_record_reuse',1),
            'loss_tokens':experiment.get('loss_tokens','all'),'learning_rate':experiment.get('train_learning_rate',5e-6),'lora_rank':experiment.get('lora_rank',16),
            'repair_mode':experiment.get('repair_mode','verified'),'step_selection':experiment.get('step_selection','ranked'),
            'objective':experiment.get('objective','opsd'),'dpo_beta':experiment.get('dpo_beta',0.1),
            'evaluation_file':experiment.get('evaluation_file','data/splits/dev_monitor.jsonl'),'evaluation_questions':experiment.get('evaluation_questions',500),
            'evaluation_interval':experiment.get('evaluation_interval',50),'bootstrap':bootstrap,
            'start_cursor':experiment.get('start_cursor',500 if bootstrap=='pilot' else 0),
            'bootstrap_questions_charged':experiment.get('bootstrap_questions_charged',500 if bootstrap=='pilot' else 0),
            'experiment_sha256':file_hash(CONFIG_PATH),'round1_sha256':file_hash(ROOT/'data/splits/round1.jsonl'),
            'dev_sha256':file_hash(ROOT/experiment.get('evaluation_file','data/splits/dev_monitor.jsonl')),
            'protocol_sha256':protocol_hash(ROOT,('collect_batch.py','evaluate.py','train_update.py','run_experiment.py')),
            'serve_sha256':file_hash(ROOT/'scripts/serve.sh'),'base_policy':BASE_POLICY,
            'evaluation_steps':'0, 1, every evaluation_interval updates, final'}
        if bootstrap=='pilot':
            self.config['pilot_verified_sha256']=file_hash(ROOT/'runs/pilot500/verified_steps.jsonl')
            self.config['bootstrap_note']='Reconstruct optimizer on the same eight pilot records used by the compatibility check; keep that check separate.'
        bind_config(RUN/'config.json',self.config)
        self.state=read_json(RUN/'state.json') or {
            'updates':0,'cursor':self.config['start_cursor'],'policy_id':BASE_POLICY,'checkpoint':None,'phase':'initializing','complete':False}
        if bootstrap=='pilot' and 'pilot_records_available' not in self.state:
            self.state.update(pilot_records_available=14,pilot_records_used=8,pilot_records_discarded_after_update=6)
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

    def current_alias(self):
        """Served-model alias and adapter path of the policy that collects the next batch."""
        if not self.state['updates']: return BASE_MODEL,None
        return 'step-'+str(self.state['updates'])+'-'+self.state['policy_id'].split(':')[1][:12],Path(self.state['checkpoint'])/'adapter'

    def evaluate(self,step=None):
        step=self.state['updates'] if step is None else step
        folder=RUN/f'evaluations/step-{step:03}'
        policy=BASE_POLICY if step==0 else json.loads((RUN/f'checkpoints/step-{step:03}/status.json').read_text())['policy_id']
        saved=read_json(folder/'summary.json',{})
        if saved and saved['policy_id']!=policy: raise ValueError('Wrong saved evaluation policy')
        if not saved.get('complete'):
            alias=BASE_MODEL if step==0 else 'step-'+str(step)+'-'+policy.split(':')[1][:12]
            adapter=RUN/f'checkpoints/step-{step:03}/adapter' if step else None
            self.status(f'evaluating internal development at update {step}')
            self.ensure_server(adapter,alias)
            self.command([ROLLOUT,'scripts/evaluate.py','--questions',self.config['evaluation_file'],'--index','data/index/wiki.sqlite',
                '--model-path',LOCAL['model_path'],'--model',alias,'--policy-id',policy,'--output',relative(folder),'--limit',str(self.config['evaluation_questions'])],folder/'run.log')
        if step and not (folder/'comparison.json').exists():
            self.command([DATA,'scripts/compare_evaluation.py','--baseline',relative(RUN/'evaluations/step-000'),
                '--updated',relative(folder),'--output',relative(folder/'comparison.json')],folder/'comparison.log')

    def bootstrap(self):
        if self.config['bootstrap']!='pilot': return
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

    def window_policies(self,next_step):
        """Policies whose records may be replayed for update `next_step` (source policies of the last W checkpoints)."""
        allowed=set()
        for step in range(max(1,next_step-self.config['replay_window_updates']),next_step):
            status=read_json(RUN/f'checkpoints/step-{step:03}/status.json')
            if status and status.get('complete'): allowed.add(status['source_policy'])
        return allowed

    def replay_candidates(self,next_step,exclude=()):
        """Records from the last W completed batches, most recent first, that are still within their reuse allowance."""
        usage=read_json(RUN/'replay-usage.json',{})
        allowed=self.window_policies(next_step)
        candidates=[]; seen=set(exclude)
        for step in range(next_step-1,max(0,next_step-self.config['replay_window_updates']-1),-1):
            folder=RUN/f'batches/step-{step:03}'
            records=read_json(folder/'fresh/records.json') or read_json(folder/'records.json') or []
            for record in records:
                if record['task_id'] in seen or record.get('policy_id') not in allowed: continue
                if len(usage.get(record['task_id'],[]))>=self.config['max_record_reuse']: continue
                if record.get('replayed_from_step') is not None: continue
                candidates.append(record|{'replayed_from_step':step}); seen.add(record['task_id'])
        return candidates

    def compose_batch(self,next_step,batch,fresh_records):
        """Fresh records first, then replayed ones; written once and bound thereafter."""
        composed=batch/'records.json'
        if composed.exists(): return json.loads(composed.read_text())
        exclude={r['task_id'] for r in fresh_records}
        replay=self.replay_candidates(next_step,exclude)[:max(0,self.config['batch_size']-len(fresh_records))]
        records=list(fresh_records)+replay
        if len(records)!=self.config['batch_size']: raise ValueError(f'Cannot compose a full batch for update {next_step}: {len(fresh_records)} fresh + {len(replay)} replayed')
        usage=read_json(RUN/'replay-usage.json',{})
        for record in records: usage.setdefault(record['task_id'],[]).append(next_step)
        atomic_json(RUN/'replay-usage.json',usage)
        atomic_json(composed,records)
        return records

    def run(self):
        if self.state['complete']: print('This experiment already completed.'); return
        self.bootstrap()
        interval=self.config['evaluation_interval']
        if self.state['updates']:
            self.evaluate(0)
            if self.state['updates']==1 or self.state['updates']%interval==0: self.evaluate()
        while self.state['updates']<self.config['max_updates'] and self.state['cursor']<self.config['max_questions']:
            next_step=self.state['updates']+1
            batch=RUN/f'batches/step-{next_step:03}'
            checkpoint=RUN/f'checkpoints/step-{next_step:03}'
            trained=read_json(checkpoint/'status.json',{})
            if not trained.get('complete'):
                batch_summary=read_json(batch/'summary.json',{})
                if not batch_summary.get('batch_ready'):
                    fresh=batch/'fresh'
                    fresh_summary=read_json(fresh/'summary.json',{})
                    if not fresh_summary.get('batch_ready'):
                        replay_available=len(self.replay_candidates(next_step))
                        needed=max(self.config['fresh_records_per_update'],self.config['batch_size']-replay_available)
                        self.status(f'collecting {needed} fresh verified repairs for update {next_step}',replay_available=replay_available)
                        alias,adapter=self.current_alias()
                        self.ensure_server(adapter,alias)
                        self.command([ROLLOUT,'scripts/collect_batch.py','--questions','data/splits/round1.jsonl','--index','data/index/wiki.sqlite',
                            '--model-path',LOCAL['model_path'],'--model',alias,'--policy-id',self.state['policy_id'],
                            '--start',str(self.state['cursor']),'--end',str(self.config['max_questions']),'--batch-size',str(needed),'--output',relative(fresh),
                            '--repair-mode',self.config['repair_mode'],'--step-selection',self.config['step_selection'],'--seed',str(self.config['seed'])],fresh/'run.log')
                        fresh_summary=json.loads((fresh/'summary.json').read_text())
                    if not fresh_summary['batch_ready']:
                        self.status('question budget exhausted without a full final batch',cursor=fresh_summary['next_cursor'],unused_final_records=fresh_summary['verified'])
                        break
                    records=self.compose_batch(next_step,batch,json.loads((fresh/'records.json').read_text()))
                    batch_summary=fresh_summary|{'batch_ready':True,'fresh':fresh_summary['verified'],'replayed':sum('replayed_from_step' in r for r in records),
                        'replayed_from_steps':sorted({r['replayed_from_step'] for r in records if 'replayed_from_step' in r}),'records':len(records)}
                    atomic_json(batch/'summary.json',batch_summary)
                self.status(f'training OPSD update {next_step}')
                self.stop_server()
                command=[TRAIN,'scripts/train_update.py','--records',relative(batch/'records.json'),'--model-path',LOCAL['model_path'],
                    '--source-policy',self.state['policy_id'],'--step',str(next_step),'--seed',str(self.config['seed']),'--output',relative(checkpoint),
                    '--learning-rate',str(self.config['learning_rate']),'--batch-size',str(self.config['batch_size']),'--loss-tokens',self.config['loss_tokens'],
                    '--objective',self.config['objective'],'--dpo-beta',str(self.config['dpo_beta'])]
                allowed=sorted(self.window_policies(next_step)-{self.state['policy_id']})
                if allowed: command += ['--allowed-policies',','.join(allowed)]
                if self.state['checkpoint']: command += ['--previous',self.state['checkpoint']]
                self.command(command,checkpoint/'run.log')
                trained=json.loads((checkpoint/'status.json').read_text())
            if not trained['complete'] or trained['source_policy']!=self.state['policy_id']: raise ValueError('Invalid completed checkpoint chain')
            if trained['records_sha256']!=file_hash(batch/'records.json'): raise ValueError('Training batch changed after update')
            if trained['policy_id']!=adapter_policy(checkpoint/'adapter'): raise ValueError('Saved checkpoint changed')
            batch_summary=json.loads((batch/'summary.json').read_text())
            self.status(f'completed update {next_step}',updates=next_step,checkpoint=str(checkpoint),policy_id=trained['policy_id'],cursor=batch_summary['next_cursor'])
            alias='step-'+str(next_step)+'-'+trained['policy_id'].split(':')[1][:12]
            self.ensure_server(checkpoint/'adapter',alias)
            self.command([DATA,'scripts/check_adapter_serving.py','--model',alias,'--records',relative(batch/'records.json'),'--output',relative(checkpoint/'serving-check.json')],checkpoint/'serving-check.log')
            if next_step==1: self.evaluate(0)
            if next_step==1 or next_step%interval==0: self.evaluate()
        self.evaluate()
        self.stop_server()
        self.status('round complete; review internal-development results before further rounds',complete=True)

if __name__=='__main__':
    experiment=None
    try:
        experiment=Experiment(); experiment.run()
    except BaseException as error:
        if experiment:
            experiment.status('stopped on error; safe to resume after resolving it',error=repr(error))
            (RUN/'traceback.txt').write_text(traceback.format_exc())
        raise
