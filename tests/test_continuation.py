import importlib.util
import json
from pathlib import Path
import pytest
from self_evolve_search.persistence import BASE_POLICY, adapter_policy, atomic_json, bind_config, file_hash, validate_training_batch

def records(policy, prefix='q', count=8):
    return [{'task_id':prefix+str(i),'policy_id':policy,'original_wins':0,'edited_wins':3} for i in range(count)]

def controller_config(**overrides):
    config={'seed':42,'max_updates':2,'max_questions':5000,'batch_size':8,'fresh_records_per_update':8,'replay_window_updates':0,'max_record_reuse':1,
            'loss_tokens':'all','learning_rate':5e-6,'evaluation_file':'data/splits/dev_monitor.jsonl','evaluation_questions':500,'evaluation_interval':50,'bootstrap':'pilot',
            'repair_mode':'verified','step_selection':'ranked','objective':'opsd','dpo_beta':0.1}
    return config|overrides

def load_controller(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('experiment_test',root/'scripts/run_experiment.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module,'RUN',tmp_path)
    monkeypatch.setattr(module,'relative',lambda path:str(path))
    return module

def test_replay_window_reuses_recent_records_within_allowance(tmp_path,monkeypatch):
    module=load_controller(tmp_path,monkeypatch)
    experiment=module.Experiment.__new__(module.Experiment)
    experiment.config=controller_config(batch_size=8,fresh_records_per_update=4,replay_window_updates=2,max_record_reuse=2)
    # completed updates 1..3 with their fresh batches (4 records each) and source policies
    for step,policy in ((1,'base'),(2,'p1'),(3,'p2')):
        atomic_json(tmp_path/f'checkpoints/step-{step:03}/status.json',{'complete':True,'source_policy':policy,'step':step})
        atomic_json(tmp_path/f'batches/step-{step:03}/fresh/records.json',records(policy,f's{step}-',4))
    atomic_json(tmp_path/'replay-usage.json',{'s3-0':[3,3]})   # exhausted its allowance
    assert experiment.window_policies(4)=={'p1','p2'}          # steps 2 and 3 only
    candidates=experiment.replay_candidates(4)
    assert [c['task_id'] for c in candidates]==['s3-1','s3-2','s3-3','s2-0','s2-1','s2-2','s2-3']   # most recent first, s3-0 excluded, step 1 outside window
    fresh=records('p3','f',4)
    composed=experiment.compose_batch(4,tmp_path/'batches/step-004',fresh)
    assert [r['task_id'] for r in composed]==['f0','f1','f2','f3','s3-1','s3-2','s3-3','s2-0']
    assert all(r['replayed_from_step'] in (2,3) for r in composed[4:])
    validate_training_batch(composed,'p3',8,experiment.window_policies(4))
    with pytest.raises(ValueError,match='Stale'): validate_training_batch(composed,'p3',8)
    usage=json.loads((tmp_path/'replay-usage.json').read_text())
    assert usage['f0']==[4] and usage['s2-0']==[4] and usage['s3-0']==[3,3]
    # composing again returns the bound batch without touching usage
    assert experiment.compose_batch(4,tmp_path/'batches/step-004',fresh)==composed
    assert json.loads((tmp_path/'replay-usage.json').read_text())==usage
    # a record that was itself replayed is never replayed again from the composed file
    atomic_json(tmp_path/f'checkpoints/step-004/status.json',{'complete':True,'source_policy':'p3','step':4})
    assert 's2-0' not in {c['task_id'] for c in experiment.replay_candidates(5)}

def test_stale_and_duplicate_training_records_are_rejected():
    batch=records('checkpoint-a')
    validate_training_batch(batch,'checkpoint-a')
    with pytest.raises(ValueError,match='Stale'): validate_training_batch(batch,'checkpoint-b')
    batch[1]=batch[0]
    with pytest.raises(ValueError,match='Duplicate'): validate_training_batch(batch,'checkpoint-a')

def test_checkpoint_identity_and_resume_configuration_detect_changes(tmp_path):
    atomic_json(tmp_path/'adapter_config.json',{'r':16})
    (tmp_path/'adapter_model.safetensors').write_bytes(b'first weights')
    before=adapter_policy(tmp_path)
    (tmp_path/'adapter_model.safetensors').write_bytes(b'new weights')
    assert before!=adapter_policy(tmp_path)
    bind_config(tmp_path/'run.json',{'policy':before})
    with pytest.raises(ValueError,match='differs'): bind_config(tmp_path/'run.json',{'policy':adapter_policy(tmp_path)})

@pytest.mark.parametrize('checkpoint_written_before_restart',[False,True])
def test_controller_collects_with_updated_weights_and_never_repeats_completed_update(tmp_path,monkeypatch,checkpoint_written_before_restart):
    module=load_controller(tmp_path,monkeypatch)
    experiment=module.Experiment.__new__(module.Experiment)
    experiment.state={'updates':0,'cursor':500,'policy_id':BASE_POLICY,'checkpoint':None,'complete':False}
    experiment.config=controller_config(max_updates=2)
    events=[]
    experiment.status=lambda phase,**fields:experiment.state.update(phase=phase,**fields)
    experiment.evaluate=lambda step=None:events.append(('evaluate',experiment.state['updates'] if step is None else step))
    experiment.stop_server=lambda:events.append(('stop',))
    experiment.ensure_server=lambda *args:events.append(('server',)+args)
    first=tmp_path/'batches/step-001'
    atomic_json(first/'records.json',records(BASE_POLICY))
    atomic_json(first/'summary.json',{'next_cursor':500,'batch_ready':True,'verified':8})
    experiment.bootstrap=lambda:None

    def checkpoint(step,source,batch):
        target=tmp_path/f'checkpoints/step-{step:03}'
        atomic_json(target/'adapter/adapter_config.json',{'r':16})
        (target/'adapter/adapter_model.safetensors').write_bytes(f'weights-{step}'.encode())
        atomic_json(target/'status.json',{'complete':True,'source_policy':source,'policy_id':adapter_policy(target/'adapter'),
            'records_sha256':file_hash(batch/'records.json'),'step':step})
        return target

    if checkpoint_written_before_restart: checkpoint(1,BASE_POLICY,first)

    def command(arguments,log):
        arguments=[str(x) for x in arguments]
        name=Path(arguments[1]).name
        def option(key): return arguments[arguments.index(key)+1]
        if name=='train_update.py':
            batch=Path(option('--records')).parent
            validate_training_batch(json.loads((batch/'records.json').read_text()),option('--source-policy'))
            step=int(option('--step')); events.append(('train',step,option('--source-policy')))
            assert option('--objective')=='opsd' and option('--loss-tokens')=='all'
            if step==2: assert option('--previous')==str(tmp_path/'checkpoints/step-001')
            checkpoint(step,option('--source-policy'),batch)
        elif name=='collect_batch.py':
            assert int(option('--start'))==500 and int(option('--batch-size'))==8
            assert option('--repair-mode')=='verified' and option('--step-selection')=='ranked'
            assert option('--policy-id')==adapter_policy(tmp_path/'checkpoints/step-001/adapter')
            target=Path(option('--output')); assert target.name=='fresh'
            atomic_json(target/'records.json',records(option('--policy-id'),'fresh'))
            atomic_json(target/'summary.json',{'batch_ready':True,'verified':8,'next_cursor':508})
            events.append(('collect',option('--policy-id')))
        elif name=='check_adapter_serving.py':
            events.append(('probe',option('--model')))
        else: raise AssertionError('Unexpected command '+name)
    experiment.command=command
    experiment.run()
    assert experiment.state['complete'] and experiment.state['updates']==2 and experiment.state['cursor']==508
    trained_steps=[e[1] for e in events if e[0]=='train']
    assert trained_steps==([2] if checkpoint_written_before_restart else [1,2])
    assert len([e for e in events if e[0]=='collect'])==1

def test_paired_evaluation_uses_question_ids_and_can_be_rerun(tmp_path):
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    metrics=('answer_em','answer_f1','support_f1','joint_f1','grounded_success')
    for role in ('base','updated'):
        folder=tmp_path/role
        atomic_json(folder/'config.json',{'questions_sha256':'same-questions','protocol_sha256':'same-protocol'})
        for number in range(2):
            value=float(number==1 or role=='updated')
            atomic_json(folder/(str(number)*24+'.json'),{'metrics':{metric:value for metric in metrics}})
        mean=.5 if role=='base' else 1.
        atomic_json(folder/'summary.json',{'complete':True,'questions':2,'policy_id':role,**{metric:mean for metric in metrics}})
    output=tmp_path/'updated/comparison.json'
    arguments=[sys.executable,str(root/'scripts/compare_evaluation.py'),'--baseline',str(tmp_path/'base'),'--updated',str(tmp_path/'updated'),'--output',str(output)]
    for _ in range(2):
        subprocess.run(arguments,check=True,capture_output=True)
        report=json.loads(output.read_text())
        assert report['questions']==2
        assert report['metrics']['joint_f1']['delta']==.5
        assert report['metrics']['joint_f1']['paired_ci95']==[0.,1.]
