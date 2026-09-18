import importlib.util
import json
from pathlib import Path
import pytest
from self_evolve_search.persistence import BASE_POLICY, adapter_policy, atomic_json, bind_config, file_hash, validate_training_batch

def records(policy, prefix='q'):
    return [{'task_id':prefix+str(i),'policy_id':policy,'original_wins':0,'edited_wins':3} for i in range(8)]

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
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('experiment_test',root/'scripts/run_experiment.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module,'RUN',tmp_path)
    monkeypatch.setattr(module,'relative',lambda path:str(path))
    experiment=module.Experiment.__new__(module.Experiment)
    experiment.state={'updates':0,'cursor':500,'policy_id':BASE_POLICY,'checkpoint':None,'complete':False}
    experiment.config={'max_updates':2,'max_questions':5000}
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
            if step==2: assert option('--previous')==str(tmp_path/'checkpoints/step-001')
            checkpoint(step,option('--source-policy'),batch)
        elif name=='collect_batch.py':
            assert int(option('--start'))==500
            assert option('--policy-id')==adapter_policy(tmp_path/'checkpoints/step-001/adapter')
            target=Path(option('--output'))
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
