"""Bind all model/evaluation code and package locks before continuing a run."""
from pathlib import Path
from self_evolve_search.persistence import bind_config, file_hash

root=Path(__file__).resolve().parents[1]
files=['config/experiment.json']
files += ['src/self_evolve_search/'+name+'.py' for name in ('agent','repair','retrieval','metrics','opsd','persistence','tokenization')]
files += ['scripts/'+name for name in ('run_experiment.py','collect_batch.py','evaluate.py','compare_evaluation.py','train_update.py','serve.sh','runtime_env.sh','check_adapter_serving.py')]
files += ['requirements-'+name+'.lock' for name in ('data','rollout','train')]
bind_config(root/'runs/round1-seed42/source-manifest.json',{name:file_hash(root/name) for name in files})
print('Model, training, evaluation sources and package locks match the frozen run.')
