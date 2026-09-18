import json
from pathlib import Path
from transformers import AutoTokenizer
from self_evolve_search.agent import Client, initial_state, validate_action, offline_guard
root=Path(__file__).resolve().parents[1]
model=json.loads((root/'config/local.json').read_text())['model_path']
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
client=Client('http://127.0.0.1:8093','Qwen/Qwen3.5-9B',tokenizer)
client.health(); offline_guard()
row=json.loads((root/'data/splits/pilot.jsonl').read_text().splitlines()[0])
state=initial_state(row['question'],2)
reply=client.chat(state['messages'])
kind=validate_action(json.loads(reply['content']),state)
result={'task_id':row['id'],'action_type':kind,'reply':reply,'scope':'single-action serving smoke; not a completed task score'}
(root/'runs/serving-smoke.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
