"""Check real checkpoint/AdamW restoration with a meta base model and CPU adapters."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
from peft import PeftModel
from safetensors.torch import load_file
from self_evolve_search.persistence import adapter_policy, atomic_json, file_hash

parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
checkpoint=Path(args.checkpoint)
status=json.loads((checkpoint/'status.json').read_text())
assert status['complete']
assert status['policy_id']==adapter_policy(checkpoint/'adapter')
assert status['optimizer_sha256']==file_hash(checkpoint/'optimizer.pt')
model_path=json.loads((root/'config/local.json').read_text())['model_path']
config=AutoConfig.from_pretrained(model_path,local_files_only=True)
with torch.device('meta'):
    model=Qwen3_5ForConditionalGeneration(config)
model=PeftModel.from_pretrained(model,str(checkpoint/'adapter'),is_trainable=True,low_cpu_mem_usage=True,torch_device='cpu')
# PEFT moves adapters back to the meta base's device. Use that exact parameter
# ordering/shapes and real saved CPU adapter tensors for the optimizer check.
weights=load_file(str(checkpoint/'adapter/adapter_model.safetensors'),device='cpu')
named=[]
for name,parameter in model.named_parameters():
    if not parameter.requires_grad: continue
    value=weights[name.replace('.default.','.')]
    assert value.shape==parameter.shape
    named.append((name,torch.nn.Parameter(value.clone())))
names=[name for name,_ in named]
assert all(parameter.device.type=='cpu' for _,parameter in named)
saved=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True)
assert names==saved['parameter_names'] and saved['step']==status['step']
optimizer=torch.optim.AdamW([parameter for _,parameter in named],lr=5e-6)
optimizer.load_state_dict(saved['optimizer'])
assert all(int(state['step'])==status['step'] for state in optimizer.state.values())
assert all(torch.isfinite(state[key]).all() for state in optimizer.state.values() for key in ('exp_avg','exp_avg_sq'))
before=load_file(str(root/'runs/opsd-smoke/adapter/adapter_model.safetensors'))
after=load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
equal=before.keys()==after.keys() and all(torch.equal(before[key],after[key]) for key in before)
report={'complete':True,'step':status['step'],'optimizer_parameter_order_matches':True,
        'optimizer_parameter_states':len(optimizer.state),'all_optimizer_steps_match':True,
        'moments_finite':True,'meta_structure_with_real_cpu_adapter_tensors':True,'first_update_matches_compatibility_adapter_exactly':equal,
        'maximum_adapter_difference_from_compatibility':max((before[key]-after[key]).abs().max().item() for key in before)}
atomic_json(args.output,report); print(json.dumps(report,indent=2))
