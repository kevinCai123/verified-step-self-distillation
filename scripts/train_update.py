"""One resumable OPSD update; optimizer state continues across fresh batches."""
import argparse
import fcntl
import gc
import json
import time
import traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model
from self_evolve_search.opsd import action_inputs, jsd_loss, teacher_messages
from self_evolve_search.persistence import BASE_POLICY, adapter_policy, atomic_json, file_hash, validate_training_batch
from self_evolve_search.tokenization import chat_ids

def main():
    parser = argparse.ArgumentParser()
    for option in ('records', 'model-path', 'source-policy', 'output'):
        parser.add_argument('--'+option, required=True)
    parser.add_argument('--previous')
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    update_lock=(out/'update.lock').open('a+')
    fcntl.flock(update_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('complete'):
        raise ValueError('Refuse to overwrite a completed update')
    status = {'complete': False, 'step': args.step, 'source_policy': args.source_policy,
              'records_sha256': file_hash(args.records), 'optimizer_updates': 0, 'seed': args.seed}
    atomic_json(out/'status.json', status); start = time.time()
    try:
        records = json.loads(Path(args.records).read_text())
        validate_training_batch(records, args.source_policy)
        previous = Path(args.previous) if args.previous else None
        if previous:
            parent = json.loads((previous/'status.json').read_text())
            if not parent['complete'] or parent['step']+1 != args.step: raise ValueError('Broken checkpoint chain')
            if adapter_policy(previous/'adapter') != args.source_policy: raise ValueError('Source adapter hash mismatch')
        elif args.step != 1 or args.source_policy != BASE_POLICY:
            raise ValueError('The first update must use the pinned base policy')
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        prepared = []
        for record in records:
            student_ids = chat_ids(tokenizer, record['student_messages'])
            teacher_ids = chat_ids(tokenizer, teacher_messages(record['student_messages'], record['guidance']))
            if len(tokenizer.encode(record['guidance'], add_special_tokens=False)) > 1024: raise ValueError('Guidance budget exceeded')
            budget = min(512, 2048-record['student_state']['generated_tokens'])
            if budget <= 0 or max(len(student_ids), len(teacher_ids))+budget > 8192: raise ValueError('Training context/budget exceeded')
            prepared.append((record, student_ids, teacher_ids, budget))
        torch.manual_seed(args.seed + 1009*(args.step-1))
        status['stage'] = 'loading model'; atomic_json(out/'status.json', status)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_path, dtype=torch.bfloat16,
                    device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True)
        if previous:
            model = PeftModel.from_pretrained(model, str(previous/'adapter'), is_trainable=True)
        else:
            targets = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
                       and 'language_model' in name and not name.endswith('lm_head')]
            if not targets: raise ValueError('No LoRA targets')
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules=targets, bias='none', task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        names = [name for name, _ in named]; parameters = [parameter for _, parameter in named]
        optimizer = torch.optim.AdamW(parameters, lr=5e-6)
        status['optimizer_restored'] = False
        if previous:
            if file_hash(previous/'optimizer.pt') != parent['optimizer_sha256']: raise ValueError('Optimizer checkpoint hash mismatch')
            saved_optimizer = torch.load(previous/'optimizer.pt', map_location='cpu', weights_only=True)
            if saved_optimizer['parameter_names'] != names: raise ValueError('Optimizer parameter order differs')
            if saved_optimizer['step'] != args.step-1: raise ValueError('Optimizer update count differs')
            optimizer.load_state_dict(saved_optimizer['optimizer'])
            status['optimizer_restored'] = True
            del saved_optimizer
        before = {name: parameter.detach().cpu().clone() for name, parameter in named}
        model.zero_grad(set_to_none=True); samples = []
        status.update(trainable_parameters=sum(p.numel() for p in parameters), record_ids=[r['task_id'] for r in records])
        for record, student_prefix, teacher_prefix, budget in prepared:
            status['stage'] = 'fresh student action and gradients: '+record['task_id']; atomic_json(out/'status.json', status)
            model.eval()
            with torch.no_grad():
                prompt = torch.tensor([student_prefix], device='cuda', dtype=torch.long)
                generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=budget,
                        do_sample=True, temperature=.7, top_p=1., top_k=0, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                target = generated[0, len(student_prefix):].tolist()
                del prompt, generated
            student_ids, student_positions = action_inputs(student_prefix, target, 'cuda')
            teacher_ids, teacher_positions = action_inputs(teacher_prefix, target, 'cuda')
            with torch.no_grad():
                teacher = model(input_ids=teacher_ids, attention_mask=torch.ones_like(teacher_ids), use_cache=False,
                                logits_to_keep=teacher_positions).logits.detach()
            model.train()
            student = model(input_ids=student_ids, attention_mask=torch.ones_like(student_ids), use_cache=False,
                            logits_to_keep=student_positions).logits
            loss = jsd_loss(student, teacher)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            (loss/len(records)).backward()
            sample = {'task_id': record['task_id'], 'prefix_tokens': len(student_prefix), 'teacher_prefix_tokens': len(teacher_prefix),
                      'target_tokens': len(target), 'student_action': tokenizer.decode(target, skip_special_tokens=True), 'loss': loss.item()}
            samples.append(sample); atomic_json(out/'samples.json', samples); print(json.dumps(sample), flush=True)
            del loss, student, teacher, student_ids, teacher_ids
            gc.collect(); torch.cuda.empty_cache()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(gradient_norm) or gradient_norm.item() == 0: raise ValueError('Invalid gradient norm')
        optimizer.step(); status['optimizer_updates'] = 1
        changed = sum(not torch.equal(before[name], parameter.detach().cpu()) for name, parameter in named)
        if not changed: raise ValueError('No adapter tensors changed')
        status.update(gradient_norm=gradient_norm.item(), changed_adapter_tensors=changed,
                      mean_jsd=sum(s['loss'] for s in samples)/len(samples), peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        model.save_pretrained(out/'adapter'); tokenizer.save_pretrained(out/'adapter')
        optimizer_path = out/'optimizer.pt.tmp'
        torch.save({'optimizer': optimizer.state_dict(), 'parameter_names': names, 'step': args.step}, optimizer_path)
        optimizer_path.replace(out/'optimizer.pt')
        saved = {name: parameter.detach().cpu().clone() for name, parameter in named}
        model.load_adapter(str(out/'adapter'), adapter_name='reload_check', is_trainable=False)
        model.set_adapter('reload_check')
        loaded = {name.replace('.reload_check.', '.default.'): parameter.detach().cpu()
                  for name, parameter in model.named_parameters() if '.reload_check.' in name}
        if not all(name in loaded and torch.equal(value, loaded[name]) for name, value in saved.items()): raise ValueError('Adapter reload mismatch')
        status.update(reload_equal=True, policy_id=adapter_policy(out/'adapter'), optimizer_sha256=file_hash(out/'optimizer.pt'), complete=True, stage='complete')
    except Exception as error:
        status.update(error=repr(error), stage='failed')
        (out/'traceback.txt').write_text(traceback.format_exc()); print(traceback.format_exc(), flush=True)
    finally:
        status['elapsed_seconds'] = time.time()-start
        atomic_json(out/'status.json', status); print(json.dumps(status), flush=True)
    if not status['complete']: raise SystemExit(1)

if __name__ == '__main__':
    main()
