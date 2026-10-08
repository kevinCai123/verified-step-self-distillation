"""One resumable OPSD update; optimizer state continues across fresh batches.

Changes after round 1: the loss can be restricted to content tokens of the sampled
action (--loss-tokens content), the learning rate is a parameter, a batch may mix
fresh records with records collected under recent checkpoints (--allowed-policies),
and every sample logs how the JSD splits between content and structural tokens plus
the most divergent tokens, so what the update is buying is visible per step.
"""
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
from self_evolve_search.opsd import action_inputs, action_type, content_token_mask, jsd_loss, jsd_per_token, teacher_messages
from self_evolve_search.persistence import BASE_POLICY, adapter_policy, atomic_json, file_hash, validate_training_batch
from self_evolve_search.repair import compact
from self_evolve_search.tokenization import chat_ids

def end_of_turn_ids(tokenizer):
    """Token IDs that end an assistant turn; generation must stop there so a sampled action never runs on into a hallucinated tool result."""
    ids = [tokenizer.convert_tokens_to_ids('<|im_end|>'), tokenizer.eos_token_id]
    return [i for i in dict.fromkeys(ids) if i is not None and i != tokenizer.unk_token_id]

def action_ids(tokenizer, action):
    """Token IDs of an action rendered as the assistant turn: compact JSON followed by the end-of-turn token."""
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if end is None or end == tokenizer.unk_token_id: end = tokenizer.eos_token_id
    return tokenizer.encode(compact(action), add_special_tokens=False) + [end]

def main():
    parser = argparse.ArgumentParser()
    for option in ('records', 'model-path', 'source-policy', 'output'):
        parser.add_argument('--'+option, required=True)
    parser.add_argument('--previous')
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--loss-tokens', choices=('content', 'all'), default='all',
                        help='content: average the JSD over value tokens of the action only; all: every action token (round-1 behaviour)')
    parser.add_argument('--allowed-policies', default='',
                        help='comma-separated policy IDs, besides --source-policy, whose records this batch may contain (replay window)')
    parser.add_argument('--objective', choices=('opsd', 'dpo', 'sft'), default='opsd',
                        help='opsd: action-token JSD against the guidance-primed teacher on a fresh student sample; dpo: preference loss, replacement over original action, reference = base model (adapter disabled); sft: cross-entropy on the replacement action tokens (no teacher pass, no sampling)')
    parser.add_argument('--dpo-beta', type=float, default=0.1)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    update_lock=(out/'update.lock').open('a+')
    fcntl.flock(update_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('complete'):
        raise ValueError('Refuse to overwrite a completed update')
    allowed = {args.source_policy} | {p for p in args.allowed_policies.split(',') if p}
    status = {'complete': False, 'step': args.step, 'source_policy': args.source_policy, 'allowed_policies': sorted(allowed),
              'records_sha256': file_hash(args.records), 'optimizer_updates': 0, 'seed': args.seed,
              'learning_rate': args.learning_rate, 'loss_tokens': args.loss_tokens, 'batch_size': args.batch_size,
              'objective': args.objective, 'dpo_beta': args.dpo_beta}
    atomic_json(out/'status.json', status); start = time.time()
    try:
        records = json.loads(Path(args.records).read_text())
        validate_training_batch(records, args.source_policy, args.batch_size, allowed)
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
            pair = None
            if args.objective == 'dpo':
                if record.get('replacement') is None or record.get('original_action') is None: raise ValueError('DPO needs a replacement and an original action')
                pair = (action_ids(tokenizer, record['replacement']), action_ids(tokenizer, record['original_action']))
                if len(student_ids)+max(map(len, pair)) > 8192: raise ValueError('Training context exceeded')
            elif args.objective == 'sft':
                if record.get('replacement') is None: raise ValueError('SFT needs a replacement action')
                pair = (action_ids(tokenizer, record['replacement']),)
                if len(student_ids)+len(pair[0]) > 8192: raise ValueError('Training context exceeded')
            prepared.append((record, student_ids, teacher_ids, budget, pair))
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
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
        status['optimizer_restored'] = False
        if previous:
            if file_hash(previous/'optimizer.pt') != parent['optimizer_sha256']: raise ValueError('Optimizer checkpoint hash mismatch')
            saved_optimizer = torch.load(previous/'optimizer.pt', map_location='cpu', weights_only=True)
            if saved_optimizer['parameter_names'] != names: raise ValueError('Optimizer parameter order differs')
            if saved_optimizer['step'] != args.step-1: raise ValueError('Optimizer update count differs')
            optimizer.load_state_dict(saved_optimizer['optimizer'])
            for group in optimizer.param_groups: group['lr'] = args.learning_rate
            status['optimizer_restored'] = True
            del saved_optimizer
        before = {name: parameter.detach().cpu().clone() for name, parameter in named}
        model.zero_grad(set_to_none=True); samples = []
        status.update(trainable_parameters=sum(p.numel() for p in parameters), record_ids=[r['task_id'] for r in records])
        for record, student_prefix, teacher_prefix, budget, pair in prepared:
            status['stage'] = 'fresh student action and gradients: '+record['task_id']; atomic_json(out/'status.json', status)
            if args.objective == 'dpo':
                chosen_ids, rejected_ids = pair
                def sequence_logprob(target):
                    ids, positions = action_inputs(student_prefix, target, 'cuda')
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=positions).logits
                    return torch.log_softmax(logits.float(), -1)[0].gather(-1, torch.tensor(target, device='cuda')[:, None]).squeeze(-1).sum()
                model.eval()
                with torch.no_grad(), model.disable_adapter():
                    reference_chosen = sequence_logprob(chosen_ids).item(); reference_rejected = sequence_logprob(rejected_ids).item()
                model.train()
                policy_chosen = sequence_logprob(chosen_ids); policy_rejected = sequence_logprob(rejected_ids)
                margin = (policy_chosen-reference_chosen)-(policy_rejected-reference_rejected)
                loss = -torch.nn.functional.logsigmoid(args.dpo_beta*margin)
                if not torch.isfinite(loss): raise ValueError('Non-finite loss')
                (loss/len(records)).backward()
                sample = {'task_id': record['task_id'], 'prefix_tokens': len(student_prefix), 'chosen_tokens': len(chosen_ids), 'rejected_tokens': len(rejected_ids),
                          'chosen': compact(record['replacement']), 'rejected': compact(record['original_action']), 'loss': loss.item(),
                          'policy_chosen_logp': policy_chosen.item(), 'policy_rejected_logp': policy_rejected.item(),
                          'reference_chosen_logp': reference_chosen, 'reference_rejected_logp': reference_rejected, 'margin': margin.item()}
                samples.append(sample); atomic_json(out/'samples.json', samples); print(json.dumps(sample), flush=True)
                del loss, policy_chosen, policy_rejected, margin
                gc.collect(); torch.cuda.empty_cache()
                continue
            if args.objective == 'sft':
                target = pair[0]
                content = content_token_mask(tokenizer, target)
                weights = torch.tensor([float(flag) for flag in content], device='cuda') if args.loss_tokens == 'content' and any(content) else torch.ones(len(target), device='cuda')
                ids, positions = action_inputs(student_prefix, target, 'cuda')
                model.train()
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=positions).logits
                token_logp = torch.log_softmax(logits.float(), -1)[0].gather(-1, torch.tensor(target, device='cuda')[:, None]).squeeze(-1)
                loss = -(token_logp*weights).sum()/weights.sum()
                if not torch.isfinite(loss): raise ValueError('Non-finite loss')
                (loss/len(records)).backward()
                nll = (-token_logp).detach().tolist()
                sample = {'task_id': record['task_id'], 'prefix_tokens': len(student_prefix), 'target_tokens': len(target),
                          'target': compact(record['replacement']), 'replacement_type': (record.get('replacement') or {}).get('action'),
                          'loss': loss.item(), 'nll_all': sum(nll)/len(nll),
                          'nll_content': (sum(v for v, f in zip(nll, content) if f)/sum(content)) if any(content) else None,
                          'content_tokens': sum(content), 'structural_tokens': len(content)-sum(content)}
                samples.append(sample); atomic_json(out/'samples.json', samples); print(json.dumps(sample), flush=True)
                del loss, logits, token_logp, ids
                gc.collect(); torch.cuda.empty_cache()
                continue
            model.eval()
            with torch.no_grad():
                prompt = torch.tensor([student_prefix], device='cuda', dtype=torch.long)
                generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=budget,
                        do_sample=True, temperature=.7, top_p=1., top_k=0, eos_token_id=end_of_turn_ids(tokenizer), use_cache=True,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                target = generated[0, len(student_prefix):].tolist()
                del prompt, generated
            content = content_token_mask(tokenizer, target)
            weights = [float(flag) for flag in content] if args.loss_tokens == 'content' else None
            student_ids, student_positions = action_inputs(student_prefix, target, 'cuda')
            teacher_ids, teacher_positions = action_inputs(teacher_prefix, target, 'cuda')
            with torch.no_grad():
                teacher = model(input_ids=teacher_ids, attention_mask=torch.ones_like(teacher_ids), use_cache=False,
                                logits_to_keep=teacher_positions).logits.detach()
            model.train()
            student = model(input_ids=student_ids, attention_mask=torch.ones_like(student_ids), use_cache=False,
                            logits_to_keep=student_positions).logits
            loss = jsd_loss(student, teacher, weights)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            (loss/len(records)).backward()
            with torch.no_grad():
                per_token = jsd_per_token(student, teacher).reshape(-1).tolist()
            content_values = [v for v, flag in zip(per_token, content) if flag]
            structural_values = [v for v, flag in zip(per_token, content) if not flag]
            ranked = sorted(range(len(per_token)), key=lambda i: -per_token[i])[:5]
            decoded = tokenizer.decode(target, skip_special_tokens=True)
            sample = {'task_id': record['task_id'], 'prefix_tokens': len(student_prefix), 'teacher_prefix_tokens': len(teacher_prefix),
                      'target_tokens': len(target), 'student_action': decoded, 'student_action_type': action_type(decoded),
                      'replacement_type': (record.get('replacement') or {}).get('action'), 'loss': loss.item(),
                      'content_tokens': len(content_values), 'structural_tokens': len(structural_values),
                      'jsd_all': sum(per_token)/len(per_token), 'jsd_content': sum(content_values)/len(content_values) if content_values else None,
                      'jsd_structural': sum(structural_values)/len(structural_values) if structural_values else None,
                      'structural_share_of_jsd': sum(structural_values)/sum(per_token) if sum(per_token) > 0 else None,
                      'top_tokens': [{'position': i, 'token': tokenizer.decode([target[i]]), 'jsd': per_token[i], 'content': content[i]} for i in ranked]}
            samples.append(sample); atomic_json(out/'samples.json', samples); print(json.dumps(sample), flush=True)
            del loss, student, teacher, student_ids, teacher_ids
            gc.collect(); torch.cuda.empty_cache()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(gradient_norm) or gradient_norm.item() == 0: raise ValueError('Invalid gradient norm')
        optimizer.step(); status['optimizer_updates'] = 1
        changed = sum(not torch.equal(before[name], parameter.detach().cpu()) for name, parameter in named)
        if not changed: raise ValueError('No adapter tensors changed')
        def mean(key):
            values = [s[key] for s in samples if s.get(key) is not None]
            return sum(values)/len(values) if values else None
        status.update(gradient_norm=gradient_norm.item(), changed_adapter_tensors=changed,
                      mean_loss=sum(s['loss'] for s in samples)/len(samples), peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        if args.objective == 'dpo':
            status.update(mean_margin=mean('margin'), reward_accuracy=sum(s['margin'] > 0 for s in samples)/len(samples))
        elif args.objective == 'sft':
            types = [s.get('replacement_type') for s in samples]
            status['replacement_types'] = {str(t): types.count(t) for t in sorted(set(types), key=str)}
            status.update(mean_nll_all=mean('nll_all'), mean_nll_content=mean('nll_content'))
        else:
            types = [s.get('student_action_type') for s in samples]
            status['sample_action_types'] = {str(t): types.count(t) for t in sorted(set(types), key=str)}
            status['sample_type_matches_replacement'] = sum(s.get('student_action_type') == s.get('replacement_type') for s in samples if s.get('replacement_type'))
            status.update(mean_jsd=status['mean_loss'], mean_jsd_all=mean('jsd_all'), mean_jsd_content=mean('jsd_content'),
                          mean_jsd_structural=mean('jsd_structural'), mean_structural_share_of_jsd=mean('structural_share_of_jsd'))
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
