"""Can the OPSD objective move the student toward verified corrections at all?

Trains a fresh LoRA on already-collected records for a fixed number of optimizer
steps and measures, on those same prefixes, whether the greedy student action moves
toward the verified replacement. This is deliberately a *memorization* test: if the
student cannot reproduce corrections on the training prefixes, no held-out gain is
possible and the loss / learning rate must be fixed before any new collection.

Runs on the GPU inside WSL with the training environment; the inference server must be stopped:

    source scripts/runtime_env.sh
    HF_HUB_OFFLINE=1 .venv-train/bin/python scripts/memorization_check.py --model-path <pinned model> \
        --output runs/memorization-check --steps 30 --learning-rate 2e-5 --loss-tokens content

Records default to every batch of runs/round1-seed42; their guidance is rewritten into the
model's compact JSON dialect so that the check reflects the fixed protocol.
"""
import argparse, gc, json, random, re, time, traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from self_evolve_search.opsd import action_inputs, content_token_mask, jsd_loss, jsd_per_token, teacher_messages
from self_evolve_search.persistence import atomic_json
from self_evolve_search.repair import compact, information_leak
from self_evolve_search.tokenization import chat_ids

def end_of_turn_ids(tokenizer):
    """Token IDs that end an assistant turn; generation must stop there so a sampled action never runs on into a hallucinated tool result."""
    ids = [tokenizer.convert_tokens_to_ids('<|im_end|>'), tokenizer.eos_token_id]
    return [i for i in dict.fromkeys(ids) if i is not None and i != tokenizer.unk_token_id]

ROOT = Path(__file__).resolve().parents[1]
STOP = set('a an the is was were are in on at of to for with and or which what who when where how did does do by from as it its this that has have had'.split())

def compact_guidance(record, mode='correction'):
    """Rebuild a record's guidance in the model's compact JSON dialect.
    mode: 'correction' (the correction alone; the fixed protocol), 'full' (correction, diagnosis and
    evidence, as round 1 wrote it, compacted), 'evidence' (evidence alone, as arm B sees it)."""
    lines = record['guidance'].split('\n'); out = []
    for line in lines:
        if line.startswith('Verified correction for this decision: '):
            out.append('Verified correction for this decision: ' + compact(record['replacement'])); continue
        try: out.append(compact(json.loads(line)))
        except ValueError: out.append(line)
    if mode == 'correction': return 'Verified correction for this decision: ' + compact(record['replacement']) + '\n'
    if mode == 'evidence':
        start = next((i for i, line in enumerate(out) if line.startswith('Retrieved evidence:')), 0)
        return '\n'.join(out[start:])
    return '\n'.join(out)

def load_records(path):
    if path:
        text = Path(path).read_text()
        return json.loads(text) if path.endswith('.json') else [json.loads(l) for l in text.splitlines() if l.strip()]
    records = []
    for batch in sorted((ROOT / 'runs/round1-seed42/batches').glob('step-*/records.json')):
        records += json.loads(batch.read_text())
    return records

def terms(action):
    if not isinstance(action, dict): return set()
    text = action.get('query') if action.get('action') == 'search' else action.get('answer', '') if action.get('action') == 'finish' else str(action.get('doc_id', ''))
    return {t for t in re.findall(r'\w+', str(text or '').casefold()) if t not in STOP}

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.

def canonical(action):
    try: return json.dumps(action, sort_keys=True, separators=(',', ':'))
    except TypeError: return None

def greedy_actions(model, tokenizer, prepared):
    model.eval(); actions = []
    for record, prefix, _, _ in prepared:
        with torch.no_grad():
            prompt = torch.tensor([prefix], device='cuda', dtype=torch.long)
            generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=96, do_sample=False, eos_token_id=end_of_turn_ids(tokenizer),
                                       pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        text = tokenizer.decode(generated[0, len(prefix):].tolist(), skip_special_tokens=True)
        try: action = json.loads(text)
        except ValueError: action = None
        actions.append({'task_id': record['task_id'], 'text': text, 'action': action}); del prompt, generated
    return actions

def score(actions, prepared):
    rows = []
    for out, (record, *_ ) in zip(actions, prepared):
        replacement, original = record['replacement'], record['original_action']
        rows.append({'task_id': record['task_id'], 'text': out['text'][:300], 'action_type': (out['action'] or {}).get('action') if isinstance(out['action'], dict) else None,
                     'replacement_type': replacement.get('action'), 'exact_replacement': canonical(out['action']) == canonical(replacement),
                     'exact_original': canonical(out['action']) == canonical(original),
                     'same_type_as_replacement': isinstance(out['action'], dict) and out['action'].get('action') == replacement.get('action'),
                     'jaccard_replacement': jaccard(terms(out['action']), terms(replacement)),
                     'jaccard_original': jaccard(terms(out['action']), terms(original)), 'valid_json': out['action'] is not None})
    n = len(rows)
    types = [r['action_type'] for r in rows]
    return {'rows': rows, 'action_types': {str(t): types.count(t) for t in sorted(set(types), key=str)},
            'exact_replacement_rate': sum(r['exact_replacement'] for r in rows) / n,
            'exact_original_rate': sum(r['exact_original'] for r in rows) / n,
            'same_type_rate': sum(r['same_type_as_replacement'] for r in rows) / n,
            'mean_jaccard_replacement': sum(r['jaccard_replacement'] for r in rows) / n,
            'mean_jaccard_original': sum(r['jaccard_original'] for r in rows) / n,
            'valid_json_rate': sum(r['valid_json'] for r in rows) / n}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True); p.add_argument('--output', required=True); p.add_argument('--records')
    p.add_argument('--steps', type=int, default=30); p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-5); p.add_argument('--loss-tokens', choices=('content', 'all'), default='content')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--limit', type=int)
    p.add_argument('--only-informationally-legal', action='store_true', help='train only on records whose replacement passes information_leak()')
    p.add_argument('--guidance', choices=('correction', 'full', 'evidence'), default='correction', help='what the teacher sees (see compact_guidance)')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    status = {'kind': 'memorization_check', 'complete': False, 'args': vars(a)}; atomic_json(out / 'status.json', status); start = time.time()
    try:
        records = load_records(a.records)
        if a.only_informationally_legal: records = [r for r in records if not information_leak(r['replacement'], r['student_state'])]
        if a.limit: records = records[:a.limit]
        if not records: raise ValueError('No records')
        tokenizer = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
        prepared = []
        for record in records:
            record = dict(record, guidance=compact_guidance(record, a.guidance))
            sp = chat_ids(tokenizer, record['student_messages']); tp = chat_ids(tokenizer, teacher_messages(record['student_messages'], record['guidance']))
            budget = min(512, 2048 - record['student_state']['generated_tokens'])
            if budget <= 0 or max(len(sp), len(tp)) + budget > 8192: continue
            prepared.append((record, sp, tp, budget))
        status.update(records=len(prepared), leaky_records=sum(bool(information_leak(r['replacement'], r['student_state'])) for r, *_ in prepared))
        torch.manual_seed(a.seed); rng = random.Random(a.seed)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model_path, dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True)
        targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'language_model' in n and not n.endswith('lm_head')]
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules=targets, bias='none', task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.enable_input_require_grads()
        parameters = [q for q in model.parameters() if q.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=a.learning_rate)
        status['stage'] = 'greedy actions before training'; atomic_json(out / 'status.json', status)
        before = score(greedy_actions(model, tokenizer, prepared), prepared)
        atomic_json(out / 'before.json', before); status['before'] = {k: v for k, v in before.items() if k != 'rows'}
        order = list(range(len(prepared))); losses = []
        for step in range(1, a.steps + 1):
            if not order or len(order) < a.batch_size: order = list(range(len(prepared))); rng.shuffle(order)
            batch = [prepared[order.pop()] for _ in range(a.batch_size)]
            model.zero_grad(set_to_none=True); step_losses = []; structural_share = []
            for record, sp, tp, budget in batch:
                model.eval()
                with torch.no_grad():
                    prompt = torch.tensor([sp], device='cuda', dtype=torch.long)
                    generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=budget, do_sample=True, temperature=.7, top_p=1., top_k=0, eos_token_id=end_of_turn_ids(tokenizer),
                                               pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                    target = generated[0, len(sp):].tolist(); del prompt, generated
                content = content_token_mask(tokenizer, target)
                weights = [float(f) for f in content] if a.loss_tokens == 'content' else None
                sids, spos = action_inputs(sp, target, 'cuda'); tids, tpos = action_inputs(tp, target, 'cuda')
                with torch.no_grad():
                    teacher = model(input_ids=tids, attention_mask=torch.ones_like(tids), use_cache=False, logits_to_keep=tpos).logits.detach()
                model.train()
                student = model(input_ids=sids, attention_mask=torch.ones_like(sids), use_cache=False, logits_to_keep=spos).logits
                loss = jsd_loss(student, teacher, weights)
                if not torch.isfinite(loss): raise ValueError('Non-finite loss')
                (loss / a.batch_size).backward(); step_losses.append(loss.item())
                with torch.no_grad():
                    per = jsd_per_token(student, teacher).reshape(-1).tolist()
                total = sum(per); structural_share.append(sum(v for v, f in zip(per, content) if not f) / total if total > 0 else 0.)
                del loss, student, teacher, sids, tids; gc.collect(); torch.cuda.empty_cache()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
            losses.append({'step': step, 'loss': sum(step_losses) / len(step_losses), 'gradient_norm': norm.item(), 'structural_share_of_jsd': sum(structural_share) / len(structural_share)})
            status.update(stage=f'step {step}/{a.steps}', losses=losses); atomic_json(out / 'status.json', status); print(json.dumps(losses[-1]), flush=True)
        status['stage'] = 'greedy actions after training'; atomic_json(out / 'status.json', status)
        after = score(greedy_actions(model, tokenizer, prepared), prepared)
        atomic_json(out / 'after.json', after); status['after'] = {k: v for k, v in after.items() if k != 'rows'}
        moved = [(b['jaccard_replacement'], x['jaccard_replacement']) for b, x in zip(before['rows'], after['rows'])]
        status['moved_toward_replacement'] = sum(x > b for b, x in moved) / len(moved)
        status['moved_away_from_replacement'] = sum(x < b for b, x in moved) / len(moved)
        status['verdict'] = ('learns: greedy actions moved toward the verified corrections on the training prefixes' if status['after']['mean_jaccard_replacement'] > status['before']['mean_jaccard_replacement'] + .05
                             else 'does not learn: the objective/learning rate cannot even memorize the corrections; fix this before collecting more data')
        status['complete'] = True
    except Exception as error:
        status['error'] = repr(error); (out / 'traceback.txt').write_text(traceback.format_exc()); print(traceback.format_exc(), flush=True)
    finally:
        status['elapsed_seconds'] = time.time() - start; atomic_json(out / 'status.json', status); print(json.dumps({k: v for k, v in status.items() if k != 'losses'}, indent=2), flush=True)
    if not status['complete']: raise SystemExit(1)

if __name__ == '__main__': main()
