"""One real OPSD update on verified records; explicitly not a full training run.

Fresh action samples come from the current model. The frozen records were
collected at its initial checkpoint; they must not be recycled as on-policy
repair records after this update. Main training must refresh repair collection.
"""
import argparse, gc, hashlib, json, time, traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from self_evolve_search.opsd import action_inputs, jsd_loss, teacher_messages
from self_evolve_search.tokenization import chat_ids

def main():
    p=argparse.ArgumentParser(); p.add_argument('--records',required=True); p.add_argument('--model-path',required=True); p.add_argument('--output',required=True); p.add_argument('--batch-size',type=int,default=8); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    status={'kind':'one_update_compatibility_test','requested_batch':a.batch_size,'complete':False,'optimizer_updates':0}
    (out/'status.json').write_text(json.dumps(status,indent=2))
    start=time.time()
    try:
        records=[json.loads(x) for x in Path(a.records).read_text().splitlines() if x.strip()]
        if not records: raise ValueError('No verified correction records; refuse invented supervision')
        tokenizer=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
        records=sorted(records,key=lambda r:len(chat_ids(tokenizer,r['student_messages'])))[:a.batch_size]
        status['actual_batch']=len(records); status['record_ids']=[r['task_id'] for r in records]
        status['records_sha256']=hashlib.sha256(Path(a.records).read_bytes()).hexdigest()
        prepared=[]
        for record in records:
            sp=chat_ids(tokenizer,record['student_messages'])
            tp=chat_ids(tokenizer,teacher_messages(record['student_messages'],record['guidance']))
            if max(len(sp),len(tp))+512>8192: raise ValueError('Scoring context exceeds model budget')
            prepared.append((record,sp,tp))
        status['stage']='loading model'
        (out/'status.json').write_text(json.dumps(status,indent=2))
        torch.manual_seed(42)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model_path,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
        targets=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear) and 'language_model' in name and not name.endswith('lm_head')]
        if not targets: raise RuntimeError('No text linear modules found for LoRA')
        (out/'lora_targets.json').write_text(json.dumps(targets,indent=2))
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        parameters=[p for p in model.parameters() if p.requires_grad]
        status['trainable_parameters']=sum(p.numel() for p in parameters)
        optimizer=torch.optim.AdamW(parameters,lr=5e-6)
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        losses=[]; samples=[]
        for record,sp,tp in prepared:
            status['stage']='sampling and scoring '+record['task_id']
            (out/'status.json').write_text(json.dumps(status,indent=2))
            model.eval()
            action_budget=min(512,2048-record['student_state']['generated_tokens'])
            if action_budget<=0: raise ValueError('No original student generation budget remains')
            with torch.no_grad():
                generated=model.generate(input_ids=torch.tensor([sp],device='cuda'),attention_mask=torch.ones((1,len(sp)),device='cuda',dtype=torch.long),max_new_tokens=action_budget,do_sample=True,temperature=.7,top_p=1.,top_k=0,use_cache=True,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            target=generated[0,len(sp):].tolist(); del generated
            sids,spos=action_inputs(sp,target,'cuda'); tids,tpos=action_inputs(tp,target,'cuda')
            with torch.no_grad():
                teacher=model(input_ids=tids,attention_mask=torch.ones_like(tids),use_cache=False,logits_to_keep=tpos).logits.detach()
            model.train()
            student=model(input_ids=sids,attention_mask=torch.ones_like(sids),use_cache=False,logits_to_keep=spos).logits
            loss=jsd_loss(student,teacher)
            if not torch.isfinite(loss): raise ValueError('Non-finite OPSD loss')
            (loss/len(records)).backward()
            losses.append(loss.item()); samples.append({'task_id':record['task_id'],'prefix_tokens':len(sp),'teacher_prefix_tokens':len(tp),'target_tokens':len(target),'student_action':tokenizer.decode(target,skip_special_tokens=True),'loss':loss.item()})
            print(json.dumps(samples[-1]),flush=True)
            del student,teacher,sids,tids,loss; gc.collect(); torch.cuda.empty_cache()
        grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(grad) or grad.item()==0: raise ValueError('Invalid/zero gradient norm')
        optimizer.step(); status['optimizer_updates']=1
        changed=sum(not torch.equal(before[n],p.detach().cpu()) for n,p in model.named_parameters() if n in before)
        if not changed: raise ValueError('Optimizer did not change any adapter tensors')
        status.update(losses=losses,gradient_norm=grad.item(),changed_adapter_tensors=changed,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        model.save_pretrained(out/'adapter'); tokenizer.save_pretrained(out/'adapter')
        (out/'samples.json').write_text(json.dumps(samples,indent=2))
        # Reload saved weights into the same base model and compare every adapter tensor exactly.
        saved={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        model.load_adapter(str(out/'adapter'),adapter_name='reload_check',is_trainable=False)
        model.set_adapter('reload_check')
        loaded={n.replace('.reload_check.','.default.'):p.detach().cpu() for n,p in model.named_parameters() if '.reload_check.' in n}
        status['reload_equal']=all(n in loaded and torch.equal(v,loaded[n]) for n,v in saved.items())
        if not status['reload_equal']: raise ValueError('Saved adapter reload mismatch')
        status['complete']=True
        status['stage']='complete'
    except Exception as e:
        status['error']=repr(e); (out/'traceback.txt').write_text(traceback.format_exc()); print(traceback.format_exc(),flush=True)
    finally:
        status['elapsed_seconds']=time.time()-start
        (out/'status.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2),flush=True)
    if not status['complete']: raise SystemExit(1)

if __name__=='__main__': main()
