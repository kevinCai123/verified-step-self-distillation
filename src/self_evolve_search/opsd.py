"""Action-only full-vocabulary JSD, with detached same-model teacher targets.

Round 1 averaged the JSD over every token of the sampled action. Most of those
tokens are JSON scaffolding ({, "action", :, ", ...) on which the student and the
guidance-primed teacher can disagree for reasons unrelated to the search decision
(the guidance dialect used ': ' separators; the model emits ':'). The helpers
below label which characters of an action carry a *value* (the action type, the
query text, a document ID, the answer, citation titles/indices) so that the loss
can be restricted to those tokens and the split can be logged. torch is imported lazily so the
pure-Python helpers (teacher_messages, the masks) work in the data environment too.
"""
import copy, math

def action_inputs(prefix_ids, target_ids, device='cpu'):
    import torch
    if not prefix_ids or not target_ids: raise ValueError('Empty prefix/target')
    ids=torch.tensor([list(prefix_ids)+list(target_ids)],dtype=torch.long,device=device)
    # Causal logits at prefix[-1] predict target[0]. Excludes every prompt/tool token.
    positions=torch.arange(len(prefix_ids)-1,len(prefix_ids)+len(target_ids)-1,device=device)
    return ids,positions

def action_type(text):
    """Action type of a generated action text, or None when it is not a JSON object with an action."""
    import json
    try: action=json.loads(text)
    except (TypeError, ValueError): return None
    return action.get('action') if isinstance(action,dict) else None

def premature_finish_share(rows, outputs):
    """Share of probed training states on which a policy answers with `finish` although the verified
    correction there is not a finish (or the student had read nothing yet). The drift tripwire of
    scripts/check_adapter_serving.py; see private_guidance() in repair.py for the failure it guards."""
    eligible=[(row,out) for row,out in zip(rows,outputs)
              if (row.get('replacement') or {}).get('action')!='finish' or not row.get('student_state',{}).get('read_docs')]
    if not eligible: return 0.
    return sum(action_type(out)=='finish' for _,out in eligible)/len(eligible)

def illegal_read_share(rows, outputs):
    """Share of probed training states on which a policy answers with a `read` of a document id that the
    student had not retrieved at that state (the agent would raise PermissionError). The second drift
    tripwire of scripts/check_adapter_serving.py, added after cycle 2: arm D seed 7 collapsed from update 25
    into reading a fabricated `doc_1` on every question, which the premature-finish share does not see."""
    import json
    if not rows: return 0.
    illegal=0
    for row,out in zip(rows,outputs):
        try: action=json.loads(out)
        except (TypeError, ValueError): continue
        if not isinstance(action,dict) or action.get('action')!='read': continue
        retrieved=[str(x) for x in (row.get('student_state') or {}).get('retrieved') or []]
        if str(action.get('doc_id')) not in retrieved: illegal+=1
    return illegal/len(rows)

def teacher_messages(student_messages,guidance):
    messages=copy.deepcopy(student_messages)
    correction=guidance.startswith('Verified correction'); evidence='Retrieved evidence:' in guidance
    what=('the verified correction and retrieved evidence' if correction and evidence else 'the verified correction' if correction else 'the retrieved evidence')
    instruction=(f'\nPRIVATE TRAINING GUIDANCE (not visible to the student): Use {what} below to assess the student\'s NEXT action. '
        'The action must obey the STUDENT permissions and currently retrieved IDs. Do not finish using facts that the student has not read.\n')
    messages[0]['content']+=instruction+guidance
    return messages

def value_character_mask(text):
    """One flag per character: True where it belongs to a JSON value (string contents, numbers,
    literals), False for structure (braces, brackets, commas, colons, quotes, key names and
    whitespace outside strings) and for any text after the top-level object closes. Tolerant of
    truncated or slightly malformed output."""
    mask=[False]*len(text); stack=[]; expect_key=False; i=0; n=len(text); opened=False
    while i<n:
        c=text[i]
        if opened and not stack: break          # the top-level object has closed: anything after it (run-on text) is not part of the action
        if c=='"':
            is_key=bool(stack) and stack[-1]=='{' and expect_key
            j=i+1
            while j<n and text[j]!='"':
                if text[j]=='\\': j+=1
                j+=1
            if not is_key:
                for k in range(i+1,min(j,n)): mask[k]=True
            if is_key: expect_key=False
            i=j+1; continue
        if c=='{': stack.append('{'); expect_key=True; opened=True
        elif c=='[': stack.append('[')
        elif c in '}]':
            if stack: stack.pop()
            expect_key=False
        elif c==',':
            if stack and stack[-1]=='{': expect_key=True
        elif c==':' or c.isspace(): pass
        else:
            j=i
            while j<n and text[j] not in '{}[],:"' and not text[j].isspace(): j+=1
            for k in range(i,j): mask[k]=True
            i=j; continue
        i+=1
    return mask

def token_spans(tokenizer,target_ids):
    """Character span of each target token inside the decoded action text."""
    spans=[]; previous=0
    for i in range(len(target_ids)):
        text=tokenizer.decode(target_ids[:i+1],skip_special_tokens=True)
        end=max(previous,len(text)); spans.append((previous,end)); previous=end
    return spans

def content_token_mask(tokenizer,target_ids):
    """True for every target token that renders at least one value character of the action."""
    text=tokenizer.decode(target_ids,skip_special_tokens=True)
    chars=value_character_mask(text)
    return [any(chars[k] for k in range(start,min(end,len(chars)))) for start,end in token_spans(tokenizer,target_ids)]

def jsd_per_token(student_logits,teacher_logits):
    import torch
    if student_logits.shape!=teacher_logits.shape: raise ValueError('Student/teacher action logits are not aligned')
    student=torch.log_softmax(student_logits.float(),dim=-1)
    teacher=torch.log_softmax(teacher_logits.detach().float(),dim=-1)
    mixture=torch.logaddexp(student,teacher)-math.log(2)
    return .5*((student.exp()*(student-mixture)).sum(-1)+(teacher.exp()*(teacher-mixture)).sum(-1))

def jsd_loss(student_logits,teacher_logits,weights=None):
    """Mean generalized JSD (beta 0.5) over action tokens; `weights` (same token layout as the
    logits' sequence axis) restricts the mean, e.g. to content tokens. Falls back to all tokens
    when no token carries weight, so a malformed sample never produces an empty loss."""
    import torch
    per_token=jsd_per_token(student_logits,teacher_logits)
    if weights is None: return per_token.mean()
    weights=torch.as_tensor(weights,dtype=per_token.dtype,device=per_token.device).reshape(per_token.shape)
    if weights.sum()<=0: return per_token.mean()
    return (per_token*weights).sum()/weights.sum()
