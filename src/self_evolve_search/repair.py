import hashlib, json, random, re
from .agent import SCHEMA, run, validate_action
from .metrics import normalize
from .persistence import BASE_POLICY

# The model natively emits compact JSON (see SCHEMA). Guidance must use the same
# dialect: round 1 used json.dumps defaults, whose ': ' and ', ' separators made
# the teacher disagree with the student at every structural token, so most of
# the JSD gradient taught whitespace instead of search decisions.
COMPACT = {'ensure_ascii': False, 'separators': (',', ':')}
DIAGNOSIS_MAX_TOKENS = 1024   # 512 truncated 360 of 439 failed diagnoses in round 1
DIAGNOSIS_MAX_PROMPT = 8192 - DIAGNOSIS_MAX_TOKENS
STOP = set('a an the is was were are in on at of to for with and or which what who when where how did does do by from as it its this that has have had'.split())

def compact(value):
    return json.dumps(value, **COMPACT)

def visible_terms(state):
    """Casefolded terms the restricted student has seen: question, its own actions and tool results (not the system prompt)."""
    terms = set()
    for message in state['messages'][1:]:
        terms |= set(re.findall(r'\w+', message['content'].casefold()))
    return terms

def novel_terms(candidate, state):
    """Every content term of a replacement search query absent from the student's context (strict audit view)."""
    if candidate.get('action') != 'search': return []
    visible = visible_terms(state)
    return [t for t in re.findall(r'\w+', str(candidate.get('query', ''))) if t.casefold() not in STOP and t.casefold() not in visible]

def information_leak(candidate, state):
    """Terms a replacement introduces that the restricted student could not know at this step.

    Search: capitalised tokens and numbers (names, titles, dates) must already appear in the
    question, the student's own earlier actions or its tool results. Generic lower-case words
    are allowed so that rephrasing ('director filmography') stays legal. Finish: the answer
    must be a span of sentences the student has read. Read is covered by validate_action.
    Round-1 corrections violated this in 80% of cases (25% contained the gold answer), so the
    student was being trained to emit facts it had no way to derive.
    """
    kind = candidate.get('action')
    if kind == 'search':
        visible = visible_terms(state)
        leaked = []
        for token in re.findall(r'\w+', str(candidate.get('query', ''))):
            if len(token) <= 2 or token.casefold() in visible: continue
            if token[0].isupper() or token.isdigit(): leaked.append(token)
        return leaked
    if kind == 'finish':
        answer = normalize(str(candidate.get('answer', '')))
        if not answer or answer in ('yes', 'no', 'noanswer'): return []
        text = normalize(' '.join(sentence for doc in state.get('read_docs', []) for _, sentence in doc['sentences']))
        return [] if answer in text else [str(candidate['answer'])]
    return []

def private_guidance(client,teacher,replacement=None,reason='',include_evidence=None):
    """Teacher-only context within 1,024 tokens.

    For a verified correction the guidance is the correction alone. The 21 September memorization
    check showed why: with the teacher's retrieved evidence (which contains the answer) in its
    context, the same-model teacher preferred to *finish*, and 30 OPSD updates at 2e-5 pulled the
    student from 88% to 9% agreement with the correction's action type, i.e. towards answering
    without evidence. The diagnosis text is kept in the record for audit, not shown to the teacher.
    With replacement=None (evidence-only control, arm B) the guidance carries the evidence alone,
    which is exactly the failure mode that arm measures.
    """
    if include_evidence is None: include_evidence=replacement is None
    prefix=('Verified correction for this decision: '+compact(replacement)+'\n' if replacement is not None else '')
    if include_evidence: prefix+='Retrieved evidence:\n'
    if client.tokens(prefix)>1024: raise ValueError('Correction exceeds guidance budget')
    text=prefix
    if include_evidence:
        for doc in teacher['state']['read_docs']:
            for i,sentence in doc['sentences']:
                line=compact([doc['title'],i,sentence])+'\n'
                if client.tokens(text+line)<=1024: text+=line
    return text

def random_steps(student,seed=42,count=3):
    """Seeded choice of up to `count` distinct replayable steps (control for the diagnostic ranking)."""
    candidates=[i for i,s in enumerate(student['trace']) if s.get('action') is not None]
    rng=random.Random(int(hashlib.sha256(f"{seed}:{student['task_id']}".encode()).hexdigest(),16))
    return sorted(rng.sample(candidates,min(count,len(candidates))))

def diagnose(student,teacher,client,forced_steps=None):
    steps=[{'step':i,'action':s.get('action'),'error':s.get('error'),'observation':s.get('observation')} for i,s in enumerate(student['trace'])]
    evidence=teacher['state']['read_docs']
    selection=('Rank up to THREE consequential steps and propose up to TWO replacements per step. ' if forced_steps is None else
        f'Propose up to TWO replacements for EACH of these step indices and no others, in this order: {list(forced_steps)}. ')
    prompt=('You are reviewing your own failed local-search trajectory. '+selection+
        'Use the original step indices. Each replacement must be legal for the restricted student: top-2 search; read only IDs already retrieved BEFORE that step; same remaining budget. '
        'A replacement search query may use only names, titles, numbers and words that already appear in the question or in the student\'s results before that step; '
        'never introduce an entity the student has not seen, even if your private evidence names it. Prefer strategy fixes: shorter entity-focused queries, reading a retrieved result instead of searching again, following a name that appears in a snippet. '
        'A final answer is allowed only after the student has READ its supporting sentences. Prefer fixing search/read decisions when evidence is missing. '
        'Your private evidence is not copied into the student. '+SCHEMA+
        ' Return {"steps":[{"step":0,"reason":"brief reason","actions":[{"action":"search","query":"..."}]}]}. Keep each reason under 30 words. No reference answer is provided.')
    def content(evidence):
        return compact({'question':student['question'],'failed_steps':steps,'private_retrieved_evidence':evidence})
    messages=[{'role':'system','content':prompt},{'role':'user','content':content(evidence)}]
    # Preserve full failed observations; reduce teacher-only evidence if needed, without consulting labels.
    while client.prompt_tokens(messages)>DIAGNOSIS_MAX_PROMPT and evidence:
        evidence=evidence[:-1]
        messages[-1]['content']=content(evidence)
    reply=client.chat(messages,seed=200,temperature=0,max_tokens=DIAGNOSIS_MAX_TOKENS,max_prompt=DIAGNOSIS_MAX_PROMPT)
    parsed=json.loads(reply['content'])
    # The diagnosis is model output: tolerate a bare list, a non-object reply, and non-object entries in
    # "steps" (a bare string entry crashed arm A at update 21 on 22 September) instead of aborting the batch.
    if isinstance(parsed,list): parsed={'steps':parsed}
    if not isinstance(parsed,dict): parsed={}
    steps=parsed.get('steps',[])
    suspects=[s for s in (steps if isinstance(steps,list) else []) if isinstance(s,dict)]
    if forced_steps is not None:
        by_step={s.get('step'):s for s in suspects if isinstance(s,dict)}
        suspects=[by_step[i] for i in forced_steps if i in by_step]
    return suspects[:3],reply

def record(row,student,teacher,client,step_index,rank,replacement,reason,mode,original_wins=None,edited_wins=None,novel=()):
    step=student['trace'][step_index]
    guidance=private_guidance(client,teacher,replacement,reason) if replacement is not None else private_guidance(client,teacher)
    return {'task_id':row['id'],'step':step_index,'rank':rank,'student_messages':step['pre_state']['messages'],'student_state':step['pre_state'],
        'original_action':step.get('action'),'replacement':replacement,'reason':reason,'guidance':guidance,'guidance_kind':'correction' if replacement is not None else 'evidence',
        'verification':'replay' if mode=='verified' else 'none','repair_mode':mode,
        'model_revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','policy_id':getattr(client,'policy_id',BASE_POLICY),
        'original_wins':original_wins,'edited_wins':edited_wins,'novel_terms':list(novel)}

def repair(row,student,teacher,client,library,mode='verified',step_selection='ranked',seed=42):
    """mode: 'verified' (screen + 3-seed paired confirmation; the method), 'unverified' (top-ranked
    legal proposal accepted without replay), 'evidence-only' (no diagnosis: a seeded random step with
    evidence-only guidance; ordinary privileged-context OPSD). step_selection: 'ranked' (the model's
    diagnostic ranking) or 'random' (seeded random steps, same proposal budget)."""
    evidence={'task_id':row['id'],'attempts':[],'confirmed':None,'diagnosis_error':None,'ambiguous_answer_excluded':False,'mode':mode,'step_selection':step_selection}
    if student['metrics']['grounded_success']: return evidence
    if student['metrics']['answer_alias_ambiguous']:
        evidence['ambiguous_answer_excluded']=True; return evidence
    if mode=='evidence-only':
        chosen=random_steps(student,seed,1)
        if chosen: evidence['confirmed']=record(row,student,teacher,client,chosen[0],None,None,'',mode)
        return evidence
    forced=random_steps(student,seed,3) if step_selection=='random' else None
    try: suspects,reply=diagnose(student,teacher,client,forced); evidence['diagnosis_reply']=reply
    except Exception as e: evidence['diagnosis_error']=str(e); return evidence
    if forced is not None: evidence['forced_steps']=forced
    seen_steps=set(); screened=[]
    for rank,suspect in enumerate(suspects,1):
        idx=suspect.get('step')
        if type(idx) is not int or idx<0 or idx>=len(student['trace']) or idx in seen_steps: continue
        seen_steps.add(idx); step=student['trace'][idx]; state=step['pre_state']; original=step.get('action')
        # Malformed original action is retained diagnostically but cannot be replayed as a valid action.
        if original is None: continue
        original_tokens=step['reply']['usage']['completion_tokens']
        baseline=None
        seen_actions=set()
        for candidate in suspect.get('actions',[])[:2]:
            att={'step':idx,'rank':rank,'replacement':candidate,'reason':str(suspect.get('reason','')),'legal':False}
            evidence['attempts'].append(att)
            try:
                validate_action(candidate,state)
                att['novel_terms']=novel_terms(candidate,state)
                leaked=information_leak(candidate,state)
                if leaked: raise ValueError('Uses information the student has not seen: '+', '.join(leaked))
                key=json.dumps(candidate,sort_keys=True)
                if key==json.dumps(original,sort_keys=True) or key in seen_actions: raise ValueError('Duplicate/original action')
                seen_actions.add(key); att['legal']=True
            except Exception as e: att['rejection']=str(e); continue
            if mode=='unverified':
                evidence['confirmed']=record(row,student,teacher,client,idx,rank,candidate,att['reason'],mode,novel=att['novel_terms'])
                return evidence
            if baseline is None: baseline=run(row,client,library,2,42,0,state,original,original_tokens)
            edited=run(row,client,library,2,42,0,state,candidate)
            att['original_screen']=baseline; att['edited_screen']=edited
            gain=int(edited['metrics']['grounded_success'])-int(baseline['metrics']['grounded_success'])
            att['screen_gain']=gain
            if gain>0: screened.append(att)
    if not screened: return evidence
    best=sorted(screened,key=lambda x:(-x['screen_gain'],x['rank'],x['step']))[0]
    step=student['trace'][best['step']]; confirmation=[]
    for seed in (1701,1702,1703):
        original=run(row,client,library,2,seed,0.7,step['pre_state'],step['action'],step['reply']['usage']['completion_tokens'])
        edited=run(row,client,library,2,seed,0.7,step['pre_state'],best['replacement'])
        confirmation.append({'seed':seed,'original':original,'edited':edited})
    evidence['selected']={'step':best['step'],'rank':best['rank'],'replacement':best['replacement'],'confirmation':confirmation}
    ow=sum(x['original']['metrics']['grounded_success'] for x in confirmation)
    ew=sum(x['edited']['metrics']['grounded_success'] for x in confirmation)
    evidence['selected'].update(original_wins=ow,edited_wins=ew)
    if ew>=2 and ow<=1:
        evidence['confirmed']=record(row,student,teacher,client,best['step'],best['rank'],best['replacement'],best['reason'],mode,ow,ew,best.get('novel_terms',[]))
    return evidence
