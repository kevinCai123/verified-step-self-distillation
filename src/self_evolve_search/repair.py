import json
from .agent import SCHEMA, run, validate_action
from .persistence import BASE_POLICY

def private_guidance(client,teacher,replacement,reason):
    prefix='Verified correction for this decision: '+json.dumps(replacement,ensure_ascii=False)+'\nDiagnosis: '+reason+'\nRetrieved evidence:\n'
    if client.tokens(prefix)>1024: raise ValueError('Correction exceeds guidance budget')
    text=prefix
    for doc in teacher['state']['read_docs']:
        for i,sentence in doc['sentences']:
            line=json.dumps([doc['title'],i,sentence],ensure_ascii=False)+'\n'
            if client.tokens(text+line)<=1024: text+=line
    return text

def diagnose(student,teacher,client):
    steps=[{'step':i,'action':s.get('action'),'error':s.get('error'),'observation':s.get('observation')} for i,s in enumerate(student['trace'])]
    evidence=teacher['state']['read_docs']
    prompt=('You are reviewing your own failed local-search trajectory. Rank up to THREE consequential steps and propose up to TWO replacements per step. '
        'Use the original step indices. Each replacement must be legal for the restricted student: top-2 search; read only IDs already retrieved BEFORE that step; same remaining budget. '
        'A final answer is allowed only after the student has READ its supporting sentences. Prefer fixing search/read decisions when evidence is missing. '
        'Your private evidence is not copied into the student. '+SCHEMA+
        ' Return {"steps":[{"step":0,"reason":"brief reason","actions":[{"action":"search","query":"..."}]}]}. No reference answer is provided.')
    messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps({'question':student['question'],'failed_steps':steps,'private_retrieved_evidence':evidence},ensure_ascii=False)}]
    # Preserve full failed observations; reduce teacher-only evidence if needed, without consulting labels.
    while client.prompt_tokens(messages)>7552 and evidence:
        evidence=evidence[:-1]
        messages[-1]['content']=json.dumps({'question':student['question'],'failed_steps':steps,'private_retrieved_evidence':evidence},ensure_ascii=False)
    reply=client.chat(messages,seed=200,temperature=0,max_tokens=512,max_prompt=7552)
    parsed=json.loads(reply['content'])
    return parsed.get('steps',[])[:3],reply

def repair(row,student,teacher,client,library):
    evidence={'task_id':row['id'],'attempts':[],'confirmed':None,'diagnosis_error':None,'ambiguous_answer_excluded':False}
    if student['metrics']['grounded_success']: return evidence
    if student['metrics']['answer_alias_ambiguous']:
        evidence['ambiguous_answer_excluded']=True; return evidence
    try: suspects,reply=diagnose(student,teacher,client); evidence['diagnosis_reply']=reply
    except Exception as e: evidence['diagnosis_error']=str(e); return evidence
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
                key=json.dumps(candidate,sort_keys=True)
                if key==json.dumps(original,sort_keys=True) or key in seen_actions: raise ValueError('Duplicate/original action')
                seen_actions.add(key); att['legal']=True
            except Exception as e: att['rejection']=str(e); continue
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
        evidence['confirmed']={'task_id':row['id'],'step':best['step'],'rank':best['rank'],'student_messages':step['pre_state']['messages'],'student_state':step['pre_state'],'original_action':step['action'],'replacement':best['replacement'],'guidance':private_guidance(client,teacher,best['replacement'],best['reason']),'model_revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','policy_id':getattr(client,'policy_id',BASE_POLICY),'original_wins':ow,'edited_wins':ew}
    return evidence
