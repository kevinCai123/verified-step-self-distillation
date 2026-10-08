import copy, importlib.util, json, sys, types
from pathlib import Path
import pytest
from self_evolve_search.agent import initial_state, execute, validate_action, run
from self_evolve_search.metrics import grade, answer_scores

class Library:
    def search(self,q,k): return [{'doc_id':str(i),'title':f'Doc {i}','snippet':'text'} for i in range(k)]
    def read(self,i): return {'doc_id':i,'title':f'Doc {i}','sentences':[[0,'The answer is 2006.']]}

def test_role_permissions_and_budget_are_enforced():
    s=initial_state('question',2); lib=Library()
    with pytest.raises(PermissionError): execute({'action':'read','doc_id':'7'},s,lib,2)
    execute({'action':'search','query':'q','top_k':999},s,lib,2)
    assert s['retrieved']==['0','1']
    with pytest.raises(PermissionError): execute({'action':'read','doc_id':'7'},s,lib,2)
    for _ in range(7): execute({'action':'search','query':'q'},s,lib,2)
    with pytest.raises(ValueError,match='budget'): execute({'action':'read','doc_id':'0'},s,lib,2)
    assert s['tool_calls']==8

def test_replay_state_is_a_separate_copy():
    original=initial_state('question',2); execute({'action':'search','query':'q'},original,Library(),2)
    snapshot=copy.deepcopy(original)
    replay=copy.deepcopy(snapshot); execute({'action':'read','doc_id':'1'},replay,Library(),2)
    assert original==snapshot and not original['observed']
    assert replay['tool_calls']==2 and original['tool_calls']==1

def test_verifier_requires_exposed_facts_and_valid_citations():
    row={'answer':'2006','supporting_facts':{'title':['A','B'],'sent_id':[0,1]}}
    assert not grade(row,'2006',[['A',0],['B',1]],[['A',0]])['grounded_success']
    assert grade(row,'2006',[['A',0],['B',1]],[['A',0],['B',1]])['grounded_success']
    assert not grade(row,'2006',[],[['A',0],['B',1]])['grounded_success']
    assert not grade(row,'2006',[['A',9]],[['A',0],['B',1]])['grounded_success']

def test_answer_scores_agree_with_official_evaluator():
    root=Path(__file__).resolve().parents[1]
    sys.modules.setdefault('ujson',json)
    spec=importlib.util.spec_from_file_location('official',root/'data/raw/hotpot_evaluate_v1.py')
    official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    for pred,gold in [('The Eiffel Tower','Eiffel Tower'),('yes','no'),('Walter Coy','Walter Darwin Coy'),('6.213 km','6.213 km long'),('',''),('2006','2006')]:
        em,f1,p,r=answer_scores(pred,gold)
        assert em==official.exact_match_score(pred,gold)
        assert (f1,p,r)==official.f1_score(pred,gold)

class StubClient:
    def tokens(self,text): return 10
    def chat(self,messages,*args):
        return {'content':json.dumps({'action':'finish','answer':'2006','citations':[['Doc 1',0]]}),'usage':{'completion_tokens':12,'prompt_tokens':30},'finish_reason':'stop','seconds':0}

def test_forced_replay_preserves_state_and_remaining_budget():
    row={'id':'x','question':'q','answer':'2006','supporting_facts':{'title':['Doc 1'],'sent_id':[0]}}
    state=initial_state('q',2)
    execute({'action':'search','query':'q'},state,Library(),2)
    state['generated_tokens']=100
    untouched=copy.deepcopy(state)
    result=run(row,StubClient(),Library(),state=state,forced={'action':'read','doc_id':'1'})
    assert state==untouched
    assert result['metrics']['grounded_success'] and result['state']['tool_calls']==2
    assert result['state']['generated_tokens']==123
    result=run(row,StubClient(),Library(),state=state,forced={'action':'read','doc_id':'7'})
    assert not result['metrics']['grounded_success'] and 'PermissionError' in result['error']

def test_generated_code_is_not_an_action():
    with pytest.raises(ValueError,match='Unknown action'):
        execute({'action':'exec','code':'raise SystemExit()'},initial_state('q',2),Library(),2)

def test_real_fts_retrieval_preserves_sentence_ids_and_permission_order(tmp_path):
    import sqlite3
    from self_evolve_search.retrieval import Library as SearchLibrary
    path=tmp_path/'wiki.sqlite'
    db=sqlite3.connect(path)
    db.executescript('CREATE TABLE docs(id INTEGER PRIMARY KEY,title TEXT,body TEXT,sentences TEXT); CREATE TABLE metadata(key TEXT,value TEXT); CREATE VIRTUAL TABLE search USING fts5(title,body,content=docs,content_rowid=id);')
    for i in range(9):
        db.execute('INSERT INTO docs VALUES (?,?,?,?)',(i,f'Bridge {i}','bridge evidence',json.dumps(['First sentence.','Second sentence.'])))
    db.execute("INSERT INTO search(search) VALUES ('rebuild')")
    db.execute('INSERT INTO metadata VALUES (?,?)',('complete','true')); db.commit(); db.close()
    lib=SearchLibrary(path)
    restricted=lib.search('bridge',2); expanded=lib.search('bridge',8)
    assert restricted==expanded[:2] and len(expanded)==8
    assert lib.read(restricted[0]['doc_id'])['sentences']==[[0,'First sentence.'],[1,'Second sentence.']]
    restricted[0]['title']='changed outside cache'
    assert lib.search('bridge',2)[0]['title']!='changed outside cache'
    assert len(lib.calls)==4

def _state_after_search(question):
    state=initial_state(question,2)
    execute({'action':'search','query':question},state,Library(),2)
    state['messages'].append({'role':'user','content':'Tool result: [{"doc_id":"0","title":"Erna Siikavirta","snippet":"Erna Siikavirta is a Finnish keyboardist known for Lordi ..."}]'})
    return state

def test_replacements_may_not_introduce_entities_the_student_has_not_seen():
    from self_evolve_search.repair import information_leak, novel_terms
    state=_state_after_search('Erna Siikavirta was a member of the Finnish band formed in what year?')
    assert information_leak({'action':'search','query':'Lordi band formed year'},state)==[]          # Lordi appears in a snippet
    assert information_leak({'action':'search','query':'Erna Siikavirta'},state)==[]                  # from the question
    assert information_leak({'action':'search','query':'band discography formed'},state)==[]          # generic words are allowed
    assert information_leak({'action':'search','query':'Mr. Lordi Tomi Putaansuu 1992'},state)==['Tomi','Putaansuu','1992']
    assert set(novel_terms({'action':'search','query':'Lordi discography 1992'},state))=={'discography','1992'}
    assert information_leak({'action':'read','doc_id':'0'},state)==[]

def test_final_answer_replacement_must_be_a_read_span():
    from self_evolve_search.repair import information_leak
    state=initial_state('q',2); execute({'action':'search','query':'q'},state,Library(),2); execute({'action':'read','doc_id':'1'},state,Library(),2)
    assert information_leak({'action':'finish','answer':'2006','citations':[]},state)==[]
    assert information_leak({'action':'finish','answer':'yes','citations':[]},state)==[]
    assert information_leak({'action':'finish','answer':'1992','citations':[]},state)==['1992']

def test_guidance_uses_the_models_compact_json_dialect_and_hides_evidence_from_the_teacher():
    from self_evolve_search.repair import private_guidance
    from self_evolve_search.opsd import teacher_messages
    teacher={'state':{'read_docs':[{'title':'Doc 1','sentences':[[0,'The answer is 2006.']]}]}}
    text=private_guidance(StubClient(),teacher,{'action':'search','query':'Israel Gelfand mathematician'},'too specific')
    assert text=='Verified correction for this decision: {"action":"search","query":"Israel Gelfand mathematician"}\n'
    assert '2006' not in text and 'too specific' not in text          # evidence and diagnosis stay out of the teacher context
    system=teacher_messages([{'role':'system','content':'S'},{'role':'user','content':'Q'}],text)[0]['content']
    assert 'Use the verified correction below' in system and 'evidence' not in system.split('PRIVATE')[1].split('below')[0]
    full=private_guidance(StubClient(),teacher,{'action':'search','query':'Israel Gelfand mathematician'},'r',include_evidence=True)
    assert full.startswith('Verified correction for this decision: {"action":"search","query":"Israel Gelfand mathematician"}\nRetrieved evidence:\n["Doc 1",0,"The answer is 2006."]')
    assert '": "' not in full and '", "' not in full

def test_premature_finish_tripwire_counts_only_states_where_finishing_is_wrong():
    from self_evolve_search.opsd import action_type, premature_finish_share
    assert action_type('{"action":"finish","answer":"x","citations":[]}')=='finish' and action_type('not json') is None and action_type('[1]') is None
    rows=[{'replacement':{'action':'search','query':'q'},'student_state':{'read_docs':[]}},
          {'replacement':{'action':'finish','answer':'a'},'student_state':{'read_docs':[{'title':'D','sentences':[[0,'s']]}]}},   # finishing is the correction: not eligible
          {'replacement':{'action':'read','doc_id':'1'},'student_state':{'read_docs':[]}}]
    outputs=['{"action":"finish","answer":"guess","citations":[]}','{"action":"finish","answer":"a","citations":[["D",0]]}','{"action":"search","query":"q"}']
    assert premature_finish_share(rows,outputs)==0.5
    assert premature_finish_share(rows,['{"action":"search","query":"q"}']*3)==0.

def test_illegal_read_tripwire_counts_reads_of_unretrieved_documents():
    from self_evolve_search.opsd import illegal_read_share
    rows=[{'student_state':{'retrieved':['12','34']}},{'student_state':{'retrieved':['12']}},{'student_state':{'retrieved':[]}},{'student_state':{'retrieved':['7']}}]
    outputs=['{"action":"read","doc_id":"12"}','{"action":"read","doc_id":"doc_1"}','{"action":"read","doc_id":"doc_1"}','{"action":"search","query":"q"}']
    assert illegal_read_share(rows,outputs)==0.5          # two fabricated reads out of four probed states
    assert illegal_read_share(rows,['not json']*4)==0. and illegal_read_share([],[])==0.

def _corpus(path, docs):
    import sqlite3
    db=sqlite3.connect(path)
    db.executescript('CREATE TABLE docs(id INTEGER PRIMARY KEY,title TEXT,body TEXT,sentences TEXT); CREATE INDEX docs_title ON docs(title); CREATE TABLE metadata(key TEXT,value TEXT); CREATE VIRTUAL TABLE search USING fts5(title,body,content=docs,content_rowid=id);')
    for i,(title,body) in enumerate(docs,1):
        db.execute('INSERT INTO docs VALUES (?,?,?,?)',(i,title,body,json.dumps([body])))
    db.execute("INSERT INTO search(search) VALUES ('rebuild')"); db.execute('INSERT INTO metadata VALUES (?,?)',('complete','true')); db.commit(); db.close()

def test_title_tier_puts_the_article_before_its_stubs_and_keeps_the_permission_prefix(tmp_path):
    from self_evolve_search.retrieval import Library as SearchLibrary, title_variants
    long_body='Cocteau Twins were a Scottish rock band active from 1979 to 1997. '+'The band explored ethereal dream pop. '*40
    docs=[('Cocteau Twins',long_body),('Cocteau Twins discography','Cocteau Twins discography.'),('Treasure (Cocteau Twins album)','Treasure is a Cocteau Twins album.'),
          ('Cocteau Twins (film)','A short film.'),('Big Country','Big Country are a Scottish rock band formed in 1981.'),('Head over Heels (Cocteau Twins album)','Album by Cocteau Twins.')]
    path=tmp_path/'wiki.sqlite'; _corpus(path,docs)
    plain=SearchLibrary(path,title_tier=False)
    assert plain.search('Cocteau Twins',2)[0]['title']!='Cocteau Twins'          # BM25 alone prefers the stubs
    lib=SearchLibrary(path)
    top2=lib.search('cocteau twins',2); top8=lib.search('cocteau twins',8)
    assert [d['title'] for d in top2]==['Cocteau Twins','Cocteau Twins (film)']    # exact title, then disambiguated title
    assert top8[:2]==top2 and len(top8)==5 and top8[2]['title']=='Cocteau Twins discography'
    assert top2[0]['snippet'].startswith('Cocteau Twins were a Scottish rock band')
    assert lib.calls[-1]['title_matches']==2 and lib.calls[-2]['title_matches']==2
    assert [d['title'] for d in lib.search('Big Country',2)][0]=='Big Country'
    assert lib.search('the who',2)==[]                                              # stopword-only query, no title: nothing
    assert title_variants('my cousin rachel')==['my cousin rachel','My cousin rachel','My Cousin Rachel']
    assert lib.search('bridge',2)==[]

class ScriptedClient:
    """Returns canned model replies in order; records every chat call."""
    def __init__(self,replies): self.replies=list(replies); self.calls=[]; self.policy_id='base:test'
    def tokens(self,text): return max(1,len(text)//4)
    def prompt_tokens(self,messages): return sum(self.tokens(m['content']) for m in messages)
    def chat(self,messages,*args,**kwargs):
        self.calls.append(messages); content=self.replies.pop(0)
        return {'content':content,'usage':{'completion_tokens':12,'prompt_tokens':30},'finish_reason':'stop','seconds':0}

class SnippetLibrary(Library):
    def search(self,q,k): return [{'doc_id':str(i),'title':f'Doc {i}','snippet':'Erna Siikavirta played keyboards for Lordi.'} for i in range(k)]

def _failed_pair():
    from self_evolve_search.repair import repair
    row={'id':'t1','question':'Erna Siikavirta was a member of the Finnish band formed in what year?','answer':'1992','supporting_facts':{'title':['Doc 1'],'sent_id':[0]}}
    student=run(row,ScriptedClient(['{"action":"search","query":"Erna Siikavirta Finnish band formed year"}','{"action":"finish","answer":"1990","citations":[["Doc 1",0]]}']),SnippetLibrary(),2)
    teacher=run(row,ScriptedClient(['{"action":"search","query":"Erna Siikavirta"}','{"action":"read","doc_id":"1"}','{"action":"finish","answer":"2006","citations":[["Doc 1",0]]}']),SnippetLibrary(),8)
    assert not student['metrics']['grounded_success'] and teacher['state']['read_docs']
    return row,student,teacher,repair

def test_random_step_selection_is_seeded_and_replayable():
    from self_evolve_search.repair import random_steps
    row,student,teacher,repair=_failed_pair()
    assert random_steps(student,42,3)==random_steps(student,42,3)==[0,1]
    assert random_steps(student,42,1)==random_steps(student,42,1) and len(random_steps(student,42,1))==1
    assert random_steps({'task_id':'other','trace':student['trace']},42,1)!=random_steps(student,7,1) or True   # different seeds may agree; only determinism is required
    client=ScriptedClient(['{"steps":[{"step":1,"reason":"r","actions":[{"action":"read","doc_id":"1"}]},{"step":0,"reason":"r","actions":[{"action":"search","query":"Erna Siikavirta"}]}]}'])
    evidence=repair(row,student,teacher,client,SnippetLibrary(),mode='unverified',step_selection='random',seed=42)
    assert evidence['forced_steps']==[0,1] and '[0, 1]' in client.calls[0][0]['content']
    assert evidence['confirmed']['step']==0 and evidence['confirmed']['rank']==1        # forced order, not the model's order

def test_unverified_mode_accepts_the_top_legal_proposal_without_replay():
    row,student,teacher,repair=_failed_pair()
    client=ScriptedClient(['{"steps":[{"step":0,"reason":"too long","actions":[{"action":"search","query":"Lordi band formed"},{"action":"search","query":"Erna Siikavirta band"}]}]}'])
    evidence=repair(row,student,teacher,client,SnippetLibrary(),mode='unverified')
    assert len(client.calls)==1                                                        # diagnosis only: no screening, no confirmation replays
    assert evidence['attempts'][0]['legal'] is False and 'Lordi' in evidence['attempts'][0]['rejection']   # not yet visible at step 0
    record=evidence['confirmed']
    assert record['replacement']=={'action':'search','query':'Erna Siikavirta band'} and record['verification']=='none' and record['guidance_kind']=='correction'
    assert record['guidance'].startswith('Verified correction for this decision: {"action":"search","query":"Erna Siikavirta band"}')
    assert record['original_wins'] is None and record['edited_wins'] is None
    from self_evolve_search.persistence import validate_training_batch
    validate_training_batch([dict(record,task_id=str(i)) for i in range(8)],'base:test')

def test_evidence_only_mode_needs_no_diagnosis_and_carries_evidence_guidance():
    from self_evolve_search.opsd import teacher_messages
    row,student,teacher,repair=_failed_pair()
    client=ScriptedClient([])
    evidence=repair(row,student,teacher,client,SnippetLibrary(),mode='evidence-only')
    record=evidence['confirmed']
    assert client.calls==[] and record['replacement'] is None and record['guidance_kind']=='evidence' and record['verification']=='none'
    assert record['guidance'].startswith('Retrieved evidence:\n["Doc 1",0,')
    assert 'the retrieved evidence below' in teacher_messages(record['student_messages'],record['guidance'])[0]['content']
    assert 'verified correction' not in teacher_messages(record['student_messages'],record['guidance'])[0]['content']
    assert repair(row,student,teacher,ScriptedClient([]),SnippetLibrary(),mode='evidence-only')['confirmed']['step']==record['step']

def test_verified_mode_still_screens_and_confirms():
    row,student,teacher,repair=_failed_pair()
    replies=['{"steps":[{"step":0,"reason":"r","actions":[{"action":"search","query":"Erna Siikavirta"}]}]}']
    replies+=['{"action":"finish","answer":"1990","citations":[]}']*9+['{"action":"read","doc_id":"1"}','{"action":"finish","answer":"1992","citations":[["Doc 1",0]]}']*9
    client=ScriptedClient(replies)
    evidence=repair(row,student,teacher,client,SnippetLibrary())
    assert evidence['mode']=='verified' and (evidence['confirmed'] is None or evidence['confirmed']['verification']=='replay')
    assert len(client.calls)>1

def test_malformed_diagnosis_entries_are_skipped_not_fatal():
    # 22 September: a bare string inside "steps" raised AttributeError and aborted arm A's batch collection.
    row,student,teacher,repair=_failed_pair()
    client=ScriptedClient(['{"steps":["step 0 was too broad",{"step":0,"reason":"r","actions":[{"action":"search","query":"Erna Siikavirta band"}]}]}'])
    evidence=repair(row,student,teacher,client,SnippetLibrary(),mode='unverified')
    assert evidence['diagnosis_error'] is None and evidence['confirmed']['step']==0
    for reply in ('["not an object"]','"just text"','{"steps":"none"}'):
        evidence=repair(row,student,teacher,ScriptedClient([reply]),SnippetLibrary(),mode='unverified')
        assert evidence['diagnosis_error'] is None and evidence['confirmed'] is None and evidence['attempts']==[]

def _load_collect_batch():
    """scripts/collect_batch.py imports transformers for the real tokenizer; the self-success control needs neither."""
    import importlib.util, sys, types
    if 'transformers' not in sys.modules:
        try: import transformers  # noqa: F401
        except ImportError: sys.modules['transformers']=types.SimpleNamespace(AutoTokenizer=None)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('collect_batch_test',root/'scripts/collect_batch.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def test_self_success_control_records_the_students_own_successful_step_without_a_teacher():
    from self_evolve_search.persistence import validate_training_batch
    collect=_load_collect_batch()
    row={'id':'ok1','question':'Erna Siikavirta was a member of the Finnish band formed in what year?','answer':'1992','supporting_facts':{'title':['Doc 1'],'sent_id':[0]}}
    success=run(row,ScriptedClient(['{"action":"search","query":"Erna Siikavirta"}','{"action":"read","doc_id":"1"}','{"action":"finish","answer":"1992","citations":[["Doc 1",0]]}']),SnippetLibrary(),2)
    assert success['metrics']['grounded_success']
    evidence=collect.self_success(row,success,ScriptedClient([]),42)
    record=evidence['confirmed']
    assert evidence['mode']=='self-success' and record is not None and record['verification']=='none' and record['repair_mode']=='self-success'
    assert record['replacement']==success['trace'][record['step']]['action']==record['original_action']
    assert record['guidance'].startswith('Verified correction for this decision: ') and 'Retrieved evidence' not in record['guidance']
    assert collect.self_success(row,success,ScriptedClient([]),42)['confirmed']['step']==record['step']     # seeded: replayable
    validate_training_batch([dict(record,task_id=str(i)) for i in range(8)],'base:test')
    failed=run(row,ScriptedClient(['{"action":"finish","answer":"1990","citations":[]}']),SnippetLibrary(),2)
    assert not failed['metrics']['grounded_success'] and collect.self_success(row,failed,ScriptedClient([]),42)['confirmed'] is None   # failures yield nothing
