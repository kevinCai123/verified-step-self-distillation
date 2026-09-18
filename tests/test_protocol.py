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
