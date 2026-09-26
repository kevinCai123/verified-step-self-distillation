import torch
from self_evolve_search.opsd import action_inputs, content_token_mask, jsd_loss, jsd_per_token, teacher_messages, value_character_mask

def test_jsd_teacher_is_detached_and_student_gets_gradient():
    s=torch.tensor([[[2.,0.,-1.]]],requires_grad=True)
    t=torch.tensor([[[0.,2.,-1.]]],requires_grad=True)
    loss=jsd_loss(s,t); loss.backward()
    assert loss.item()>0 and torch.isfinite(loss)
    assert s.grad is not None and s.grad.abs().sum()>0 and t.grad is None

def test_equal_distributions_have_zero_loss():
    s=torch.randn(2,3,9)
    assert abs(jsd_loss(s,s).item())<1e-6

def test_different_prefix_lengths_have_identical_targets():
    s,sp=action_inputs([4,5],[7,8,9])
    t,tp=action_inputs([1,2,3,4,5],[7,8,9])
    assert s[0,sp+1].tolist()==t[0,tp+1].tolist()==[7,8,9]
    assert sp.tolist()==[1,2,3] and tp.tolist()==[4,5,6]

def test_private_guidance_does_not_mutate_student():
    s=[{'role':'system','content':'Student rules'},{'role':'user','content':'Question'}]
    t=teacher_messages(s,'private correction')
    assert 'private correction' not in str(s) and 'private correction' in str(t)

def test_value_characters_exclude_json_structure_and_keys():
    text='{"action":"finish","answer":"The Regency era","citations":[["Richard Cosway",0]]}'
    mask=value_character_mask(text)
    values=''.join(c for c,flag in zip(text,mask) if flag)
    assert values=='finishThe Regency eraRichard Cosway0'
    spaced='{"action": "search", "query": "Erna Siikavirta"}'
    assert ''.join(c for c,flag in zip(spaced,value_character_mask(spaced)) if flag)=='searchErna Siikavirta'
    truncated='{"action":"search","query":"unterminated'
    assert ''.join(c for c,flag in zip(truncated,value_character_mask(truncated)) if flag)=='searchunterminated'

class CharTokenizer:
    def decode(self,ids,skip_special_tokens=True): return ''.join(ids)

def test_content_token_mask_marks_tokens_that_render_a_value_character():
    tokens=['{"','action','":"','search','","','query','":"','Erna',' Siika','virta','"}']
    assert content_token_mask(CharTokenizer(),tokens)==[False,False,False,True,False,False,False,True,True,True,False]   # the closing "} carries no value character

def test_weighted_loss_averages_only_content_tokens_and_falls_back_when_empty():
    s=torch.randn(1,4,7); t=torch.randn(1,4,7)
    per=jsd_per_token(s,t)[0]
    assert torch.allclose(jsd_loss(s,t,[0,1,0,1]),(per[1]+per[3])/2)
    assert torch.allclose(jsd_loss(s,t,[0,0,0,0]),per.mean())
    assert torch.allclose(jsd_loss(s,t),per.mean())

def test_value_mask_ignores_text_after_the_action_closes():
    text='{"action":"search","query":"F-segment car"}\n\nuser Tool result: [{"doc_id": "118"}]'
    assert ''.join(c for c,flag in zip(text,value_character_mask(text)) if flag)=='searchF-segment car'
