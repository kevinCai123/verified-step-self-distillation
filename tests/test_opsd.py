import torch
from self_evolve_search.opsd import action_inputs, jsd_loss, teacher_messages

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
