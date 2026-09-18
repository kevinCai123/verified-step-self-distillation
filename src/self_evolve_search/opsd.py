"""Action-only full-vocabulary JSD, with detached same-model teacher targets."""
import copy, math
import torch

def action_inputs(prefix_ids, target_ids, device='cpu'):
    if not prefix_ids or not target_ids: raise ValueError('Empty prefix/target')
    ids=torch.tensor([list(prefix_ids)+list(target_ids)],dtype=torch.long,device=device)
    # Causal logits at prefix[-1] predict target[0]. Excludes every prompt/tool token.
    positions=torch.arange(len(prefix_ids)-1,len(prefix_ids)+len(target_ids)-1,device=device)
    return ids,positions

def teacher_messages(student_messages,guidance):
    messages=copy.deepcopy(student_messages)
    instruction=('\nPRIVATE TRAINING GUIDANCE (not visible to the student): Use the verified correction and retrieved evidence below to assess the student\'s NEXT action. '
        'The action must obey the STUDENT permissions and currently retrieved IDs. Do not finish using facts that the student has not read.\n')
    messages[0]['content']+=instruction+guidance
    return messages

def jsd_loss(student_logits,teacher_logits):
    if student_logits.shape!=teacher_logits.shape: raise ValueError('Student/teacher action logits are not aligned')
    student=torch.log_softmax(student_logits.float(),dim=-1)
    teacher=torch.log_softmax(teacher_logits.detach().float(),dim=-1)
    mixture=torch.logaddexp(student,teacher)-math.log(2)
    per_token=.5*((student.exp()*(student-mixture)).sum(-1)+(teacher.exp()*(teacher-mixture)).sum(-1))
    return per_token.mean()
