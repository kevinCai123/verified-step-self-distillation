"""Official HotpotQA normalization/answer/support scoring, plus exposure checks."""
import collections, re, string

def normalize(s):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())

def answer_scores(pred,gold):
    p,g=normalize(pred),normalize(gold)
    em=float(p==g)
    if (p in ('yes','no','noanswer') or g in ('yes','no','noanswer')) and p!=g: return em,0.,0.,0.
    pt,gt=p.split(),g.split(); common=sum((collections.Counter(pt)&collections.Counter(gt)).values())
    if not common: return em,0.,0.,0.
    precision,recall=common/len(pt),common/len(gt)
    return em,2*precision*recall/(precision+recall),precision,recall

def supporting(row):
    s=row['supporting_facts']
    return set(zip(s['title'],s['sent_id'])) if isinstance(s,dict) else set(map(tuple,s))

def grade(row, answer, citations, observed):
    ae,af,ap,ar=answer_scores(answer,row['answer'])
    gold=supporting(row)
    predicted=set()
    for item in citations:
        if isinstance(item,list) and len(item)==2 and isinstance(item[0],str) and isinstance(item[1],int): predicted.add(tuple(item))
    common=len(gold&predicted)
    sp=common/len(predicted) if predicted else 0.
    sr=common/len(gold) if gold else 0.
    sf=2*sp*sr/(sp+sr) if sp+sr else 0.
    jp,jr=ap*sp,ar*sr
    obs=set(map(tuple,observed))
    valid=bool(predicted) and len(predicted)==len(citations) and predicted<=obs
    coverage=gold<=obs
    return {'answer_em':ae,'answer_f1':af,'support_em':float(predicted==gold),'support_f1':sf,'joint_em':ae*float(predicted==gold),'joint_f1':2*jp*jr/(jp+jr) if jp+jr else 0.,'citation_valid':valid,'support_exposed':coverage,'grounded_success':bool(ae and valid and coverage),'answer_alias_ambiguous':bool(not ae and af>0)}
