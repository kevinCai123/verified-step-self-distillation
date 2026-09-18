import argparse, json, re, sqlite3
from pathlib import Path

def normalize(s): return ' '.join(s.split())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--questions',required=True); p.add_argument('--index',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    db=sqlite3.connect('file:'+a.index+'?mode=ro',uri=True)
    rows=[json.loads(x) for x in Path(a.questions).read_text().splitlines()]
    missing=[]; mismatch=[]; facts=0; matched=0
    for row in rows:
        contexts=dict(zip(row['context']['title'],row['context']['sentences']))
        for title,i in zip(row['supporting_facts']['title'],row['supporting_facts']['sent_id']):
            facts+=1
            docs=db.execute('SELECT sentences FROM docs WHERE title=?',(title,)).fetchall()
            if not docs: missing.append([row['id'],title,i]); continue
            expected=contexts.get(title,[])
            if i>=len(expected): mismatch.append([row['id'],title,i,'missing source sentence']); continue
            if any(i<len(ss:=json.loads(doc[0])) and normalize(ss[i])==normalize(expected[i]) for doc in docs): matched+=1
            else: mismatch.append([row['id'],title,i,'sentence differs'])
    report={'questions':len(rows),'facts':facts,'matched_facts':matched,'coverage':matched/facts,'missing_titles':missing,'sentence_mismatches':mismatch}
    Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)},indent=2))

if __name__=='__main__': main()
