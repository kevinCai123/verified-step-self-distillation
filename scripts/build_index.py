"""Stream the official tar/bz2 corpus into SQLite; never select using QA labels."""
import argparse, bz2, hashlib, io, json, sqlite3, tarfile, time
from pathlib import Path

def sentences(text):
    # Release schema contains paragraphs -> sentence strings.
    if text and isinstance(text[0], list): return [s for paragraph in text for s in paragraph]
    return text

def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive', required=True); p.add_argument('--output', required=True); a=p.parse_args()
    out=Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): raise FileExistsError('Refuse to overwrite an existing index: ' + str(out))
    db=sqlite3.connect(out)
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-131072; CREATE TABLE docs(id INTEGER PRIMARY KEY, title TEXT, body TEXT, sentences TEXT); CREATE INDEX docs_title ON docs(title); CREATE VIRTUAL TABLE search USING fts5(title,body,content=docs,content_rowid=id,tokenize="unicode61"); CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);')
    start=time.time(); count=0; batch=[]
    with tarfile.open(a.archive, mode='r|bz2') as tar:
        for member in tar:
            if not member.isfile(): continue
            stream=tar.extractfile(member)
            if member.name.endswith('.bz2'): stream=bz2.BZ2File(stream)
            for line in stream:
                if not line.strip(): continue
                doc=json.loads(line)
                ss=sentences(doc['text'])
                batch.append((int(doc['id']),doc['title'],' '.join(ss),json.dumps(ss,ensure_ascii=False)))
                if len(batch)>=5000:
                    db.executemany('INSERT INTO docs VALUES (?,?,?,?)',batch); db.commit(); count+=len(batch); batch=[]
                    if count%100000==0: print(json.dumps({'documents':count,'seconds':round(time.time()-start)}),flush=True)
    if batch: db.executemany('INSERT INTO docs VALUES (?,?,?,?)',batch); count+=len(batch); db.commit()
    print('Building full-text postings',flush=True)
    db.execute("INSERT INTO search(search) VALUES ('rebuild')"); db.commit()
    meta={'complete':True,'documents':count,'source':'official HotpotQA fullwiki introductory paragraphs','archive_md5':'01edf64cd120ecc03a2745352779514c','elapsed_seconds':time.time()-start}
    for k,v in meta.items(): db.execute('INSERT INTO metadata VALUES (?,?)',(k,json.dumps(v)))
    db.commit(); db.close()
    out.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)

if __name__ == '__main__': main()
