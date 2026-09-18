import json, re, sqlite3, time
from functools import lru_cache

STOP = set('a an the is was were are in on at of to for with and or which what who when where how did does do by from as it its this that has have had'.split())

class Library:
    def __init__(self, path):
        self.calls=[]
        self.db=sqlite3.connect('file:' + str(path) + '?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-65536')
        try:
            complete=self.db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
            if not complete or not json.loads(complete[0]): raise ValueError('Incomplete corpus index')
        except sqlite3.OperationalError as e: raise ValueError('Missing index completion marker') from e

    def search(self, query, top_k):
        start=time.perf_counter()
        terms=[t for t in re.findall(r'\w+',str(query).casefold()) if t not in STOP][:12]
        if not terms:
            self.calls.append({'action':'search','seconds':time.perf_counter()-start,'results':0})
            return []
        match=' OR '.join('"'+t+'"' for t in terms)
        found=self._search(match,top_k)
        self.calls.append({'action':'search','seconds':time.perf_counter()-start,'results':len(found)})
        return [{'doc_id':str(i),'title':t,'snippet':s} for i,t,s in found]

    @lru_cache(maxsize=2048)
    def _search(self,match,top_k):
        # Fixed corpus: identical queries/replays return identical results.
        return self.db.execute('SELECT rowid,title,snippet(search,1,\'\',\'\',\' ... \',32) FROM search WHERE search MATCH ? ORDER BY bm25(search,3,1),rowid LIMIT ?', (match,top_k)).fetchall()

    def read(self, doc_id):
        start=time.perf_counter()
        row=self.db.execute('SELECT id,title,sentences FROM docs WHERE id=?',(doc_id,)).fetchone()
        if row is None: raise KeyError('Unknown document')
        self.calls.append({'action':'read','seconds':time.perf_counter()-start})
        return {'doc_id':str(row[0]),'title':row[1],'sentences':[[i,s] for i,s in enumerate(json.loads(row[2]))]}
