import json, re, sqlite3, time
from functools import lru_cache

STOP = set('a an the is was were are in on at of to for with and or which what who when where how did does do by from as it its this that has have had'.split())
CONNECTORS = {'of', 'the', 'and', 'de', 'la', 'du', 'von', 'van', 'a', 'an', 'in', 'on', 'at', 'for', 'to'}

def title_variants(query):
    """Casings under which a query is looked up as an article title (indexed, case-sensitive lookups)."""
    text = ' '.join(str(query).split())
    if not text: return []
    words = text.split(' ')
    variants = [text, text[:1].upper() + text[1:],
                ' '.join(w if i and w.casefold() in CONNECTORS else (w[:1].upper() + w[1:]) for i, w in enumerate(words))]
    return list(dict.fromkeys(v for v in variants if v))

class Library:
    """Read-only access to the fixed corpus.

    search() returns, in this order and truncated to top_k: articles whose title equals the query,
    articles whose title is the query plus a parenthesised disambiguator (shortest title first),
    then the BM25 ranking. Every tier is ordered independently of top_k, so a top-2 result is
    always the prefix of the top-8 result for the same query (the student/teacher contract).
    The title tiers were added after round 1: with `bm25(search,3,1)` alone an entity's own stubs
    outranked its article ("Cocteau Twins" ranked 5th behind its discography and album pages), so
    no legal top-2 query could reach the evidence for a large share of failures.
    """
    def __init__(self, path, title_tier=True):
        self.calls=[]
        self.title_tier=title_tier
        self.db=sqlite3.connect('file:' + str(path) + '?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-65536')
        try:
            complete=self.db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
            if not complete or not json.loads(complete[0]): raise ValueError('Incomplete corpus index')
        except sqlite3.OperationalError as e: raise ValueError('Missing index completion marker') from e

    def search(self, query, top_k):
        start=time.perf_counter()
        terms=[t for t in re.findall(r'\w+',str(query).casefold()) if t not in STOP][:12]
        found=list(self._titles(' '.join(str(query).split()))) if self.title_tier else []
        if terms:
            match=' OR '.join('"'+t+'"' for t in terms)
            seen={row[0] for row in found}
            found += [row for row in self._search(match,top_k+len(found)) if row[0] not in seen][:max(0,top_k-len(found))]
        found=found[:top_k]
        self.calls.append({'action':'search','seconds':time.perf_counter()-start,'results':len(found),'title_matches':sum(1 for row in found if row[3])})
        return [{'doc_id':str(i),'title':t,'snippet':s} for i,t,s,_ in found]

    @lru_cache(maxsize=2048)
    def _titles(self,query):
        """(id, title, snippet, True) rows for exact-title and disambiguated-title matches, deterministic order."""
        exact=[]; disambiguated=[]
        for variant in title_variants(query):
            exact += self.db.execute('SELECT id,title,body FROM docs WHERE title=? ORDER BY id',(variant,)).fetchall()
            disambiguated += self.db.execute('SELECT id,title,body FROM docs WHERE title>=? AND title<? ORDER BY length(title),title,id',(variant+' (',variant+' )')).fetchall()
        rows=[]; seen=set()
        for i,t,body in exact+sorted(disambiguated,key=lambda r:(len(r[1]),r[1],r[0])):
            if i in seen: continue
            seen.add(i); rows.append((i,t,' '.join(body.split()[:32])+(' ...' if len(body.split())>32 else ''),True))
        return tuple(rows)

    @lru_cache(maxsize=2048)
    def _search(self,match,top_k):
        # Fixed corpus: identical queries/replays return identical results.
        return tuple((i,t,s,False) for i,t,s in self.db.execute('SELECT rowid,title,snippet(search,1,\'\',\'\',\' ... \',32) FROM search WHERE search MATCH ? ORDER BY bm25(search,3,1),rowid LIMIT ?', (match,top_k)).fetchall())

    def read(self, doc_id):
        start=time.perf_counter()
        row=self.db.execute('SELECT id,title,sentences FROM docs WHERE id=?',(doc_id,)).fetchone()
        if row is None: raise KeyError('Unknown document')
        self.calls.append({'action':'read','seconds':time.perf_counter()-start})
        return {'doc_id':str(row[0]),'title':row[1],'sentences':[[i,s] for i,s in enumerate(json.loads(row[2]))]}
