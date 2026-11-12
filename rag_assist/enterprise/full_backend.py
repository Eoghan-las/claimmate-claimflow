"""Immutable full-corpus release on the existing local PostgreSQL/OS/Neo4j services."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

from rag_assist.localstack.backends import BackendStore,_json
from rag_assist.localstack.retrieval import fuse_routes


def verify_snapshot(expected,observed,label):
    if set(expected)!=set(observed):raise ValueError(label+'_ID_COVERAGE')
    if any(expected[key]!=observed[key] for key in expected):raise ValueError(label+'_SOURCE_MISMATCH')
    return len(observed)


def source_aliases(unit):
    return list(dict.fromkeys(a for a in [unit['title'],unit.get('metadata',{}).get('clause_no')]
                              if isinstance(a,str) and len(a)>=2))


class FullCorpusStore(BackendStore):
    def __init__(self,settings,knowledge):
        super().__init__(settings)
        self.knowledge=knowledge;self.bundle=knowledge.release_id
        suffix=self.bundle.removeprefix('full-bge-')
        self.text_index='claimmate-full-text-'+suffix;self.layout_index='claimmate-full-layout-'+suffix
        self.windows={};self.rows={}
        for sid,doc in knowledge.corpus.documents.items():
            for w in doc['artifact']['windows']:
                if not w['retrieval_eligible']:continue
                row={'window_id':w['window_id'],'evidence_id':w['unit_id'],'product_id':knowledge.sources[sid].version_id,
                     'source_id':sid,'text':w['search_text'],'body_text':w['body_text'],
                     'input_sha256':sha256(w['search_text'].encode()).hexdigest(),
                     'kind':w['unit_kind'],'title':knowledge.unit_metadata[w['unit_id']]['title'],
                     'source_role':knowledge.catalogue['documents'][sid]['source_role']}
                self.windows[w['window_id']]=w;self.rows[w['window_id']]=row

    def archive(self,folder):
        folder=Path(folder);(folder/'originals').mkdir(parents=True,exist_ok=True)
        with self._pg() as c:
            self._schema(c)
            for sid,doc in self.knowledge.corpus.documents.items():
                src=self.knowledge.sources[sid];pid=src.version_id
                target=folder/'originals'/(sid+'.pdf')
                shutil.copyfile(self.knowledge.corpus.root/doc['record']['pdf_path'],target)
                if sha256(target.read_bytes()).hexdigest()!=src.raw_file_sha256:raise ValueError('ARCHIVE_PDF_HASH_MISMATCH')
                self._put(c,pid,'full_source',sid,{k:doc[k] for k in ('record','build','physical','artifact')})
                self._put(c,pid,'catalogue_document',sid,self.knowledge.catalogue['documents'][sid])
                for page in doc['physical']['pages']:
                    self._put(c,pid,'full_page',sid+':'+str(page['page_number']),{**page,'source_id':sid})
                for u in doc['artifact']['business_document']['units']:
                    self._put(c,pid,'full_unit',u['unit_id'],u)
                for w in doc['artifact']['windows']:
                    self._put(c,pid,'full_window',w['window_id'],w)
            for pid,p in self.knowledge.products.items():self._put(c,pid,'product',pid,p.model_dump(mode='json'))
        self._state('ARCHIVED_REFERENCE_ONLY',self.knowledge.corpus.counts)

    def ingest(self,models,folder):
        if models.status()['embedding']['model_id']!='BAAI/bge-m3':raise ValueError('INDEX_EMBEDDING_SPACE_MISMATCH')
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        if self._published_state()=='READY':
            return self.audit()
        self.archive(folder)
        self._state('BUILDING',self.knowledge.corpus.counts)
        mapping={'settings':{'index':{'knn':True,'number_of_shards':1,'number_of_replicas':0}},
            'mappings':{'properties':{'bundle_id':{'type':'keyword'},'product_id':{'type':'keyword'},
                'source_id':{'type':'keyword'},'source_role':{'type':'keyword'},'window_id':{'type':'keyword'},
                'evidence_id':{'type':'keyword'},'input_sha256':{'type':'keyword'},'text':{'type':'text'},'title':{'type':'text'},
                'vector':{'type':'knn_vector','dimension':1024,'method':{'name':'hnsw','engine':'lucene','space_type':'cosinesimil'}}}}}
        lm={'settings':{'number_of_shards':1,'number_of_replicas':0},'mappings':{'properties':{
            'bundle_id':{'type':'keyword'},'product_id':{'type':'keyword'},'source_id':{'type':'keyword'},
            'evidence_id':{'type':'keyword'},'text':{'type':'text'},'title':{'type':'text'}}}}
        for index,body in ((self.text_index,mapping),(self.layout_index,lm)):
            try:self._os(index)
            except RuntimeError as e:
                if str(e)!='BACKEND_HTTP_404':raise
                self._os(index,'PUT',body)
        import numpy as np
        cache=self.knowledge.corpus.root/'runtime/full-rag/vector-cache';cache.mkdir(parents=True,exist_ok=True)
        rows=list(self.rows.values())
        for start in range(0,len(rows),128):
            batch=rows[start:start+128]
            key=sha256((self.knowledge.corpus.model_fingerprint+''.join(r['input_sha256'] for r in batch)).encode()).hexdigest()
            target=cache/(key+'.npz')
            if target.exists():
                with np.load(target,allow_pickle=False) as saved:vectors=saved['vectors']
            else:
                vectors=np.asarray(models.encode([r['text'] for r in batch]),dtype=np.float32)
                np.savez_compressed(target,vectors=vectors)
            if vectors.shape!=(len(batch),1024) or not np.isfinite(vectors).all():raise ValueError('FULL_VECTOR_ALIGNMENT')
            bulk=[]
            for row,vector in zip(batch,vectors):
                bulk.extend([_json({'index':{'_index':self.text_index,'_id':row['window_id']}}),
                             _json({**row,'bundle_id':self.bundle,'vector':vector.tolist()})])
            response=self._os('_bulk?refresh=true','POST','\n'.join(bulk)+'\n','application/x-ndjson')
            if response.get('errors'):raise RuntimeError('FULL_TEXT_BULK_ERROR')
            print(_json({'stage':'embedding_and_index','completed':min(start+len(batch),len(rows)),'total':len(rows)}),flush=True)
        layout=[]
        for sid,business in self.knowledge.business.items():
            for u in business['units']:
                if u['kind'].startswith('TABLE') or u['kind']=='FOOTNOTE_CANDIDATE':
                    layout.append({'evidence_id':u['unit_id'],'source_id':sid,'product_id':self.knowledge.sources[sid].version_id,
                                   'text':u['projection_text'],'title':u['title'],'kind':u['kind']})
        for start in range(0,len(layout),100):
            bulk=[]
            for row in layout[start:start+100]:bulk.extend([_json({'index':{'_index':self.layout_index,'_id':row['evidence_id']}}),_json({**row,'bundle_id':self.bundle})])
            if bulk and self._os('_bulk?refresh=true','POST','\n'.join(bulk)+'\n','application/x-ndjson').get('errors'):
                raise RuntimeError('FULL_LAYOUT_BULK_ERROR')
        nodes=[];edges=[]
        self._neo('CREATE INDEX full_unit_release IF NOT EXISTS FOR (u:FullUnit) ON (u.bundle_id,u.unit_id)',{})
        for sid,business in self.knowledge.business.items():
            pid=self.knowledge.sources[sid].version_id
            for u in business['units']:
                nodes.append({'unit_id':u['unit_id'],'product_id':pid,'source_id':sid,'title':u['title'],'kind':u['kind'],
                              'aliases':source_aliases(u),'eligible':u.get('retrieval_eligible',False)})
                if u['parent_id']:edges.append({'from_id':u['unit_id'],'to_id':u['parent_id'],'product_id':pid,'kind':'PARENT','proof':'SOURCE_STRUCTURE'})
                for context in u['shared_context_refs']:edges.append({'from_id':u['unit_id'],'to_id':context,'product_id':pid,'kind':'SHARED','proof':'SOURCE_STRUCTURE'})
            for edge in business['dependencies']:
                if edge['to_unit_id']:edges.append({'from_id':edge['from_unit_id'],'to_id':edge['to_unit_id'],'product_id':pid,
                    'kind':edge['relation'],'proof':_json(edge['source_spans'])})
        with self._pg() as c:
            for row in nodes:self._put(c,row['product_id'],'full_node',row['unit_id'],row)
        for start in range(0,len(nodes),200):
            self._neo('''UNWIND $rows AS row MERGE (u:FullUnit {bundle_id:$bundle,unit_id:row.unit_id})
                SET u.product_id=row.product_id,u.source_id=row.source_id,u.title=row.title,u.kind=row.kind,
                    u.aliases=row.aliases,u.eligible=row.eligible''',{'bundle':self.bundle,'rows':nodes[start:start+200]})
        for start in range(0,len(edges),200):
            self._neo('''UNWIND $rows AS row MATCH (a:FullUnit {bundle_id:$bundle,unit_id:row.from_id,product_id:row.product_id})
                MATCH (b:FullUnit {bundle_id:$bundle,unit_id:row.to_id,product_id:row.product_id})
                MERGE (a)-[r:FULL_CONTEXT {bundle_id:$bundle,kind:row.kind}]->(b)
                SET r.proof=row.proof,r.state='SOURCE_BOUND_CANDIDATE' ''',{'bundle':self.bundle,'rows':edges[start:start+200]})
        audit=self.audit()
        self._state('READY',audit)
        return audit

    def audit(self):
        observed={};response=self._os(self.text_index+'/_search?scroll=1m','POST',{'size':500,'query':{'match_all':{}},'_source':['window_id','input_sha256','product_id','source_id','text']})
        while True:
            hits=response.get('hits',{}).get('hits',[])
            if not hits:break
            for h in hits:
                row=h['_source']
                if row['window_id'] in observed:raise ValueError('FULL_INDEX_DUPLICATE_ID')
                observed[row['window_id']]=row
            response=self._os('_search/scroll','POST',{'scroll':'1m','scroll_id':response['_scroll_id']})
        fields=('window_id','input_sha256','product_id','source_id','text')
        verify_snapshot({k:{f:r[f] for f in fields} for k,r in self.rows.items()},observed,'FULL_INDEX')
        vectors=self._os(self.text_index+'/_count','POST',{'query':{'exists':{'field':'vector'}}})['count']
        mapping=self._os(self.text_index+'/_mapping')[self.text_index]['mappings']['properties']['vector']
        if vectors!=len(self.rows) or mapping['dimension']!=1024:raise ValueError('FULL_VECTOR_COVERAGE')
        expected_layout={};expected_nodes={};expected_edges={}
        for sid,business in self.knowledge.business.items():
            pid=self.knowledge.sources[sid].version_id
            for u in business['units']:
                uid=u['unit_id']
                expected_nodes[uid]=[pid,sid,u['title'],u['kind'],source_aliases(u),u.get('retrieval_eligible',False)]
                if u['kind'].startswith('TABLE') or u['kind']=='FOOTNOTE_CANDIDATE':
                    expected_layout[uid]={'evidence_id':uid,'source_id':sid,'product_id':pid,'text':u['projection_text'],'title':u['title'],'kind':u['kind'],'bundle_id':self.bundle}
                if u['parent_id']:expected_edges[(uid,u['parent_id'],'PARENT')]='SOURCE_STRUCTURE'
                for context in u['shared_context_refs']:expected_edges[(uid,context,'SHARED')]='SOURCE_STRUCTURE'
            for e in business['dependencies']:
                if e['to_unit_id']:expected_edges[(e['from_unit_id'],e['to_unit_id'],e['relation'])]=_json(e['source_spans'])
        actual_layout={};response=self._os(self.layout_index+'/_search?scroll=1m','POST',{'size':500,'query':{'match_all':{}}})
        while response.get('hits',{}).get('hits',[]):
            for hit in response['hits']['hits']:actual_layout[hit['_id']]=hit['_source']
            response=self._os('_search/scroll','POST',{'scroll':'1m','scroll_id':response['_scroll_id']})
        verify_snapshot(expected_layout,actual_layout,'FULL_LAYOUT')
        nodes=self._neo('''MATCH (u:FullUnit {bundle_id:$bundle})
            RETURN u.unit_id,u.product_id,u.source_id,u.title,u.kind,u.aliases,u.eligible''',{'bundle':self.bundle})
        actual_nodes={r['row'][0]:r['row'][1:] for result in nodes for r in result['data']}
        verify_snapshot(expected_nodes,actual_nodes,'FULL_GRAPH_NODE')
        edges=self._neo('''MATCH (a)-[r:FULL_CONTEXT {bundle_id:$bundle}]->(b)
            RETURN a.unit_id,b.unit_id,r.kind,r.proof,a.product_id,b.product_id,a.bundle_id,b.bundle_id''',{'bundle':self.bundle})
        actual_edges={}
        for result in edges:
            for row in result['data']:
                a,b,kind,proof,ap,bp,ab,bb=row['row']
                if ap!=bp or ab!=self.bundle or bb!=self.bundle:raise ValueError('FULL_GRAPH_EDGE_SCOPE')
                key=(a,b,kind)
                if key in actual_edges:raise ValueError('FULL_GRAPH_DUPLICATE_EDGE')
                actual_edges[key]=proof
        verify_snapshot(expected_edges,actual_edges,'FULL_GRAPH_EDGE')
        with self._pg() as c:
            counts=dict(c.execute('SELECT kind,count(*) FROM smallstack_items WHERE bundle_id=%s GROUP BY kind',(self.bundle,)).fetchall())
        expected={'full_source':len(self.knowledge.sources),'full_page':self.knowledge.corpus.counts['pages'],'full_unit':self.knowledge.corpus.counts['units'],
                  'full_window':self.knowledge.corpus.counts['windows'],'full_node':len(expected_nodes),
                  'catalogue_document':len(self.knowledge.sources),'product':len(self.knowledge.products)}
        if any(counts.get(k)!=v for k,v in expected.items()):raise ValueError('FULL_ARCHIVE_COUNT_MISMATCH')
        with self._pg() as c:
            for sid,doc in self.knowledge.corpus.documents.items():
                row=c.execute("SELECT payload FROM smallstack_items WHERE bundle_id=%s AND kind='full_source' AND item_id=%s",(self.bundle,sid)).fetchone()
                if not row or row[0]!={k:doc[k] for k in ('record','build','physical','artifact')}:raise ValueError('FULL_ARCHIVE_SOURCE_MISMATCH')
        return {'release_id':self.bundle,'indexed_windows':len(observed),'archive_counts':counts,
                'layout_objects':len(actual_layout),'graph_nodes':len(actual_nodes),'graph_edges':len(actual_edges),
                'text_index':self.text_index,'layout_index':self.layout_index,'input_alignment_verified':True,
                'semantic_complete':False,'production_ready':False}

    def retrieve_full(self,task,plan,models):
        if self._published_state()!='READY':raise ValueError('FULL_RELEASE_NOT_READY')
        pid=task.version_id;question=task.variants[0].text
        scopes=[{'term':{'bundle_id':self.bundle}},{'term':{'product_id':pid}}]
        routes={};windows={}
        def convert(hits):
            result=[];seen=set()
            for h in hits:
                row=h['_source'];uid=row['evidence_id']
                if row.get('product_id')!=pid:raise ValueError('FULL_RETRIEVAL_SCOPE_MISMATCH')
                if uid not in seen:
                    result.append({'evidence_id':uid,'match_text':row['text'],'window_id':row.get('window_id')});seen.add(uid)
                    if row.get('window_id'):windows.setdefault(uid,row)
            return result
        text=[]
        for variant in task.variants:
            vector=models.encode([variant.text])[0]
            if len(vector)!=1024:raise ValueError('INDEX_EMBEDDING_SPACE_MISMATCH')
            dense=self._os(self.text_index+'/_search','POST',{'size':30,'query':{'knn':{'vector':{'vector':vector,'k':30,'filter':{'bool':{'filter':scopes}}}}}})
            text+=convert(dense['hits']['hits'])
            if variant.channel!='DENSE_TEXT_ONLY':
                bm=self._os(self.text_index+'/_search','POST',{'size':30,'query':{'bool':{'filter':scopes,'must':[{'match':{'text':variant.text}}]}}})
                text+=convert(bm['hits']['hits'])
        unique={r['evidence_id']:r for r in text};text=list(unique.values())[:task.budget['text']]
        routes['text']={'status':'OK' if text else 'EMPTY','backend':'OpenSearch/BGE-M3','candidates':text}
        with self._pg() as c:
            records=c.execute('''SELECT payload FROM smallstack_items WHERE bundle_id=%s AND product_id=%s AND kind='full_node'
                AND (payload->>'eligible')::boolean AND EXISTS(SELECT 1 FROM jsonb_array_elements_text(payload->'aliases') AS a
                    WHERE a=ANY(%s) OR position(a in %s)>0) ORDER BY length(payload->>'title'),item_id LIMIT %s''',
                (self.bundle,pid,[task.topic],question,task.budget['structured'])).fetchall()
        structured=[{'evidence_id':r[0]['unit_id']} for r in records]
        routes['structured']={'status':'OK' if structured else 'EMPTY','backend':'PostgreSQL/source_headings','candidates':structured}
        layout=self._os(self.layout_index+'/_search','POST',{'size':task.budget['layout'],'query':{'bool':{'filter':scopes,
            'must':[{'multi_match':{'query':question,'fields':['title^2','text']}}]}}})
        spatial=convert(layout['hits']['hits'])
        routes['layout']={'status':'OK' if spatial else 'EMPTY','backend':'OpenSearch/source_layout','candidates':spatial}
        graph=self._neo('''MATCH (s:FullUnit {bundle_id:$bundle,product_id:$product}) WHERE s.eligible AND
            any(a IN s.aliases WHERE a IN $terms OR $question CONTAINS a)
            WITH s ORDER BY s.unit_id LIMIT 40
            MATCH path=(s)-[:FULL_CONTEXT*0..2]->(u:FullUnit {bundle_id:$bundle,product_id:$product})
            WHERE u.eligible AND all(n IN nodes(path) WHERE n.bundle_id=$bundle AND n.product_id=$product)
            RETURN DISTINCT u.unit_id ORDER BY u.unit_id LIMIT $limit''',
            {'bundle':self.bundle,'product':pid,'terms':[task.topic],'question':question,'limit':task.budget['graph']})
        graph_rows=[{'evidence_id':item['row'][0]} for data in graph for item in data['data']]
        routes['graph']={'status':'OK' if graph_rows else 'EMPTY','backend':'Neo4j/source_bound_context','candidates':graph_rows}
        fused=fuse_routes(routes);candidates=[]
        for row in fused[:60]:
            uid=row['evidence_id'];unit=self.knowledge.unit_metadata[uid]
            window=windows.get(uid)
            if window is None:
                window=next((r for r in self.rows.values() if r['evidence_id']==uid),None)
            if window:
                candidates.append({**row,'text':window['body_text'] if 'body_text' in window else window['text'],
                                   'match_text':window['text'],'window_id':window.get('window_id')})
        ranked=models.rerank(question,candidates)
        if any(r.get('rerank_status')!='OK' for r in ranked):raise ValueError('FULL_RERANK_INCOMPLETE')
        return {'routes':routes,'candidate_ids':[r['evidence_id'] for r in ranked],
                'matches':[{k:r.get(k) for k in ('evidence_id','window_id','rerank_score')} for r in ranked],
                'backend_mode':'PUBLIC_FULL_CORPUS_BGE','diagnostics':[],'release_id':self.bundle,
                'models':{'embedding':'BAAI/bge-m3','reranker':'BAAI/bge-reranker-v2-m3','visual_embedding_executed':False}}
