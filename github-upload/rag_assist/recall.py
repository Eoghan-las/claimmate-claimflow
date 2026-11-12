"""Four independent evidence producers and provenance-preserving fusion."""
import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from chunker.indexer import tokenize
from chunker.structure import cn2int

from . import ROOT
from .knowledge import article_of,digest

CHANNELS=("text","structured","layout","graph")


def article_numbers(question):
    numbers={cn2int(x) for x in re.findall(r'第([一二三四五六七八九十百]+)条',question)}
    for match in re.finditer(r'第([一二三四五六七八九十百]+)条\s*(?:到|至|—|-)\s*第([一二三四五六七八九十百]+)条',question):
        a,b=map(cn2int,match.groups())
        if not a or a>b or b-a>100:
            raise ValueError('invalid article range')
        numbers.update(range(a,b+1))
    return numbers


class FourRouteRecall:
    def __init__(self,db_path,units,indexes,edges,sources):
        self.db_path=Path(db_path)
        self.units,self.indexes,self.edges,self.sources=units,indexes,edges,sources
        with self.connect() as db:
            required={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'layout_blocks','graph_entities'}<=required:
                raise ValueError('four-route index schema missing: build a new bundle')
            self.manifest=json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
        self.incoming={}
        for src,targets in edges.items():
            for dst,role in targets:
                if role=='DEFINITION':
                    self.incoming.setdefault(dst,[]).append((src,'DEFINITION_USED_BY'))

    def connect(self):
        db=sqlite3.connect(self.db_path,timeout=15)
        db.row_factory=sqlite3.Row
        return db

    def allowed(self,uid,tenant,corpus,version):
        u=self.units.get(uid)
        return bool(u and tenant and u['tenant']==tenant and u['corpus']==corpus and u['version']==version)

    def structured(self,q,tenant,corpus,version):
        numbers=article_numbers(q)
        kinds=[]
        for pattern,kind in [(r'免责|责任免除','EXCLUSION'),(r'等待期','WAITING'),(r'比例|社保|耗材','TABLE'),(r'免赔额','DEDUCTIBLE'),(r'定义|释义','DEFINITION')]:
            if re.search(pattern,q): kinds.append(kind)
        all_clauses=bool(re.search(r'(全部|所有)(保险)?条款|条款清单|条款全文',q))
        if not numbers and not kinds and not all_clauses:
            return {'candidates':[],'query_kind':'PARAMETERIZED_SQL'}
        sql='SELECT unit_id,local_id,unit_type FROM units WHERE tenant=? AND corpus=? AND version=? AND searchable=1'
        values=[tenant,corpus,version]
        if not numbers and kinds:
            sql+=' AND unit_type IN ('+','.join('?' for _ in kinds)+')';values+=kinds
        sql+=' ORDER BY rowid'
        with self.connect() as db: rows=[dict(r) for r in db.execute(sql,values)]
        if numbers: rows=[r for r in rows if cn2int(article_of(r['local_id'])) in numbers]
        return {'candidates':[{'unit_id':r['unit_id'],'score':1.0,'reason':'EXACT_ARTICLE' if numbers else 'REGISTERED_METADATA'} for r in rows],
                'query_kind':'PARAMETERIZED_SQL','requested_articles':sorted(numbers)}

    def layout(self,q,tenant,corpus,version):
        with self.connect() as db:
            rows=[dict(r) for r in db.execute('SELECT * FROM layout_blocks WHERE tenant=? AND corpus=? AND version=?',(tenant,corpus,version))]
        if not rows or not re.search(r'表|比例|百分|社保|耗材|脚注|自费药|目录',q):
            return {'candidates':[],'tables':[]}
        texts=[]
        for row in rows:
            block=json.loads(row['data_json'])
            texts.append(' '.join(block['headers'])+' '+json.dumps(block['rows'],ensure_ascii=False)+' '+block['footnote'])
        scores=np.asarray(BM25Okapi([tokenize(t) for t in texts]).get_scores(tokenize(q)))
        # A single-table BM25 corpus can have negative IDF; rank and lexical
        # overlap decide whether there is evidence, never score>0 alone.
        overlap=[len(set(tokenize(q)) & set(tokenize(t))) for t in texts]
        order=sorted(range(len(rows)),key=lambda i:(-overlap[i],-float(scores[i])))[:8]
        candidates,tables=[],[]
        for i in order:
            if not overlap[i]: continue
            row=rows[i];block=json.loads(row['data_json'])
            candidates.append({'unit_id':row['unit_id'],'score':float(overlap[i]),'reason':'HEADER_CELL_FOOTNOTE_MATCH'})
            tables.append({'block_id':row['block_id'],'evidence_id':row['unit_id'],**block})
        return {'candidates':candidates,'tables':tables}

    def graph(self,q,tenant,corpus,version):
        with self.connect() as db:
            entities=[dict(r) for r in db.execute('SELECT * FROM graph_entities WHERE tenant=? AND corpus=? AND version=?',(tenant,corpus,version))]
        seeds=[e for e in entities if any(alias in q for alias in json.loads(e['aliases_json']))]
        candidates,paths,seen=[],[],set()
        queue=[(e['seed_unit_id'],[e['entity_id'],e['seed_unit_id']],0) for e in seeds]
        while queue and len(seen)<60:
            uid,path,depth=queue.pop(0)
            if uid in seen or not self.allowed(uid,tenant,corpus,version): continue
            seen.add(uid)
            candidates.append({'unit_id':uid,'score':1/(depth+1),'reason':'ENTITY_SEED' if depth==0 else 'RELATION_PATH'})
            paths.append({'unit_id':uid,'path':path,'hops':depth})
            if depth<2:
                for target,role in self.edges.get(uid,[])+self.incoming.get(uid,[]):
                    if self.allowed(target,tenant,corpus,version): queue.append((target,path+[role,target],depth+1))
        return {'candidates':candidates,'paths':paths,'matched_entities':[e['label'] for e in seeds],'truncated':bool(queue)}

    def retrieve(self,q,*,tenant,corpus,version,as_of='2026-11-01',channels=None):
        selected=tuple(CHANNELS if channels is None else channels)
        if any(name not in CHANNELS for name in selected) or len(set(selected))!=len(selected):
            raise ValueError('unknown or duplicate retrieval channel')
        source=self.sources.get(corpus)
        if not source or not tenant or tenant!=source['tenant'] or version!=source['version'] or as_of<source['effective_from']:
            return {'fused':[],'channels':{name:{'status':'DENIED','candidates':[],'candidate_count':0,'elapsed_ms':0} for name in selected},'plan':{'channels':list(selected)}}
        result={}
        for name in selected:
            t0=time.perf_counter()
            if name=='text':
                value={'candidates':self.indexes[corpus].search(q,k=20),'backend':getattr(self.indexes[corpus],'backend','BM25+TFIDF')}
                value['query_encoding']='ONLINE_MODEL_INFERENCE' if getattr(self.indexes[corpus],'encoder',None) else 'TFIDF_TRANSFORM'
                value['query_encoding_ms']=self.indexes[corpus].last_query_encoding_ms
            else: value=getattr(self,name)(q,tenant,corpus,version)
            value['candidates']=[x for x in value['candidates'] if self.allowed(x['unit_id'],tenant,corpus,version)]
            value.update(status='OK' if value['candidates'] else 'EMPTY',candidate_count=len(value['candidates']),elapsed_ms=round((time.perf_counter()-t0)*1000,3))
            result[name]=value
        merged={}
        for name,value in result.items():
            for rank,candidate in enumerate(value['candidates'],1):
                uid=candidate['unit_id']
                item=merged.setdefault(uid,{'unit_id':uid,'local_id':self.units[uid]['local_id'],'score':0.0,'via':[],'ranks':{}})
                item['score']+=1/(60+rank)
                item['via'].append(name);item['ranks'][name]=rank
        target_types=[];role_groups=[]
        for pattern,kinds in [(r'定义|界定|什么叫',['DEFINITION']),
                              (r'比例|进口耗材|社保',['TABLE','RATIO']),
                              (r'保险责任|保障范围|住院医疗费用保险金|住院津贴|赔不赔|能报',['LIABILITY']),
                              (r'责任免除|免责',['EXCLUSION'])]:
            if re.search(pattern,q):
                target_types.extend(kinds);role_groups.append(kinds)
        for x in merged.values(): x['primary_role_match']=self.units[x['unit_id']]['unit_type'] in target_types
        fused=sorted(merged.values(),key=lambda x:(-int(x['primary_role_match']),-x['score'],min(x['ranks'].values()),x['unit_id']))
        anchors=[]
        for group in role_groups:
            found=next((x for x in fused if self.units[x['unit_id']]['unit_type'] in group),None)
            if found and found['unit_id'] not in {x['unit_id'] for x in anchors}: anchors.append(found)
        anchor_ids={x['unit_id'] for x in anchors}
        fused=anchors+[x for x in fused if x['unit_id'] not in anchor_ids]
        return {'fused':fused,'channels':result,'plan':{'channels':list(selected),'fusion':'ROLE_COVERAGE_THEN_RRF','expected_primary_types':target_types,
                'rank_constant':60,'scope':{'corpus':corpus,'version':version},'authorization':'PRE_RETRIEVAL'}}

    def case_records(self,application_id,tenant,as_of,version=None):
        if not isinstance(application_id,str) or not re.fullmatch(r'[A-Za-z0-9_:-]{1,64}',application_id):
            raise ValueError('invalid application_id')
        file=ROOT/'claimMate/claimMate_uw.db'
        source_hash=digest(file.read_bytes())
        if source_hash!=self.manifest['business_source_sha256']:
            raise ValueError('business source changed after bundle publication')
        if as_of!=self.manifest['business_snapshot_as_of']:
            return {'context':None,'records':[],'scope_error':'HISTORICAL_CASE_SNAPSHOT_UNAVAILABLE'}
        with sqlite3.connect(file.as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            sql='SELECT app_id,product_id,product_version,apply_date FROM application WHERE app_id=? AND tenant_id=? AND apply_date<=?'
            params=[application_id,tenant,as_of]
            if version is not None: sql+=' AND product_version=?';params.append(version)
            context=db.execute(sql,params).fetchone()
            if not context: return {'context':None,'records':[]}
            context=dict(context)
            views={
                'v_uw_case':('SELECT * FROM v_uw_case WHERE app_id=? AND tenant_id=?',[application_id,tenant]),
                'uw_fact':('SELECT f.* FROM uw_fact f JOIN application a ON a.app_id=f.app_id WHERE a.app_id=? AND a.tenant_id=?',[application_id,tenant]),
                'document':('SELECT d.* FROM document d JOIN application a ON a.app_id=d.app_id WHERE a.app_id=? AND a.tenant_id=?',[application_id,tenant]),
                'v_uw_disclosure_detail':('SELECT v.* FROM v_uw_disclosure_detail v JOIN application a ON a.app_id=v.app_id WHERE a.app_id=? AND a.tenant_id=?',[application_id,tenant]),
                'v_uw_decision_trace':('SELECT v.* FROM v_uw_decision_trace v JOIN application a ON a.app_id=v.app_id WHERE a.app_id=? AND a.tenant_id=?',[application_id,tenant]),
                'uw_rule':('SELECT r.* FROM uw_rule r JOIN product p ON p.product_id=r.product_id WHERE p.tenant_id=? AND p.product_id=? AND p.version=?',[tenant,context['product_id'],context['product_version']]),
                'clause':('SELECT c.* FROM clause c JOIN product p ON p.product_id=c.product_id AND p.version=c.product_version WHERE p.tenant_id=? AND p.product_id=? AND p.version=?',[tenant,context['product_id'],context['product_version']])}
            records=[]
            for table,(sql,params) in views.items():
                for row in db.execute(sql,params):
                    values=dict(row);text=json.dumps(values,ensure_ascii=False,sort_keys=True)
                    h=digest(text);uid='SQL:'+digest(tenant+application_id+table+h)
                    records.append({'evidence_id':uid,'local_id':table+':'+h[:12],'corpus':'structured_medical_demo',
                        'version':context['product_version'],'role':'STRUCTURED_RECORD','source_text':text,'anchor':table+':'+h[:12],
                        'source_spans':[{'location_kind':'SQL_ROW_KEY','table':table,'application_id':application_id,'row_hash':h}],
                        'content_sha256':h,'source_sha256':source_hash,'source_classification':'SYNTHETIC',
                        'source_url':'/v1/evidence/'+uid,'applicability':'SOURCE_RECORD_NOT_AUTOMATIC_UW_AUTHORITY','row':values})
        return {'context':context,'records':records}
