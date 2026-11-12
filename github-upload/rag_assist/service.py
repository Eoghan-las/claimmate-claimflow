"""Queryable evidence, source-complete answers, and audited human review."""
import hashlib
import json
import re
import sqlite3
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from .decisions import demo_rules, evaluate_rules, validate_case_facts
from .knowledge import article_of, digest
from chunker.structure import cn2int
from .search import ScopedIndex,BGEEncoder
from .recall import FourRouteRecall


def now():
    return datetime.now(timezone.utc).isoformat()


class InsuranceRAG:
    def __init__(self, db_path: Path,embedding_mode='tfidf'):
        if embedding_mode not in ('tfidf','bge'): raise ValueError('unsupported embedding mode')
        self.db_path = Path(db_path).resolve()
        if not self.db_path.is_file():
            raise ValueError("index missing: run python -m rag_assist build")
        self.lock = threading.RLock()
        self.execution_code_hash=digest(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))))
        with self.connect() as db:
            self.bundle_id = db.execute("SELECT value FROM meta WHERE key='bundle_id'").fetchone()[0]
            self.sources = {r["corpus"]:dict(r) for r in db.execute("SELECT * FROM sources")}
            self.units = {r["unit_id"]:dict(r) for r in db.execute("SELECT * FROM units")}
            self.edges = {}
            for row in db.execute("SELECT * FROM edges"):
                self.edges.setdefault(row["src"],[]).append((row["dst"],row["role"]))
        self.encoder=BGEEncoder() if embedding_mode=='bge' else None
        self.indexes = {name:ScopedIndex([u for u in self.units.values() if u["corpus"]==name and u["searchable"]],
                                        encoder=self.encoder,cache_dir=self.db_path.parent/'embedding-cache')
                        for name in self.sources}
        self.recall=FourRouteRecall(self.db_path,self.units,self.indexes,self.edges,self.sources)

    def connect(self):
        db = sqlite3.connect(self.db_path,timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _authorized(self, unit, tenant, corpus, version):
        return bool(tenant) and unit["tenant"]==tenant and unit["corpus"]==corpus and unit["version"]==version

    def _item(self, unit, role=None):
        return {"evidence_id":unit["unit_id"],"local_id":unit["local_id"],"corpus":unit["corpus"],
                "version":unit["version"],"role":role or unit["role"],"source_text":unit["source_text"],
                "anchor":unit["anchor"],"source_spans":json.loads(unit["spans"]),
                "content_sha256":unit["content_sha256"],"source_sha256":self.sources[unit["corpus"]]["source_hash"],
                "source_classification":self.sources[unit["corpus"]]["classification"],
                "source_url":"/v1/sources/"+unit["corpus"],
                "applicability":"REFERENCE_REQUIRES_FACT_CHECK"}

    def _save(self, result, tenant, kind, request, db=None, actor="local-cli"):
        result["execution_code_hash"]=self.execution_code_hash
        own = db is None
        db = db or self.connect()
        try:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",(result["run_id"],tenant,self.bundle_id,kind,
                       json.dumps({**request,"actor":actor},ensure_ascii=False),json.dumps(result,ensure_ascii=False),now()))
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",(uuid.uuid4().hex,result["run_id"],tenant,
                       "RESPONSE_FINALIZED",json.dumps({"status":result["status"],"bundle_id":self.bundle_id,"actor":actor},ensure_ascii=False),now()))
            if own:
                db.commit()
        finally:
            if own:
                db.close()

    def ask(self, question, *, corpus, tenant, version=None, as_of="2026-11-01",actor="local-cli",application_id=None):
        if not isinstance(question,str) or not question.strip() or len(question)>4000:
            raise ValueError("question must contain 1 to 4000 characters")
        date.fromisoformat(as_of)
        result = {"run_id":uuid.uuid4().hex,"bundle_id":self.bundle_id,"status":"ANSWERED",
                  "route":"B_RULE","question":question,"evidence":[],"parents":[],"citations":[],
                  "ranked_local_ids":[],"completeness":"INCOMPLETE","answer":"", "missing_facts":[],
                  "recommendation":None,"generation_mode":"VERIFIED_EXTRACTIVE_TEMPLATE","trace":[],
                  "completeness_scope":"CONFIGURED_SOURCE_DOCUMENT_CONTEXT; NOT_MEDICAL_UW_SUFFICIENCY",
                  "recall_channels":{},"recall_plan":{},"recall_provenance":{},"structured_records":[],
                  "text_backend":getattr(next(iter(self.indexes.values())),'backend','')}
        source = self.sources.get(corpus)
        if application_id is None and tenant=='T_008':
            references=set(re.findall(r'(?<![A-Za-z0-9_])APP_\d{4}_\d+(?![A-Za-z0-9_])',question))
            if len(references)>1:
                result.update(status='NEED_INFO',answer='请指定本次需要核对的一个投保申请。',missing_facts=['application_id'])
                self._save(result,tenant,'QUERY',{'question':question,'corpus':corpus,'version':version,'as_of':as_of},actor=actor)
                return result
            if references: application_id=next(iter(references))
        if application_id is not None and tenant=='T_008':
            return self._case_question(result,application_id,corpus,tenant,as_of,actor,version=version)
        if not tenant or tenant!="T_008":
            result.update(status="DENIED",answer="无权访问请求的资料。")
        elif not source:
            result.update(status="NEED_INFO",answer="请选择产品资料库和版本。",missing_facts=["product_context"])
        elif version and version!=source["version"]:
            result.update(status="NO_EVIDENCE",answer="指定范围没有可用版本。")
        elif as_of<source["effective_from"]:
            result.update(status="NO_EVIDENCE",answer="该资料在指定业务时点尚未生效。")
        elif re.search(r"核保建议|是否承保|能否承保|能否投保|能买吗|加费|拒保",question):
            result.update(status="NO_EVIDENCE",answer="当前资料库没有经授权的真实核保手册，不能据合同条款给出承保建议。")
        else:
            version = source["version"]
            exact = re.findall(r"第([一二三四五六七八九十百]+)条",question)
            all_articles = not exact and bool(re.search(r"(?:全部|所有)(?:保险)?条款|(?:条款全文|条款清单)",question))
            recall=self.recall.retrieve(question,tenant=tenant,corpus=corpus,version=version,as_of=as_of,
                                        channels=['structured'] if exact or all_articles else None)
            ranked=recall['fused']
            result['recall_channels']=recall['channels']
            result['recall_plan']=recall['plan']
            result['recall_provenance']={r['unit_id']:{'via':r['via'],'ranks':r['ranks'],'rrf_score':r['score']} for r in ranked}
            result["ranked_local_ids"] = [x["local_id"] for x in ranked]
            if not ranked and not exact and not all_articles:
                result.update(status="NO_EVIDENCE",answer="当前产品资料未找到足够依据，请补充条款名称或具体保险问题。")
            else:
                main_ids = [r["unit_id"] for r in ranked[:5]]
                if exact:
                    result["route"]="B_LOOKUP"
                    requested={cn2int(x) for x in exact}
                    for span in re.finditer(r"第([一二三四五六七八九十百]+)条\s*(?:到|至|—|-)\s*第([一二三四五六七八九十百]+)条",question):
                        a,b=map(cn2int,span.groups())
                        if a>b or b-a>100:
                            raise ValueError("invalid article range")
                        requested.update(range(a,b+1))
                    main_ids = [u["unit_id"] for u in self.units.values() if u["searchable"] and
                                self._authorized(u,tenant,corpus,version) and cn2int(article_of(u["local_id"])) in requested]
                    actual={cn2int(article_of(self.units[uid]["local_id"])) for uid in main_ids}
                    if not requested<=actual:
                        result.update(status="NO_EVIDENCE",answer="当前版本没有该条款号。")
                        main_ids=[]
                if all_articles:
                    result["route"]="B_LOOKUP"
                    main_ids=[u["unit_id"] for u in self.units.values() if u["searchable"] and self._authorized(u,tenant,corpus,version)]
                if exact or all_articles:
                    result["ranked_local_ids"]=[self.units[uid]["local_id"] for uid in main_ids]
                if re.search(r"(?:全部|所有|逐项|完整).{0,10}(?:责任免除|免责)|(?:责任免除|免责).{0,10}(?:全部|所有|逐项)",question):
                    exclusion_ids = [u["unit_id"] for u in self.units.values() if u["searchable"] and
                                     u["unit_type"]=="EXCLUSION" and self._authorized(u,tenant,corpus,version)]
                    main_ids = list(dict.fromkeys(main_ids+exclusion_ids))
                seen, queue, roles = set(),list(main_ids),{}
                for uid in main_ids:
                    roles[uid] = self.units[uid]["role"]
                while queue:
                    uid = queue.pop(0)
                    if uid in seen:
                        continue
                    unit = self.units.get(uid)
                    if not unit or not self._authorized(unit,tenant,corpus,version):
                        result.update(status="REVIEW_REQUIRED",answer="必需证据无法在当前权限和版本内取得。")
                        break
                    if len(seen)>=200:
                        result.update(status="REVIEW_REQUIRED",answer="相关证据超出本次预算，需要人工拆分问题。")
                        break
                    seen.add(uid)
                    result["evidence"].append(self._item(unit,roles.get(uid)))
                    for target,role in self.edges.get(uid,[]):
                        roles.setdefault(target,role)
                        queue.append(target)
                        result["trace"].append({"step":"EVIDENCE_EXPANDED","src":uid,"dst":target,"role":role})
                parent_seen = set()
                for uid in seen:
                    pid = self.units[uid]["parent_id"]
                    parent = self.units.get(pid)
                    if parent and pid not in parent_seen:
                        if not self._authorized(parent,tenant,corpus,version):
                            result.update(status="REVIEW_REQUIRED",answer="上下文权限校验失败。")
                            continue
                        parent_seen.add(pid)
                        result["parents"].append(self._item(parent,"PARENT_CONTEXT"))
                if result["status"]=="ANSWERED":
                    result["completeness"]="COMPLETE"
                    prefix = "依据当前版本的原文与相关约束整理如下。条款解释不构成个案核保决定。"
                    if source["classification"]=="SYNTHETIC":
                        prefix="【合成演示条款】"+prefix
                    result["answer"]=prefix+"\n\n"+"\n\n".join("["+e["local_id"]+" · "+e["role"]+"]\n"+e["source_text"] for e in result["evidence"])
                for e in result["evidence"]+result["parents"]:
                    result["citations"].append({"evidence_id":e["evidence_id"],"anchor":e["anchor"],
                                                "quote":e["source_text"],"content_sha256":e["content_sha256"]})
        result["trace"].insert(0,{"step":"SCOPE_PINNED","corpus":corpus,"version":version,"as_of":as_of})
        # Persist the same body the caller receives; failure propagates, never a
        # successful unlogged answer. Sensitive values stay in this local DB.
        self._save(result,tenant,"QUERY",{"question":question,"corpus":corpus,"version":version,"as_of":as_of},actor=actor)
        return result

    def evidence(self, evidence_id, *, tenant):
        unit = self.units.get(evidence_id)
        if not unit:
            with self.connect() as db:
                record=db.execute('SELECT payload_json FROM record_evidence WHERE evidence_id=? AND tenant=?',(evidence_id,tenant)).fetchone()
            if record:
                value=json.loads(record[0])
                if digest(value['source_text'])!=value['content_sha256']: raise ValueError('record integrity failed')
                return value
        if not unit or not tenant or unit["tenant"]!=tenant:
            raise PermissionError("evidence unavailable")
        source = self.sources[unit["corpus"]]
        if digest(unit["source_text"])!=unit["content_sha256"]:
            raise ValueError("source content integrity failed")
        return self._item(unit)

    def audit(self, run_id, *, tenant):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=? AND tenant=?",(run_id,tenant)).fetchone()
            if not row:
                raise PermissionError("run unavailable")
            events = [dict(e) for e in db.execute("SELECT * FROM events WHERE run_id=? AND tenant=? ORDER BY created_at",(run_id,tenant))]
        return {"run_id":run_id,"bundle_id":row["bundle_id"],"actor":json.loads(row["request_json"]).get("actor"),
                "response":json.loads(row["response_json"]),"events":events}

    def evaluate_case(self, case_id, facts, *, tenant, expected_revision,actor="local-cli"):
        if tenant!="T_008" or case_id!="DEMO_CASE_01":
            raise PermissionError("only the explicitly authorized synthetic case is available")
        validate_case_facts(facts)
        rules = demo_rules()
        result = evaluate_rules(rules,facts)
        result.update(run_id=uuid.uuid4().hex,bundle_id=self.bundle_id,case_id=case_id,
                      rule_set="SYNTHETIC_RULEBOOK_V1",fact_snapshot_hash=digest(json.dumps(facts,sort_keys=True)),
                      warning="仅验证合成核保规则执行，不可用于真实承保。")
        with self.lock, self.connect() as db:
            row = db.execute("SELECT * FROM cases WHERE case_id=? AND tenant=?",(case_id,tenant)).fetchone()
            revision = row["revision"] if row else 0
            if expected_revision!=revision:
                raise ValueError("case revision conflict")
            result["case_revision"]=revision+1
            self._save(result,tenant,"CASE_EVALUATION",{"facts":facts,"rules":rules,"case_id":case_id},db,actor=actor)
            db.execute("INSERT INTO cases VALUES (?,?,?,?) ON CONFLICT(case_id) DO UPDATE SET revision=excluded.revision,current_run=excluded.current_run",
                       (case_id,tenant,revision+1,result["run_id"]))
        return result

    def review(self, run_id, action, *, tenant, expected_revision, idempotency_key, reason,actor="local-cli"):
        if not reason.strip() or action not in ("CONFIRM","REQUEST_INFO","ESCALATE") or not idempotency_key:
            raise ValueError("valid review action, reason, and idempotency key required")
        payload = {"run_id":run_id,"action":action,"expected_revision":expected_revision,"reason":reason,"actor":actor}
        request_hash = digest(json.dumps(payload,sort_keys=True))
        with self.lock,self.connect() as db:
            prior = db.execute("SELECT * FROM reviews WHERE tenant=? AND idem_key=?",(tenant,idempotency_key)).fetchone()
            if prior:
                if prior["request_hash"]!=request_hash:
                    raise ValueError("idempotency conflict")
                return json.loads(prior["payload_json"])
            run = db.execute("SELECT * FROM runs WHERE run_id=? AND tenant=? AND kind='CASE_EVALUATION'",(run_id,tenant)).fetchone()
            if not run:
                raise PermissionError("case run unavailable")
            response = json.loads(run["response_json"])
            case = db.execute("SELECT * FROM cases WHERE case_id=? AND tenant=?",(response["case_id"],tenant)).fetchone()
            if case["revision"]!=expected_revision or case["current_run"]!=run_id:
                raise ValueError("stale case revision")
            if action=="CONFIRM" and not response.get("recommendation"):
                raise ValueError("cannot confirm missing or conflicting recommendation")
            event = {"event_id":uuid.uuid4().hex,**payload,"case_revision":case["revision"]+1,
                     "data_class":"SYNTHETIC","core_sync":"LOCAL_DEMO_ONLY"}
            db.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?)",(event["event_id"],run_id,tenant,idempotency_key,request_hash,json.dumps(event,ensure_ascii=False)))
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",(event["event_id"],run_id,tenant,"HUMAN_REVIEW",json.dumps(event,ensure_ascii=False),now()))
            db.execute("UPDATE cases SET revision=revision+1 WHERE case_id=? AND tenant=?",(response["case_id"],tenant))
        return event

    def status(self, *, tenant):
        if tenant!="T_008":
            raise PermissionError("scope unavailable")
        return {"bundle_id":self.bundle_id,"mode":"LOCAL_DEMO","generation_mode":"VERIFIED_EXTRACTIVE_TEMPLATE",
                "retrieval":"FOUR_ROUTE_RRF","text_backend":getattr(next(iter(self.indexes.values())),'backend','BM25+TFIDF'),
                "recall_channels":["text","structured","layout","graph"],
                "embedding_model":{'path':str(self.encoder.path),'encoder_fingerprint':self.encoder.model_sha256,
                                   'dimension':self.encoder.dim,'online_query_calls':self.encoder.query_calls} if self.encoder else None,
                "sources":[{k:v for k,v in s.items() if k not in ("raw_text","path","physical_pages")} for s in self.sources.values()],
                "searchable_units":sum(u["searchable"] for u in self.units.values())}

    def data_query(self,view,filters,*,tenant,select=None,order_by=None,limit=100,actor="local-cli"):
        from .structured import query_data
        result=query_data(tenant,view,filters,select=select,order_by=order_by,limit=limit)
        result.update(run_id=uuid.uuid4().hex,bundle_id=self.bundle_id,status="ANSWERED",route="A_DATA")
        self._save(result,tenant,"DATA_QUERY",{"view":view,"filters":filters,"select":select,"order_by":order_by,"limit":limit},actor=actor)
        return result

    def _case_question(self,result,application_id,corpus,tenant,as_of,actor,version=None):
        found=self.recall.case_records(application_id,tenant,as_of,version=version)
        records=found['records']
        result.update(route='C_CASE',structured_records=records,case_context=found['context'],evidence=records,
                      recall_plan={'channels':['structured'],'reason':'CASE_FACTS_FIRST'},
                      recall_channels={'structured':{'status':'OK' if records else 'EMPTY','candidate_count':len(records),'query_kind':'PARAMETERIZED_13_TABLE_SQL'}},
                      case_scope_error=found.get('scope_error'))
        if not records:
            result.update(status='NO_EVIDENCE',answer='当前授权范围和业务时点没有可用案件资料。')
        else:
            expected=self.recall.manifest.get(corpus,{})
            ctx=found['context']
            mismatch=(ctx['product_version']!=expected.get('version') or ctx['product_id']!=expected.get('product_id'))
            result.update(status='REVIEW_REQUIRED',context_mismatch='CASE_AND_KNOWLEDGE_PRODUCT_VERSION_MISMATCH' if mismatch else None,
                          answer='已取得结构化案件证据；'+('案件与所选知识产品/版本不一致，已阻断跨版本融合。' if mismatch else '真实核保手册尚未发布，需要人工核对。'),
                          completeness_scope='AUTHORIZED_STRUCTURED_CASE_RECORDS_ONLY; NOT_MEDICAL_UW_SUFFICIENCY')
            result['citations']=[{k:e[k] for k in ['evidence_id','anchor','content_sha256']}|{'quote':e['source_text']} for e in records]
        with self.connect() as db:
            for e in records:
                db.execute('INSERT OR IGNORE INTO record_evidence VALUES (?,?,?,?)',(e['evidence_id'],tenant,application_id,json.dumps(e,ensure_ascii=False)))
            self._save(result,tenant,'CASE_QUERY',{'question':result['question'],'application_id':application_id,'corpus':corpus,'as_of':as_of,'version':version},db,actor=actor)
        return result
