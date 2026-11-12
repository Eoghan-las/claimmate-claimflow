"""One LangGraph execution per turn; atomic business commit precedes delivery."""
from time import perf_counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from rag_assist.localstack.app import _grounded
from .planning import add_hypotheses, make_plan
from .contracts import Understanding,QueryVariant
from .state import Conflict


class Turn(TypedDict, total=False):
    actor: object
    request: object
    memory: dict
    plan: object
    diagnostics: list
    retrieval: list
    evidence: list
    results: list
    output: dict
    condition_package: dict


class RagService:
    def __init__(self, store, knowledge, gateway=None):
        self.store, self.knowledge, self.gateway = store, knowledge, gateway
        graph = StateGraph(Turn)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("answer", self._answer)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", END)
        self.graph = graph.compile()

    @staticmethod
    def _model_contexts(evidence):
        contexts=[]
        ordered=sorted(evidence,key=lambda e:(e["role"]!="CONTEXT",e["version_id"],e["page_number"],e["start"]))
        for e in ordered:
            if any(p["source_id"]==e["source_id"] and p["version_id"]==e["version_id"] and
                   p["start"]<=e["start"] and e["end"]<=p["end"] and
                   p["text"][e["start"]-p["start"]:e["end"]-p["start"]]==e["text"] for p in contexts):
                continue
            contexts.append({**e,"evidence_id":"E"+str(len(contexts)+1),"title":e["filename"],
                             "canonical_start":e["start"],"raw_pdf_sha256":e["raw_file_sha256"]})
        return contexts

    def _plan(self, state):
        understanding,diagnostics=None,[]
        if self.gateway is not None and hasattr(self.gateway,"interpret"):
            try:
                history=[]
                for turn in self.store.history(state['actor'],'chat',state['request'].conversation_id):
                    safe={'question':str(turn.get('question',''))[:2400],'answer':''}
                    try:
                        self.knowledge.validate_output(state['actor'],turn)
                        if all(e.get('data_class')=='PUBLIC' and self.knowledge.sources[e['source_id']].data_class=='PUBLIC'
                               for e in turn.get('evidence',[])):
                            safe['answer']=str(turn.get('answer',''))[:2400]
                    except (ValueError,PermissionError):pass
                    history.append(safe)
                memory={**state["memory"],"recent_turns":history}
                understanding=Understanding.model_validate(self.gateway.interpret(state["request"].question,memory,
                    [p.model_dump(mode="json") for p in self.knowledge.catalog(state["actor"])]))
            except Exception:
                diagnostics.append("MODEL_PLAN_UNAVAILABLE_OR_INVALID")
        plan = make_plan(self.knowledge,state["actor"],state["request"],state["memory"],understanding)
        diagnostics += add_hypotheses(plan,self.knowledge,self.gateway,actor=state['actor'])
        if getattr(self.knowledge,'preserve_original_query',False):
            for task in plan.subtasks:
                if len(task.variants)<3 and not any(v.kind=='ORIGINAL' for v in task.variants):
                    task.variants.append(QueryVariant(kind='ORIGINAL',text=state['request'].question))
        return {"plan":plan,"diagnostics":diagnostics}

    def _retrieve(self, state):
        plan=state["plan"]
        if plan.intent != "PRODUCT_QA":
            return {"retrieval":[],"evidence":[],"results":[]}
        retrieval, results, roots, diagnostics = [], [], list(plan.anchor_ids), list(state["diagnostics"])
        try:
            for task in plan.subtasks:
                found=self.knowledge.search(state["actor"],task,plan)
                retrieval.append({"version_id":task.version_id,"topic":task.topic,**found})
                if set(found["routes"]) != {"text","structured","layout","graph"} or any(
                        r["status"] not in {"OK","EMPTY"} for r in found["routes"].values()):
                    raise ValueError("RETRIEVAL_CHANNEL_FAILED")
                diagnostics.extend(found.get("diagnostics",[]))
                candidates=[]
                for uid in found["candidate_ids"]:
                    u=self.knowledge.units.get(uid)
                    if u is None or u.version_id != task.version_id:
                        raise ValueError("RETRIEVAL_SCOPE_MISMATCH")
                    if uid not in plan.exclude_primary_ids and (getattr(self.knowledge,'soft_topic_labels',False) or plan.related_search or
                            (task.topic in u.topics and u.role in {"CLAUSE","TABLE"})):
                        candidates.append(u)
                unclassified=False
                if not candidates:
                    # Labels aid discovery; missing labels cannot veto actual scoped originals.
                    candidates=[self.knowledge.units[uid] for uid in found["candidate_ids"] if
                        uid not in plan.exclude_primary_ids and self.knowledge.units[uid].role!="CONTEXT"]
                    unclassified=bool(candidates)
                    if unclassified:diagnostics.append("UNCLASSIFIED_SEARCH_CANDIDATES")
                if not candidates:
                    raise ValueError("PRIMARY_EVIDENCE_MISSING")
                if 'selected_primary_ids' in found:
                    selected=set(found['selected_primary_ids'])
                    primary=[u for u in candidates if u.unit_id in selected]
                else:
                    limit=getattr(self.knowledge,'primary_limit',2 if plan.related_search else 1)
                    primary=candidates[:limit]
                for u in primary:
                    if u.unit_id in {r["unit_id"] for r in results}:
                        continue
                    p=self.knowledge.products[u.version_id]
                    results.append({"unit_id":u.unit_id,"version_id":u.version_id,"family_id":p.family_id,
                                    "title":p.title,"topic":task.topic,
                                    "match_status":"UNCLASSIFIED_SEARCH_CANDIDATE" if unclassified else "REGISTERED_TOPIC_CANDIDATE"})
                    roots.append(u.unit_id)
            if hasattr(self.knowledge,'package'):
                packet=self.knowledge.package(state['actor'],list(dict.fromkeys(roots)),plan.as_of,plan.plan_id)
                scoped=getattr(self.knowledge,'scoped_evidence',None)
                if scoped is not None and self.gateway is not None and hasattr(self.gateway,'select_evidence'):
                    legacy=packet
                    try:
                        packet=scoped.package(state['actor'],plan,retrieval,state['request'].question,self.gateway,legacy=legacy)
                        diagnostics=[d for d in diagnostics if d!='EVIDENCE_SELECTION_BUDGET_EXCEEDED']
                        if legacy.get('missing_info'):diagnostics.append('LEGACY_CLOSURE_REPLACED_BY_SCOPED_PROOFS')
                    except Exception as error:
                        packet={**legacy,'generation_allowed':False,'missing_info':sorted(set(legacy['missing_info']+['SCOPED_EVIDENCE_FAILED']))}
                        diagnostics.append('SCOPED_EVIDENCE_FAILED')
                        code=str(error)
                        if isinstance(error,ValueError):diagnostics.append(code)
                elif 'EVIDENCE_SELECTION_BUDGET_EXCEEDED' in diagnostics:
                    packet['generation_allowed']=False
                    packet['missing_info']=sorted(set(packet['missing_info']+['EVIDENCE_SELECTION_BUDGET_EXCEEDED']))
                evidence=packet['evidence']
            else:
                evidence=self.knowledge.hydrate(state["actor"],list(dict.fromkeys(roots)),plan.as_of,plan.plan_id)
                packet={}
            return {"retrieval":retrieval,"results":results,"evidence":evidence,"diagnostics":diagnostics,'condition_package':packet}
        except (ValueError,PermissionError,RuntimeError,OSError,TimeoutError) as error:
            if isinstance(error,PermissionError):
                raise
            code=str(error) if isinstance(error,ValueError) else "RETRIEVAL_UNAVAILABLE"
            return {"retrieval":retrieval,"results":[],"evidence":[],"diagnostics":diagnostics+[code]}

    def _answer(self, state):
        plan=state["plan"]
        base={"question":state["request"].question,"plan":plan.model_dump(mode="json"),
              "release_id":self.knowledge.release_id,"authorization_revision":state["actor"].authorization_revision,
              "evidence":state["evidence"],"results":state["results"],"retrieval":state["retrieval"],
              "diagnostics":state["diagnostics"],"missing_info":plan.missing_info,
              "semantic_complete":False,"requires_human_review":True,"core_signable":False,
              "model_mode":"EXTRACTIVE_NO_LLM","citations":[],
              "completeness":"REGISTERED_DEPENDENCIES_ONLY"}
        base["scope_validity_verified"]=all(self.knowledge.products[v].validity_verified for v in plan.version_ids)
        if plan.intent == "CLARIFY":
            return {"output":{**base,"status":"CLARIFY","answer":"请补充或选择："+"、".join(plan.missing_info)}}
        if plan.intent == "CASE_ASSIST":
            return {"output":{**base,"status":"HUMAN_REVIEW","answer":"需要已授权的规则和经核实的案件事实，请交人工核保。"}}
        if plan.intent == "GENERAL":
            answer="通用概念答疑需要结合具体保险责任理解；查询产品参数请补充产品、版本及计划。"
            if self.gateway is not None and hasattr(self.gateway,"general"):
                try:
                    answer=self.gateway.general(state["request"].question)["answer"]
                    base["model_mode"]="BASE_MODEL_GENERAL"
                except Exception:
                    base["diagnostics"].append("GENERAL_MODEL_UNAVAILABLE")
            return {"output":{**base,"status":"GENERAL","answer":answer,"missing_info":["具体产品版本及计划"]}}
        if not state["evidence"]:
            return {"output":{**base,"status":"HUMAN_REVIEW","answer":"证据或必要上下文不齐，未形成业务回答，请核对原文。"}}
        citations=[{k:e[k] for k in ("unit_id","source_id","version_id","filename","page_number","start","end","content_sha256")}
                   for e in state["evidence"]]
        answer="\n\n".join(f"[{e['unit_id']}] {e['text']}" for e in state["evidence"])
        output={**base,"status":"SOURCE_EVIDENCE","answer":answer,"citations":citations}
        if state.get('condition_package'):
            packet=state['condition_package']
            output.update(condition_packages=packet['condition_packages'],generation_allowed=packet['generation_allowed'],
                          missing_info=sorted(set(output['missing_info']+packet['missing_info'])),completeness=packet['completeness'])
            if 'risk_dispositions' in packet:
                output.update(risk_dispositions=packet['risk_dispositions'],scope_note=packet['scope_note'],
                              context_characters=packet['context_characters'],legacy_context_characters=packet.get('legacy_context_characters'))
                output['evidence_review_attempts']=packet.get('evidence_review_attempts',[])
            if not packet['generation_allowed']:
                output['diagnostics'].append('CONDITION_PACKAGE_REQUIRES_REVIEW')
                return {'output':output}
        if self.gateway is not None and hasattr(self.gateway,"answer"):
            try:
                contexts=self._model_contexts(state["evidence"])
                feedback=None
                for attempt in range(2):
                    scoped=getattr(self.knowledge,'scoped_evidence',None)
                    if scoped is not None and any('scoped_page_id' in e for e in state['evidence']):
                        scoped.validate_evidence(state['actor'],plan,state['evidence'])
                    draft=self.gateway.answer(plan.normalized_question,contexts,**({"feedback":feedback} if feedback else {}))
                    try:
                        claims, generated_citations=_grounded(draft,contexts)
                        break
                    except ValueError as error:
                        if attempt or str(error) not in {"QUOTE_NOT_IN_SOURCE","UNSUPPORTED_NUMBER","EXPLICIT_QUALIFIER_OMITTED","CITATION_OUTSIDE_AUDITED_SUPPORT"}:
                            raise
                        feedback={"error":str(error),"instruction":"逐字复制对应来源audited_quotes中的连续原文；每个数字与前置限定须在同条引文和解释中保留，不猜测或拼接。"}
                output["validation_retry_count"]=attempt
                if scoped is not None and any('scoped_page_id' in e for e in state['evidence']):
                    scoped.validate_evidence(state['actor'],plan,state['evidence'])
                verification=self.gateway.verify(plan.normalized_question,draft,contexts)
                output['verification']={'method':'MODEL_SECOND_PASS','semantic_guarantee':False,**verification}
                checks=verification.get("checks",[])
                if (verification.get("sufficient_to_answer") is not True or verification.get("missing_conditions",[]) or
                        len(checks)!=len(claims) or {c.get("claim_index") for c in checks}!=set(range(len(claims))) or
                        any(any(c.get(k) is not True for k in ("supported","product_plan_match","conditions_preserved")) for c in checks)):
                    raise ValueError("MODEL_SUPPORT_CHECK_FAILED")
                output.update(status="ANSWERED",answer="\n\n".join(c["text"] for c in claims),claims=claims,
                              citations=generated_citations,model_mode="GROUNDED_MODEL",model=self.gateway.model,
                              verification={"method":"MODEL_SECOND_PASS","semantic_guarantee":False,**verification})
            except Exception as error:
                output["diagnostics"].append("MODEL_DRAFT_WITHHELD")
                code=str(error)
                import re
                if re.fullmatch(r"[A-Z_0-9]{3,100}",code):output["diagnostics"].append(code)
                # Retain verifiable originals; never release an invalid generated draft.
        return {"output":output}

    def ask(self, actor, request):
        if "UW" not in actor.roles:
            raise PermissionError("ROLE_NOT_ALLOWED")
        payload=request.model_dump(mode="json")
        cached=self.store.cached(actor,"chat",request.conversation_id,request.request_id,payload)
        if cached:
            self.knowledge.validate_output(actor,cached)
            return cached
        revision,memory=self.store.load(actor,"chat",request.conversation_id)
        if revision != request.expected_revision:
            raise Conflict("REVISION_CONFLICT")
        started=perf_counter()
        state=self.graph.invoke({"actor":actor,"request":request,"memory":memory})
        output=state["output"]
        self.knowledge.validate_output(actor,output)
        next_memory=dict(memory,pending_clarification=output["missing_info"])
        if output["status"] in {"SOURCE_EVIDENCE","ANSWERED"}:
            plan=state["plan"]
            next_memory.update(version_ids=plan.version_ids,plan_id=plan.plan_id,as_of=plan.as_of.isoformat(),
                topics=plan.topics,results=output["results"],release_id=self.knowledge.release_id,
                presented_families=sorted(set(memory.get("presented_families",[]))|{r["family_id"] for r in output["results"]}),
                presented_units=sorted(set(memory.get("presented_units",[]))|{r["unit_id"] for r in output["results"]}))
        output["elapsed_ms"]=round((perf_counter()-started)*1000,2)
        return self.store.commit(actor,"chat",request.conversation_id,revision,request.request_id,payload,next_memory,output)
