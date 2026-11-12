"""Versioned claims preparation: whitelisted tools, complete rules, exact money."""
from copy import deepcopy
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .state import Conflict


def money(value):
    if not isinstance(value,str) or not re.fullmatch(r"(?:0|[1-9]\d{0,10})(?:\.\d{1,2})?",value):
        raise ValueError("INVALID_MONEY")
    return Decimal(value).quantize(Decimal("0.01"))


class ClaimExecution(TypedDict, total=False):
    actor: object
    case: dict
    policy: dict
    tools: list
    evidence: list
    rule_results: list
    calculation: dict | None
    missing_info: list
    diagnostics: list
    status: str
    done: bool


class ClaimsService:
    def __init__(self, store, knowledge, policies, gateway=None):
        self.store,self.knowledge,self.policies,self.gateway=store,knowledge,deepcopy(policies),gateway
        graph=StateGraph(ClaimExecution)
        graph.add_node("tool_step",self._step)
        graph.add_edge(START,"tool_step")
        graph.add_conditional_edges("tool_step",lambda s:END if s["done"] else "tool_step")
        self.graph=graph.compile()

    def _policy(self, actor, policy_id):
        policy=self.policies.get(policy_id)
        if ("CLAIMS" not in actor.roles or not policy or policy["tenant_id"]!=actor.tenant_id or
                actor.user_id not in policy["authorized_users"]):
            raise PermissionError("POLICY_UNAVAILABLE")
        self.knowledge.allowed(actor,policy["version_id"],date.fromisoformat(policy["incident_date"]),policy["plan_id"])
        return deepcopy(policy)

    def _case(self, actor, case_id):
        revision,state=self.store.load(actor,"case",case_id)
        if not revision:
            raise KeyError("CASE_UNAVAILABLE")
        current=self._policy(actor,state["policy_id"])
        if current["snapshot_id"]!=state["policy_snapshot_id"]:
            raise Conflict("POLICY_SNAPSHOT_CHANGED_RECONCILIATION_REQUIRED")
        return revision,state

    @staticmethod
    def _materials(materials, facts, actor):
        rows=[m.model_dump() for m in materials]
        if len({m["material_id"] for m in rows})!=len(rows):
            raise ValueError("DUPLICATE_MATERIAL_ID")
        by_id={m["material_id"]:m for m in rows}
        confirmed={k:v.model_dump() for k,v in facts.items()}
        for name,f in confirmed.items():
            if name not in {"eligible_expense","other_paid","waiting_met","hospital_eligible"}:
                raise ValueError("FACT_FIELD_NOT_REGISTERED")
            if any(mid not in by_id or not by_id[mid]["confirmed"] for mid in f["material_ids"]):
                raise ValueError("FACT_PROVENANCE_UNCONFIRMED")
            if name in {"eligible_expense","other_paid"}:
                money(f["value"])
            elif type(f["value"]) is not bool:
                raise ValueError("BOOLEAN_FACT_REQUIRED")
            f["confirmed_by"]=actor.user_id
        for m in rows:
            m["content_sha256"]=sha256(m["text"].encode()).hexdigest()
            m["confirmed_by"]=actor.user_id if m["confirmed"] else None
        return rows,confirmed

    def create(self, actor, request):
        policy=self._policy(actor,request.policy_id)
        payload=request.model_dump(mode="json")
        cached=self.store.cached(actor,"case",request.case_id,request.request_id,payload)
        if cached:
            return cached
        rows,facts=self._materials(request.materials,request.confirmed_facts,actor)
        state={"case_id":request.case_id,"policy_id":request.policy_id,"policy_snapshot_id":policy["snapshot_id"],
               "material_revision":1,"materials":rows,"confirmed_facts":facts,"last_report":None,
               "status":"READY","release_id":self.knowledge.release_id}
        return self.store.commit(actor,"case",request.case_id,0,request.request_id,payload,state,
                                 {"case_id":request.case_id,"status":"READY","material_revision":1})

    def replace_materials(self, actor, case_id, request):
        revision,state=self._case(actor,case_id)
        payload=request.model_dump(mode="json")
        cached=self.store.cached(actor,"case",case_id,request.request_id,payload)
        if cached:
            return cached
        if revision!=request.expected_revision:
            raise Conflict("REVISION_CONFLICT")
        rows,facts=self._materials(request.materials,request.confirmed_facts,actor)
        state.update(materials=rows,confirmed_facts=facts,material_revision=state["material_revision"]+1,
                     last_report=None,status="READY",release_id=self.knowledge.release_id)
        invalidated=["evidence","rules","calculation","draft"]
        return self.store.commit(actor,"case",case_id,revision,request.request_id,payload,state,
                                 {"case_id":case_id,"status":"READY","material_revision":state["material_revision"],"invalidated":invalidated})

    def _allowed_tools(self, state):
        done=set(state["tools"])
        if "load_policy" not in done:
            return ["load_policy"]
        if "check_materials" not in done:
            return ["check_materials"]
        if state["status"]!="PROCESSING":
            if state["status"].startswith("AWAITING_") and "retrieve_evidence" not in done:
                return ["retrieve_evidence"]
            return ["draft_review"]
        choices=[t for t in ("retrieve_evidence","evaluate_rules") if t not in done]
        if choices:
            return choices
        if "calculate" not in done:
            return ["calculate"]
        return ["draft_review"]

    def _step(self, state):
        trace=list(state["tools"])
        if len(trace)>=8:
            return {"status":"HUMAN_REVIEW","diagnostics":["TOOL_BUDGET_EXCEEDED"],"done":True}
        allowed=self._allowed_tools(state)
        tool=allowed[0]
        if self.gateway is not None and hasattr(self.gateway,"choose_tool"):
            try:
                tool=self.gateway.choose_tool(allowed,{"completed_tools":trace,"status":state["status"],
                                                       "missing_info":state["missing_info"]})
            except Exception:
                return {"status":"HUMAN_REVIEW","diagnostics":["TOOL_SELECTOR_UNAVAILABLE"],"done":True}
        if tool not in allowed:
            return {"status":"HUMAN_REVIEW","diagnostics":["TOOL_NOT_ALLOWED"],"done":True}
        trace.append(tool)
        update={"tools":trace}
        policy,case=state["policy"],state["case"]
        if tool=="load_policy":
            if case["policy_snapshot_id"]!=policy["snapshot_id"]:
                update.update(status="HUMAN_REVIEW",diagnostics=["POLICY_SNAPSHOT_CHANGED"])
        elif tool=="check_materials":
            issues=[]
            by_kind={}
            for m in case["materials"]:
                by_kind.setdefault(m["kind"],[]).append(m)
                if m["subject_id"]!=policy["subject_id"]:
                    issues.append("subject_mismatch:"+m["material_id"])
                if not m["confirmed"]:
                    issues.append("material_unconfirmed:"+m["material_id"])
            missing=[k for k in policy["required_materials"] if k not in by_kind]
            if issues:
                update.update(status="AWAITING_FACT_REVIEW",missing_info=issues+missing)
            elif missing:
                update.update(status="AWAITING_SUPPLEMENT",missing_info=missing)
        elif tool=="retrieve_evidence":
            try:
                update["evidence"]=self.knowledge.hydrate(state["actor"],policy["evidence_ids"],
                                                          date.fromisoformat(policy["incident_date"]),policy["plan_id"])
            except (ValueError,PermissionError):
                update.update(status="HUMAN_REVIEW",diagnostics=["CLAIMS_EVIDENCE_UNAVAILABLE"])
        elif tool=="evaluate_rules":
            results=[]
            for rule in policy["rules"]:
                if rule["field"]=="incident_in_period":
                    value=policy["start_date"]<=policy["incident_date"]<=policy["end_date"]
                else:
                    value=case["confirmed_facts"].get(rule["field"],{}).get("value")
                result="UNKNOWN" if value is None else "PASS" if type(value) is bool and value==rule["expected"] else "FAIL"
                results.append({"rule_id":rule["rule_id"],"result":result,"field":rule["field"],
                                "evidence_ids":rule["evidence_ids"],"rule_version":policy["rule_version"]})
            update["rule_results"]=results
            if any(r["result"]=="UNKNOWN" for r in results):
                update.update(status="AWAITING_FACT_REVIEW",missing_info=[r["field"] for r in results if r["result"]=="UNKNOWN"])
            elif any(r["result"]=="FAIL" for r in results):
                update.update(status="AWAITING_HUMAN_REVIEW",missing_info=["rule_failed_requires_manual_liability_review"])
        elif tool=="calculate":
            fs=case["confirmed_facts"]
            missing=[f for f in ("eligible_expense","other_paid") if f not in fs]
            if missing:
                update.update(status="AWAITING_FACT_REVIEW",missing_info=missing)
            else:
                try:
                    config=policy["benefit"]
                    if config["formula"]!="DEMO_NET_REIMBURSEMENT_V1":
                        raise ValueError("UNREGISTERED_BENEFIT_FORMULA")
                    expense,other=money(fs["eligible_expense"]["value"]),money(fs["other_paid"]["value"])
                    deductible,limit,rate=money(config["deductible"]),money(config["limit"]),Decimal(config["rate"])
                    if other>expense or not rate.is_finite() or not 0<=rate<=1:
                        raise ValueError("AMOUNT_CONFLICT")
                    net=max(Decimal("0.00"),expense-other-deductible)
                    estimated=min(limit,net*rate).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
                    update["calculation"]={"estimated_amount":str(estimated),"eligible_expense":str(expense),
                        "other_paid":str(other),"deductible":str(deductible),"rate":str(rate),"limit":str(limit),
                        "formula":config["formula"],"config_version":config["version"],
                        "evidence_ids":config["evidence_ids"],"material_revision":case["material_revision"],
                        "rounding":"ROUND_HALF_UP_0.01","formal_payment":False}
                except (ValueError,KeyError):
                    update.update(status="AWAITING_FACT_REVIEW",missing_info=["amount_or_benefit_configuration"])
        elif tool=="draft_review":
            update.update(status="AWAITING_HUMAN_REVIEW" if state["status"]=="PROCESSING" else state["status"],done=True)
        return update

    def run(self, actor, case_id, request):
        revision,case=self._case(actor,case_id)
        payload=request.model_dump(mode="json")
        cached=self.store.cached(actor,"case",case_id,request.request_id,payload)
        if cached:
            if cached.get("material_revision")!=case["material_revision"]:
                raise Conflict("MATERIAL_REVISION_CHANGED")
            self.knowledge.validate_output(actor,cached)
            return cached
        if revision!=request.expected_revision:
            raise Conflict("REVISION_CONFLICT")
        policy=self._policy(actor,case["policy_id"])
        state=self.graph.invoke({"actor":actor,"case":case,"policy":policy,"tools":[],"evidence":[],
            "rule_results":[],"calculation":None,"missing_info":[],"diagnostics":[],"status":"PROCESSING","done":False},
            {"recursion_limit":20})
        report={"case_id":case_id,"status":state["status"],"material_revision":case["material_revision"],
            "policy_snapshot_id":policy["snapshot_id"],"tool_trace":state["tools"],"rule_results":state["rule_results"],
            "calculation":state["calculation"],"missing_info":state["missing_info"],"diagnostics":state["diagnostics"],
            "evidence":state["evidence"],"release_id":self.knowledge.release_id,"as_of":policy["incident_date"],
            "plan_id":policy["plan_id"],"formal_decision":False,"requires_human_review":True,
            "agent_mode":"MODEL_TOOL_SELECTION" if self.gateway else "DETERMINISTIC_TOOL_WORKFLOW",
            "review_task_id":"local-review-"+sha256(f"{actor.tenant_id}:{actor.user_id}:{case_id}:{revision}".encode()).hexdigest()[:20],
            "material_sources":[{k:m[k] for k in ("material_id","filename","content_sha256","confirmed_by")} for m in case["materials"]],
            "draft":"初审准备结果："+state["status"]+"。请人工核对材料、责任依据、规则与试算；未作正式判赔。"}
        self.knowledge.validate_output(actor,report)
        case.update(last_report=report,status=report["status"])
        return self.store.commit(actor,"case",case_id,revision,request.request_id,payload,case,report)

    def get(self, actor, case_id):
        revision,state=self._case(actor,case_id)
        if state.get("last_report"):
            self.knowledge.validate_output(actor,state["last_report"])
        return {**state,"revision":revision}
