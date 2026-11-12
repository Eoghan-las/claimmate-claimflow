"""Business regressions use explicitly synthetic originals, never production metrics."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from rag_assist.enterprise.app import create_app
from rag_assist.enterprise.contracts import Actor, Ask, CaseCreate, CaseRun, MaterialsUpdate
from rag_assist.enterprise.demo import demo_services
from rag_assist.enterprise.state import Conflict, StateStore


@pytest.fixture
def system(tmp_path):
    return demo_services(tmp_path / "state.sqlite")


def ask(s, question="甲医疗险等待期有什么约定？", **kwargs):
    return s.rag.ask(s.actor, Ask(conversation_id="chat1", request_id="r1", expected_revision=0,
                                question=question, as_of="2026-01-01", **kwargs))


def follow(s, previous, question, **kwargs):
    return s.rag.ask(s.actor, Ask(conversation_id="chat1", request_id="r" + str(previous["revision"] + 1),
                                expected_revision=previous["revision"], question=question, **kwargs))


def test_source_bundle_contains_registered_context_and_original_spans(system):
    r = ask(system)
    assert r["status"] == "SOURCE_EVIDENCE"
    assert r["model_mode"] == "EXTRACTIVE_NO_LLM"
    assert {"a.wait", "a.definition", "a.exception", "a.footnote"} <= {x["unit_id"] for x in r["evidence"]}
    for x in r["evidence"]:
        assert x["text"] == system.knowledge.sources[x["source_id"]].text[x["start"]:x["end"]]
        assert x["filename"] and x["page_number"] > 0 and x["content_sha256"]
    assert set(r["retrieval"][0]["routes"]) == {"text", "structured", "layout", "graph"}
    assert r["semantic_complete"] is False and r["requires_human_review"] is True


def test_other_products_then_ordinal_uses_committed_order(system):
    first = ask(system)
    second = follow(system, first, "再列出两个其他产品的等待期条款")
    assert {r["family_id"] for r in second["results"]} == {"beta", "gamma"}
    third = follow(system, second, "第二个有哪些例外？")
    assert third["plan"]["version_ids"] == [second["results"][1]["version_id"]]
    assert third["plan"]["as_of"] == "2026-01-01"
    assert third["plan"]["field_sources"]["version_ids"] == "committed_result_ordinal"


def test_ambiguous_reference_stops_before_retrieval(system):
    first = ask(system)
    second = follow(system, first, "再列出两个其他产品的等待期条款")
    third = follow(system, second, "那它呢？")
    assert third["status"] == "CLARIFY" and not third["retrieval"]
    assert third["missing_info"]


def test_related_search_keeps_dependencies_even_when_already_presented(system):
    first = ask(system)
    second = follow(system, first, "除了这类条款还有哪些关联条款？")
    assert second["plan"]["strategy"] == "HYDE"
    assert "a.wait" not in {x["unit_id"] for x in second["results"]}
    assert {"a.wait", "a.definition", "a.exception"} <= {x["unit_id"] for x in second["evidence"]}
    assert all(x["version_id"] == "alpha-2025" for x in second["evidence"])


def test_subquestions_budget_and_missing_plan(system):
    r = ask(system, "甲医疗险的等待期和免赔额分别如何约定？",plan_id="basic")
    assert r["plan"]["strategy"] == "SUB_QUERY"
    assert len(r["plan"]["subtasks"]) == 2
    for task in r["plan"]["subtasks"]:
        assert task["budget"] == {"text": 80, "structured": 20, "layout": 30, "graph": 30}
    r2 = system.rag.ask(system.actor, Ask(conversation_id="c2", request_id="p1", expected_revision=0,
        question="甲医疗险报销比例是多少？", as_of="2026-01-01"))
    assert r2["status"] == "CLARIFY" and "plan_id" in r2["missing_info"]


def test_stepback_and_excessive_scope_are_explicit(system):
    r = ask(system, "甲医疗险住院治疗的费用符合什么原则？", strategy="STEP_BACK")
    assert r["plan"]["strategy"] == "STEP_BACK"
    r2 = system.rag.ask(system.actor, Ask(conversation_id="c2", request_id="large", expected_revision=0,
        question="比较甲医疗险乙医疗险丙医疗险的等待期和免赔额", as_of="2026-01-01"))
    assert r2["status"] == "CLARIFY" and "subtask_budget" in r2["missing_info"]


def test_hypothesis_cannot_invent_numeric_conditions(system):
    # Keep the fixture eligible for external-source planning so this test reaches
    # the numeric-drift guard rather than stopping at source classification.
    system.knowledge.sources={sid:s.model_copy(update={'data_class':'PUBLIC'})
                              for sid,s in system.knowledge.sources.items()}
    class Model:
        model = "TEST_DOUBLE"
        def hypothesis(self, *args):
            return "该产品等待期999天，报销比例99%。"
    system.rag.gateway = Model()
    first = ask(system)
    second = follow(system, first, "还有哪些关联条款？")
    assert "HYPOTHESIS_FIELD_DRIFT" in second["diagnostics"]
    assert all("999" not in v["text"] for t in second["plan"]["subtasks"] for v in t["variants"])


def test_missing_registered_dependency_blocks_generation(system):
    del system.knowledge.units["a.exception"]
    r = ask(system)
    assert r["status"] == "HUMAN_REVIEW"
    assert "DEPENDENCY_MISSING" in r["diagnostics"]


def test_replay_is_idempotent_and_rechecks_revocations(system):
    first = ask(system)
    assert ask(system) == first
    with pytest.raises(Conflict):
        ask(system, "乙医疗险等待期是什么？")
    system.knowledge.revoked_sources.add("source-alpha")
    with pytest.raises(PermissionError):
        ask(system)


def test_new_asof_reselects_version_instead_of_reusing_expired_identity(system):
    r = ask(system)
    changed = follow(system, r, "按2024年的版本查询", as_of="2024-06-01")
    assert changed["status"] == "CLARIFY"
    assert not changed["evidence"]


def test_session_ownership_and_atomic_revision(tmp_path):
    db = StateStore(tmp_path / "db.sqlite")
    a = Actor(tenant_id="t1", user_id="u1", families=["alpha"])
    b = Actor(tenant_id="t1", user_id="u2", families=["alpha"])
    def commit(i):
        try:
            return db.commit(a, "chat", "c", 0, "r"+str(i), {"i":i}, {"value":i}, {"status":"ok"})
        except Conflict:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(commit, [1, 2]))
    assert sum(x is not None for x in rows) == 1
    assert db.load(a, "chat", "c")[0] == 1
    assert db.load(b, "chat", "c") == (0, {})


def materials(complete=True):
    names = ["invoice", "discharge", "settlement"] if complete else ["invoice"]
    return [{"material_id":m, "kind":m, "subject_id":"person-demo", "confirmed":True,
             "text":"合成材料" + m, "filename":m + ".txt"} for m in names]


def facts():
    return {k:{"value":v,"material_ids":["invoice"]} for k,v in {
        "eligible_expense":"1000.01", "other_paid":"100.00", "waiting_met":True,
        "hospital_eligible":True}.items()}


def create_case(s, mats=None, confirmed=None):
    return s.claims.create(s.actor, CaseCreate(case_id="case1", request_id="create1", policy_id="policy-demo",
        materials=materials() if mats is None else mats, confirmed_facts=facts() if confirmed is None else confirmed))


def test_claims_exact_decimal_and_complete_rule_registry(system):
    created = create_case(system)
    r = system.claims.run(system.actor, "case1", CaseRun(request_id="run1", expected_revision=created["revision"]))
    assert r["status"] == "AWAITING_HUMAN_REVIEW"
    # Independent arithmetic: (1000.01 - 100.00 - 100.00) * .8 = 640.008 -> 640.01.
    assert Decimal(r["calculation"]["estimated_amount"]) == Decimal("640.01")
    assert len(r["rule_results"]) == len(system.claims.policies["policy-demo"]["rules"])
    assert r["formal_decision"] is False
    assert {"load_policy","check_materials","retrieve_evidence","evaluate_rules","calculate","draft_review"} <= set(r["tool_trace"])


def test_missing_material_then_revision_resume_invalidates_calculation(system):
    created = create_case(system, materials(False))
    first = system.claims.run(system.actor,"case1", CaseRun(request_id="run1",expected_revision=created["revision"]))
    assert first["status"] == "AWAITING_SUPPLEMENT" and first["calculation"] is None
    updated = system.claims.replace_materials(system.actor,"case1", MaterialsUpdate(request_id="update1",
        expected_revision=first["revision"],materials=materials(),confirmed_facts=facts()))
    assert updated["material_revision"] == 2 and updated["invalidated"] == ["evidence","rules","calculation","draft"]
    second = system.claims.run(system.actor,"case1",CaseRun(request_id="run2",expected_revision=updated["revision"]))
    assert second["status"] == "AWAITING_HUMAN_REVIEW" and second["calculation"]["material_revision"] == 2


@pytest.mark.parametrize("removed", ["eligible_expense","waiting_met","hospital_eligible"])
def test_unknown_facts_do_not_default_to_zero_or_pass(system, removed):
    fs = facts(); del fs[removed]
    c = create_case(system, confirmed=fs)
    r = system.claims.run(system.actor,"case1",CaseRun(request_id="run1",expected_revision=c["revision"]))
    assert r["status"] == "AWAITING_FACT_REVIEW" and r["calculation"] is None


def test_wrong_subject_and_unconfirmed_material_block_processing(system):
    mats=materials();mats[0]["subject_id"]="another-person"
    c=create_case(system,mats)
    r=system.claims.run(system.actor,"case1",CaseRun(request_id="run1",expected_revision=c["revision"]))
    assert r["status"] == "AWAITING_FACT_REVIEW" and r["calculation"] is None


def test_model_cannot_skip_business_prerequisites(system):
    class BadSelector:
        model="TEST_DOUBLE"
        def choose_tool(self, allowed, summary): return "calculate"
    system.claims.gateway=BadSelector()
    c=create_case(system)
    r=system.claims.run(system.actor,"case1",CaseRun(request_id="run1",expected_revision=c["revision"]))
    assert r["status"] == "HUMAN_REVIEW" and "TOOL_NOT_ALLOWED" in r["diagnostics"]


def test_api_authentication_validation_and_case_isolation(system):
    app=create_app(system, token_registry={"one":system.actor,"two":Actor(tenant_id="other",user_id="other",families=["alpha"])})
    with TestClient(app) as c:
        assert c.get("/api/enterprise/catalog").status_code == 401
        req={"conversation_id":"x","request_id":"r","expected_revision":0,"question":"甲医疗险等待期？","as_of":"2026-01-01"}
        assert c.post("/api/enterprise/ask",json={**req,"tenant_id":"other"},headers={"Authorization":"Bearer one"}).status_code == 422
        r=c.post("/api/enterprise/ask",json=req,headers={"Authorization":"Bearer one"})
        assert r.status_code==200 and r.json()["revision"]==1
        assert c.get("/api/enterprise/cases/case1",headers={"Authorization":"Bearer two"}).status_code==404
        assert c.get("/").status_code==200


def test_role_and_authorization_rechecked_on_cached_result(system):
    r=ask(system)
    limited=system.actor.model_copy(update={"families":["beta"]})
    with pytest.raises(PermissionError):
        system.rag.ask(limited,Ask(conversation_id="chat1",request_id="r1",expected_revision=0,
                                 question=r["question"],as_of="2026-01-01"))


def test_canonical_tampering_outside_quote_is_detected(system):
    source=system.knowledge.sources["source-alpha"]
    source.text=source.text+"意外增加的未发布条款"
    r=ask(system)
    assert r["status"]=="HUMAN_REVIEW" and "SOURCE_INTEGRITY" in r["diagnostics"]


def test_registered_rule_evidence_is_retained_if_model_checks_rules_first(system):
    class Selector:
        model="TEST_DOUBLE"
        def choose_tool(self,allowed,summary): return "evaluate_rules" if "evaluate_rules" in allowed else allowed[0]
    system.claims.gateway=Selector()
    fs=facts();del fs["waiting_met"]
    c=create_case(system,confirmed=fs)
    r=system.claims.run(system.actor,"case1",CaseRun(request_id="run",expected_revision=c["revision"]))
    assert r["status"]=="AWAITING_FACT_REVIEW" and "a.wait" in {e["unit_id"] for e in r["evidence"]}


def test_policy_revision_change_blocks_replaying_old_calculation(system):
    c=create_case(system)
    request=CaseRun(request_id="run",expected_revision=c["revision"])
    system.claims.run(system.actor,"case1",request)
    system.claims.policies["policy-demo"]["snapshot_id"]="policy-demo-v2"
    with pytest.raises(Conflict):
        system.claims.run(system.actor,"case1",request)


def test_model_topic_understanding_supports_nonliteral_user_question(system):
    class Interpreter:
        model="TEST_DOUBLE"
        def interpret(self,question,context,catalog):
            return {"intent":"PRODUCT_QA","topics":["免赔额"],"strategy":"DIRECT"}
    system.rag.gateway=Interpreter()
    r=ask(system,"甲医疗险需要自己先承担多少费用？",plan_id="basic")
    assert r["status"]=="SOURCE_EVIDENCE" and r["plan"]["topics"]==["免赔额"]
    assert r["plan"]["field_sources"]["topics"]=="validated_model_classification"


def test_material_revision_prevents_old_calculation_replay(system):
    c=create_case(system)
    old=CaseRun(request_id="run1",expected_revision=c["revision"])
    r=system.claims.run(system.actor,"case1",old)
    system.claims.replace_materials(system.actor,"case1",MaterialsUpdate(request_id="update1",
        expected_revision=r["revision"],materials=materials(False),confirmed_facts=facts()))
    with pytest.raises(Conflict,match="MATERIAL_REVISION_CHANGED"):
        system.claims.run(system.actor,"case1",old)


def test_irrelevant_old_topic_is_not_forced_into_new_evidence_bundle(system):
    first=ask(system,plan_id="basic")
    second=follow(system,first,"免赔额如何约定？")
    assert "a.wait" not in {e["unit_id"] for e in second["evidence"]}


def test_state_failure_never_releases_generated_result(system,monkeypatch):
    def unavailable(*args,**kwargs): raise OSError("state unavailable")
    monkeypatch.setattr(system.store,"commit",unavailable)
    app=create_app(system,token_registry={"test":system.actor})
    with TestClient(app) as c:
        r=c.post('/api/enterprise/ask',headers={"Authorization":"Bearer test"},json={
            "conversation_id":"c","request_id":"r","expected_revision":0,"question":"甲医疗险等待期", "as_of":"2026-01-01"})
        assert r.status_code==503 and "evidence" not in r.json()


def test_missing_business_label_does_not_discard_retrieved_original(system):
    for u in system.knowledge.units.values():
        if u.version_id=="alpha-2025":u.topics=[]
    def retrieved(task,plan):
        return {"candidate_ids":["a.wait"],"routes":{k:{"status":"OK","candidates":[{"evidence_id":"a.wait"}]} for k in
            ("text","structured","layout","graph")},"diagnostics":[],"backend_mode":"TEST_DOUBLE"}
    system.knowledge.retriever=retrieved
    r=ask(system)
    assert r["status"]=="SOURCE_EVIDENCE" and r["results"][0]["match_status"]=="UNCLASSIFIED_SEARCH_CANDIDATE"
    assert "a.exception" in {e["unit_id"] for e in r["evidence"]}
    assert system.knowledge.units["a.wait"].topics==[]


def test_filtered_product_title_does_not_dominate_semantic_query(system):
    r=ask(system)
    assert "甲医疗险" not in r["plan"]["subtasks"][0]["variants"][0]["text"]
    assert "甲医疗险" in r["plan"]["normalized_question"]


def test_quote_repair_is_bounded_and_uses_verifiable_original(system):
    class RepairModel:
        model="TEST_DOUBLE"
        def __init__(self):self.calls=0
        def answer(self,question,contexts,feedback=None):
            self.calls+=1
            e=contexts[0]
            return {"claims":[{"text":e["text"],"sources":[{"evidence_id":e["evidence_id"],
                "quote":e["text"] if feedback else "不存在于该原件的引文内容"}]}],"missing_info":[]}
        def verify(self,question,draft,contexts):
            return {"sufficient_to_answer":True,"missing_conditions":[],"checks":[{"claim_index":0,
                "supported":True,"product_plan_match":True,"conditions_preserved":True}]}
    model=RepairModel();system.rag.gateway=model
    r=ask(system)
    assert r["status"]=="ANSWERED" and model.calls==2 and r["validation_retry_count"]==1


def test_retriever_runtime_failure_returns_committed_manual_path(system):
    def failed(task,plan):raise RuntimeError("backend timeout")
    system.knowledge.retriever=failed
    r=ask(system)
    assert r["status"]=="HUMAN_REVIEW" and r["revision"]==1
    assert "RETRIEVAL_UNAVAILABLE" in r["diagnostics"]


def test_unregistered_topic_can_search_originals_without_inventing_classification(system):
    r=ask(system,"甲医疗险发生争议时的管辖约定在哪里？")
    assert r["plan"]["intent"]=="PRODUCT_QA" and r["plan"]["topics"]==["原文查阅"]
    assert r["status"] in {"SOURCE_EVIDENCE","HUMAN_REVIEW"}


def test_model_receives_short_aliases_and_lossless_parent_context(system):
    from rag_assist.enterprise.contracts import Evidence
    from rag_assist.enterprise.knowledge import digest
    source=system.knowledge.sources["source-alpha"]
    system.knowledge.units["a.page"]=Evidence(unit_id="a.page",source_id=source.source_id,version_id=source.version_id,
        text=source.text,content_sha256=digest(source.text),start=0,end=len(source.text),page_number=1,topics=[],role="CONTEXT")
    system.knowledge.units["a.wait"].dependencies.append("a.page")
    class Capture:
        model="TEST_DOUBLE"
        def answer(self,question,contexts,feedback=None):
            assert len(contexts)==1 and contexts[0]["evidence_id"]=="E1"
            assert "a.page"==contexts[0]["unit_id"] and source.text==contexts[0]["text"]
            return {"claims":[{"text":contexts[0]["text"][:30],"sources":[{"evidence_id":"E1","quote":contexts[0]["text"][:30]}]}]}
        def verify(self,*args):
            return {"sufficient_to_answer":True,"checks":[{"claim_index":0,"supported":True,"product_plan_match":True,"conditions_preserved":True}]}
    system.rag.gateway=Capture()
    assert ask(system)["status"]=="ANSWERED"


def test_deductible_needs_confirmed_plan_and_plan_is_visible_to_generation(system):
    r=ask(system,"甲医疗险免赔额如何约定？")
    assert r["status"]=="CLARIFY" and "plan_id" in r["missing_info"]
    r=system.rag.ask(system.actor,Ask(conversation_id="c2",request_id="r1",expected_revision=0,
        question="甲医疗险免赔额如何约定？",plan_id="basic",as_of="2026-01-01"))
    assert r["status"]=="SOURCE_EVIDENCE" and "计划basic" in r["plan"]["normalized_question"]


def test_second_product_is_not_second_clause_of_first_product(system):
    first=ask(system,"比较甲医疗险和乙医疗险的等待期与免赔额",plan_id="basic")
    assert len(first["results"])==4
    second=follow(system,first,"第二个产品的免赔额是什么？")
    assert second["plan"]["version_ids"]==["beta-2025"]
    assert all(e["version_id"]=="beta-2025" for e in second["evidence"])
