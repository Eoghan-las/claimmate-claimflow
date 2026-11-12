"""Resolve catalog and committed references before producing bounded queries."""
from datetime import date
import re

from .contracts import QueryPlan, QueryVariant, Subtask
from .knowledge import TOPICS


def make_plan(knowledge, actor, request, memory, understanding=None):
    q = request.question.strip()
    day = request.as_of or date.fromisoformat(memory.get("as_of", date.today().isoformat()))
    plan = QueryPlan(normalized_question=q, as_of=day, plan_id=request.plan_id or memory.get("plan_id"),
                     field_sources={"as_of":"request" if request.as_of else "memory_or_current_date"})
    catalog = knowledge.catalog(actor)
    explicit = request.version_ids or [p.version_id for p in catalog if any(a in q for a in [p.title,*p.aliases])]
    visible_ids = {p.version_id for p in catalog}
    if any(i not in visible_ids for i in request.version_ids):
        raise PermissionError("PRODUCT_UNAVAILABLE")
    previous = memory.get("results", [])
    ordinal = re.search(r"第([一二三四1234])个", q)
    others = bool(re.search(r"其他产品|其它产品|别的产品", q))
    related = bool(re.search(r"关联|相关条款|其他条款|这类条款", q))
    if explicit:
        plan.version_ids = list(dict.fromkeys(explicit))
        plan.field_sources["version_ids"] = "request_or_catalog_alias"
        if set(explicit) != set(memory.get("version_ids", [])) and request.plan_id is None:
            plan.plan_id = None
    elif ordinal:
        index = "一二三四".find(ordinal[1]) if ordinal[1] in "一二三四" else int(ordinal[1])-1
        pool=previous
        if q[ordinal.end():].lstrip().startswith("产品"):
            pool=[]
            seen=set()
            for row in previous:
                if row["family_id"] not in seen:
                    pool.append(row);seen.add(row["family_id"])
        if index >= len(pool):
            plan.missing_info.append("result_ordinal")
        else:
            chosen = pool[index]
            if q[ordinal.end():].lstrip().startswith("产品") and len({r["version_id"] for r in previous if r["family_id"]==chosen["family_id"]})>1:
                plan.missing_info.append("product_version")
            plan.version_ids = [chosen["version_id"]]
            plan.anchor_ids = [chosen["unit_id"]]
            plan.field_sources["version_ids"] = "committed_result_ordinal"
    elif others:
        presented = set(memory.get("presented_families", []))
        eligible = [p for p in catalog if p.scope_kind=="PRODUCT_VERSION" and p.family_id not in presented and
                    (not p.validity_verified or ((p.valid_from is None or p.valid_from <= day) and
                    (p.valid_to is None or day <= p.valid_to)))]
        # One applicable version per family; overlapping versions need catalog clarification.
        by_family = {}
        for p in eligible:
            by_family.setdefault(p.family_id, []).append(p.version_id)
        if any(len(v) != 1 for v in by_family.values()):
            plan.missing_info.append("product_version")
        else:
            count = 2 if re.search(r"两|2",q) else 1
            plan.version_ids = [v[0] for v in by_family.values()][:count]
            if len(plan.version_ids) < count:
                plan.missing_info.append("other_product_coverage")
        plan.plan_id = request.plan_id
        plan.field_sources["version_ids"] = "catalog_excluding_presented_families"
    else:
        if len(previous) > 1 and re.search(r"它|这个|这条|那条", q) and not related:
            plan.missing_info.append("ambiguous_reference")
        else:
            plan.version_ids = memory.get("version_ids", [])
            plan.anchor_ids = [r["unit_id"] for r in previous]
            plan.field_sources["version_ids"] = "committed_memory"
    plan.topics = [topic for topic,aliases in TOPICS.items() if any(a in q for a in aliases)]
    if understanding and understanding.topics:
        plan.topics = understanding.topics
        plan.field_sources["topics"] = "validated_model_classification"
    if plan.topics and plan.topics != memory.get("topics",[]) and not related:
        plan.anchor_ids = []
    if not plan.topics:
        if related or ordinal or others or re.search(r"它|这个|这条|那条|呢|例外",q):
            plan.topics = memory.get("topics", [])
        elif plan.version_ids:
            residue=q
            for product in catalog:
                for name in [product.title,*product.aliases]:residue=residue.replace(name,"")
            if len(residue.strip("？?，。 "))>=4:
                plan.topics=["原文查阅"]
    if related:
        plan.related_search = True
        plan.anchor_ids = [r["unit_id"] for r in previous if r["version_id"] in plan.version_ids]
        plan.exclude_primary_ids = memory.get("presented_units", [])
        if not plan.anchor_ids:
            plan.missing_info.append("clause_anchor")
    if (understanding and understanding.intent=="CASE_ASSIST") or (
            re.search(r"客户|患者|被保险人",q) and re.search(r"承保|拒保|加费|延期",q)):
        plan.intent = "CASE_ASSIST"
        plan.missing_info.append("authorized_medical_underwriting_rules_and_confirmed_case")
        return plan
    if re.search(r"公司规则|内部规则|核保手册|核保细则",q):
        plan.intent = "CASE_ASSIST"
        plan.missing_info.append("authorized_company_rules")
        return plan
    if not plan.version_ids:
        if not related and not ordinal and not others and (re.search(r"什么是|解释.{0,8}概念|概念",q) or
                (understanding and understanding.intent=="GENERAL")):
            plan.intent = "GENERAL"
            return plan
        plan.missing_info.append("product_version")
    for vid in plan.version_ids:
        try:
            knowledge.allowed(actor,vid,day,plan.plan_id)
        except ValueError as error:
            plan.missing_info.append(str(error))
    if not plan.topics:
        plan.missing_info.append("topic")
    if {"给付比例","免赔额"} & set(plan.topics) and plan.plan_id is None and any(knowledge.products[v].plan_ids for v in plan.version_ids):
        plan.missing_info.append("plan_id")
    if len(plan.version_ids)*len(plan.topics) > 4:
        plan.missing_info.append("subtask_budget")
    if plan.missing_info:
        plan.intent = "CLARIFY"
        return plan
    n = len(plan.version_ids)*len(plan.topics)
    requested=request.strategy if request.strategy!="AUTO" else understanding.strategy if understanding else "AUTO"
    plan.strategy = "SUB_QUERY" if n > 1 else (requested if requested not in {"AUTO","SUB_QUERY"} else
                    "HYDE" if related else "DIRECT")
    for vid in plan.version_ids:
        for topic in plan.topics:
            semantic_question=q
            for product in catalog:
                for name in sorted([product.title,product.version_id,*product.aliases],key=len,reverse=True):
                    semantic_question=semantic_question.replace(name,"")
            normalized = f"{topic}；" + semantic_question
            variants = [QueryVariant(kind="NORMALIZED", text=normalized)]
            if plan.strategy == "STEP_BACK":
                variants.append(QueryVariant(kind="ABSTRACT",text=f"{knowledge.products[vid].title} {topic}的定义、适用范围、前提和例外"))
            plan.subtasks.append(Subtask(version_id=vid,topic=topic,variants=variants))
    plan.normalized_question = "\n".join(f"{knowledge.products[t.version_id].title}；版本{t.version_id}；"
        f"计划{plan.plan_id or '未指定'}；业务时点{day.isoformat()}；"
        f"有效期{'已登记' if knowledge.products[t.version_id].validity_verified else '未核验，仅查原文'}；{t.variants[0].text}" for t in plan.subtasks)
    return plan


def add_hypotheses(plan, knowledge, gateway, *, actor=None):
    diagnostics=[]
    if plan.strategy != "HYDE" or gateway is None or not hasattr(gateway,"hypothesis"):
        return diagnostics
    for task in plan.subtasks[:2]:
        try:
            ids=[a for a in plan.anchor_ids if a in knowledge.units and knowledge.units[a].version_id==task.version_id]
            if ids and actor is None:raise PermissionError('ANCHOR_AUTHORIZATION_REQUIRED')
            rows=knowledge.evidence(actor,ids,plan.as_of,plan.plan_id) if ids else []
            if any(r['data_class']!='PUBLIC' for r in rows):raise PermissionError('ANCHOR_NOT_PUBLIC')
            anchors=[r['text'] for r in rows]
            text=gateway.hypothesis(task.variants[0].text,anchors)
            allowed_numbers=set(re.findall(r"\d+(?:\.\d+)?", " ".join(anchors+[task.variants[0].text])))
            if not isinstance(text,str) or not 10<=len(text)<=1000:
                diagnostics.append("HYPOTHESIS_FORMAT_INVALID")
                continue
            if (not set(re.findall(r"\d+(?:\.\d+)?",text)) <= allowed_numbers or
                    any(p.title in text for p in knowledge.products.values() if p.version_id != task.version_id)):
                diagnostics.append("HYPOTHESIS_FIELD_DRIFT")
                continue
            task.variants.append(QueryVariant(kind="HYPOTHETICAL",text=text,channel="DENSE_TEXT_ONLY"))
        except Exception as error:
            diagnostics.append("HYPOTHESIS_FORMAT_INVALID" if str(error)=="HYPOTHESIS_FORMAT_INVALID" else "HYPOTHESIS_UNAVAILABLE")
    return diagnostics
