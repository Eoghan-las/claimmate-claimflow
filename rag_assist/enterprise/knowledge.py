"""Canonical evidence, scope checks and dependency hydration for all retrievers."""
from hashlib import sha256
import re

from .contracts import Evidence, ProductVersion, Source
from rag_assist.localstack.retrieval import fuse_routes


TOPICS = {
    "等待期": ("等待期", "观察期"),
    "免赔额": ("免赔额", "免赔", "起付线"),
    "给付比例": ("给付比例", "报销比例", "赔付比例", "报多少"),
    "责任范围": ("责任范围", "保障范围", "住院", "治疗", "医疗费用"),
    "责任免除": ("责任免除", "除外责任", "不赔"),
}


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


class Knowledge:
    def __init__(self, products, sources, units, *, release_id, retriever=None):
        self.products = {p.version_id:p for p in map(ProductVersion.model_validate, products)}
        self.sources = {s.source_id:s for s in map(Source.model_validate, sources)}
        self.units = {u.unit_id:u for u in map(Evidence.model_validate, units)}
        self.release_id = release_id
        self.revoked_sources = set()
        self.retriever = retriever
        for source in self.sources.values():
            if digest(source.text) != source.sha256 or source.version_id not in self.products:
                raise ValueError("SOURCE_INTEGRITY")
        for unit in self.units.values():
            self.validate_unit(unit)

    def validate_unit(self, u):
        source = self.sources[u.source_id]
        if digest(source.text) != source.sha256:
            raise ValueError("SOURCE_INTEGRITY")
        if (u.version_id != source.version_id or u.end-u.start != len(u.text) or
                source.text[u.start:u.end] != u.text or digest(u.text) != u.content_sha256):
            raise ValueError("EVIDENCE_INTEGRITY")

    def catalog(self, actor):
        return [p for p in self.products.values() if p.family_id in actor.families and
                any(s.version_id == p.version_id and s.source_id not in self.revoked_sources for s in self.sources.values())]

    def allowed(self, actor, version_id, as_of, plan_id=None):
        p = self.products.get(version_id)
        if p is None or p.family_id not in actor.families:
            raise PermissionError("PRODUCT_UNAVAILABLE")
        if p.validity_verified and ((p.valid_from is not None and as_of < p.valid_from) or (p.valid_to is not None and as_of > p.valid_to)):
            raise ValueError("VERSION_NOT_APPLICABLE")
        if plan_id is not None and plan_id not in p.plan_ids:
            raise ValueError("PLAN_NOT_APPLICABLE")
        if not any(s.version_id == version_id and s.source_id not in self.revoked_sources for s in self.sources.values()):
            raise PermissionError("SOURCE_REVOKED")
        return p

    def evidence(self, actor, ids, as_of, plan_id):
        rows = []
        for uid in ids:
            if uid not in self.units:
                raise ValueError("DEPENDENCY_MISSING")
            unit = self.units[uid]
            self.allowed(actor, unit.version_id, as_of, plan_id)
            if unit.source_id in self.revoked_sources:
                raise PermissionError("SOURCE_REVOKED")
            if unit.plan_ids and (plan_id is None or plan_id not in unit.plan_ids):
                raise ValueError("DEPENDENCY_PLAN_MISMATCH")
            self.validate_unit(unit)
            source = self.sources[unit.source_id]
            rows.append({**unit.model_dump(), "filename":source.filename, "source_url":source.source_url,
                         "source_sha256":source.sha256,"raw_file_sha256":source.raw_file_sha256,"data_class":source.data_class})
        return rows

    def hydrate(self, actor, roots, as_of, plan_id, *, max_chars=18000):
        visited, pending = [], list(roots)
        while pending:
            uid = pending.pop(0)
            if uid in visited:
                continue
            if uid not in self.units:
                raise ValueError("DEPENDENCY_MISSING")
            unit = self.units[uid]
            # Dependencies may not cross product versions, including through a cycle.
            if any(d in self.units and self.units[d].version_id != unit.version_id for d in unit.dependencies):
                raise ValueError("DEPENDENCY_SCOPE_MISMATCH")
            visited.append(uid)
            pending.extend(unit.dependencies)
        evidence = self.evidence(actor, visited, as_of, plan_id)
        if sum(len(e["text"]) for e in evidence) > max_chars:
            raise ValueError("CONTEXT_BUDGET_EXCEEDED")
        return evidence

    def validate_output(self, actor, output):
        if output.get("release_id") != self.release_id:
            raise PermissionError("KNOWLEDGE_RELEASE_CHANGED")
        from datetime import date
        plan = output.get("plan") or {}
        day = date.fromisoformat(plan.get("as_of", output.get("as_of", date.today().isoformat())))
        for row in output.get("evidence", []):
            self.evidence(actor, [row["unit_id"]], day, plan.get("plan_id", output.get("plan_id")))
        for row in output.get("results", []):
            self.allowed(actor, row["version_id"], day, plan.get("plan_id"))

    def search(self, actor, task, plan):
        self.allowed(actor, task.version_id, plan.as_of, plan.plan_id)
        if self.retriever:
            return self.retriever(task, plan)
        # ponytail: this explicit synthetic registry tests orchestration, not dense/visual recall.
        candidates = [u for u in self.units.values() if u.version_id == task.version_id and
                      (not u.plan_ids or plan.plan_id in u.plan_ids) and u.source_id not in self.revoked_sources]
        baseline = [v.text for v in task.variants if v.channel == "TEXT"]
        def overlap(u):
            return max((len(set(re.findall(r"[\u4e00-\u9fff]{2}", text)) &
                            set(re.findall(r"[\u4e00-\u9fff]{2}", u.text))) for text in baseline), default=0)
        matched = [u for u in candidates if task.topic in u.topics]
        routes = {}
        routes["text"] = sorted(matched, key=lambda u:(-overlap(u),u.unit_id))
        routes["structured"] = matched
        routes["layout"] = [u for u in candidates if u.layout_kind and task.topic in u.topics]
        graph_ids, queue = set(), [(i,0) for i in plan.anchor_ids] + [(u.unit_id,0) for u in matched]
        while queue and len(graph_ids) < 40:
            uid, depth = queue.pop(0)
            if uid in graph_ids or uid not in self.units:
                continue
            unit = self.units[uid]
            if unit.version_id != task.version_id:
                continue
            graph_ids.add(uid)
            if depth < 2:
                queue.extend((i,depth+1) for i in unit.dependencies+unit.related)
        routes["graph"] = [u for u in candidates if u.unit_id in graph_ids]
        result = {name:{"status":"OK" if rows else "EMPTY", "backend":"SYNTHETIC_REGISTRY",
                        "candidates":[{"evidence_id":u.unit_id} for u in rows[:task.budget[name]]],
                        "executed":True} for name,rows in routes.items()}
        fused = fuse_routes(result)
        return {"routes":result, "candidate_ids":[r["evidence_id"] for r in fused],
                "backend_mode":"SYNTHETIC_FUNCTIONAL", "diagnostics":[]}
