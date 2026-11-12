"""Fail-closed reference assembly against an independently reviewed manifest."""
from datetime import date
from hashlib import sha256

from .contracts import (AssessmentRequest, AccessScope, EngineLimits, KnowledgeSnapshot,
                        EvidenceBinding, SourceBinding)

CHECKS = (
    "source_integrity", "source_coverage", "root_scope_coverage",
    "dependency_coverage", "relation_integrity", "scope_consistency",
    "fact_state", "rule_state", "conflict_state", "citation_state", "claim_support",
)


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def assess(snapshot: KnowledgeSnapshot, request: AssessmentRequest,
           access: AccessScope, *, limits=EngineLimits()):
    if request.mode not in {"SOURCE_LOOKUP", "CONDITION_EXPLANATION", "CASE_ASSIST"}:
        raise ValueError("unsupported assessment mode")
    if request.root_selection not in {"EXPLICIT", "ALL_APPROVED", "SEARCH_CANDIDATES"}:
        raise ValueError("unsupported responsibility selection")
    as_of = date.fromisoformat(request.as_of)
    result = {
        "decision_status": "EVIDENCE_BLOCKED", "checks": {k: "NOT_CHECKED" for k in CHECKS},
        "production_readiness": "BLOCKED", "recommendation": None, "core_signable": False,
        "human_review_required": request.mode == "CASE_ASSIST",
        "evidence": [], "missing_evidence": [], "invalid_evidence": [],
        "missing_relations": [], "unexpected_relations": [], "missing_facts": [],
        "unresolved_responsibilities": [], "reason_codes": [], "manifest_ids": [],
        "bundle_id": request.bundle_id, "mode": request.mode,
    }
    checks = result["checks"]
    if snapshot.sources and all(s.data_class == "SYNTHETIC" for s in snapshot.sources):
        result["production_readiness"] = "DEMO_ONLY"

    def stop(status, code):
        result["decision_status"] = status
        result["reason_codes"].append(code)
        if status in {"REVIEW_REQUIRED", "EVIDENCE_BLOCKED"}:
            result["human_review_required"] = True
        if status == "DENIED":
            for field in ("evidence", "missing_evidence", "invalid_evidence", "missing_relations",
                          "unexpected_relations", "manifest_ids", "unresolved_responsibilities"):
                result[field] = []
            result["reason_codes"] = ["OBJECT_UNAVAILABLE_IN_SCOPE"]
        return result

    if not access.actor_id or not access.tenant_id:
        return stop("DENIED", "MISSING_IDENTITY")
    if not request.product_release_id:
        result["missing_facts"] = ["product.release_id"]
        return stop("NEED_INFO", "PRODUCT_CONTEXT_REQUIRED")
    if request.product_release_id not in access.product_release_ids:
        return stop("DENIED", "PRODUCT_NOT_AUTHORIZED")
    owned = [s for s in snapshot.sources if s.tenant_id == access.tenant_id
             and s.product_release_id == request.product_release_id]
    if not owned:
        return stop("DENIED", "OBJECT_UNAVAILABLE_IN_SCOPE")
    catalog = snapshot.catalog
    if snapshot.revoked:
        return stop("EVIDENCE_BLOCKED", "BUNDLE_REVOKED")
    if (catalog.product_release_id, catalog.bundle_id) != (request.product_release_id, request.bundle_id):
        return stop("EVIDENCE_BLOCKED", "PINNED_BUNDLE_UNAVAILABLE")
    collections = [
        [s.revision_id for s in snapshot.sources],
        [u.unit_id for u in snapshot.units],
        [m.manifest_id for m in snapshot.manifests],
        [c for c, _ in catalog.responsibilities],
    ]
    if any(len(ids) != len(set(ids)) for ids in collections):
        return stop("EVIDENCE_BLOCKED", "DUPLICATE_ID")
    sources = {s.revision_id: s for s in snapshot.sources}
    units = {u.unit_id: u for u in snapshot.units}
    unit_bindings, registry_bindings = {}, {}
    roots = dict(catalog.responsibilities)
    if catalog.approved is not True or request.root_selection == "SEARCH_CANDIDATES":
        checks["root_scope_coverage"] = "UNKNOWN"
        result["missing_facts"] = ["responsibility.selection"]
        return stop("NEED_INFO", "RESPONSIBILITY_SCOPE_UNCONFIRMED")
    chosen = set(roots) if request.root_selection == "ALL_APPROVED" else set(request.responsibility_ids)
    if not chosen or not chosen <= set(roots):
        checks["root_scope_coverage"] = "UNKNOWN"
        result["missing_facts"] = ["responsibility.selection"]
        return stop("NEED_INFO", "RESPONSIBILITY_NOT_IN_APPROVED_CATALOG")
    for coverage in chosen:
        root = units.get(roots[coverage])
        if root is not None and coverage not in root.responsibility_ids:
            checks["root_scope_coverage"] = checks["scope_consistency"] = "FAILED"
            return stop("EVIDENCE_BLOCKED", "CATALOG_ROOT_SCOPE_MISMATCH")
    checks["root_scope_coverage"] = "VERIFIED"
    root_ids = {roots[c] for c in chosen}
    if request.mode != "SOURCE_LOOKUP" and not request.plan_id:
        checks["scope_consistency"] = "UNKNOWN"
        result["missing_facts"] = ["product.plan_id"]
        return stop("NEED_INFO", "PLAN_REQUIRED_NO_DEFAULT")

    # Every quote is re-read against an immutable source, not a retrieval prefix.
    def inspect(ids, approved_hashes=None, approved_sources=None):
        evidence, missing, invalid = [], [], []
        for uid in sorted(ids, key=lambda x: (x not in root_ids, x)):
            unit = units.get(uid)
            if unit is None:
                missing.append(uid)
                continue
            source = sources.get(unit.source_revision_id)
            if source is None:
                invalid.append(uid)
                continue
            if source.tenant_id != access.tenant_id or source.visibility not in access.visibility:
                raise PermissionError("unavailable object")
            scope_ok = (
                source.product_release_id == request.product_release_id
                and source.bundle_id == request.bundle_id
                and (not unit.responsibility_ids or bool(unit.responsibility_ids & chosen))
                and (not unit.plan_ids or request.plan_id in unit.plan_ids)
            )
            valid_time = date.fromisoformat(source.valid_from) <= as_of
            if source.valid_to:
                valid_time = valid_time and as_of < date.fromisoformat(source.valid_to)
            published = (source.data_class in {"PRIVATE_APPROVED", "PUBLIC_REFERENCE"}
                         and source.publication_state == "PUBLISHED") or (
                source.data_class == "SYNTHETIC" and source.publication_state == "DEMO")
            original_ok = (
                type(unit.start) is int and type(unit.end) is int
                and 0 <= unit.start < unit.end <= len(source.text)
                and source.text[unit.start:unit.end] == unit.text
                and digest(source.text) == source.sha256
                and digest(unit.text) == unit.content_sha256
            )
            if approved_hashes is not None:
                actual_binding = EvidenceBinding(unit.unit_id,unit.source_revision_id,unit.start,unit.end,
                                                 unit.content_sha256,unit.responsibility_ids,unit.plan_ids)
                original_ok = (original_ok and approved_hashes.get(uid) == unit.content_sha256
                               and unit_bindings.get(uid) == actual_binding)
            if approved_sources is not None:
                actual_source = SourceBinding(source.revision_id,source.tenant_id,source.product_release_id,
                                              source.bundle_id,source.sha256,source.valid_from,source.valid_to,
                                              source.visibility,source.data_class,source.publication_state)
                original_ok = (original_ok and approved_sources.get(source.revision_id) == source.sha256
                               and registry_bindings.get(source.revision_id) == actual_source)
            if not scope_ok:
                checks["scope_consistency"] = "FAILED"
            if not (original_ok and scope_ok and valid_time and published):
                invalid.append(uid)
                continue
            evidence.append({
                "evidence_id": uid, "source_revision_id": source.revision_id,
                "source_text": unit.text, "content_sha256": unit.content_sha256,
                "source_sha256": source.sha256, "span": {"start": unit.start, "end": unit.end},
                "responsibility_ids": sorted(unit.responsibility_ids),
                "plan_ids": sorted(unit.plan_ids), "data_class": source.data_class,
            })
        return evidence, missing, invalid

    def checked_inspect(ids, hashes=None, source_hashes=None):
        try:
            values = inspect(ids, hashes, source_hashes)
        except PermissionError:
            return None
        result["evidence"] = values[0]
        result["missing_evidence"] = sorted(set(values[1]) | set(values[2]))
        result["invalid_evidence"] = values[2]
        if values[1] or values[2]:
            checks["source_integrity"] = "FAILED" if values[2] else "UNKNOWN"
            checks["citation_state"] = "FAILED" if values[2] else "VERIFIED"
        else:
            checks["source_integrity"] = "VERIFIED"
            checks["citation_state"] = "VERIFIED"
        if checks["scope_consistency"] != "FAILED":
            checks["scope_consistency"] = "VERIFIED"
        checks["claim_support"] = "VERIFIED"  # Only exact source quotes are emitted.
        return values

    def source_only(code):
        if len(root_ids) > limits.max_units or sum(len(units[u].text) for u in root_ids if u in units) > limits.max_characters:
            return stop("EVIDENCE_BLOCKED", "EVIDENCE_BUDGET_EXCEEDED")
        if checked_inspect(root_ids) is None:
            return stop("DENIED", "OBJECT_UNAVAILABLE_IN_SCOPE")
        if result["missing_evidence"] or result["invalid_evidence"]:
            return stop("EVIDENCE_BLOCKED", "ROOT_SOURCE_UNAVAILABLE_OR_INVALID")
        checks["fact_state"] = checks["rule_state"] = "NOT_APPLICABLE"
        return stop("SOURCE_ONLY", code)

    if request.mode == "SOURCE_LOOKUP":
        return source_only("EXACT_QUOTE_ONLY_NO_BUSINESS_COMPLETENESS")

    selected = {}
    for responsibility in sorted(chosen):
        candidates = [
            m for m in snapshot.manifests
            if m.product_release_id == request.product_release_id
            and m.bundle_id == request.bundle_id and m.plan_id == request.plan_id
            and responsibility in m.responsibility_ids
            and set(m.responsibility_ids) <= chosen
            and m.approved is True and m.author_id and m.reviewer_id
            and m.author_id != m.reviewer_id and m.source_coverage_verified is True
            and not m.unresolved_refs
        ]
        if len(candidates) > 1:
            return stop("EVIDENCE_BLOCKED", "AMBIGUOUS_DEPENDENCY_MANIFEST")
        if not candidates:
            result["unresolved_responsibilities"].append(responsibility)
        else:
            selected[candidates[0].manifest_id] = candidates[0]
    if not selected:
        checks["dependency_coverage"] = checks["source_coverage"] = "UNKNOWN"
        return source_only("APPROVED_DEPENDENCY_MANIFEST_UNAVAILABLE")

    hashes, source_hashes, expected_edges = {}, {}, set()
    for manifest in selected.values():
        expected_root_pairs = {c: roots[c] for c in manifest.responsibility_ids}
        if len(dict(manifest.expected_roots)) != len(manifest.expected_roots) or dict(manifest.expected_roots) != expected_root_pairs:
            checks["root_scope_coverage"] = checks["scope_consistency"] = "FAILED"
            return stop("EVIDENCE_BLOCKED", "APPROVED_ROOT_BINDING_MISMATCH")
        b_ids = [b.unit_id for b in manifest.expected_bindings]
        s_ids = [s.revision_id for s in manifest.expected_sources]
        if (len(b_ids) != len(set(b_ids)) or set(b_ids) != {uid for uid, _ in manifest.expected_evidence}
                or len(s_ids) != len(set(s_ids)) or set(s_ids) != {sid for sid, _ in manifest.source_revisions}):
            checks["source_integrity"] = "FAILED"
            return stop("EVIDENCE_BLOCKED", "APPROVAL_BINDING_INCOMPLETE")
        if not {b.source_revision_id for b in manifest.expected_bindings} <= set(s_ids):
            checks["source_integrity"] = "FAILED"
            return stop("EVIDENCE_BLOCKED", "MANIFEST_SOURCE_APPROVAL_MISSING")
        for binding in manifest.expected_bindings:
            old = unit_bindings.get(binding.unit_id)
            if old is not None and old != binding:
                return stop("EVIDENCE_BLOCKED", "CONFLICTING_EVIDENCE_BINDING")
            unit_bindings[binding.unit_id] = binding
        for binding in manifest.expected_sources:
            old = registry_bindings.get(binding.revision_id)
            if old is not None and old != binding:
                return stop("EVIDENCE_BLOCKED", "CONFLICTING_SOURCE_BINDING")
            registry_bindings[binding.revision_id] = binding
        manifest_scope = set(manifest.responsibility_ids)
        for uid, _ in manifest.expected_evidence:
            unit = units.get(uid)
            if unit and unit.responsibility_ids and not (unit.responsibility_ids & manifest_scope):
                checks["scope_consistency"] = "FAILED"
                return stop("EVIDENCE_BLOCKED", "MANIFEST_UNIT_SCOPE_MISMATCH")
        for edge in manifest.expected_relations:
            if edge.scope_id not in manifest_scope:
                checks["scope_consistency"] = "FAILED"
                return stop("EVIDENCE_BLOCKED", "MANIFEST_RELATION_SCOPE_MISMATCH")
            for uid in (edge.from_unit, edge.to_unit):
                unit = units.get(uid)
                if unit and unit.responsibility_ids and edge.scope_id not in unit.responsibility_ids:
                    checks["scope_consistency"] = "FAILED"
                    return stop("EVIDENCE_BLOCKED", "RELATION_ENDPOINT_SCOPE_MISMATCH")
        if len(dict(manifest.expected_evidence)) != len(manifest.expected_evidence):
            return stop("EVIDENCE_BLOCKED", "DUPLICATE_MANIFEST_EVIDENCE")
        for key, value in manifest.expected_evidence:
            if key in hashes and hashes[key] != value:
                return stop("EVIDENCE_BLOCKED", "CONFLICTING_MANIFEST_HASH")
            hashes[key] = value
        for key, value in manifest.source_revisions:
            if key in source_hashes and source_hashes[key] != value:
                return stop("EVIDENCE_BLOCKED", "CONFLICTING_SOURCE_HASH")
            source_hashes[key] = value
        expected_edges.update(e.key for e in manifest.expected_relations)
    expected_ids = set(hashes)
    known_root_ids = {roots[c] for m in selected.values() for c in m.responsibility_ids}
    result["manifest_ids"] = sorted(selected)
    checks["source_coverage"] = "VERIFIED"
    if not known_root_ids <= expected_ids or any(
        src not in expected_ids or dst not in expected_ids or scope not in chosen
        for src, dst, kind, scope, predicate in expected_edges
    ):
        return stop("EVIDENCE_BLOCKED", "MANIFEST_REFERENCE_INVALID")
    # The expected graph, rather than the observed graph, determines obligations.
    reached = set(known_root_ids)
    while True:
        expanded = reached | {dst for src, dst, *_ in expected_edges if src in reached}
        if expanded == reached:
            break
        reached = expanded
    if reached != expected_ids:
        return stop("EVIDENCE_BLOCKED", "UNREACHABLE_MANIFEST_EVIDENCE")
    if len(expected_ids) > limits.max_units or sum(len(units[u].text) for u in expected_ids if u in units) > limits.max_characters:
        return stop("EVIDENCE_BLOCKED", "EVIDENCE_BUDGET_EXCEEDED")
    if checked_inspect(expected_ids, hashes, source_hashes) is None:
        return stop("DENIED", "OBJECT_UNAVAILABLE_IN_SCOPE")
    checks["dependency_coverage"] = "FAILED" if result["missing_evidence"] else "VERIFIED"
    actual_edges = {
        e.key for e in snapshot.relations
        if e.approved is True and e.scope_id in chosen
        and (e.from_unit in expected_ids or e.to_unit in expected_ids)
    }
    result["missing_relations"] = sorted("|".join(e) for e in expected_edges - actual_edges)
    result["unexpected_relations"] = sorted("|".join(e) for e in actual_edges - expected_edges)
    checks["relation_integrity"] = "FAILED" if result["missing_relations"] or result["unexpected_relations"] else "VERIFIED"
    checks["conflict_state"] = "VERIFIED"  # No alternative manifest or edge tuple survived.
    checks["fact_state"] = checks["rule_state"] = "NOT_APPLICABLE"
    if result["invalid_evidence"] or result["missing_evidence"]:
        return stop("EVIDENCE_BLOCKED", "MANDATORY_SOURCE_UNAVAILABLE_OR_INVALID")
    if checks["relation_integrity"] == "FAILED":
        return stop("EVIDENCE_BLOCKED", "APPROVED_RELATIONS_INCONSISTENT")
    if result["unresolved_responsibilities"]:
        checks["dependency_coverage"] = checks["source_coverage"] = "UNKNOWN"
        checks["relation_integrity"] = checks["conflict_state"] = "UNKNOWN"
        return source_only("APPROVED_DEPENDENCY_MANIFEST_UNAVAILABLE")
    if request.mode == "CASE_ASSIST":
        checks["fact_state"] = "UNKNOWN"
        checks["rule_state"] = "NOT_CHECKED"
        return stop("REVIEW_REQUIRED", "CASE_RULE_EXECUTION_NOT_CONFIGURED")
    return stop("ANSWERED_REFERENCE", "REGISTERED_SCOPE_AND_EXACT_SOURCE_QUOTES")
