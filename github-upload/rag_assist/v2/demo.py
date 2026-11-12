"""Self-contained synthetic originals; these approvals have no production meaning."""
from .assessments import digest
from .contracts import (DependencyManifest, EvidenceBinding, EvidenceUnit,
                        KnowledgeSnapshot, Relation, ResponsibilityCatalog,
                        SourceBinding, SourceRevision)

DEMO_NOTICE = "合成资料演示；审批只用于演示引用完整性，不是生产审批，不提供实际核保建议。"


def demo_snapshot():
    specs = (
        ("DEMO_RESP_A", "演示责任甲：本段仅演示引用链，不承诺任何保险责任。", "DEMO_COV_A", False),
        ("DEMO_WAIT", "演示等待期：仅为合成示例，等待期需结合本责任核对。", "DEMO_COV_A", False),
        ("DEMO_RATIO", "演示计划甲参数：本段不给出给付比例或核保意见。", "DEMO_COV_A", True),
        ("DEMO_FOOT", "演示脚注：本参数仅属于演示计划甲，不能外推到其他计划。", "DEMO_COV_A", True),
        ("DEMO_RESP_B", "演示责任乙：独立范围，未提供完整依赖审批。", "DEMO_COV_B", False),
    )
    text = "【合成演示原件；非保险条款】\n" + "\n".join(body for _, body, _, _ in specs)
    source = SourceRevision("DEMO_SOURCE_1", "DEMO_TENANT_A", "DEMO_PRODUCT_1", "DEMO_BUNDLE_1",
                            text, digest(text), "2025-01-01", data_class="SYNTHETIC", publication_state="DEMO")
    units = tuple(EvidenceUnit(uid, source.revision_id, body, digest(body), text.index(body),
                               text.index(body) + len(body), frozenset({coverage}),
                               frozenset({"DEMO_PLAN_A"}) if plan_specific else frozenset())
                  for uid, body, coverage, plan_specific in specs)
    edges = (
        Relation("DEMO_RESP_A", "DEMO_WAIT", "REQUIRES", "DEMO_COV_A", digest("DEMO_WAIT_CONDITION"), True),
        Relation("DEMO_RESP_A", "DEMO_RATIO", "PARAMETERIZED_BY", "DEMO_COV_A", digest("DEMO_PLAN_A_PARAMETER"), True),
        Relation("DEMO_RATIO", "DEMO_FOOT", "FOOTNOTE_FOR", "DEMO_COV_A", digest("DEMO_PLAN_A_FOOTNOTE"), True),
    )
    bindings = tuple(EvidenceBinding(u.unit_id, u.source_revision_id, u.start, u.end,
                                     u.content_sha256, u.responsibility_ids, u.plan_ids) for u in units[:4])
    source_binding = SourceBinding(source.revision_id, source.tenant_id, source.product_release_id,
                                   source.bundle_id, source.sha256, source.valid_from, source.valid_to,
                                   source.visibility, source.data_class, source.publication_state)
    manifest = DependencyManifest(
        "DEMO_MANIFEST_A", source.product_release_id, source.bundle_id, "DEMO_PLAN_A", ("DEMO_COV_A",),
        tuple((u.unit_id, u.content_sha256) for u in units[:4]), edges, ((source.revision_id, source.sha256),),
        "DEMO_AUTHOR_SYNTHETIC_ONLY", "DEMO_REVIEWER_SYNTHETIC_ONLY", approved=True,
        source_coverage_verified=True, expected_bindings=bindings, expected_sources=(source_binding,),
        expected_roots=(("DEMO_COV_A", "DEMO_RESP_A"),))
    catalog = ResponsibilityCatalog(source.product_release_id, source.bundle_id,
                                     (("DEMO_COV_A", "DEMO_RESP_A"), ("DEMO_COV_B", "DEMO_RESP_B")), True)
    return KnowledgeSnapshot((source,), units, edges, catalog, (manifest,))
