"""Immutable internal contracts for the V2 reference-evidence slice.

These objects are trusted application inputs, never accepted as approval claims
from an HTTP request. Production publication and case decision engines are not
implemented by this module.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SourceRevision:
    revision_id: str
    tenant_id: str
    product_release_id: str
    bundle_id: str
    text: str
    sha256: str
    valid_from: str
    valid_to: Optional[str] = None
    visibility: str = "UW_INTERNAL"
    data_class: str = "UNVERIFIED"
    publication_state: str = "DRAFT"


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    source_revision_id: str
    text: str
    content_sha256: str
    start: int
    end: int
    responsibility_ids: frozenset[str]
    plan_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Relation:
    from_unit: str
    to_unit: str
    relation_type: str
    scope_id: str
    predicate_hash: str
    approved: bool = False

    @property
    def key(self):
        return (self.from_unit, self.to_unit, self.relation_type,
                self.scope_id, self.predicate_hash)

    @property
    def label(self):
        return "|".join(self.key)


@dataclass(frozen=True)
class EvidenceBinding:
    """Exact evidence identity included in the independent approval."""
    unit_id: str
    source_revision_id: str
    start: int
    end: int
    content_sha256: str
    responsibility_ids: frozenset[str]
    plan_ids: frozenset[str]


@dataclass(frozen=True)
class SourceBinding:
    """Registry scope and validity are pinned separately from the file hash."""
    revision_id: str
    tenant_id: str
    product_release_id: str
    bundle_id: str
    sha256: str
    valid_from: str
    valid_to: Optional[str]
    visibility: str
    data_class: str
    publication_state: str


@dataclass(frozen=True)
class DependencyManifest:
    manifest_id: str
    product_release_id: str
    bundle_id: str
    plan_id: str
    responsibility_ids: tuple[str, ...]
    expected_evidence: tuple[tuple[str, str], ...]
    expected_relations: tuple[Relation, ...]
    source_revisions: tuple[tuple[str, str], ...]
    author_id: str
    reviewer_id: str
    approved: bool = False
    source_coverage_verified: bool = False
    unresolved_refs: tuple[str, ...] = ()
    expected_bindings: tuple[EvidenceBinding, ...] = ()
    expected_sources: tuple[SourceBinding, ...] = ()
    expected_roots: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ResponsibilityCatalog:
    product_release_id: str
    bundle_id: str
    responsibilities: tuple[tuple[str, str], ...]
    approved: bool = False


@dataclass(frozen=True)
class KnowledgeSnapshot:
    sources: tuple[SourceRevision, ...]
    units: tuple[EvidenceUnit, ...]
    relations: tuple[Relation, ...]
    catalog: ResponsibilityCatalog
    manifests: tuple[DependencyManifest, ...]
    revoked: bool = False


@dataclass(frozen=True)
class AccessScope:
    actor_id: str
    tenant_id: str
    product_release_ids: frozenset[str]
    visibility: frozenset[str]


@dataclass(frozen=True)
class AssessmentRequest:
    mode: str
    product_release_id: Optional[str]
    bundle_id: Optional[str]
    plan_id: Optional[str]
    responsibility_ids: tuple[str, ...]
    root_selection: str
    as_of: str


@dataclass(frozen=True)
class EngineLimits:
    max_units: int = 1000
    max_characters: int = 800000

    def __post_init__(self):
        if type(self.max_units) is not int or self.max_units < 1:
            raise ValueError("max_units must be a positive integer")
        if type(self.max_characters) is not int or self.max_characters < 1:
            raise ValueError("max_characters must be a positive integer")
