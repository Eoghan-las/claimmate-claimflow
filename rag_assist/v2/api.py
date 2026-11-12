"""Local DEMO_ONLY API. Server-owned identity and committed SQLite audit gate every result.

No real documents, SSO, production approval, case rules, or core issuance are supported.
"""
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from .assessments import assess
from .contracts import AccessScope, AssessmentRequest, KnowledgeSnapshot
from .demo import DEMO_NOTICE, demo_snapshot

DEMO_FLAGS = dict(production_readiness="DEMO_ONLY", core_signable=False,
                  approval_kind="DEMO_SYNTHETIC_ONLY", demo_notice=DEMO_NOTICE)


@dataclass(frozen=True)
class Principal:
    actor_id: str
    tenant_id: str
    role: str
    product_release_ids: frozenset[str]
    visibility: frozenset[str]

    def __post_init__(self):
        if not self.actor_id or not self.tenant_id or self.role not in {"UW_OFFICER", "UW_REVIEWER", "AUDITOR"}:
            raise ValueError("A server actor, tenant, and supported role are required")


Identifier = Annotated[StrictStr, Field(min_length=1, max_length=160)]


class ReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["SOURCE_LOOKUP", "CONDITION_EXPLANATION", "CASE_ASSIST"]
    product_release_id: Identifier | None = None
    bundle_id: Identifier | None = None
    plan_id: Identifier | None = None
    responsibility_ids: Annotated[list[Identifier], Field(max_length=32)]
    root_selection: Literal["EXPLICIT"] = "EXPLICIT"
    as_of: date


def _require_demo(snapshot):
    if not isinstance(snapshot, KnowledgeSnapshot) or not snapshot.sources:
        raise ValueError("Only nonempty SYNTHETIC/DEMO snapshots are supported")
    source_records = (*snapshot.sources, *(s for m in snapshot.manifests for s in m.expected_sources))
    if any(s.data_class != "SYNTHETIC" or s.publication_state != "DEMO" for s in source_records):
        raise ValueError("Only SYNTHETIC data with DEMO publication is supported; real documents are rejected")
    if any(not m.author_id.startswith("DEMO_") or not m.reviewer_id.startswith("DEMO_")
           or m.author_id == m.reviewer_id for m in snapshot.manifests):
        raise ValueError("Only distinct explicit DEMO author/reviewer identities are supported")


def create_app(*, token_registry: dict[str, Principal], audit_path="runtime/v2-audit.sqlite",
               snapshot: KnowledgeSnapshot | None = None):
    snapshot = demo_snapshot() if snapshot is None else snapshot
    _require_demo(snapshot)
    # Copy the registry so later caller mutation cannot change an active demo's permissions.
    tokens = dict(token_registry)
    if any(not isinstance(token, str) or not token or not isinstance(actor, Principal)
           for token, actor in tokens.items()):
        raise ValueError("The token registry must contain server-owned Principals")
    audit_path = Path(audit_path)
    app = FastAPI(title="V2 synthetic reference demo", version="0.1.0", description=DEMO_NOTICE)
    bearer = HTTPBearer(auto_error=False)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        principal = tokens.get(credentials.credentials) if credentials else None
        if principal is None:
            raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
        return principal

    def unavailable():
        return HTTPException(404, "Object unavailable in scope")

    def in_scope(principal, product_id):
        owned = [s for s in snapshot.sources
                 if s.tenant_id == principal.tenant_id and s.product_release_id == product_id]
        return (product_id in principal.product_release_ids and bool(owned)
                and all(s.visibility in principal.visibility for s in owned))

    @contextmanager
    def audit_connection():
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(audit_path, timeout=2.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("""CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY, event_kind TEXT NOT NULL,
                tenant_id TEXT NOT NULL, actor_id TEXT NOT NULL, role TEXT NOT NULL,
                created_at TEXT NOT NULL, request_json TEXT NOT NULL,
                output_json TEXT NOT NULL, citations_json TEXT NOT NULL, result_status TEXT NOT NULL
            )""")
            yield connection

    def record(kind, principal, request, output):
        event_id = str(uuid4())
        output = dict(output, **DEMO_FLAGS)
        output["audit_event_id" if kind == "audit_read" else "run_id"] = event_id
        citations = output.get("evidence", output.get("output", {}).get("evidence", []))
        try:
            with audit_connection() as connection, connection:
                connection.execute("INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    event_id, kind, principal.tenant_id, principal.actor_id, principal.role,
                    datetime.now(timezone.utc).isoformat(), json.dumps(request, ensure_ascii=False),
                    json.dumps(output, ensure_ascii=False), json.dumps(citations, ensure_ascii=False),
                    output.get("decision_status", "AUDIT_READ" if kind == "audit_read" else "CATALOG_READ")))
            # Both the INSERT and commit have completed before any business output is returned.
        except (sqlite3.Error, OSError):
            raise HTTPException(503, "Audit unavailable; no business result released") from None
        return output

    @app.get("/health")
    def health():
        return dict(status="ok", **DEMO_FLAGS)

    @app.get("/v2/catalog")
    def catalog(principal: Principal = Depends(authenticate)):
        catalog = snapshot.catalog
        if not in_scope(principal, catalog.product_release_id):
            raise unavailable()
        body = dict(product_release_id=catalog.product_release_id, bundle_id=catalog.bundle_id,
                    responsibilities=[dict(responsibility_id=c, root_evidence_id=r)
                                      for c, r in catalog.responsibilities],
                    plans=sorted({m.plan_id for m in snapshot.manifests
                                  if m.product_release_id == catalog.product_release_id
                                  and m.bundle_id == catalog.bundle_id}))
        return record("catalog_read", principal, {"method": "GET", "path": "/v2/catalog"}, body)

    @app.post("/v2/reference-assessments")
    def reference_assessment(body: ReferenceRequest, principal: Principal = Depends(authenticate)):
        if principal.role not in {"UW_OFFICER", "UW_REVIEWER"}:
            raise HTTPException(403, "Role cannot create reference assessments")
        if body.product_release_id is not None and not in_scope(principal, body.product_release_id):
            raise unavailable()
        request = AssessmentRequest(body.mode, body.product_release_id, body.bundle_id, body.plan_id,
                                    tuple(body.responsibility_ids), body.root_selection, body.as_of.isoformat())
        access = AccessScope(principal.actor_id, principal.tenant_id, principal.product_release_ids,
                             principal.visibility)
        result = assess(snapshot, request, access)
        if result["decision_status"] == "DENIED":
            raise unavailable()
        return record("reference_assessment", principal, body.model_dump(mode="json"), result)

    @app.get("/v2/runs/{run_id}")
    def run(run_id: str, principal: Principal = Depends(authenticate)):
        try:
            with audit_connection() as connection:
                row = connection.execute(
                    "SELECT * FROM audit_events WHERE event_id = ? AND tenant_id = ? "
                    "AND event_kind = 'reference_assessment' AND (actor_id = ? OR ? IN ('UW_REVIEWER', 'AUDITOR'))",
                    (run_id, principal.tenant_id, principal.actor_id, principal.role)).fetchone()
        except (sqlite3.Error, OSError):
            raise HTTPException(503, "Audit unavailable; no business result released") from None
        if row is None:
            raise unavailable()
        request = json.loads(row["request_json"])
        saved_output = json.loads(row["output_json"])
        product_id = request["product_release_id"]
        if product_id is None:
            # A clarification belongs to this demo's catalog scope, even before
            # product selection. Never use this fallback to release evidence.
            if saved_output.get("decision_status") != "NEED_INFO" or saved_output.get("evidence"):
                raise unavailable()
            product_id = snapshot.catalog.product_release_id
        if not in_scope(principal, product_id):
            raise unavailable()
        output = dict(run_id=row["event_id"], actor_id=row["actor_id"], tenant_id=row["tenant_id"],
                      role=row["role"], created_at=row["created_at"], request=request,
                      output=saved_output)
        return record("audit_read", principal, {"method": "GET", "run_id": run_id}, output)

    return app
