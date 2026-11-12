"""Loopback demo API. Tokens are server-side identities; bodies cannot set them."""
from pathlib import Path
import hashlib
from typing import Any, Dict, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from . import ROOT
from .knowledge import digest
from .service import InsuranceRAG


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Query(StrictBody):
    question: str = Field(min_length=1,max_length=4000)
    corpus: Literal["medical","liability"]
    version: Optional[str] = None
    as_of: str = "2026-11-01"
    application_id: Optional[str] = Field(default=None,max_length=64)


class CaseEvaluation(StrictBody):
    facts: Dict[str, Any]
    expected_revision: int = Field(ge=0)


class Review(StrictBody):
    run_id: str
    action: Literal["CONFIRM","REQUEST_INFO","ESCALATE"]
    reason: str = Field(min_length=1,max_length=1000)
    expected_revision: int = Field(ge=0)


class DataQuery(StrictBody):
    view_id: str
    filters: list = Field(default_factory=list)
    select: Optional[list] = None
    order_by: Optional[dict] = None
    limit: int = Field(default=100,ge=1,le=200)


def create_app(db_path: Path, tokens,embedding_mode='tfidf'):
    app = FastAPI(title="核保证据工作台 · 本地演示",version="1.0",docs_url="/api-docs")
    service = InsuranceRAG(db_path,embedding_mode=embedding_mode)
    security = HTTPBearer(auto_error=False)

    def identity(credential:Optional[HTTPAuthorizationCredentials]=Depends(security)):
        if not credential or credential.scheme.lower()!="bearer" or credential.credentials not in tokens:
            raise HTTPException(401,"Valid local demo token required")
        user = dict(tokens[credential.credentials])
        user.setdefault("actor","local-token-"+hashlib.sha256(credential.credentials.encode()).hexdigest()[:12])
        if not user.get("tenant") or not set(user.get("roles",[])) & {"UW_OFFICER","UW_REVIEWER","AUDITOR"}:
            raise HTTPException(403,"Scope denied")
        return user

    def checked(call):
        try:
            return call()
        except PermissionError:
            raise HTTPException(403,"Object unavailable in current scope")
        except ValueError as error:
            raise HTTPException(409 if "conflict" in str(error) or "stale" in str(error) else 422,str(error))

    @app.middleware("http")
    async def boundaries(request,call_next):
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length)>65536):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail":"request too large"},status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"]="no-store"
        response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["Content-Security-Policy"]="default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.get("/")
    def home():
        return FileResponse(ROOT/"rag_assist/static/index.html")

    @app.get("/workbench.js")
    def script():
        return FileResponse(ROOT/"rag_assist/static/workbench.js",media_type="application/javascript")

    @app.get("/workbench.css")
    def stylesheet():
        return FileResponse(ROOT/"rag_assist/static/workbench.css",media_type="text/css")

    @app.get("/health")
    def health():
        return {"status":"ok","mode":"LOCAL_DEMO"}

    @app.get("/v1/status")
    def status(user=Depends(identity)):
        return checked(lambda:service.status(tenant=user["tenant"]))

    @app.post("/v1/queries")
    def query(body:Query,user=Depends(identity)):
        return checked(lambda:service.ask(body.question,corpus=body.corpus,tenant=user["tenant"],version=body.version,as_of=body.as_of,actor=user["actor"],application_id=body.application_id))

    @app.post("/v1/data-queries")
    def data_query(body:DataQuery,user=Depends(identity)):
        return checked(lambda:service.data_query(body.view_id,body.filters,tenant=user["tenant"],select=body.select,order_by=body.order_by,limit=body.limit,actor=user["actor"]))

    @app.get("/v1/evidence/{evidence_id}")
    def evidence(evidence_id:str,user=Depends(identity)):
        return checked(lambda:service.evidence(evidence_id,tenant=user["tenant"]))

    @app.get("/v1/sources/{corpus}")
    def source(corpus:Literal["medical","liability"],user=Depends(identity)):
        if user["tenant"]!="T_008":
            raise HTTPException(403,"Scope denied")
        source = service.sources[corpus]
        allowed = {"medical":ROOT/"claimMate/chunker/samples/clause_medical.txt",
                   "liability":ROOT/"claimMate/samples_pdf/iachina_示范条款_2020.pdf"}
        file=allowed[corpus]
        if digest(file.read_bytes())!=source["source_hash"]:
            raise HTTPException(409,"Original file hash differs from published bundle")
        return FileResponse(file,media_type="text/plain; charset=utf-8" if corpus=="medical" else "application/pdf")

    @app.get("/v1/runs/{run_id}/audit")
    def audit(run_id:str,user=Depends(identity)):
        return checked(lambda:service.audit(run_id,tenant=user["tenant"]))

    @app.post("/v1/cases/DEMO_CASE_01/evaluate")
    def evaluate(body:CaseEvaluation,user=Depends(identity)):
        if not set(user["roles"]) & {"UW_OFFICER","UW_REVIEWER"}:
            raise HTTPException(403,"Underwriter role required")
        return checked(lambda:service.evaluate_case("DEMO_CASE_01",body.facts,tenant=user["tenant"],expected_revision=body.expected_revision,actor=user["actor"]))

    @app.post("/v1/reviews")
    def review(body:Review,idempotency_key:Optional[str]=Header(None),user=Depends(identity)):
        if "UW_REVIEWER" not in user["roles"]:
            raise HTTPException(403,"Human reviewer role required")
        return checked(lambda:service.review(body.run_id,body.action,tenant=user["tenant"],expected_revision=body.expected_revision,
                      idempotency_key=idempotency_key or "",reason=body.reason,actor=user["actor"]))

    return app
