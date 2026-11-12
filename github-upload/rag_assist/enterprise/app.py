"""Authenticated local API. Identity and product authority never come from a prompt."""
import os
from pathlib import Path
import sqlite3

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .contracts import Ask, CaseCreate, CaseRun, MaterialsUpdate
from .state import Conflict


def create_app(services, *, token_registry=None):
    if token_registry is None:
        token=os.environ.get("CLAIMMATE_ACCESS_TOKEN")
        if not token:
            raise ValueError("CLAIMMATE_ACCESS_TOKEN_REQUIRED")
        token_registry={token:services.actor}
    tokens=dict(token_registry)
    app=FastAPI(title="ClaimMate / ClaimFlow",version="0.2.0")
    bearer=HTTPBearer(auto_error=False)
    app.state.services=services

    def authenticate(value:HTTPAuthorizationCredentials | None=Depends(bearer)):
        actor=tokens.get(value.credentials) if value else None
        if actor is None:
            raise HTTPException(401,"Authentication required")
        return actor

    def call(operation):
        try:
            return operation()
        except Conflict as e:
            raise HTTPException(409,str(e)) from None
        except PermissionError:
            raise HTTPException(403,"Object or source unavailable in current authority") from None
        except KeyError:
            raise HTTPException(404,"Object unavailable") from None
        except ValueError as e:
            raise HTTPException(422,str(e)) from None
        except (sqlite3.Error,OSError):
            raise HTTPException(503,"Business state unavailable; result not released") from None

    @app.get("/health")
    def health():
        return {"status":"OK","mode":services.mode,"state_backend":services.store.backend,
                "release_id":services.knowledge.release_id,"production_ready":False,
                "model_connected":services.rag.gateway is not None,
                "configured_model":getattr(services.rag.gateway,'model',None),
                "context_id":getattr(getattr(services,'backend',None),'context_id',None)}

    @app.get("/api/enterprise/catalog")
    def catalog(actor=Depends(authenticate)):
        visible=services.knowledge.catalog(actor);ids={p.version_id for p in visible}
        documents=getattr(services.knowledge,'catalogue',{}).get('documents',{})
        return {"products":[p.model_dump(mode="json") for p in services.knowledge.catalog(actor)],
                "documents":{sid:meta for sid,meta in documents.items() if meta['version_id'] in ids and sid not in services.knowledge.revoked_sources},
                "mode":services.mode,"release_id":services.knowledge.release_id,"claims_enabled":bool(services.claims.policies)}

    @app.post("/api/enterprise/ask")
    def ask(body:Ask,actor=Depends(authenticate)):
        return call(lambda:services.rag.ask(actor,body))

    @app.get("/api/enterprise/conversations/{conversation_id}")
    def history(conversation_id:str,actor=Depends(authenticate)):
        def read():
            rows=services.store.history(actor,"chat",conversation_id)
            for row in rows:
                services.knowledge.validate_output(actor,row)
            return {"turns":rows}
        return call(read)

    @app.post("/api/enterprise/cases")
    def create_case(body:CaseCreate,actor=Depends(authenticate)):
        return call(lambda:services.claims.create(actor,body))

    @app.get("/api/enterprise/cases/{case_id}")
    def get_case(case_id:str,actor=Depends(authenticate)):
        return call(lambda:services.claims.get(actor,case_id))

    @app.put("/api/enterprise/cases/{case_id}/materials")
    def materials(case_id:str,body:MaterialsUpdate,actor=Depends(authenticate)):
        return call(lambda:services.claims.replace_materials(actor,case_id,body))

    @app.post("/api/enterprise/cases/{case_id}/run")
    def run_case(case_id:str,body:CaseRun,actor=Depends(authenticate)):
        return call(lambda:services.claims.run(actor,case_id,body))

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return app
