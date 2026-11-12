"""Bridge the existing real four-route index without relabeling its BGE/VL status."""
from pathlib import Path
import re
from types import SimpleNamespace

from rag_assist.localstack.backends import TEXT_INDEX,BackendStore
from rag_assist.localstack.config import Settings
from rag_assist.localstack.__main__ import prepare
from rag_assist.localstack.models import LocalModels
from rag_assist.localstack.retrieval import fuse_routes
from .claims import ClaimsService
from .contracts import Actor
from .knowledge import Knowledge,TOPICS
from .rag import RagService
from .state import StateStore


class PublicFourRoutes:
    def __init__(self,store,models):
        if models.status()["embedding"]["model_id"]!="BAAI/bge-m3":
            raise ValueError("INDEX_EMBEDDING_SPACE_MISMATCH")
        self.store,self.models=store,models

    def __call__(self,task,plan):
        baseline=task.variants[0].text
        vector=self.models.encode([baseline])[0]
        # This is the existing 1024-dimensional BGE index, not a Qwen index.
        if len(vector)!=1024:
            raise ValueError("INDEX_EMBEDDING_SPACE_MISMATCH")
        result=self.store.retrieve(task.version_id,baseline,[task.topic],[],vector,limit=20)
        routes=result["routes"]
        diagnostics=[]
        for variant in task.variants[1:]:
            scoped=[{"term":{"bundle_id":self.store.bundle}},{"term":{"product_id":task.version_id}}]
            qvec=self.models.encode([variant.text])[0]
            if len(qvec)!=1024:
                raise ValueError("INDEX_EMBEDDING_SPACE_MISMATCH")
            hits=[]
            dense=self.store._os(TEXT_INDEX+"/_search","POST",{"size":20,"query":{"knn":{"vector":{
                "vector":qvec,"k":20,"filter":{"bool":{"filter":scoped}}}}}})
            hits.extend(dense["hits"]["hits"])
            if variant.channel!="DENSE_TEXT_ONLY":
                bm=self.store._os(TEXT_INDEX+"/_search","POST",{"size":20,"query":{"bool":{
                    "filter":scoped,"must":[{"match":{"text":variant.text}}]}}})
                hits.extend(bm["hits"]["hits"])
            # Variants share the one text-route vote and its bounded candidate set.
            existing={r["evidence_id"] for r in routes["text"]["candidates"]}
            for hit in hits:
                uid=hit["_source"]["evidence_id"]
                if uid not in existing and len(existing)<task.budget["text"]:
                    routes["text"]["candidates"].append({"evidence_id":uid,"match_text":hit["_source"]["text"]})
                    existing.add(uid)
            if routes["text"]["candidates"] and routes["text"]["status"]=="EMPTY":
                routes["text"]["status"]="OK"
        fused=fuse_routes(routes)
        rows={r["evidence_id"]:r for r in self.store.evidence([r["evidence_id"] for r in fused],task.version_id)}
        candidates=[]
        for item in fused[:60]:
            if item["evidence_id"] not in rows:
                raise ValueError("INDEX_EVIDENCE_MISSING")
            row={**rows[item["evidence_id"]],**item}
            for match in item["route_matches"]:
                if match.get("match_text") and match["match_text"] in row["text"]:
                    row["match_text"]=match["match_text"]
                    break
            candidates.append(row)
        ranked=self.models.rerank(baseline,candidates)
        if any(r.get("rerank_status")!="OK" for r in ranked):
            raise ValueError("RERANK_INCOMPLETE")
        return {"routes":routes,"candidate_ids":[r["evidence_id"] for r in ranked],
                "backend_mode":"PUBLIC_LIVE_BGE","diagnostics":diagnostics,
                "models":{"embedding":"BAAI/bge-m3","reranker":"BAAI/bge-reranker-v2-m3",
                          "visual_embedding_executed":False},"reranked_candidates":len(ranked)}


def public_services(path=None,*,gateway=None,postgres_state=True):
    backend=BackendStore(Settings.from_runtime())
    if not backend.health()["ready"]:
        raise RuntimeError("PUBLIC_BACKENDS_NOT_READY")
    documents,seed,_,_=prepare()
    products,sources,units=[],[],[]
    for p,doc in zip(seed["products"],documents):
        pid=p["product_id"]
        products.append({"version_id":pid,"family_id":pid,"title":p["title"],"aliases":p["aliases"],
                         "plan_ids":[],"valid_from":"0001-01-01","validity_verified":False})
        sources.append({"source_id":p["source_id"],"version_id":pid,"filename":Path(p["pdf_path"]).name,
                        "text":doc["canonical_text"],"sha256":doc["canonical_sha256"],"data_class":"PUBLIC",
                        "raw_file_sha256":p["raw_pdf_sha256"],"source_url":p["source_url"]})
        records=[r for r in seed["clause_records"] if r["product_id"]==pid]
        layouts=[r for r in seed["layout_items"] if r["product_id"]==pid]
        page_ids={page["page_number"]:"context:"+pid+":"+str(page["page_number"]) for page in doc["pages"]}
        for page in doc["pages"]:
            if page["page_number"] in p["selected_pages"] and page["text"].strip():
                units.append({"unit_id":page_ids[page["page_number"]],"source_id":p["source_id"],"version_id":pid,
                    "text":page["text"],"start":page["start"],"end":page["end"],"page_number":page["page_number"],
                    "content_sha256":page["content_sha256"],"topics":[],"role":"CONTEXT",
                    "blocks":[{"text":u["text"],"bbox":u["bbox"]} for u in doc["units"] if u["page_number"]==page["page_number"]]})
        for block in doc["units"]:
            if block["page_number"] not in p["selected_pages"] or not block["text"].strip():
                continue
            owned=[r for r in records if block["unit_id"] in r["evidence_ids"]]
            spatial=[r for r in layouts if block["unit_id"] in r["evidence_ids"]]
            registered=" ".join(r["title"]+" "+" ".join(r["aliases"]) for r in owned)
            topics=[topic for topic,aliases in TOPICS.items() if any(a in registered for a in aliases)]
            pages={block["page_number"]}|{n for r in owned+spatial for n in r.get("context_pages",[])}
            units.append({"unit_id":block["unit_id"],"source_id":p["source_id"],"version_id":pid,"text":block["text"],
                "start":block["start"],"end":block["end"],"page_number":block["page_number"],"content_sha256":block["content_sha256"],
                "topics":topics,"role":"CLAUSE","bbox":block["bbox"],"dependencies":[page_ids[n] for n in sorted(pages)],
                "layout_kind":spatial[0]["kind"] if spatial else None})
    knowledge=Knowledge(products,sources,units,release_id=seed["bundle_id"],retriever=PublicFourRoutes(backend,LocalModels()))
    store=StateStore(connection_factory=backend._pg) if postgres_state else StateStore(path)
    actor=Actor(tenant_id="public-reference",user_id="local-reviewer",families=[p["family_id"] for p in products],roles=["UW"])
    return SimpleNamespace(actor=actor,knowledge=knowledge,store=store,rag=RagService(store,knowledge,gateway),
                           claims=ClaimsService(store,knowledge,{}),mode="PUBLIC_LIVE_BGE",backend=backend)
