"""Source-bound condition closure; projections and graph candidates are not quotations."""
from collections import defaultdict
from datetime import date
from hashlib import sha256
import re

from rag_assist.v2.business_chunks import _identity,collect_business_evidence,FLAGS
from .catalogue import compact
from .contracts import Evidence,ProductVersion,Source
from .knowledge import Knowledge,TOPICS,digest


def condition_document(original):
    units=original["units"];clauses=defaultdict(list);definitions={}
    for u in units:
        if u["kind"]=="CLAUSE_CANDIDATE":
            number=compact(u.get("metadata",{}).get("clause_no", ""))
            if number:clauses[number].append(u)
        path=" ".join(u.get("heading_path",[]))
        name=compact(u["title"])
        foot=re.match(r"^\d+([^：:]{2,24})[：:]",compact(u["projection_text"])) if u["kind"]=="FOOTNOTE_CANDIDATE" else None
        if foot:name=foot[1]
        if ("释义" in path or foot) and 2<=len(name)<=24 and not re.fullmatch(r"[一二三四五六七八九十百\d.]+",name):
            definitions.setdefault(name,[]).append(u)
    edges=list(original["dependencies"]);seen={(e["from_unit_id"],e["to_unit_id"]) for e in edges}
    for u in units:
        if not u.get("retrieval_eligible") or not u["source_spans"]:continue
        text=compact(u["projection_text"])
        references=set(re.findall(r'[“"《]\s*(\d+(?:\.\d+)+)',text))
        references.update(re.findall(r'第(\d+(?:\.\d+)+)条',text))
        targets=[]
        for number in references:
            found=clauses.get(number,[])
            targets.append((found[0] if len(found)==1 else None,"EXPLICIT_CLAUSE_REFERENCE_CANDIDATE",number))
        for name,found in definitions.items():
            if name in text and len(found)==1 and found[0]["unit_id"]!=u["unit_id"]:
                targets.append((found[0],"DEFINITION_MENTION_CANDIDATE",name))
        for target,relation,reference in targets:
            uid=target["unit_id"] if target else None
            if uid==u["unit_id"] or (u["unit_id"],uid) in seen:continue
            edge={"from_unit_id":u["unit_id"],"to_unit_id":uid,"relation":relation,
                  "source_spans":u["source_spans"],"target_ref":reference,
                  "state":"SOURCE_BOUND_CANDIDATE",**FLAGS}
            edge["dependency_id"]=_identity("business-edge-v1:",edge)
            edges.append(edge);seen.add((u["unit_id"],uid))
    result={**original,"dependencies":edges}
    result["document_id"]=_identity("business-document-v1:",{k:v for k,v in result.items() if k!="document_id"})
    return result


class CorpusKnowledge(Knowledge):
    primary_limit=3
    soft_topic_labels=True

    def __init__(self,corpus,catalogue,retriever=None):
        self.corpus,self.catalogue=corpus,catalogue
        self.products={p["version_id"]:ProductVersion.model_validate(p) for p in catalogue["products"]}
        self.sources={};self.units={};self.business={};self.unit_sources={};self.unit_metadata={}
        self.release_id=catalogue["release_id"];self.revoked_sources=set();self.retriever=retriever
        for sid,doc in corpus.documents.items():
            physical=doc["physical"];meta=catalogue["documents"][sid];vid=meta["version_id"]
            self.sources[sid]=Source(source_id=sid,version_id=vid,filename=corpus.root.joinpath(doc["record"]["pdf_path"]).name,
                text=physical["canonical_text"],sha256=physical["canonical_sha256"],source_url=doc["record"]["source_url"],
                raw_file_sha256=physical["raw_pdf_sha256"],data_class="PUBLIC")
            self.business[sid]=condition_document(doc["artifact"]["business_document"])
            for u in self.business[sid]["units"]:
                if u["kind"]=="DOCUMENT" or not u["source_spans"]:continue
                span=u["source_spans"][0];text=physical["canonical_text"][span["start"]:span["end"]]
                heading=compact(" ".join([u["title"],*u.get("heading_path",[])]))
                topics=[topic for topic,aliases in TOPICS.items() if any(a in heading for a in aliases)]
                role="TABLE" if u["kind"].startswith("TABLE") else "FOOTNOTE" if u["kind"]=="FOOTNOTE_CANDIDATE" else "CLAUSE"
                self.units[u["unit_id"]]=Evidence(unit_id=u["unit_id"],source_id=sid,version_id=vid,text=text,
                    start=span["start"],end=span["end"],content_sha256=digest(text),page_number=span["page_number"],
                    topics=topics,aliases=[u["title"]],role=role,bbox=span.get("bbox"))
                self.unit_sources[u["unit_id"]]=sid
                self.unit_metadata[u["unit_id"]]=u

    def package(self,actor,roots,as_of,plan_id,*,max_chars=18000):
        grouped=defaultdict(list)
        for root in roots:
            if root not in self.units:raise ValueError("DEPENDENCY_MISSING")
            u=self.units[root];self.allowed(actor,u.version_id,as_of,plan_id)
            if u.source_id in self.revoked_sources:raise PermissionError("SOURCE_REVOKED")
            grouped[u.source_id].append(root)
        packages=[];evidence=[];missing=[];total=0
        for sid,ids in grouped.items():
            doc=self.corpus.documents[sid];business=self.business[sid]
            packet=collect_business_evidence(doc["view"],business,ids,product_id=business["product_id"],max_chars=max_chars)
            source=self.sources[sid]
            if digest(source.text)!=source.sha256:raise ValueError("SOURCE_INTEGRITY")
            if self.catalogue["documents"][sid]["attachment_version_state"]=="NAME_MATCH_REFERENCE_ONLY" and \
                    self.catalogue["documents"][sid]["source_role"]!="TERMS":
                packet["risks"]+=['ATTACHMENT_VERSION_REVIEW_REQUIRED']
                packet["status"]='HUMAN_REVIEW'
            for part in packet["source_parts"]:
                text=source.text[part["start"]:part["end"]]
                if part["text"]!=text or digest(text)!=part["content_sha256"]:raise ValueError("PACKAGE_SOURCE_SPAN_MISMATCH")
                eid="quote:"+sha256(f"{sid}:{part['start']}:{part['end']}".encode()).hexdigest()
                evidence.append({"unit_id":eid,"source_id":sid,"version_id":source.version_id,"text":text,
                    "start":part["start"],"end":part["end"],"page_number":part["page_number"],"content_sha256":digest(text),
                    "filename":source.filename,"source_url":source.source_url,"source_sha256":source.sha256,
                    "raw_file_sha256":source.raw_file_sha256,"data_class":"PUBLIC","role":"CONTEXT","bbox":part.get("bbox"),
                    "blocks":[{"text":text,"bbox":part.get("bbox",[])}],"condition_package_ref":sid})
            total+=packet["generation_context_characters"]
            missing.extend(packet["missing_references"]);missing.extend(packet["risks"])
            packages.append({"source_id":sid,"version_id":source.version_id,**packet})
        if total>max_chars:missing.append("CONTEXT_BUDGET_EXCEEDED")
        return {"evidence":evidence,"condition_packages":packages,"missing_info":sorted(set(missing)),
                "generation_allowed":bool(packages) and total<=max_chars and all(p["status"]=="READY_FOR_SOURCE_REVIEW" for p in packages),
                "semantic_complete":False,"requires_human_review":True,"context_characters":total,
                "completeness":"REGISTERED_STRUCTURE_AND_LITERAL_REFERENCES_ONLY"}

    def hydrate(self,actor,roots,as_of,plan_id,*,max_chars=18000):
        return self.package(actor,roots,as_of,plan_id,max_chars=max_chars)["evidence"]

    def validate_output(self,actor,output):
        if output.get("release_id")!=self.release_id:raise PermissionError("KNOWLEDGE_RELEASE_CHANGED")
        plan=output.get("plan") or {};day=date.fromisoformat(plan.get("as_of",date.today().isoformat()))
        for e in output.get("evidence",[]):
            self.allowed(actor,e["version_id"],day,plan.get("plan_id"))
            source=self.sources.get(e["source_id"])
            if source is None or source.source_id in self.revoked_sources:raise PermissionError("SOURCE_REVOKED")
            if source.version_id!=e["version_id"] or digest(source.text)!=source.sha256 or \
                    source.text[e["start"]:e["end"]]!=e["text"] or digest(e["text"])!=e["content_sha256"]:
                raise ValueError("REPLAY_SOURCE_MISMATCH")
        for row in output.get("results",[]):self.allowed(actor,row["version_id"],day,plan.get("plan_id"))
