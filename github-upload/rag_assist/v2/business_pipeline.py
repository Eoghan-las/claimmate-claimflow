"""Local business-chunk build. No LLM, database writes, or embedding inference."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
from time import perf_counter

from .business_chunks import build_business_document, build_business_windows
from .business_layout import extract_business_layout
from .document_structure import extract_document_structure
from .parsers import adapt_public_document


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def run_public_business_chunking(prepared, *, root, max_tokens=256):
    from transformers import AutoTokenizer
    from rag_assist.localstack.models import EMBEDDING
    root=Path(root).resolve()
    started=perf_counter()
    documents,seed,_,identities=prepared
    assets={name:sha256((EMBEDDING/name).read_bytes()).hexdigest() for name in
            ("tokenizer.json","tokenizer_config.json","sentencepiece.bpe.model") if (EMBEDDING/name).is_file()}
    tokenizer=AutoTokenizer.from_pretrained(str(EMBEDDING),local_files_only=True)
    versions={name:metadata.version(name) for name in ("transformers","tokenizers")}
    tokenizer_id="BGE_M3_BUSINESS_TOKENIZER:"+sha256(json.dumps({"assets":assets,"versions":versions},sort_keys=True).encode()).hexdigest()
    count=lambda text:len(tokenizer.encode(text,add_special_tokens=True,truncation=False))
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder=root/"runtime/business-chunking"/stamp
    report={"kind":"LOCAL_BUSINESS_CHUNKING_NOT_BUSINESS_APPROVAL","started_at_utc":datetime.now(timezone.utc).isoformat(),
            "build_dir":str(folder),"tokenizer_id":tokenizer_id,"tokenizer_assets":assets,"versions":versions,
            "max_tokens":max_tokens,"llm_calls":0,"embedding_inference_calls":0,"online_index_rebuilt":False,
            "documents":[],"errors":[],"semantic_complete":False,"answer_ceiling":"SOURCE_ONLY"}
    report["implementation_sha256"]={name:sha256((root/"rag_assist/v2"/name).read_bytes()).hexdigest()
                                     for name in ("business_chunks.py","business_layout.py","business_pipeline.py","document_structure.py")}
    by_source={d["source_id"]:d for d in documents}
    for product in seed["products"]:
        pid=product["product_id"]
        try:
            physical=by_source[product["source_id"]]
            view=adapt_public_document(physical)
            layouts=[r for r in seed["layout_items"] if r["product_id"]==pid]
            layout=extract_business_layout(root/product["pdf_path"],view,pages=product["selected_pages"],registered_layouts=layouts)
            structure=extract_document_structure(root/product["pdf_path"],view,pages=product["selected_pages"])
            business=build_business_document(view,product_id=pid,layout=layout,structure=structure,pages=product["selected_pages"])
            windows=build_business_windows(view,business,count_tokens=count,tokenizer_id=tokenizer_id,max_tokens=max_tokens)
            artifact={"business_document":business,"windows":windows,"layout":layout,"structure":structure,
                      "source_identity":next(r for r in identities if r["source_id"]==view.source_id)}
            target=folder/(pid+".json")
            digest=_write(target,artifact)
            report["documents"].append({"product_id":pid,"source_id":view.source_id,"selected_pages":product["selected_pages"],
                "artifact_path":str(target.relative_to(root)),"artifact_sha256":digest,"document_id":business["document_id"],
                "units":dict(Counter(u["kind"] for u in business["units"])),"windows":len(windows),
                "max_actual_tokens":max((w["token_count"] for w in windows),default=0),"coverage":business["coverage"],
                "dependencies":len(business["dependencies"]),"unresolved_dependencies":sum(e["to_unit_id"] is None for e in business["dependencies"]),
                "risks":business["risks"],"structure_engine":structure["engine"],"registered_clause_overrides":False,
                "layout_cells":sum(len(t["cells"]) for o in layout["objects"] for t in o.get("metadata",{}).get("tables",[]))})
        except Exception as error:
            report["errors"].append({"product_id":pid,"type":type(error).__name__,"error":str(error)})
    report["seconds"]=round(perf_counter()-started,3)
    report["status"]="BUILT_FOR_SOURCE_REVIEW" if not report["errors"] and len(report["documents"])==len(seed["products"]) else "FAILED"
    _write(folder/"build-report.json",report)
    _write(root/"runtime/business-chunking/build-report.json",report)
    if report["status"]=="BUILT_FOR_SOURCE_REVIEW":
        _write(root/"runtime/business-chunking/latest.json",{"report_path":str((folder/"build-report.json").relative_to(root)),
               "report_sha256":sha256((folder/"build-report.json").read_bytes()).hexdigest()})
    return report
