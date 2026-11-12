"""Load the frozen full corpus without changing canonical text or compiler output."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from rag_assist.v2.business_chunks import validate_business_document
from rag_assist.v2.parsers import adapt_public_document
from rag_assist.v2.public_index import _validate
from rag_assist.v2.structure_pipeline import verify_windows


def sha(value):
    return sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked(root,relative,expected=None):
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError("CORPUS_PATH_ESCAPE")
    data=path.read_bytes()
    if expected and sha(data)!=expected:raise ValueError("CORPUS_FILE_HASH_MISMATCH")
    return data


@dataclass
class Corpus:
    root: Path
    documents: dict
    counts: dict
    fingerprint: str
    model_fingerprint: str


def validate_source(document,count_tokens=None):
    physical,artifact=document["physical"],document["artifact"]
    _validate(physical)
    if artifact["input"]["source_id"]!=physical["source_id"] or artifact["input"]["raw_pdf_sha256"]!=physical["raw_pdf_sha256"]:
        raise ValueError("CORPUS_SOURCE_IDENTITY_MISMATCH")
    view=adapt_public_document(physical)
    business=artifact["business_document"]
    validate_business_document(view,business,product_id=business["product_id"])
    for window in artifact["windows"]:
        if window["product_id"]!=business["product_id"] or window["text_revision_id"]!=business["text_revision_id"]:
            raise ValueError("CORPUS_WINDOW_SCOPE_MISMATCH")
    if count_tokens:verify_windows(view,business,artifact["windows"],count_tokens)
    document["view"]=view
    return document


def load_corpus(root):
    root=Path(root).resolve()
    marker=read_json(root/"runtime/structure/full-corpus-latest.json")
    report_data=checked(root,marker["report_path"],marker["report_sha256"])
    report=json.loads(report_data)
    ledger_data=checked(root,"evaluation/results/v2-public-physical-coverage.json",report["input_ledger_sha256"])
    ledger=json.loads(ledger_data)
    builds={d["source_id"]:d for d in report["documents"]}
    if set(builds)!={d["source_id"] for d in ledger["documents"]}:raise ValueError("CORPUS_DOCUMENT_SET_MISMATCH")
    from tokenizers import Tokenizer
    from rag_assist.localstack.models import EMBEDDING
    for name,digest in report["tokenizer_assets"].items():
        if sha((EMBEDDING/name).read_bytes())!=digest:raise ValueError("CORPUS_TOKENIZER_MISMATCH")
    tokenizer=Tokenizer.from_file(str(EMBEDDING/"tokenizer.json"));tokenizer.no_truncation()
    documents={};counts=dict(documents=0,pages=0,units=0,windows=0,indexable_windows=0,image_review_pages=0)
    for record in ledger["documents"]:
        sid=record["source_id"];build=builds[sid]
        checked(root,record["pdf_path"],record["expected_sha256"])
        physical=json.loads(checked(root,record["derived_path"],record["derived_sha256"]))
        artifact=json.loads(checked(root,build["artifact_path"],build["artifact_sha256"]))
        doc=validate_source({"record":record,"build":build,"physical":physical,"artifact":artifact},
                            lambda text:len(tokenizer.encode(text).ids))
        documents[sid]=doc
        counts["documents"]+=1;counts["pages"]+=physical["physical_page_count"]
        counts["units"]+=len(artifact["business_document"]["units"])
        counts["windows"]+=len(artifact["windows"])
        counts["indexable_windows"]+=sum(w["retrieval_eligible"] for w in artifact["windows"])
        counts["image_review_pages"]+=len(build.get("image_review_pages",[]))
    model_manifest=checked(root,"runtime/smallstack/models/bge-m3-readonly-manifest.json")
    return Corpus(root,documents,counts,sha(report_data+ledger_data),sha(model_manifest))
