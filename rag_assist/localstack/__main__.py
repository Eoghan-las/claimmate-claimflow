"""Local-stack entry point; no dependency or global-environment mutation."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .config import ROOT, Settings


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def checked_file(relative, prefix, expected_hash=None):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to((ROOT / prefix).resolve()):
        raise ValueError("INPUT_PATH_OUT_OF_SCOPE")
    data = path.read_bytes()
    if expected_hash and sha256(data).hexdigest() != expected_hash:
        raise ValueError("INPUT_HASH_MISMATCH")
    return data


def prepare():
    """Reuse the source-bound R1.2 windows; check their immutable input ledger."""
    from rag_assist.v2.public_index import _validate
    seed = read_json(ROOT / "data/smallstack/seed.json")
    ledger = read_json(ROOT / "evaluation/results/v2-public-physical-coverage.json")
    corpus = read_json(ROOT / "runtime/r1-unit-windows/corpus-final.json")
    physical = {d["source_id"]: d for d in ledger["documents"]}
    artifacts = {d["source_id"]: d for d in corpus["documents"]}
    documents, rows = [], []
    identities = []
    for product in seed["products"]:
        source = product["source_id"]
        record, artifact_record = physical[source], artifacts[source]
        raw = checked_file(product["pdf_path"], "data/public_medical", product["raw_pdf_sha256"])
        if sha256(raw).hexdigest() != record["expected_sha256"]:
            raise ValueError("PDF_LEDGER_MISMATCH")
        derived = checked_file(product["physical_artifact_path"], "runtime/v2-public-ingestion", record["derived_sha256"])
        document = json.loads(derived)
        _validate(document)
        if document["source_id"] != source or document["canonical_sha256"] != product["canonical_sha256"]:
            raise ValueError("SOURCE_IDENTITY_MISMATCH")
        data = checked_file(artifact_record["artifact_path"], "runtime/r1-unit-windows/artifacts-final", artifact_record["artifact_sha256"])
        artifact = json.loads(data)
        if artifact["source_id"] != source or artifact_record["status"] != "PASS" or not artifact_record["unit_window_coverage_complete"]:
            raise ValueError("WINDOW_ARTIFACT_NOT_VALIDATED")
        units = {u["unit_id"]: u for u in document["units"]}
        assembled = {u["unit_id"]: u for u in artifact["assembly"]["units"]}
        seen = set()
        for window in artifact["windows"]:
            leaf = assembled[window["unit_id"]]
            if len(leaf["block_ids"]) != 1:
                raise ValueError("EXPECTED_SINGLE_BLOCK_WINDOW")
            unit = units[leaf["block_ids"][0]]
            if unit["page_number"] not in product["selected_pages"]:
                continue
            if unit["text"][window["start"]:window["end"]] != window["text"] or sha256(window["text"].encode()).hexdigest() != window["input_sha256"]:
                raise ValueError("WINDOW_SOURCE_SPAN_MISMATCH")
            if window["token_count"] > 256:
                raise ValueError("WINDOW_TOKEN_BUDGET_EXCEEDED")
            if not window["text"].strip():
                continue
            seen.add(unit["unit_id"])
            rows.append({"window_id": window["window_id"], "evidence_id": unit["unit_id"],
                         "product_id": product["product_id"], "source_id": source,
                         "page_number": unit["page_number"], "text": window["text"],
                         "token_count": window["token_count"], "tokenizer_id": window["tokenizer_id"]})
        required = {u["unit_id"] for u in document["units"] if u["page_number"] in product["selected_pages"] and u["text"].strip()}
        if seen != required:
            raise ValueError("SELECTED_SEARCHABLE_BLOCK_MISSING")
        documents.append({**document, **{k: product[k] for k in ("product_id", "title", "source_url", "selected_pages")}})
        identities.append({"source_id": source, "raw_pdf_sha256": product["raw_pdf_sha256"],
                           "derived_sha256": record["derived_sha256"], "artifact_sha256": artifact_record["artifact_sha256"],
                           "selected_pages": product["selected_pages"], "searchable_units": len(required)})
    return documents, seed, rows, identities


def main():
    parser = argparse.ArgumentParser(description="ClaimMate 本机小批量系统")
    parser.add_argument("command", choices=["prepare", "chunk", "ingest", "health", "serve"])
    args = parser.parse_args()
    if args.command == "chunk":
        from rag_assist.v2.business_pipeline import run_public_business_chunking
        report = run_public_business_chunking(prepare(), root=ROOT)
        print(json.dumps({"status": report["status"], "documents": len(report["documents"]),
                          "windows": sum(d["windows"] for d in report["documents"]), "seconds": report["seconds"],
                          "errors": report["errors"], "report": str(ROOT / "runtime/business-chunking/build-report.json")}, ensure_ascii=False, indent=2))
        if report["status"] == "FAILED":
            raise SystemExit(1)
        return
    if args.command == "prepare":
        documents, seed, rows, identities = prepare()
        report = {"status": "INPUTS_VERIFIED", "products": len(seed["products"]), "windows": len(rows), "sources": identities,
                  "semantic_complete": False, "answer_ceiling": "SOURCE_ONLY"}
        write_json(ROOT / "runtime/smallstack/prepared-inputs.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    settings = Settings.from_runtime()
    from .backends import BackendStore
    store = BackendStore(settings)
    if args.command == "health":
        print(json.dumps(store.health(), ensure_ascii=False, indent=2))
    elif args.command == "ingest":
        from .models import LocalModels
        documents, seed, rows, identities = prepare()
        models = LocalModels()
        vectors = models.encode([row["text"] for row in rows])
        report = store.ingest(documents, seed, rows, vectors)
        report.update(sources=identities, actual_models=models.status(), created_at_utc=datetime.now(timezone.utc).isoformat(),
                      seed_sha256=sha256((ROOT / "data/smallstack/seed.json").read_bytes()).hexdigest(),
                      semantic_complete=False, answer_ceiling="SOURCE_ONLY")
        write_json(ROOT / "runtime/smallstack/ingest-report.json", report)
        print(json.dumps({k:v for k,v in report.items() if k != "actual_models"}, ensure_ascii=False, indent=2))
    else:
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(settings, store=store), host="127.0.0.1", port=8767, access_log=False)


if __name__ == "__main__":
    main()
