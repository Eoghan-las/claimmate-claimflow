"""Reproducible offline structure/chunk build over the registered public corpus.

Run: python -B -m rag_assist.v2.structure_pipeline
No registered clause boundaries, LLM calls, embedding inference or index writes.
"""
import argparse
from collections import Counter, defaultdict
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


def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _contained(root, path):
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError("INPUT_PATH_OUTSIDE_WORKSPACE")
    return result


def verify_windows(view, business, windows, count_tokens):
    """Independently read back quotes, budgets, complete physical coverage."""
    units = {u["unit_id"]: u for u in business["units"]}
    ranges = defaultdict(list)
    for window in windows:
        unit = units[window["unit_id"]]
        parts = []
        for ref in window["source_spans"]:
            if not any(p["block_id"] == ref["block_id"] and p["start"] <= ref["start"] and p["end"] >= ref["end"]
                       for p in unit["source_spans"]):
                raise ValueError("WINDOW_QUOTE_OUTSIDE_UNIT")
            text = view.canonical_text[ref["start"]:ref["end"]]
            if sha256(text.encode()).hexdigest() != ref["content_sha256"]:
                raise ValueError("WINDOW_QUOTE_HASH_MISMATCH")
            parts.append(text)
            ranges[ref["block_id"]].append((ref["start"], ref["end"]))
        body = "\n".join(parts)
        if body != window["body_text"] or window["search_text"] != window["retrieval_prefix"] + body:
            raise ValueError("WINDOW_READING_PROJECTION_MISMATCH")
        if count_tokens(window["search_text"]) != window["token_count"] or window["token_count"] > window["max_tokens"]:
            raise ValueError("WINDOW_TOKEN_BUDGET_MISMATCH")
    for block in view.blocks:
        if block.span.page_number not in business["selected_pages"] or not block.text.strip():
            continue
        cursor = block.span.start
        for left, right in sorted(ranges[block.unit_id]):
            if left > cursor:
                raise ValueError(f"WINDOW_SOURCE_COVERAGE_GAP:{block.unit_id}:{cursor}:{left}")
            cursor = max(cursor, right)
        if cursor != block.span.end:
            raise ValueError(f"WINDOW_SOURCE_COVERAGE_GAP:{block.unit_id}:{cursor}:{block.span.end}")


def run_structure_corpus(root, *, source_ids=None, docling_cache=None, run_scenarios=True):
    from transformers import AutoTokenizer
    from rag_assist.localstack.models import EMBEDDING
    root = Path(root).resolve()
    started = perf_counter()
    ledger_path = root / "evaluation/results/v2-public-physical-coverage.json"
    ledger = _read(ledger_path)
    if source_ids:
        unknown = set(source_ids) - {r["source_id"] for r in ledger["documents"]}
        if unknown: raise ValueError("UNKNOWN_SOURCE_IDS: " + ",".join(sorted(unknown)))
    rows = [r for r in ledger["documents"] if not source_ids or r["source_id"] in source_ids]
    seed = _read(root / "data/smallstack/seed.json")
    known = {p["source_id"]: p["product_id"] for p in seed["products"]}
    tokenizer = AutoTokenizer.from_pretrained(str(EMBEDDING), local_files_only=True)
    # This code counts tokens; it does not send oversized text into the model.
    tokenizer.model_max_length = 10**9
    assets = {name: sha256((EMBEDDING / name).read_bytes()).hexdigest()
              for name in ("tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model") if (EMBEDDING / name).is_file()}
    versions = {name: metadata.version(name) for name in ("transformers", "tokenizers", "PyMuPDF")}
    tokenizer_id = "BGE_M3_STRUCTURE:" + sha256(json.dumps({"assets": assets, "versions": versions}, sort_keys=True).encode()).hexdigest()
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = root / "runtime/structure" / ("full-corpus-" + stamp)
    report = {"kind": "AUTOMATIC_STRUCTURE_SCENARIOS_NOT_BUSINESS_RECALL", "build_dir": str(folder),
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "input_ledger_sha256": sha256(ledger_path.read_bytes()).hexdigest(),
              "tokenizer_id": tokenizer_id, "tokenizer_assets": assets, "versions": versions,
              "registered_clause_overrides": False, "registered_layout_overrides": False,
              "llm_calls": 0, "embedding_inference_calls": 0, "online_index_rebuilt": False,
              "semantic_complete": False, "answer_ceiling": "SOURCE_ONLY", "documents": []}
    names = ("business_chunks.py", "business_layout.py", "document_structure.py", "docling_adapter.py", "structure_pipeline.py")
    report["implementation_sha256"] = {name: sha256((root / "rag_assist/v2" / name).read_bytes()).hexdigest() for name in names}
    artifacts = {}
    for index, row in enumerate(rows, 1):
        tick = perf_counter()
        entry = {"source_id": row["source_id"], "title": row["title"], "pdf_path": row["pdf_path"],
                 "source_url": row["source_url"], "raw_pdf_sha256": row["expected_sha256"]}
        artifact = {"input": entry.copy()}
        try:
            pdf, derived = _contained(root, row["pdf_path"]), _contained(root, row["derived_path"])
            if row["status"] != "PHYSICALLY_MATERIALIZED" or sha256(pdf.read_bytes()).hexdigest() != row["expected_sha256"]:
                raise ValueError("RAW_PDF_MANIFEST_MISMATCH")
            payload = derived.read_bytes()
            if sha256(payload).hexdigest() != row["derived_sha256"]: raise ValueError("DERIVED_MANIFEST_MISMATCH")
            view = adapt_public_document(json.loads(payload))
            if view.source_id != row["source_id"] or view.raw_pdf_sha256 != row["expected_sha256"]:
                raise ValueError("CANONICAL_SOURCE_SCOPE_MISMATCH")
            scope = known.get(view.source_id, "document:" + view.source_id)
            evidence = None
            if docling_cache is not None:
                from .docling_adapter import find_cached_docling_evidence
                evidence = find_cached_docling_evidence(pdf, cache_dir=Path(docling_cache))
            structure = extract_document_structure(pdf, view, docling_document=evidence)
            layout = extract_business_layout(pdf, view)
            business = build_business_document(view, product_id=scope, layout=layout, structure=structure)
            windows = build_business_windows(view, business, count_tokens=count, tokenizer_id=tokenizer_id)
            artifact.update(structure=structure, layout=layout, business_document=business, windows=windows)
            verify_windows(view, business, windows, count)
            artifacts[view.source_id] = artifact
            entry.update(status="BUILT_FOR_SOURCE_REVIEW", scope_id=scope,
                         scope_binding="REGISTERED_PRODUCT_DOCUMENT" if view.source_id in known else "ISOLATED_DOCUMENT_NOT_PRODUCT_MATCHED",
                         pages_processed=view.physical_page_count, physical_blocks=len(view.blocks),
                         business_units=len(business["units"]), unit_kinds=dict(Counter(u["kind"] for u in business["units"])),
                         structure_engine=structure["engine"], structure_elements=len(structure["elements"]),
                         structure_kinds=dict(Counter(e["kind"] for e in structure["elements"])),
                         windows=len(windows), retrieval_eligible_windows=sum(w["retrieval_eligible"] for w in windows),
                         max_actual_tokens=max((w["token_count"] for w in windows), default=0),
                         source_character_coverage=business["coverage"], risks=business["risks"],
                         image_review_pages=[p.page_number for p in view.pages if p.image_review_required],
                         ocr_required_pages=[p.page_number for p in view.pages if p.coverage_state == "OCR_REQUIRED"])
        except Exception as error:
            entry.update(status="FAILED", error={"type": type(error).__name__, "code": str(error)})
        entry["seconds"] = round(perf_counter() - tick, 3)
        path = folder / (row["source_id"] + ".json")
        entry["artifact_sha256"] = _save(path, artifact)
        entry["artifact_path"] = str(path.relative_to(root))
        report["documents"].append(entry)
        _save(folder / "report.json", report)
        print(json.dumps({"index": index, "source_id": row["source_id"], "status": entry["status"],
                          "units": entry.get("business_units"), "windows": entry.get("windows"),
                          "error": entry.get("error"), "seconds": entry["seconds"]}, ensure_ascii=False), flush=True)
    if run_scenarios:
        from evaluation.smallstack.verify_structure_scenarios import run_scenarios as verify_scenarios
        scenarios = verify_scenarios(root, artifacts)
        report["scenario_report_path"] = str((folder / "scenarios.json").relative_to(root))
        report["scenario_report_sha256"] = _save(folder / "scenarios.json", scenarios)
        report["scenario_aggregate"] = scenarios["aggregate"]
    report["summary"] = {"files_requested": len(rows), "pages_requested": sum(r["physical_pages"] for r in rows),
                         "files_built": sum(d["status"] == "BUILT_FOR_SOURCE_REVIEW" for d in report["documents"]),
                         "pages_built": sum(d.get("pages_processed", 0) for d in report["documents"]),
                         "business_units": sum(d.get("business_units", 0) for d in report["documents"]),
                         "windows": sum(d.get("windows", 0) for d in report["documents"]),
                         "engines": dict(Counter(d.get("structure_engine", "FAILED") for d in report["documents"])),
                         "seconds": round(perf_counter() - started, 3)}
    report["status"] = "BUILT_FOR_SOURCE_REVIEW" if report["summary"]["files_built"] == len(rows) else "FAILED"
    report_hash = _save(folder / "report.json", report)
    _save(root / "runtime/structure/full-corpus-latest.json", {"report_path": str((folder / "report.json").relative_to(root)), "report_sha256": report_hash})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--docling-cache", type=Path)
    parser.add_argument("--skip-scenarios", action="store_true")
    args = parser.parse_args()
    report = run_structure_corpus(args.root, source_ids=args.sources, docling_cache=args.docling_cache, run_scenarios=not args.skip_scenarios)
    print(json.dumps({"summary": report["summary"], "scenarios": report.get("scenario_aggregate"), "folder": report["build_dir"]}, ensure_ascii=False))
    failed = report["status"] == "FAILED" or report.get("scenario_aggregate", {}).get("failed", 0)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
