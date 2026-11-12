"""Physical PDF preservation, never automatic insurance-rule interpretation.

This is the fallback reader while the layout/OCR bake-off is pending. Library
reading order, visual glyph fidelity, table structure and semantics are NOT
certified. Every physical page is accounted for, including pages needing OCR.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import fitz


EXTRACTOR = "pymupdf-physical-fallback-v1"


def load_public_manifest(path) -> list[dict]:
    """Normalize acquired inventories; this does not certify website authority.

    Original fields are retained. Both acquisition schemas have been used by
    the project; inconsistent aliases, upgraded states and escaped paths fail.
    File bytes and actual PDF page count are checked by the extraction stage.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("EMPTY_MANIFEST")
    result, seen_ids = [], set()

    def alias(record, *names):
        values = [record[n] for n in names if n in record]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError("CONFLICTING_MANIFEST_FIELD: " + "/".join(names))
        return values[0] if values else None

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("INVALID_MANIFEST_RECORD")
        source_id = alias(record, "doc_id", "document_id")
        # These identifiers become local filenames, so Windows drive/ADS syntax is forbidden.
        if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", source_id) or source_id in {".", ".."}:
            raise ValueError("INVALID_SOURCE_ID")
        if source_id in seen_ids:
            raise ValueError("DUPLICATE_SOURCE_ID: " + source_id)
        seen_ids.add(source_id)
        scope = record.get("data_class", record.get("acquisition_scope",
                     manifest.get("data_class", manifest.get("acquisition_scope"))))
        publication = record.get("publication_state", manifest.get("publication_state"))
        if scope != "PUBLIC_REFERENCE" or publication != "REFERENCE_ONLY" or record.get("status", "DOWNLOADED") != "DOWNLOADED":
            raise ValueError("UNTRUSTED_MANIFEST_STATE")
        relative = alias(record, "pdf_path", "pdf_file")
        if not isinstance(relative, str) or not relative:
            raise ValueError("INVALID_PDF_PATH")
        relative_path = Path(relative)
        pdf_path = (path.parent / relative_path).resolve()
        if relative_path.is_absolute() or relative_path.drive or not pdf_path.is_relative_to(path.parent):
            raise ValueError("MANIFEST_PATH_ESCAPE")
        if not pdf_path.is_file():
            raise ValueError("PDF_FILE_MISSING")
        expected_sha = record.get("sha256")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha):
            raise ValueError("INVALID_SOURCE_HASH")
        pages = record.get("page_count")
        if pages is not None and (type(pages) is not int or pages < 1):
            raise ValueError("INVALID_PAGE_COUNT")
        source_url = record.get("url", record.get("source_url"))
        if not isinstance(source_url, str):
            raise ValueError("INVALID_SOURCE_URL")
        url = urlsplit(source_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("INVALID_SOURCE_URL")
        title = record.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("INVALID_SOURCE_TITLE")
        result.append(dict(source_id=source_id, title=title, pdf_path=pdf_path, source_url=source_url,
                           expected_sha256=expected_sha.lower(), expected_pages=pages,
                           acquisition_record=record))
    return result


@dataclass(frozen=True)
class PdfLimits:
    max_pages: int = 2000
    max_bytes: int = 100_000_000
    max_characters: int = 10_000_000

    def __post_init__(self):
        for name in ("max_pages", "max_bytes", "max_characters"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")


def extract_public_pdf(path, *, source_id: str, expected_sha256: str,
                       limits: PdfLimits = PdfLimits()) -> dict:
    """Return unreviewed page parents and exact text-block quote spans.

    `expected_sha256` binds the downloaded original. Canonical text has its own
    hash and offsets, explicitly distinct from PDF byte offsets. A budget error
    rejects the whole document rather than returning a partial success.
    """
    if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", source_id):
        raise ValueError("INVALID_SOURCE_ID")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ValueError("INVALID_SOURCE_HASH")
    path = Path(path)
    if path.stat().st_size > limits.max_bytes:
        raise ValueError("FILE_SIZE_LIMIT")
    raw = path.read_bytes()
    if len(raw) > limits.max_bytes:
        raise ValueError("FILE_SIZE_LIMIT")
    original_hash = sha256(raw).hexdigest()
    if original_hash != expected_sha256.lower():
        raise ValueError("SOURCE_HASH_MISMATCH")
    namespace = f"{source_id}:{original_hash}:{EXTRACTOR}"
    pages, units, canonical_parts = [], [], []
    offset = 0
    with fitz.open(stream=raw, filetype="pdf") as document:
        if document.is_encrypted or document.needs_pass:
            raise ValueError("ENCRYPTED_PDF")
        if len(document) > limits.max_pages:
            raise ValueError("PAGE_LIMIT")
        for page_index, page in enumerate(document):
            if page_index:
                canonical_parts.append("\f")
                offset += 1
            page_id = f"{namespace}:p{page_index + 1}"
            page_start = offset
            page_parts = []
            # Avoid embedding decoded image bytes in the text extraction result.
            flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
            layout = page.get_text("dict", flags=flags, sort=True)
            for block_index, block in enumerate(layout["blocks"]):
                if block.get("type") != 0:
                    continue
                lines = block.get("lines", [])
                text = "\n".join("".join(span["text"] for span in line.get("spans", []))
                                 for line in lines)
                if not text:
                    continue
                if page_parts:
                    page_parts.append("\n")
                    offset += 1
                start = offset
                offset += len(text)
                if offset > limits.max_characters:
                    raise ValueError("CHARACTER_LIMIT")
                page_parts.append(text)
                units.append({
                    "unit_id": f"{page_id}:b{block_index + 1}",
                    "source_id": source_id, "raw_pdf_sha256": original_hash,
                    "parent_page_id": page_id, "page_number": page_index + 1,
                    "bbox": list(block["bbox"]),
                    "line_bboxes": [list(line["bbox"]) for line in lines],
                    "text": text, "start": start, "end": offset,
                    "offset_basis": "CANONICAL_TEXT_UNICODE_CODEPOINTS",
                    "content_sha256": sha256(text.encode("utf-8")).hexdigest(),
                    "business_labels": [], "semantic_review_state": "UNREVIEWED",
                })
            if offset > limits.max_characters:
                raise ValueError("CHARACTER_LIMIT")
            page_text = "".join(page_parts)
            canonical_parts.append(page_text)
            images = [{"bbox": list(item["bbox"]),
                       "state": "IMAGE_CONTENT_UNREVIEWED"}
                      for item in page.get_image_info(hashes=False)]
            if page_text.strip():
                state = "TEXT_EXTRACTED_UNREVIEWED"
            else:
                state = "OCR_REQUIRED" if images else "NO_TEXT_REVIEW_REQUIRED"
            pages.append({
                "page_id": page_id, "page_number": page_index + 1,
                "width": page.rect.width, "height": page.rect.height,
                "text": page_text, "start": page_start, "end": offset,
                "content_sha256": sha256(page_text.encode("utf-8")).hexdigest(),
                "coverage_state": state, "image_regions": images,
                "image_review_required": bool(images),
            })
    canonical_text = "".join(canonical_parts)
    return {
        "source_id": source_id, "raw_pdf_sha256": original_hash,
        "raw_pdf_bytes": len(raw), "canonical_text": canonical_text,
        "canonical_sha256": sha256(canonical_text.encode("utf-8")).hexdigest(),
        "physical_page_count": len(pages), "pages": pages, "units": units,
        "data_class": "PUBLIC_REFERENCE", "publication_state": "REFERENCE_ONLY",
        "answer_ceiling": "SOURCE_ONLY", "semantic_complete": False,
        "production_readiness": "NO_GO", "extractor": EXTRACTOR,
        "extractor_library_version": fitz.VersionBind,
        "quality_flags": ["LIBRARY_SORTED_READING_ORDER_UNVERIFIED",
                          "VISUAL_TEXT_FIDELITY_UNVERIFIED", "TABLE_STRUCTURE_UNVERIFIED",
                          "BUSINESS_SEMANTICS_UNREVIEWED"],
    }
