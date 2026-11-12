"""Adapt preserved physical output without OCR, databases, or business labels."""
from collections.abc import Mapping
import math
import re
from typing import Any

from .base import (OFFSET_BASIS, BlockView, CanonicalTextView, ImageRegion,
                   PageView, SourceSpan, _bbox, _hash, _require, _revision_id)


def _reference_state(record):
    _require(isinstance(record, Mapping))
    expected = dict(data_class="PUBLIC_REFERENCE", publication_state="REFERENCE_ONLY",
                    answer_ceiling="SOURCE_ONLY", semantic_complete=False,
                    production_readiness="NO_GO", semantic_review_state="UNREVIEWED",
                    business_labels=[], approved=False, business_approved=False,
                    core_approved=False, core_signable=False,
                    unresolved_dependencies="NOT_ASSESSED")
    for key, value in expected.items():
        if key in record:
            _require(type(record[key]) is type(value) and record[key] == value,
                     "UNTRUSTED_REFERENCE_STATE")


def _span(record, canonical, revision, page_number, bbox):
    start, end, text = record["start"], record["end"], record["text"]
    _require(type(start) is int and type(end) is int and 0 <= start <= end <= len(canonical))
    _require(isinstance(text, str) and canonical[start:end] == text)
    _require(record["content_sha256"] == _hash(text))
    return SourceSpan(revision, page_number, start, end, bbox, record["content_sha256"])


def adapt_public_document(document: Mapping[str, Any]) -> CanonicalTextView:
    """Validate and freeze the existing public_sources physical-output schema.

    Original PDF bytes and visual fidelity remain the extractor's responsibility.
    This adapter validates the recorded identities, text coverage and positions;
    it cannot certify glyphs, reading order, table structure or business meaning.
    """
    try:
        return _adapt(document)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("SOURCE_INTEGRITY") from error


def _adapt(document):
    _reference_state(document)
    for key in ("data_class", "publication_state", "answer_ceiling", "semantic_complete", "production_readiness"):
        _require(key in document)
    source_id, raw_hash = document["source_id"], document["raw_pdf_sha256"]
    _require(isinstance(source_id, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", source_id))
    _require(isinstance(raw_hash, str) and re.fullmatch(r"[a-f0-9]{64}", raw_hash))
    _require(type(document["raw_pdf_bytes"]) is int and document["raw_pdf_bytes"] > 0)
    extractor, library_version = document["extractor"], document["extractor_library_version"]
    _require(isinstance(extractor, str) and bool(extractor.strip()))
    _require(isinstance(library_version, str) and bool(library_version.strip()))
    canonical = document["canonical_text"]
    _require(isinstance(canonical, str) and document["canonical_sha256"] == _hash(canonical))
    flags = document["quality_flags"]
    _require(isinstance(flags, (list, tuple)) and all(isinstance(flag, str) for flag in flags))
    pages, units = document["pages"], document["units"]
    _require(isinstance(pages, (list, tuple)) and isinstance(units, (list, tuple)))
    _require(type(document["physical_page_count"]) is int and
             document["physical_page_count"] == len(pages) and len(pages) > 0)
    revision = _revision_id(source_id, raw_hash, extractor, library_version, document["canonical_sha256"])
    namespace = f"{source_id}:{raw_hash}:{extractor}"
    by_page, unit_ids = {}, set()
    for unit in units:
        _reference_state(unit)
        _require(unit["source_id"] == source_id and unit["raw_pdf_sha256"] == raw_hash)
        number = unit["page_number"]
        _require(type(number) is int and 1 <= number <= len(pages))
        parent = f"{namespace}:p{number}"
        _require(unit["parent_page_id"] == parent)
        unit_id = unit["unit_id"]
        _require(isinstance(unit_id, str) and re.fullmatch(re.escape(parent) + r":b[1-9][0-9]*", unit_id))
        _require(unit_id not in unit_ids)
        unit_ids.add(unit_id)
        _require(unit["offset_basis"] == OFFSET_BASIS)
        _require(unit["business_labels"] == [] and unit["semantic_review_state"] == "UNREVIEWED")
        span = _span(unit, canonical, revision, number, _bbox(unit["bbox"]))
        _require(bool(unit["text"]))
        _require(isinstance(unit["line_bboxes"], (list, tuple)))
        block = BlockView(unit_id, source_id, raw_hash, parent, unit["text"], span,
                          tuple(_bbox(box) for box in unit["line_bboxes"]))
        by_page.setdefault(number, []).append(block)
    result, offset = [], 0
    for number, page in enumerate(pages, 1):
        _reference_state(page)
        _require(type(page["page_number"]) is int and page["page_number"] == number)
        _require(page["page_id"] == f"{namespace}:p{number}")
        width, height = page["width"], page["height"]
        _require(all(type(n) in (int, float) and math.isfinite(n) and n > 0 for n in (width, height)))
        span = _span(page, canonical, revision, number, (0, 0, width, height))
        _require(page["start"] == offset)
        _require(isinstance(page["image_regions"], (list, tuple)))
        _require(type(page["image_review_required"]) is bool and
                 page["image_review_required"] == bool(page["image_regions"]))
        images = []
        for image in page["image_regions"]:
            _reference_state(image)
            _require(image["state"] == "IMAGE_CONTENT_UNREVIEWED")
            images.append(ImageRegion(_bbox(image["bbox"])))
        expected_coverage = ("TEXT_EXTRACTED_UNREVIEWED" if page["text"].strip() else
                             "OCR_REQUIRED" if images else "NO_TEXT_REVIEW_REQUIRED")
        _require(page["coverage_state"] == expected_coverage)
        blocks = tuple(sorted(by_page.get(number, ()), key=lambda block: block.span.start))
        cursor = page["start"]
        for position, block in enumerate(blocks):
            _require(block.span.start == cursor + bool(position) and block.span.end <= page["end"])
            cursor = block.span.end
        _require(cursor == page["end"] and "\n".join(block.text for block in blocks) == page["text"])
        result.append(PageView(page["page_id"], number, width, height, page["text"], span,
                               page["coverage_state"], tuple(images), page["image_review_required"], blocks))
        offset = page["end"] + (number < len(pages))
    _require("\f".join(page.text for page in result) == canonical)
    immutable_pages = tuple(result)
    return CanonicalTextView(source_id, raw_hash, document["raw_pdf_bytes"], extractor,
                             library_version, canonical, document["canonical_sha256"], revision,
                             len(result), immutable_pages,
                             tuple(block for page in immutable_pages for block in page.blocks), tuple(flags))
