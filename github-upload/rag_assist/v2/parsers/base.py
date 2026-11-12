"""Immutable physical text records. Offsets count Unicode codepoints."""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re


OFFSET_BASIS = "CANONICAL_TEXT_UNICODE_CODEPOINTS"


def _require(condition, message="SOURCE_INTEGRITY"):
    if not condition:
        raise ValueError(message)


def _hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _bbox(value):
    _require(isinstance(value, (list, tuple)) and len(value) == 4)
    _require(all(type(n) in (int, float) and math.isfinite(n) for n in value))
    _require(value[0] <= value[2] and value[1] <= value[3])
    return tuple(value)


def _revision_id(source_id, raw_hash, extractor, library_version, text_hash):
    identity = json.dumps([source_id, raw_hash, extractor, library_version, text_hash],
                          ensure_ascii=False, separators=(",", ":"))
    return "text-view-v1:" + _hash(identity)


def _source_identity(source_id, raw_hash):
    _require(type(source_id) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", source_id))
    _require(type(raw_hash) is str and re.fullmatch(r"[a-f0-9]{64}", raw_hash))


def _span_text(span, text):
    _require(type(span) is SourceSpan and type(text) is str)
    _require(span.end - span.start == len(text) and span.content_sha256 == _hash(text))


@dataclass(frozen=True, slots=True)
class SourceSpan:
    text_revision_id: str
    page_number: int
    start: int
    end: int
    bbox: tuple[float, float, float, float]
    content_sha256: str
    offset_basis: str = OFFSET_BASIS

    def __post_init__(self):
        _require(type(self.text_revision_id) is str and bool(self.text_revision_id))
        _require(type(self.page_number) is int and self.page_number >= 1)
        _require(type(self.start) is int and type(self.end) is int and 0 <= self.start <= self.end)
        _require(type(self.bbox) is tuple)
        _bbox(self.bbox)
        _require(type(self.content_sha256) is str and re.fullmatch(r"[a-f0-9]{64}", self.content_sha256))
        _require(type(self.offset_basis) is str and self.offset_basis == OFFSET_BASIS)


@dataclass(frozen=True, slots=True)
class ImageRegion:
    bbox: tuple[float, float, float, float]
    state: str = "IMAGE_CONTENT_UNREVIEWED"

    def __post_init__(self):
        _require(type(self.bbox) is tuple)
        _bbox(self.bbox)
        _require(type(self.state) is str and self.state == "IMAGE_CONTENT_UNREVIEWED")


@dataclass(frozen=True, slots=True)
class BlockView:
    unit_id: str
    source_id: str
    raw_pdf_sha256: str
    parent_page_id: str
    text: str
    span: SourceSpan
    line_bboxes: tuple[tuple[float, float, float, float], ...]
    business_labels: tuple[str, ...] = ()
    semantic_review_state: str = "UNREVIEWED"

    def __post_init__(self):
        _source_identity(self.source_id, self.raw_pdf_sha256)
        _span_text(self.span, self.text)
        _require(bool(self.text))
        prefix = f"{self.source_id}:{self.raw_pdf_sha256}:"
        suffix = f":p{self.span.page_number}"
        _require(type(self.parent_page_id) is str and self.parent_page_id.startswith(prefix) and
                 self.parent_page_id.endswith(suffix) and len(self.parent_page_id) > len(prefix) + len(suffix))
        _require(type(self.unit_id) is str and
                 re.fullmatch(re.escape(self.parent_page_id) + r":b[1-9][0-9]*", self.unit_id))
        _require(type(self.line_bboxes) is tuple)
        for box in self.line_bboxes:
            _require(type(box) is tuple)
            _bbox(box)
        _require(type(self.business_labels) is tuple and self.business_labels == ())
        _require(type(self.semantic_review_state) is str and self.semantic_review_state == "UNREVIEWED")


@dataclass(frozen=True, slots=True)
class PageView:
    page_id: str
    page_number: int
    width: float
    height: float
    text: str
    span: SourceSpan
    coverage_state: str
    image_regions: tuple[ImageRegion, ...]
    image_review_required: bool
    blocks: tuple[BlockView, ...]

    def __post_init__(self):
        _require(type(self.page_id) is str and bool(self.page_id))
        _require(type(self.page_number) is int and self.page_number >= 1)
        _require(all(type(n) in (int, float) and math.isfinite(n) and n > 0
                     for n in (self.width, self.height)))
        _span_text(self.span, self.text)
        _require(self.span.page_number == self.page_number and self.span.bbox == (0, 0, self.width, self.height))
        _require(type(self.image_regions) is tuple and all(type(image) is ImageRegion for image in self.image_regions))
        _require(type(self.image_review_required) is bool and self.image_review_required == bool(self.image_regions))
        expected_coverage = ("TEXT_EXTRACTED_UNREVIEWED" if self.text.strip() else
                             "OCR_REQUIRED" if self.image_regions else "NO_TEXT_REVIEW_REQUIRED")
        _require(type(self.coverage_state) is str and self.coverage_state == expected_coverage)
        _require(type(self.blocks) is tuple and all(type(block) is BlockView for block in self.blocks))
        cursor, seen_ids = self.span.start, set()
        for position, block in enumerate(self.blocks):
            _require(block.parent_page_id == self.page_id and block.span.page_number == self.page_number)
            _require(block.span.text_revision_id == self.span.text_revision_id and block.unit_id not in seen_ids)
            _require(block.span.start == cursor + bool(position) and block.span.end <= self.span.end)
            seen_ids.add(block.unit_id)
            cursor = block.span.end
        _require(cursor == self.span.end and "\n".join(block.text for block in self.blocks) == self.text)


@dataclass(frozen=True, slots=True)
class CanonicalTextView:
    source_id: str
    raw_pdf_sha256: str
    raw_pdf_bytes: int
    extractor: str
    extractor_library_version: str
    canonical_text: str
    canonical_sha256: str
    text_revision_id: str
    physical_page_count: int
    pages: tuple[PageView, ...]
    blocks: tuple[BlockView, ...]
    quality_flags: tuple[str, ...]
    data_class: str = "PUBLIC_REFERENCE"
    publication_state: str = "REFERENCE_ONLY"
    answer_ceiling: str = "SOURCE_ONLY"
    semantic_complete: bool = False
    production_readiness: str = "NO_GO"

    def __post_init__(self):
        _source_identity(self.source_id, self.raw_pdf_sha256)
        _require(type(self.raw_pdf_bytes) is int and self.raw_pdf_bytes > 0)
        _require(type(self.extractor) is str and bool(self.extractor.strip()))
        _require(type(self.extractor_library_version) is str and bool(self.extractor_library_version.strip()))
        _require(type(self.canonical_text) is str and type(self.canonical_sha256) is str and
                 self.canonical_sha256 == _hash(self.canonical_text))
        _require(type(self.text_revision_id) is str and self.text_revision_id == _revision_id(
            self.source_id, self.raw_pdf_sha256, self.extractor,
            self.extractor_library_version, self.canonical_sha256))
        _require(type(self.pages) is tuple and all(type(page) is PageView for page in self.pages))
        _require(type(self.blocks) is tuple and all(type(block) is BlockView for block in self.blocks))
        _require(type(self.quality_flags) is tuple and all(type(flag) is str for flag in self.quality_flags))
        _require(type(self.physical_page_count) is int and self.physical_page_count == len(self.pages) > 0)
        for value, expected in ((self.data_class, "PUBLIC_REFERENCE"),
                                (self.publication_state, "REFERENCE_ONLY"),
                                (self.answer_ceiling, "SOURCE_ONLY"),
                                (self.semantic_complete, False),
                                (self.production_readiness, "NO_GO")):
            _require(type(value) is type(expected) and value == expected, "UNTRUSTED_REFERENCE_STATE")
        namespace = f"{self.source_id}:{self.raw_pdf_sha256}:{self.extractor}"
        offset = 0
        for number, page in enumerate(self.pages, 1):
            _require(page.page_id == f"{namespace}:p{number}" and page.page_number == number)
            _require(page.span.text_revision_id == self.text_revision_id and page.span.start == offset)
            _require(all(block.source_id == self.source_id and block.raw_pdf_sha256 == self.raw_pdf_sha256
                         for block in page.blocks))
            offset = page.span.end + (number < len(self.pages))
        _require("\f".join(page.text for page in self.pages) == self.canonical_text)
        _require(self.blocks == tuple(block for page in self.pages for block in page.blocks))


def resolve_quote(view: CanonicalTextView, span: SourceSpan) -> str:
    """Resolve only an exact registered whole-page or whole-block location.

    A matching substring alone is not a physical alignment. New subranges need
    a separately verified mapping before they can become registered spans.
    """
    _require(type(view) is CanonicalTextView and type(span) is SourceSpan, "INVALID_QUOTE")
    _require(span.text_revision_id == view.text_revision_id, "TEXT_REVISION_MISMATCH")
    _require(1 <= span.page_number <= len(view.pages) and span.end <= len(view.canonical_text), "INVALID_QUOTE")
    page = view.pages[span.page_number - 1]
    _require(span == page.span or any(span == block.span for block in page.blocks), "UNREGISTERED_QUOTE")
    quote = view.canonical_text[span.start:span.end]
    _require(_hash(quote) == span.content_sha256, "QUOTE_HASH_MISMATCH")
    return quote
