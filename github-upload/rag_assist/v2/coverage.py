"""A physical text disposition ledger, never semantic or visual certification.

Report DTOs carry claims, not approval. Consumers must call
validate_coverage_report with trusted source, assembly, windows and tokenizer
before using coverage booleans or pending-review information.
"""
from collections.abc import Callable
from dataclasses import dataclass

from .parsers.base import CanonicalTextView, ImageRegion, _require
from .units import (UnitAssembly, _identity, _string, _strings,
                    _validate_assembly, resolve_unit_text)
from .windows import RetrievalWindow, _count


def _record(identifier, total, covered, window_ids):
    _string(identifier)
    _require(type(total) is int and total > 0 and type(covered) is int and 0 <= covered <= total,
             "INVALID_COVERAGE_COUNTS")
    _strings(window_ids)
    _require(bool(window_ids) == bool(covered), "INVALID_COVERAGE_WINDOWS")


@dataclass(frozen=True, slots=True)
class BlockCoverage:
    block_id: str
    total_chars: int
    covered_chars: int
    window_ids: tuple[str, ...]

    def __post_init__(self):
        _record(self.block_id, self.total_chars, self.covered_chars, self.window_ids)

    @property
    def state(self) -> str:
        return "MAPPED" if self.covered_chars == self.total_chars else "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class UnitWindowCoverage:
    unit_id: str
    total_chars: int
    covered_chars: int
    window_ids: tuple[str, ...]

    def __post_init__(self):
        _record(self.unit_id, self.total_chars, self.covered_chars, self.window_ids)

    @property
    def state(self) -> str:
        return "MAPPED" if self.covered_chars == self.total_chars else "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class PendingReview:
    page_number: int
    state: str
    image_regions: tuple[ImageRegion, ...]

    def __post_init__(self):
        _require(type(self.page_number) is int and self.page_number > 0, "INVALID_PAGE")
        _require(type(self.state) is str and self.state in
                 ("OCR_REQUIRED", "NO_TEXT_REVIEW_REQUIRED", "IMAGE_CONTENT_UNREVIEWED"), "INVALID_REVIEW_STATE")
        _require(type(self.image_regions) is tuple and
                 all(type(region) is ImageRegion for region in self.image_regions), "INVALID_IMAGE_REGIONS")
        for region in self.image_regions:
            region.__post_init__()
        _require(bool(self.image_regions) == (self.state != "NO_TEXT_REVIEW_REQUIRED"), "INVALID_REVIEW_STATE")


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Immutable ledger data; local invariants cannot prove its completeness.

    Ordinary construction/replace can omit records or change context identities.
    The derived flags and report_id therefore remain untrusted until the entire
    report passes validate_coverage_report against trusted processing inputs.
    """
    assembly_id: str
    text_revision_id: str
    tokenizer_id: str
    block_records: tuple[BlockCoverage, ...]
    unit_records: tuple[UnitWindowCoverage, ...]
    pending_reviews: tuple[PendingReview, ...]
    invalid_window_ids: tuple[str, ...]
    answer_ceiling: str = "SOURCE_ONLY"
    semantic_review_state: str = "UNREVIEWED"
    semantic_complete: bool = False
    production_readiness: str = "NO_GO"

    def __post_init__(self):
        for value in (self.assembly_id, self.text_revision_id, self.tokenizer_id):
            _string(value)
        for records, cls, key in ((self.block_records, BlockCoverage, "block_id"),
                                  (self.unit_records, UnitWindowCoverage, "unit_id"),
                                  (self.pending_reviews, PendingReview, "page_number")):
            _require(type(records) is tuple and all(type(item) is cls for item in records), "INVALID_COVERAGE_RECORDS")
            for item in records:
                item.__post_init__()
            _require(len({getattr(item, key) for item in records}) == len(records), "DUPLICATE_COVERAGE_RECORD")
        _strings(self.invalid_window_ids)
        for value, expected in ((self.answer_ceiling, "SOURCE_ONLY"),
                                (self.semantic_review_state, "UNREVIEWED"),
                                (self.semantic_complete, False),
                                (self.production_readiness, "NO_GO")):
            _require(type(value) is type(expected) and value == expected, "UNTRUSTED_REFERENCE_STATE")

    @property
    def uncovered_block_ids(self) -> tuple[str, ...]:
        return tuple(record.block_id for record in self.block_records if record.state == "UNRESOLVED")

    @property
    def missing_window_unit_ids(self) -> tuple[str, ...]:
        """Includes units with partial coverage, not just units with zero windows."""
        return tuple(record.unit_id for record in self.unit_records if record.state == "UNRESOLVED")

    @property
    def text_coverage_complete(self) -> bool:
        return bool(self.block_records) and not self.uncovered_block_ids

    @property
    def unit_window_coverage_complete(self) -> bool:
        return bool(self.unit_records) and not self.missing_window_unit_ids

    @property
    def report_id(self) -> str:
        return _identity("coverage-v1:", (self.assembly_id, self.text_revision_id, self.tokenizer_id,
                         tuple((r.block_id, r.total_chars, r.covered_chars, r.window_ids) for r in self.block_records),
                         tuple((r.unit_id, r.total_chars, r.covered_chars, r.window_ids) for r in self.unit_records),
                         tuple((r.page_number, r.state, tuple(image.bbox for image in r.image_regions))
                               for r in self.pending_reviews), self.invalid_window_ids))


def _covered(intervals):
    total, cursor = 0, -1
    for start, end in sorted(intervals):
        total += max(0, end - max(start, cursor))
        cursor = max(cursor, end)
    return total


def check_coverage(view: CanonicalTextView, assembly: UnitAssembly,
                   windows: tuple[RetrievalWindow, ...], *,
                   count_tokens: Callable[[str], int], tokenizer_id: str) -> CoverageReport:
    """Independently recheck each window before counting source or unit characters."""
    _validate_assembly(view, assembly)
    _string(tokenizer_id)
    _require(callable(count_tokens), "TOKEN_COUNTER_INVALID")
    _require(type(windows) is tuple and all(type(window) is RetrievalWindow for window in windows),
             "INVALID_WINDOWS")
    units = {unit.unit_id: unit for unit in assembly.units}
    texts = {unit.unit_id: resolve_unit_text(view, unit) for unit in assembly.units if unit.searchable}
    unit_ranges = {unit_id: [] for unit_id in texts}
    unit_windows = {unit_id: [] for unit_id in texts}
    block_ranges = {block.unit_id: [] for block in view.blocks}
    block_windows = {block.unit_id: [] for block in view.blocks}
    invalid = []
    for window in windows:
        # Constructor checks alone cannot establish membership in this view.
        try:
            window.__post_init__()
            _require(window.assembly_id == assembly.assembly_id and
                     window.text_revision_id == view.text_revision_id and
                     window.tokenizer_id == tokenizer_id and window.unit_id in texts, "WINDOW_CONTEXT_MISMATCH")
            text = texts[window.unit_id]
            _require(window.end <= len(text) and window.text == text[window.start:window.end],
                     "WINDOW_SLICE_MISMATCH")
        except ValueError:
            _string(window.window_id)
            invalid.append(window.window_id)
            continue
        actual = _count(count_tokens, window.text)
        if actual != window.token_count or actual > window.max_tokens:
            invalid.append(window.window_id)
            continue
        unit = units[window.unit_id]
        unit_ranges[unit.unit_id].append((window.start, window.end))
        unit_windows[unit.unit_id].append(window.window_id)
        global_start = unit.spans[0].start + window.start
        global_end = unit.spans[0].start + window.end
        for block_id, span in zip(unit.block_ids, unit.spans):
            start, end = max(global_start, span.start), min(global_end, span.end)
            if start < end:
                block_ranges[block_id].append((start, end))
                block_windows[block_id].append(window.window_id)
    records = tuple(BlockCoverage(block.unit_id, len(block.text), _covered(block_ranges[block.unit_id]),
                                 tuple(dict.fromkeys(block_windows[block.unit_id]))) for block in view.blocks)
    unit_records = tuple(UnitWindowCoverage(unit_id, len(text), _covered(unit_ranges[unit_id]),
                                           tuple(dict.fromkeys(unit_windows[unit_id]))) for unit_id, text in texts.items())
    pending = tuple(PendingReview(page.page_number,
                    "IMAGE_CONTENT_UNREVIEWED" if page.text.strip() else page.coverage_state,
                    page.image_regions) for page in view.pages if page.image_regions or not page.text.strip())
    return CoverageReport(assembly.assembly_id, view.text_revision_id, tokenizer_id,
                          records, unit_records, pending, tuple(dict.fromkeys(invalid)))


def validate_coverage_report(view: CanonicalTextView, assembly: UnitAssembly,
                             windows: tuple[RetrievalWindow, ...], report: CoverageReport, *,
                             count_tokens: Callable[[str], int], tokenizer_id: str) -> CoverageReport:
    """Recompute a supplied report against trusted inputs before consuming claims.

    Compare every stored field after exact-type and nested-record validation;
    checking a candidate's own dynamically derived report_id proves nothing.
    Return the freshly recomputed report, never the supplied DTO.
    """
    _require(type(report) is CoverageReport, "INVALID_COVERAGE_REPORT")
    report.__post_init__()
    expected = check_coverage(view, assembly, windows, count_tokens=count_tokens,
                              tokenizer_id=tokenizer_id)
    _require(report == expected, "COVERAGE_REPORT_MISMATCH")
    return expected
