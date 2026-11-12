"""Budgeted original-text projections. Offsets belong to unit text, never PDF geometry."""
from collections.abc import Callable
from dataclasses import dataclass
import re

from .parsers.base import CanonicalTextView, _hash, _require
from .units import UnitAssembly, _identity, _string, _validate_assembly, resolve_unit_text


OFFSET_BASIS = "UNIT_TEXT_UNICODE_CODEPOINTS"


def _count(count_tokens, text):
    try:
        value = count_tokens(text)
    except Exception as error:
        raise ValueError("TOKEN_COUNTER_FAILED") from error
    _require(type(value) is int and value >= 0, "TOKEN_COUNTER_INVALID")
    return value


def _parameters(tokenizer_id, max_tokens, overlap_chars, max_windows):
    _string(tokenizer_id)
    _require(type(max_tokens) is int and max_tokens > 0, "INVALID_MAX_TOKENS")
    _require(type(overlap_chars) is int and overlap_chars >= 0, "INVALID_OVERLAP")
    _require(type(max_windows) is int and max_windows > 0, "INVALID_MAX_WINDOWS")


def _window_id(window):
    return _identity("window-v1:", (window.unit_id, window.assembly_id, window.text_revision_id,
                     window.start, window.end, window.input_sha256, window.tokenizer_id,
                     window.token_count, window.max_tokens, window.overlap_chars,
                     window.max_windows, window.offset_basis))


@dataclass(frozen=True, slots=True)
class RetrievalWindow:
    window_id: str
    unit_id: str
    assembly_id: str
    text_revision_id: str
    start: int
    end: int
    text: str
    input_sha256: str
    tokenizer_id: str
    token_count: int
    max_tokens: int
    overlap_chars: int = 32
    max_windows: int = 100000
    offset_basis: str = OFFSET_BASIS

    def __post_init__(self):
        for value in (self.window_id, self.unit_id, self.assembly_id, self.text_revision_id):
            _string(value)
        _parameters(self.tokenizer_id, self.max_tokens, self.overlap_chars, self.max_windows)
        _require(type(self.start) is int and type(self.end) is int and 0 <= self.start < self.end,
                 "INVALID_WINDOW_RANGE")
        _require(type(self.text) is str and len(self.text) == self.end - self.start, "INVALID_WINDOW_TEXT")
        _require(type(self.input_sha256) is str and re.fullmatch(r"[a-f0-9]{64}", self.input_sha256) and
                 self.input_sha256 == _hash(self.text), "WINDOW_HASH_MISMATCH")
        _require(type(self.token_count) is int and 0 <= self.token_count <= self.max_tokens,
                 "INVALID_TOKEN_COUNT")
        _require(type(self.offset_basis) is str and self.offset_basis == OFFSET_BASIS, "INVALID_OFFSET_BASIS")
        _require(self.window_id == _window_id(self), "WINDOW_ID_MISMATCH")


def build_windows(view: CanonicalTextView, assembly: UnitAssembly, *,
                  count_tokens: Callable[[str], int], tokenizer_id: str,
                  max_tokens: int = 256, overlap_chars: int = 32,
                  max_windows: int = 100000) -> tuple[RetrievalWindow, ...]:
    """Measure complete candidate strings, shrinking without assuming BPE monotonicity.

    Halving is a length heuristic only. If even the smallest trial fails, check
    untried lengths before declaring failure: merges can make longer text fit.
    A failed budget/counter/limit aborts the entire call; no partial result escapes.
    """
    _validate_assembly(view, assembly)
    _parameters(tokenizer_id, max_tokens, overlap_chars, max_windows)
    _require(callable(count_tokens), "TOKEN_COUNTER_INVALID")
    result = []
    for unit in assembly.units:
        if not unit.searchable:
            continue
        text = resolve_unit_text(view, unit)
        start, limit = 0, len(text)
        while start < len(text):
            _require(len(result) < max_windows, "MAX_WINDOWS_EXCEEDED")
            length = min(limit, len(text) - start)
            tried = set()
            while True:
                tried.add(length)
                candidate = text[start:start + length]
                count = _count(count_tokens, candidate)
                if count <= max_tokens:
                    break
                if length > 1:
                    length = max(1, length // 2)
                    continue
                for alternative in range(2, len(text) - start + 1):
                    if alternative in tried:
                        continue
                    alternative_count = _count(count_tokens, text[start:start + alternative])
                    if alternative_count <= max_tokens:
                        length, count = alternative, alternative_count
                        candidate = text[start:start + length]
                        break
                else:
                    raise ValueError("TOKEN_BUDGET_IMPOSSIBLE")
                break
            end = start + length
            values = dict(unit_id=unit.unit_id, assembly_id=assembly.assembly_id,
                          text_revision_id=view.text_revision_id, start=start, end=end,
                          text=candidate, input_sha256=_hash(candidate), tokenizer_id=tokenizer_id,
                          token_count=count, max_tokens=max_tokens, overlap_chars=overlap_chars,
                          max_windows=max_windows)
            identity = _identity("window-v1:", (unit.unit_id, assembly.assembly_id, view.text_revision_id,
                                 start, end, values["input_sha256"], tokenizer_id, count, max_tokens,
                                 overlap_chars, max_windows, OFFSET_BASIS))
            result.append(RetrievalWindow(window_id=identity, **values))
            if end == len(text):
                break
            start = end - min(overlap_chars, length - 1)
            limit = max(1, length * 2)
    return tuple(result)
