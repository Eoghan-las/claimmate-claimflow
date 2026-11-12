"""Explicit structural candidates over registered physical text, without semantics."""
from dataclasses import dataclass
import json

from .parsers.base import CanonicalTextView, SourceSpan, _hash, _require


KINDS = frozenset(("DOCUMENT", "PAGE_CONTEXT", "SECTION_CANDIDATE",
                   "CLAUSE_CANDIDATE", "PARAGRAPH", "BLOCK"))


def _string(value):
    _require(type(value) is str and bool(value.strip()), "INVALID_STRUCTURE")


def _strings(value):
    _require(type(value) is tuple, "INVALID_STRUCTURE")
    for item in value:
        _string(item)
    _require(len(set(value)) == len(value), "DUPLICATE_ID")


def _identity(prefix, values):
    return prefix + _hash(json.dumps(values, ensure_ascii=False, separators=(",", ":")))


def _role(key, parent, kind, block_ids, searchable):
    _string(key)
    if parent is not None:
        _string(parent)
    _require(type(kind) is str and kind in KINDS, "INVALID_UNIT_KIND")
    _strings(block_ids)
    _require(type(searchable) is bool, "INVALID_SEARCHABLE")
    _require((parent is None) == (kind == "DOCUMENT"), "INVALID_ROOT")
    _require(not searchable or (kind not in ("DOCUMENT", "PAGE_CONTEXT") and bool(block_ids)),
             "INVALID_SEARCHABLE")


def _unit_id(revision, key, parent_id, kind, block_ids, spans, searchable):
    locations = [(span.text_revision_id, span.page_number, span.start, span.end,
                  span.bbox, span.content_sha256, span.offset_basis) for span in spans]
    return _identity("unit-v1:", (revision, key, parent_id, kind, block_ids, locations, searchable))


@dataclass(frozen=True, slots=True)
class UnitSpec:
    key: str
    parent_key: str | None
    kind: str
    block_ids: tuple[str, ...]
    searchable: bool

    def __post_init__(self):
        _role(self.key, self.parent_key, self.kind, self.block_ids, self.searchable)


@dataclass(frozen=True, slots=True)
class UnitDraft:
    unit_id: str
    text_revision_id: str
    key: str
    parent_id: str | None
    kind: str
    block_ids: tuple[str, ...]
    spans: tuple[SourceSpan, ...]
    searchable: bool

    def __post_init__(self):
        _string(self.unit_id)
        _string(self.text_revision_id)
        _role(self.key, self.parent_id, self.kind, self.block_ids, self.searchable)
        _require(type(self.spans) is tuple and all(type(span) is SourceSpan for span in self.spans),
                 "INVALID_SPANS")
        _require(len(self.spans) == len(self.block_ids), "INVALID_SPANS")
        cursor = -1
        for span in self.spans:
            span.__post_init__()
            _require(span.text_revision_id == self.text_revision_id and span.start >= cursor and
                     span.start < span.end, "INVALID_SPANS")
            cursor = span.end
        _require(self.unit_id == _unit_id(self.text_revision_id, self.key, self.parent_id,
                                         self.kind, self.block_ids, self.spans, self.searchable),
                 "UNIT_ID_MISMATCH")


@dataclass(frozen=True, slots=True)
class UnitAssembly:
    assembly_id: str
    text_revision_id: str
    units: tuple[UnitDraft, ...]

    def __post_init__(self):
        _string(self.assembly_id)
        _string(self.text_revision_id)
        _require(type(self.units) is tuple and bool(self.units) and
                 all(type(unit) is UnitDraft for unit in self.units), "INVALID_UNITS")
        for unit in self.units:
            unit.__post_init__()
            _require(unit.text_revision_id == self.text_revision_id, "TEXT_REVISION_MISMATCH")
        by_id = {unit.unit_id: unit for unit in self.units}
        _require(len(by_id) == len(self.units) and
                 len({unit.key for unit in self.units}) == len(self.units), "DUPLICATE_ID")
        roots = tuple(unit for unit in self.units if unit.parent_id is None)
        _require(len(roots) == 1 and roots[0].kind == "DOCUMENT", "INVALID_ROOT")
        root = roots[0]
        positions = {block_id: i for i, block_id in enumerate(root.block_ids)}
        root_spans = dict(zip(root.block_ids, root.spans))
        for unit in self.units:
            _require(all(block_id in positions for block_id in unit.block_ids), "UNKNOWN_BLOCK")
            indices = tuple(positions[block_id] for block_id in unit.block_ids)
            _require(not indices or indices == tuple(range(indices[0], indices[-1] + 1)),
                     "NONCONTIGUOUS_BLOCKS")
            _require(unit.spans == tuple(root_spans[block_id] for block_id in unit.block_ids),
                     "UNREGISTERED_SPANS")
            if unit.parent_id is not None:
                _require(unit.parent_id in by_id, "ORPHAN_UNIT")
                _require(set(unit.block_ids).issubset(by_id[unit.parent_id].block_ids),
                         "CHILD_OUTSIDE_PARENT")
            seen, current = set(), unit
            while current.parent_id is not None:
                _require(current.unit_id not in seen, "CYCLIC_PARENT_CHAIN")
                seen.add(current.unit_id)
                _require(current.parent_id in by_id, "ORPHAN_UNIT")
                current = by_id[current.parent_id]
            _require(current.unit_id == root.unit_id, "INVALID_ROOT")
        _require(self.assembly_id == _identity("assembly-v1:",
                 (self.text_revision_id, tuple(unit.unit_id for unit in self.units))), "ASSEMBLY_ID_MISMATCH")


def assemble_units(view: CanonicalTextView, specs: tuple[UnitSpec, ...] | None = None) -> UnitAssembly:
    """Use explicit parent links, or preserve document → page → every block."""
    _require(type(view) is CanonicalTextView, "INVALID_VIEW")
    blocks = {block.unit_id: block for block in view.blocks}
    if specs is None:
        result = [UnitSpec("document", None, "DOCUMENT", tuple(blocks), False)]
        for page in view.pages:
            key = f"page:{page.page_number}"
            result.append(UnitSpec(key, "document", "PAGE_CONTEXT",
                                   tuple(block.unit_id for block in page.blocks), False))
            result.extend(UnitSpec(block.unit_id, key, "BLOCK", (block.unit_id,), True)
                          for block in page.blocks)
        specs = tuple(result)
    _require(type(specs) is tuple and bool(specs) and all(type(spec) is UnitSpec for spec in specs),
             "INVALID_SPECS")
    for spec in specs:
        spec.__post_init__()
    by_key = {spec.key: spec for spec in specs}
    _require(len(by_key) == len(specs), "DUPLICATE_KEY")
    roots = tuple(spec for spec in specs if spec.parent_key is None)
    _require(len(roots) == 1 and roots[0].kind == "DOCUMENT" and
             roots[0].block_ids == tuple(blocks), "ROOT_MUST_REGISTER_ALL_BLOCKS")
    _require(all(spec.parent_key is None or spec.parent_key in by_key for spec in specs), "ORPHAN_UNIT")
    drafts, pending = {}, list(specs)
    while pending:
        remaining = []
        for spec in pending:
            if spec.parent_key is not None and spec.parent_key not in drafts:
                remaining.append(spec)
                continue
            _require(all(block_id in blocks for block_id in spec.block_ids), "UNKNOWN_BLOCK")
            spans = tuple(blocks[block_id].span for block_id in spec.block_ids)
            parent_id = None if spec.parent_key is None else drafts[spec.parent_key].unit_id
            unit_id = _unit_id(view.text_revision_id, spec.key, parent_id, spec.kind,
                               spec.block_ids, spans, spec.searchable)
            drafts[spec.key] = UnitDraft(unit_id, view.text_revision_id, spec.key, parent_id,
                                         spec.kind, spec.block_ids, spans, spec.searchable)
        _require(len(remaining) < len(pending), "CYCLIC_PARENT_CHAIN")
        pending = remaining
    units = tuple(drafts[spec.key] for spec in specs)
    return UnitAssembly(_identity("assembly-v1:", (view.text_revision_id, tuple(unit.unit_id for unit in units))),
                        view.text_revision_id, units)


def resolve_unit_text(view: CanonicalTextView, unit: UnitDraft) -> str:
    """Resolve registered whole blocks; local window offsets are never PDF spans."""
    _require(type(view) is CanonicalTextView and type(unit) is UnitDraft, "INVALID_UNIT")
    unit.__post_init__()
    _require(view.text_revision_id == unit.text_revision_id, "TEXT_REVISION_MISMATCH")
    blocks = {block.unit_id: (i, block) for i, block in enumerate(view.blocks)}
    indices = []
    for block_id, span in zip(unit.block_ids, unit.spans):
        _require(block_id in blocks and blocks[block_id][1].span == span, "UNREGISTERED_SPANS")
        indices.append(blocks[block_id][0])
    _require(not indices or indices == list(range(indices[0], indices[-1] + 1)), "NONCONTIGUOUS_BLOCKS")
    return "" if not unit.spans else view.canonical_text[unit.spans[0].start:unit.spans[-1].end]


def parent_chain(assembly: UnitAssembly, unit_id: str) -> tuple[UnitDraft, ...]:
    """Return the current candidate followed by all its ancestors through root."""
    _require(type(assembly) is UnitAssembly, "INVALID_ASSEMBLY")
    assembly.__post_init__()
    _string(unit_id)
    by_id = {unit.unit_id: unit for unit in assembly.units}
    _require(unit_id in by_id, "UNKNOWN_UNIT")
    result, current = [], by_id[unit_id]
    while True:
        result.append(current)
        if current.parent_id is None:
            return tuple(result)
        current = by_id[current.parent_id]


def _validate_assembly(view, assembly):
    _require(type(view) is CanonicalTextView and type(assembly) is UnitAssembly, "INVALID_ASSEMBLY")
    assembly.__post_init__()
    _require(view.text_revision_id == assembly.text_revision_id, "TEXT_REVISION_MISMATCH")
    root = next(unit for unit in assembly.units if unit.parent_id is None)
    _require(root.block_ids == tuple(block.unit_id for block in view.blocks) and
             root.spans == tuple(block.span for block in view.blocks), "UNREGISTERED_SPANS")
