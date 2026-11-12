"""Local PDF geometry candidates bound to an immutable physical text view."""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import math
from pathlib import Path
import re

import fitz

from .parsers import CanonicalTextView, resolve_quote


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _risk(code, **context):
    return dict(code=code, state="REVIEW_REQUIRED", **context)


def _inside(inner, outer, tolerance=1):
    return (inner[0] >= outer[0] - tolerance and inner[1] >= outer[1] - tolerance
            and inner[2] <= outer[2] + tolerance and inner[3] <= outer[3] + tolerance)


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _normal(text):
    return re.sub(r"\s+", "", text or "")


def _span(block):
    return dict(asdict(block.span), block_id=block.unit_id, text=block.text)


def _source_parts(page_view, box):
    """Use original line text; geometry is a locator, never a new quote source."""
    parts = []
    for block in page_view.blocks:
        cursor = block.span.start
        lines = block.text.split("\n")
        if len(lines) != len(block.line_bboxes):
            continue
        for line, line_box in zip(lines, block.line_bboxes):
            cx, cy = (line_box[0] + line_box[2]) / 2, (line_box[1] + line_box[3]) / 2
            if line and box[0] - .5 <= cx <= box[2] + .5 and box[1] - .5 <= cy <= box[3] + .5:
                parts.append(dict(block_id=block.unit_id, text_revision_id=block.span.text_revision_id,
                                  page_number=page_view.page_number, start=cursor,
                                  end=cursor + len(line), bbox=list(line_box), text=line,
                                  content_sha256=sha256(line.encode("utf-8")).hexdigest()))
            cursor += len(line) + 1
    # Reading order inside a cell can differ from the containing multi-cell block.
    return sorted(parts, key=lambda p: (round(p["bbox"][1], 1), p["bbox"][0], p["start"]))


def _sources(parts, blocks):
    return [_span(blocks[key]) for key in dict.fromkeys(p["block_id"] for p in parts)]


def _edges(values):
    result = []
    for value in sorted(values):
        if not result or value - result[-1] > .5:
            result.append(value)
    return result


def _table_geometry(table, page_view, blocks):
    boxes = [box for row in table.rows for box in row.cells if box is not None]
    xs = _edges(v for box in boxes for v in (box[0], box[2]))
    ys = _edges(v for box in boxes for v in (box[1], box[3]))
    extracted = table.extract()
    cells, grid, risks = [], [], []
    for raw_row, row in enumerate(table.rows):
        slots = []
        for raw_col, box in enumerate(row.cells):
            if box is None:
                slots.append(None)  # A missing slot is never filled from a neighbour.
                continue
            col = min(range(len(xs)), key=lambda i: abs(xs[i] - box[0]))
            end_col = min(range(len(xs)), key=lambda i: abs(xs[i] - box[2]))
            top = min(range(len(ys)), key=lambda i: abs(ys[i] - box[1]))
            bottom = min(range(len(ys)), key=lambda i: abs(ys[i] - box[3]))
            parts = _source_parts(page_view, box)
            expected = extracted[raw_row][raw_col]
            text = "\n".join(part["text"] for part in parts)
            cell_risks = []
            if _normal(text) != _normal(expected):
                cell_risks.append(_risk("CELL_SOURCE_UNRESOLVED", page_number=page_view.page_number,
                                        row=top, col=col))
                parts, text = [], None
            sources = _sources(parts, blocks)
            cells.append(dict(row=top, col=col, rowspan=bottom - top, colspan=end_col - col,
                              bbox=list(box), text=text, source_parts=parts, source_spans=sources,
                              source_block_ids=[s["block_id"] for s in sources],
                              is_header=False, risks=cell_risks))
            slots.append(len(cells) - 1)
            risks.extend(cell_risks)
        grid.append(slots)
    # PyMuPDF's first-row/external header inference is also a candidate.
    header_parts = _source_parts(page_view, table.header.bbox) if table.header.bbox else []
    first_text = " ".join(c["text"] or "" for c in cells if c["row"] == 0)
    header_rows = [0] if re.search(r"保险计划|计划[一二三四]|各项保险责任|保障目录|\bHeader\b", first_text) else []
    for cell in cells:
        cell["is_header"] = cell["row"] in header_rows
    all_header_parts = header_parts + [p for c in cells if c["is_header"] for p in c["source_parts"]]
    header_sources = _sources(all_header_parts, blocks)
    risks.append(_risk("HEADER_INTERPRETATION_REVIEW_REQUIRED", page_number=page_view.page_number))
    return dict(page_number=page_view.page_number, bbox=list(table.bbox),
                row_count=table.row_count, col_count=table.col_count,
                grid_row_count=len(ys) - 1, grid_col_count=len(xs) - 1,
                cells=cells, grid=grid, header_rows=header_rows,
                header_source_spans=header_sources,
                header=dict(bbox=list(table.header.bbox) if table.header.bbox else None,
                            external=bool(table.header.external), source_parts=header_parts,
                            source_spans=_sources(header_parts, blocks)), risks=risks)


def _registered(record, view, blocks):
    key, product = record.get("layout_id", record.get("key")), record.get("product_id")
    _require(isinstance(key, str) and isinstance(product, str) and product and
             key.startswith(product + ":"), "REGISTERED_PRODUCT_MISMATCH")
    _require(record.get("source_id", view.source_id) == view.source_id and
             record.get("raw_pdf_sha256", view.raw_pdf_sha256) == view.raw_pdf_sha256 and
             record.get("text_revision_id", view.text_revision_id) == view.text_revision_id,
             "REGISTERED_SOURCE_MISMATCH")
    ids = record.get("evidence_ids", record.get("block_ids"))
    _require(isinstance(ids, (list, tuple)) and ids and all(isinstance(key, str) for key in ids)
             and len(set(ids)) == len(ids)
             and all(key in blocks for key in ids), "REGISTERED_BLOCK_MISMATCH")
    pages = record.get("context_pages", sorted({blocks[key].span.page_number for key in ids}))
    _require(isinstance(pages, (list, tuple)) and all(type(n) is int and 1 <= n <= len(view.pages) for n in pages)
             and {blocks[key].span.page_number for key in ids} <= set(pages), "REGISTERED_PAGE_MISMATCH")
    regions = record.get("reviewed_bbox_regions", [])
    _require(isinstance(regions, (list, tuple)), "REGISTERED_REGION_MISMATCH")
    verified_regions = []
    for region in regions:
        _require(isinstance(region, dict), "REGISTERED_REGION_MISMATCH")
        number, box = region.get("page_number"), region.get("bbox")
        _require(type(number) is int and number in pages and isinstance(box, (list, tuple))
                 and len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box)
                 and box[0] <= box[2] and box[1] <= box[3], "REGISTERED_REGION_MISMATCH")
        _require(any(blocks[key].span.page_number == number and _inside(blocks[key].span.bbox, box)
                     for key in ids), "REGISTERED_REGION_MISMATCH")
        verified_regions.append(dict(page_number=number, bbox=list(box)))
    if regions:
        _require(all(any(region["page_number"] == blocks[key].span.page_number
                         and _inside(blocks[key].span.bbox, region["bbox"])
                         for region in verified_regions) for key in ids), "REGISTERED_REGION_MISMATCH")
    kind = str(record.get("kind", "")).upper()
    _require(kind in {"TABLE", "FOOTNOTE"}, "REGISTERED_KIND_INVALID")
    for name, expected in (("semantic_review_state", "UNREVIEWED"), ("answer_ceiling", "SOURCE_ONLY"),
                           ("data_class", "PUBLIC_REFERENCE")):
        _require(record.get(name, expected) == expected, "UNTRUSTED_REFERENCE_STATE")
    for name in ("approved", "business_approved", "semantic_complete"):
        _require(record.get(name, False) is False, "UNTRUSTED_REFERENCE_STATE")
    metadata = dict(provenance="REGISTERED", product_id=product, semantic_review_state="UNREVIEWED",
                    answer_ceiling="SOURCE_ONLY", source_id=view.source_id,
                    text_revision_id=view.text_revision_id, source_spans=[_span(blocks[key]) for key in ids],
                    reviewed_bbox_regions=verified_regions,
                    layout_review=record.get("layout_review", "UNREVIEWED"))
    return dict(key=key, kind=kind, title=record.get("title", ""), block_ids=list(ids),
                context_pages=list(pages), metadata=metadata, risks=[])


def _automatic_footnotes(page, page_view, excluded):
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
    raw_blocks = [b for b in page.get_text("dict", flags=flags, sort=True)["blocks"] if b.get("type") == 0]
    sizes = Counter()
    for raw in raw_blocks:
        for line in raw.get("lines", []):
            for span in line["spans"]:
                sizes[round(span["size"], 1)] += len(span["text"].strip())
    total = sum(sizes.values())
    body_sizes = [size for size, count in sizes.items() if count >= total * .15]
    body = max(body_sizes, default=0)
    result = []
    for block in page_view.blocks:
        if block.unit_id in excluded or block.span.bbox[1] < page_view.height * .5:
            continue
        if not re.match(r"^\s*(?:\d{1,3}[^\d\s.（(]|[*†‡])", block.text) or not re.search(r"[:：]", block.text[:80]):
            continue
        matches = [raw for raw in raw_blocks if _normal("\n".join(
            "".join(s["text"] for s in line["spans"]) for line in raw.get("lines", []))) == _normal(block.text)
                   and _inside(raw["bbox"], block.span.bbox)]
        font_sizes = [s["size"] for raw in matches for line in raw["lines"]
                      for s in line["spans"] if s["text"].strip()]
        if not font_sizes or max(font_sizes) > body * .9:
            continue
        result.append(dict(key=block.unit_id + ":footnote", kind="FOOTNOTE", title=block.text.split("\n")[0][:80],
                           block_ids=[block.unit_id], context_pages=[page_view.page_number],
                           metadata=dict(provenance="FONT_GEOMETRY_CANDIDATE", semantic_review_state="UNREVIEWED",
                                         answer_ceiling="SOURCE_ONLY", source_spans=[_span(block)]),
                           risks=[_risk("FOOTNOTE_REFERENCE_REVIEW_REQUIRED", block_id=block.unit_id)]))
    return result


def _note_number(text):
    match = re.match(r"^\s*(\d{1,3})(?!\d)(?=[^\d\s.（(])", text)
    return match.group(1) if match and re.search(r"[:：]", text[:80]) else None


def _footnote_references(pdf, view, objects, selected):
    """Locate unique source footnotes; these edges remain semantic candidates."""
    targets, definitions = {}, {}
    blocks = {block.unit_id: block for block in view.blocks}
    for block in view.blocks:
        number = _note_number(block.text)
        if number:
            definitions.setdefault(number, set()).add(block.unit_id)
    note_blocks = {key for obj in objects if obj["kind"] == "FOOTNOTE" for key in obj["block_ids"]}
    for obj in objects:
        if obj["kind"] == "FOOTNOTE" and obj["block_ids"]:
            number = _note_number(blocks[obj["block_ids"][0]].text)
            if number:
                targets.setdefault(number, []).append(obj)
    tables = [table for obj in objects for table in obj["metadata"].get("tables", [])]
    references, risks, seen = [], [], set()
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
    for number in sorted(selected):
        raw_blocks = [raw for raw in pdf[number - 1].get_text("dict", flags=flags, sort=True)["blocks"]
                      if raw.get("type") == 0]
        sizes = Counter()
        for raw in raw_blocks:
            for line in raw.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
        body = max((size for size, count in sizes.items() if count >= sum(sizes.values()) * .15), default=0)
        for raw in raw_blocks:
            raw_text = "\n".join("".join(s["text"] for s in line["spans"]) for line in raw.get("lines", []))
            matching = [block for block in view.pages[number - 1].blocks
                        if block.text == raw_text and _inside(raw["bbox"], block.span.bbox)
                        and _inside(block.span.bbox, raw["bbox"])]
            # Leading definition numbers are not references back to themselves.
            if _note_number(raw_text) or any(block.unit_id in note_blocks for block in matching):
                continue
            for line in raw.get("lines", []):
                for span in line["spans"]:
                    marker = span["text"].strip()
                    if not re.fullmatch(r"\d{1,3}", marker):
                        continue
                    if not (span["flags"] & 1 or span["size"] <= body * .7):
                        continue
                    if any(t["page_number"] == number and _inside(span["bbox"], t["bbox"]) for t in tables):
                        continue
                    candidates = targets.get(marker, [])
                    if len(matching) != 1 or len(candidates) != 1 or len(definitions.get(marker, ())) != 1:
                        risks.append(_risk("UNRESOLVED_FOOTNOTE_REFERENCE", marker=marker, page_number=number,
                                           bbox=list(span["bbox"]), source_block_ids=[b.unit_id for b in matching]))
                        continue
                    source, target = matching[0], candidates[0]
                    identity = (source.unit_id, target["key"])
                    if identity not in seen:
                        references.append(dict(from_block_id=source.unit_id, to_key=target["key"],
                                               reason="SUPERSCRIPT_FOOTNOTE_REFERENCE_CANDIDATE",
                                               source_block_ids=[source.unit_id]))
                        seen.add(identity)
    return references, risks


def extract_business_layout(pdf_path, view, *, pages=None, registered_layouts=()):
    """Validate original bytes and emit source-bound, unreviewed layout objects."""
    _require(type(view) is CanonicalTextView, "INVALID_CANONICAL_VIEW")
    raw = Path(pdf_path).read_bytes()
    _require(sha256(raw).hexdigest() == view.raw_pdf_sha256, "SOURCE_HASH_MISMATCH")
    chosen = list(range(1, len(view.pages) + 1)) if pages is None else list(pages)
    _require(all(type(n) is int and 1 <= n <= len(view.pages) for n in chosen), "INVALID_LAYOUT_PAGE")
    selected = set(chosen)
    blocks = {b.unit_id: b for b in view.blocks}
    for block in view.blocks:
        resolve_quote(view, block.span)
    objects = [_registered(r, view, blocks) for r in registered_layouts]
    _require(len({o["key"] for o in objects}) == len(objects), "DUPLICATE_LAYOUT_KEY")
    objects = [o for o in objects if selected.intersection(o["context_pages"])]
    # A selected part of a registered object always expands to its whole source.
    selected.update(n for obj in objects for n in obj["context_pages"])
    excluded = {key for obj in objects if obj["kind"] == "FOOTNOTE" for key in obj["block_ids"]}
    risks, detected = [], []
    with fitz.open(stream=raw, filetype="pdf") as pdf:
        _require(len(pdf) == view.physical_page_count, "SOURCE_PAGE_COUNT_MISMATCH")
        for number in sorted(selected):
            page, page_view = pdf[number - 1], view.pages[number - 1]
            tables = page.find_tables().tables
            # Native strict mode excludes the filled highlight rectangles that
            # otherwise turn ordinary paragraphs into apparent cell grids.
            strict = page.find_tables(strategy="lines_strict").tables
            kept = []
            for index, table in enumerate(tables):
                if any(_inside(table.bbox, parent.bbox) and _area(table.bbox) < _area(parent.bbox) * .85
                       for parent in tables):
                    risks.append(_risk("NESTED_TABLE_CANDIDATE_SUPPRESSED", page_number=number,
                                       detector_index=index, bbox=list(table.bbox)))
                    continue
                if not any(_inside(table.bbox, other.bbox, 2) and _inside(other.bbox, table.bbox, 2)
                           for other in strict):
                    risks.append(_risk("SHADING_TABLE_CANDIDATE_SUPPRESSED", page_number=number,
                                       detector_index=index, bbox=list(table.bbox)))
                    continue
                kept.append(table)
            for index, table in enumerate(kept):
                geometry = _table_geometry(table, page_view, blocks)
                parts = _source_parts(page_view, table.bbox) + geometry["header"]["source_parts"]
                ids = list(dict.fromkeys(p["block_id"] for p in parts))
                matching = [o for o in objects if o["kind"] == "TABLE" and number in o["context_pages"]
                            and ids and set(ids) <= set(o["block_ids"])]
                if matching:
                    obj = matching[0]
                    obj["metadata"].setdefault("tables", []).append(geometry)
                    obj["risks"].extend(geometry["risks"])
                else:
                    obj = dict(key=f"{page_view.page_id}:table:{index + 1}", kind="TABLE", title="",
                               block_ids=ids, context_pages=[number],
                               metadata=dict(provenance="PDF_GEOMETRY_CANDIDATE", semantic_review_state="UNREVIEWED",
                                             answer_ceiling="SOURCE_ONLY", tables=[geometry]), risks=list(geometry["risks"]))
                    objects.append(obj)
                detected.append((obj, geometry))
            objects.extend(_automatic_footnotes(page, page_view, excluded))
        references, reference_risks = _footnote_references(pdf, view, objects, selected)
        risks.extend(reference_risks)
    for first, a in detected:
        for second, b in detected:
            if b["page_number"] != a["page_number"] + 1:
                continue
            if abs(a["bbox"][0] - b["bbox"][0]) > 3 or abs(a["bbox"][2] - b["bbox"][2]) > 3:
                continue
            if first is second:  # Registered context, never semantic approval.
                first["metadata"]["continuation_provenance"] = "REGISTERED"
            else:
                candidate = dict(from_key=first["key"], to_key=second["key"],
                                 from_page=a["page_number"], to_page=b["page_number"],
                                 col_counts=[a["col_count"], b["col_count"]])
                first["metadata"].setdefault("continuation_candidates", []).append(candidate)
                risks.append(_risk("TABLE_CONTINUATION_REVIEW_REQUIRED", **candidate))
    for obj in objects:
        if obj["kind"] == "TABLE" and not obj["metadata"].get("tables"):
            obj["risks"].append(_risk("REGISTERED_TABLE_GEOMETRY_UNRESOLVED", key=obj["key"]))
        risks.extend(obj["risks"])
    return dict(objects=objects, references=references, risks=risks)
