"""Reconstruct source-bound PDF structure without rewriting canonical quotations.

Geometry and typography support candidates, never semantic approval. Elements
partition block characters (the canonical inter-block/page separators are not
block characters). Their list order is a derived reading order.
"""
from collections import Counter
from hashlib import sha256
import math
from pathlib import Path
import re

import fitz

from .parsers import CanonicalTextView


_DIGITS = r"\d(?:\s*\d)?"
_NUMBER = re.compile(r"^\s*(" + _DIGITS + r"(?:\s*[.．]\s*" + _DIGITS + r"){1,3})(?!\s*[\d.．])(?:\s+|(?=[\u4e00-\u9fff]))(.*)$")
_BARE_NUMBER = re.compile(r"^\s*(\d{1,2}(?:\s*[.．]\s*\d{1,2}){1,3}|\d{1,2}\s*[.．])\s*$")
_CHAPTER = re.compile(r"^第[一二三四五六七八九十百零〇\d]+(?:部分|章|节|篇)")
_CHINESE_HEADING = re.compile(r"^([一二三四五六七八九十百]+)[、．.]\s*(.*)$")
_SECTION = re.compile(r"^(?:附[录件表](?:[一二三四五六七八九十\dA-Z]*)?|释义|健康告知|财务告知|投保须知|重要提示|声明及签名栏)$")
_QUESTION = re.compile(r"^\s*(\d(?:\s*\d)?)(?:\s*[、．]\s*|\s*\.(?!\s*\d)\s*|\s+)(.+)$")
_NAV = re.compile(r"阅\s*读\s*指\s*引|条款目录|^目\s*录(?:[（(].*?[）)])?\s*$|\bCONTENTS\b", re.I)
_LEADERS = re.compile(r"[.．…·]{4,}")
_SENTENCE = re.compile(r"[。；;！？!?：:]")
_AMOUNT = re.compile(r"^(?:万?元|亿元|%|％|周岁|个月|公斤|厘米|倍|条(?:所|约|规|定|及|中)|天(?:$|[内后前])|岁(?:$|[至以上下])|年(?:$|[内后前]))")


def _risk(code, **context):
    return dict(code=code, state="REVIEW_REQUIRED", **context)


def _normal(text):
    return re.sub(r"\s+", "", text or "")


def _clean(text):
    return re.sub(r"^[\uf000-\uf8ff◆◇●■]+", "", text.strip()).strip()


def _title(text):
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", text)) and len(_normal(text)) <= 55 and not _SENTENCE.search(text) and not _AMOUNT.match(text.strip())


def _overlap(a, b):
    return min(a[2], b[2]) >= max(a[0], b[0]) - 2 and min(a[3], b[3]) >= max(a[1], b[1]) - 2


def _atoms(page_view, page):
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
    raw = [line for block in page.get_text("dict", flags=flags, sort=True)["blocks"]
           if block.get("type") == 0 for line in block.get("lines", [])]
    number_rows = [line for line in raw if _BARE_NUMBER.fullmatch("".join(s["text"] for s in line["spans"]))]
    table_error = None
    try:
        # One-row boxes commonly frame a chapter banner rather than a data grid.
        table_boxes = [tuple(table.bbox) for table in page.find_tables().tables
                       if table.row_count >= 2 and table.col_count >= 2]
    except Exception as error:
        table_boxes, table_error = [], type(error).__name__
    result = []
    for block in page_view.blocks:
        lines = block.text.split("\n")
        aligned = len(lines) == len(block.line_bboxes)
        cursor = block.span.start
        for index, text in enumerate(lines if aligned else [block.text]):
            end = cursor + len(text) + (aligned and index < len(lines) - 1)
            if end == cursor:
                continue
            box = block.line_bboxes[index] if aligned else block.span.bbox
            candidates = [line for line in raw if _overlap(box, line["bbox"]) and
                          "".join(s["text"] for s in line["spans"]) == text]
            font = max((s["size"] for line in candidates for s in line["spans"] if s["text"].strip()), default=0)
            bold = any(s.get("flags", 0) & 16 for line in candidates for s in line["spans"] if s["text"].strip())
            segments = [(text, box, font, bold)]
            # Some PDFs concatenate a left title and right body into ONE line.
            # Split only an exact raw-span boundary with a gutter number, a gap,
            # and changed typography. Neither detected text nor whitespace is rewritten.
            if len(candidates) == 1 and any(abs(n["bbox"][1] - box[1]) <= 4
                                           and n["bbox"][0] < box[0] - 10 for n in number_rows):
                spans = candidates[0]["spans"]
                for cut in range(1, len(spans)):
                    prefix = "".join(s["text"] for s in spans[:cut])
                    suffix = "".join(s["text"] for s in spans[cut:])
                    left, right = spans[cut - 1], spans[cut]
                    style_change = left["font"] != right["font"] or abs(left["size"] - right["size"]) > .5
                    if (_title(prefix) and len(re.findall(r"[\u4e00-\u9fffA-Za-z]", suffix)) >= 8
                            and right["bbox"][0] - left["bbox"][2] > 3 and style_change):
                        segments = []
                        for part, quote in ((spans[:cut], prefix), (spans[cut:], suffix)):
                            part_box = (min(s["bbox"][0] for s in part), min(s["bbox"][1] for s in part),
                                        max(s["bbox"][2] for s in part), max(s["bbox"][3] for s in part))
                            segments.append((quote, part_box, max(s["size"] for s in part),
                                             any(s.get("flags", 0) & 16 for s in part)))
                        break
            part_start = cursor
            for position, (quote, part_box, size, part_bold) in enumerate(segments):
                part_end = part_start + len(quote) if position < len(segments) - 1 else end
                result.append(dict(block=block, start=part_start, end=part_end, text=quote, bbox=tuple(part_box),
                                   page=page_view.page_number, size=size, bold=part_bold, aligned=aligned,
                                   table=any(b[0] - 1 <= (part_box[0] + part_box[2]) / 2 <= b[2] + 1
                                             and b[1] - 1 <= (part_box[1] + part_box[3]) / 2 <= b[3] + 1 for b in table_boxes),
                                   table_error=table_error))
                part_start = part_end
            cursor = end
    return result


def _span(atom, view):
    text = view.canonical_text[atom["start"]:atom["end"]]
    return dict(block_id=atom["block"].unit_id, text_revision_id=view.text_revision_id,
                page_number=atom["page"], start=atom["start"], end=atom["end"],
                bbox=list(atom["bbox"]), content_sha256=sha256(text.encode("utf-8")).hexdigest(),
                offset_basis="CANONICAL_TEXT_UNICODE_CODEPOINTS", text=text)


def _element(kind, atoms, view, *, title="", clause_no=None, signals=(), risks=()):
    canonical = sorted(atoms, key=lambda a: (a["start"], a["end"]))
    identity = f"{view.text_revision_id}:{kind}:" + ",".join(f"{a['start']}-{a['end']}" for a in canonical)
    return dict(element_id="structure:" + sha256(identity.encode()).hexdigest(), kind=kind,
                source_spans=[_span(a, view) for a in canonical], projection_spans=[_span(a, view) for a in atoms],
                title=title, clause_no=clause_no, signals=list(signals), risks=list(risks))


def _navigation(atoms):
    """Recognize TOC areas and dense columnar continuations, independent of page index."""
    visible = [a for a in atoms if a["text"].strip()]
    anchors = [a for a in visible if _NAV.search(_clean(a["text"]))]
    leaders = [a for a in visible if _LEADERS.search(a["text"])]
    numbered = [a for a in visible if _NUMBER.match(a["text"]) and _title(_NUMBER.match(a["text"]).group(2))]
    columns = {round(a["bbox"][0] / 30) for a in numbered}
    dense = len(numbered) >= 8 and len(columns) >= 2 and len(numbered) >= len(visible) * .55
    # A TOC may share a physical page with the beginning of the contract. An
    # explicit chapter followed by prose supplies a source-supported stop.
    body_starts = [a["bbox"][1] for a in visible if _CHAPTER.match(_clean(a["text"]))
                   and not _SENTENCE.search(a["text"])
                   and any(a["bbox"][1] + 5 < b["bbox"][1] < a["bbox"][1] + 160
                           and _SENTENCE.search(b["text"])
                           and len(re.findall(r"[\u4e00-\u9fffA-Za-z]", b["text"])) >= 8
                           for b in visible)]
    cutoff = min(body_starts, default=math.inf)
    # Explicit guides are navigation from the marker onward. A dense continued
    # TOC can omit its marker; repeated short entries in separate columns support it.
    if dense or (anchors and len(leaders) >= 3):
        return {a["start"] for a in atoms if a["bbox"][1] < cutoff - 2}
    if anchors:
        top = min(a["bbox"][1] for a in anchors)
        return {a["start"] for a in atoms if top - 2 <= a["bbox"][1] < cutoff - 2}
    return {a["start"] for a in leaders}


def _docling_signals(evidence, view, atoms, risks):
    accepted = {}
    if evidence is None:
        return accepted
    if not isinstance(evidence, dict) or evidence.get("raw_pdf_sha256") != view.raw_pdf_sha256:
        risks.append(_risk("DOCLING_SOURCE_MISMATCH"))
        return accepted
    for risk in evidence.get("risks", []):
        if isinstance(risk, str):
            risks.append(_risk(risk, provenance="DOCLING_EVIDENCE"))
        elif isinstance(risk, dict) and isinstance(risk.get("code"), str):
            risks.append(dict(risk, provenance="DOCLING_EVIDENCE"))
        else:
            risks.append(_risk("DOCLING_EVIDENCE_INVALID"))
    entries = evidence.get("elements", [])
    if not isinstance(entries, (list, tuple)):
        risks.append(_risk("DOCLING_EVIDENCE_INVALID"))
        return accepted
    for entry in entries:
        if not isinstance(entry, dict):
            risks.append(_risk("DOCLING_EVIDENCE_INVALID"))
            continue
        number, box = entry.get("page_number", entry.get("page_no")), entry.get("bbox")
        text, label = entry.get("text"), str(entry.get("label", "")).lower()
        valid = (type(number) is int and isinstance(box, (list, tuple)) and len(box) == 4
                 and all(type(n) in (int, float) and math.isfinite(n) for n in box)
                 and box[0] <= box[2] and box[1] <= box[3] and isinstance(text, str) and _normal(text))
        if not valid:
            risks.append(_risk("DOCLING_EVIDENCE_INVALID"))
            continue
        candidates = [a for a in atoms if a["page"] == number and _overlap(a["bbox"], box)
                      and _normal(a["text"]) == _normal(text)]
        if len(candidates) != 1:
            risks.append(_risk("DOCLING_SOURCE_ALIGNMENT_UNRESOLVED", page_number=number, label=label))
        elif label in {"section_header", "title", "heading", "section", "list_item", "text", "page_header", "page_footer"}:
            accepted[candidates[0]["start"]] = label
        else:
            risks.append(_risk("DOCLING_LABEL_UNSUPPORTED", page_number=number, label=label))
    return accepted


def _page_elements(atoms, page_view, view, evidence, repeated, next_page_atoms=()):
    navigation = _navigation(atoms)
    used, elements, risks = set(), [], []
    visible = [a for a in atoms if a["text"].strip()]
    prose = [a for a in visible if _SENTENCE.search(a["text"])]
    body_size = Counter(a["size"] for a in (prose or visible) if a["size"] > 0).most_common(1)
    body_size = body_size[0][0] if body_size else 0
    questionnaire = any(re.search(r"健康.*告知|健康及财务告知|健康询问", a["text"]) for a in atoms)
    question_candidates = []
    for a in visible:
        match = _QUESTION.match(a["text"])
        if match and (questionnaire or re.search(r"是否|有无|何种|多少|\?", match.group(2))):
            question_candidates.append(a)
    questionnaire = questionnaire or bool(question_candidates)

    def emit(kind, group, **kwargs):
        used.update(a["start"] for a in group)
        elements.append((_element(kind, group, view, **kwargs), group))

    for a in atoms:
        if a["start"] in navigation:
            emit("NAVIGATION", [a], signals=["TOC_OR_READING_GUIDE"])
        elif not a["text"].strip():
            emit("FURNITURE", [a], signals=["WHITESPACE"])
        elif (a["bbox"][1] < page_view.height * .07 or a["bbox"][3] > page_view.height * .95) and (
                re.fullmatch(r"\s*(?:第?\s*\d+\s*页?|\d+\s*/\s*\d+)\s*", a["text"])
                or _normal(a["text"]) in repeated):
            emit("FURNITURE", [a], signals=["REPEATED_PAGE_MARGIN_OR_PAGE_NUMBER"])
    for a in visible:
        if a["start"] in used:
            continue
        text = _clean(a["text"])
        if a["text"].strip() in {"\uf075", "◆", "■"}:
            titles = [b for b in visible if b["start"] not in used and b is not a
                      and abs(b["bbox"][1] - a["bbox"][1]) <= 4 and b["bbox"][0] > a["bbox"][0]
                      and _title(b["text"]) and b["size"] > body_size * 1.1]
            if titles:
                title_atom = min(titles, key=lambda b: b["bbox"][0])
                emit("SECTION", [a, title_atom], title=_clean(title_atom["text"]), signals=["SECTION_MARKER", "TYPOGRAPHY"])
                continue
        if (_CHAPTER.match(text) and len(text) <= 45 and not re.search(r"[。；;]", text)) or _SECTION.fullmatch(text):
            emit("SECTION", [a], title=text, signals=["EXPLICIT_SECTION_MARKER"])
        elif a in question_candidates and questionnaire:
            match = _QUESTION.match(a["text"])
            emit("QUESTION", [a], title=match.group(2).strip(), clause_no=_normal(match.group(1)),
                 signals=["QUESTIONNAIRE_ITEM_NUMBER", "QUESTION_OR_FORM_CONTEXT"])
        elif (_CHINESE_HEADING.match(text) and _title(_CHINESE_HEADING.match(text).group(2))):
            match = _CHINESE_HEADING.match(text)
            emit("HEADING", [a], title=match.group(2).strip(), clause_no=match.group(1),
                 signals=["CHINESE_NUMBERED_TITLE"])
        elif (a["text"].strip().startswith(("\uf075", "◆", "■")) and _title(text)
              and a["size"] > body_size * 1.1):
            emit("SECTION", [a], title=text, signals=["SECTION_MARKER", "TYPOGRAPHY"])

    # A bare number in a left gutter plus its same-row title and right-column
    # body is strong geometric support even if the extractor interleaves them.
    for number in visible:
        if number["start"] in used or number["table"]:
            continue
        match = _BARE_NUMBER.fullmatch(number["text"])
        if not match:
            continue
        x, y = number["bbox"][0], number["bbox"][1]
        same_row = [a for a in visible if a["start"] not in used and a is not number
                    and abs(a["bbox"][1] - y) <= 4 and a["bbox"][0] > x + 12]
        same_row.sort(key=lambda a: a["bbox"][0])
        first = same_row[0] if same_row else None
        if first is None or not _title(first["text"]):
            continue
        tx = first["bbox"][0]
        body = [a for a in visible if a["start"] not in used and a["bbox"][0] > tx + 25
                and y - 4 <= a["bbox"][1] <= y + 40 and a is not first
                and len(re.findall(r"[\u4e00-\u9fffA-Za-z]", a["text"])) >= 8]
        if not body:
            continue
        bx = min(a["bbox"][0] for a in body)
        group = [number, first]
        next_number_y = min((a["bbox"][1] for a in visible if a is not number
                             and abs(a["bbox"][0] - x) < 4 and a["bbox"][1] > y + 4
                             and _BARE_NUMBER.fullmatch(a["text"])), default=math.inf)
        for continuation in sorted(visible, key=lambda a: (a["bbox"][1], a["start"])):
            if continuation["start"] in used or continuation in group:
                continue
            if (abs(continuation["bbox"][0] - tx) <= 4 and
                    4 < continuation["bbox"][1] - group[-1]["bbox"][1] <= 24
                    and continuation["bbox"][1] < next_number_y - 4
                    and continuation["bbox"][0] < bx - 15 and _title(continuation["text"])
                    and not _NUMBER.match(continuation["text"]) and not _BARE_NUMBER.match(continuation["text"])):
                group.append(continuation)
        clause_no = _normal(match.group(1)).replace("．", ".").rstrip(".")
        emit("HEADING" if "." in clause_no else "SECTION", group,
             title="".join(a["text"].strip() for a in group[1:]), clause_no=clause_no,
             signals=["NUMBERED_GUTTER", "TITLE_COLUMN", "ASSOCIATED_BODY_COLUMN"])

    for a in visible:
        if a["start"] in used or a["table"]:
            continue
        match = _NUMBER.match(a["text"])
        if match and _title(match.group(2)):
            following = [b for b in visible if b["start"] not in used and b is not a
                         and b["bbox"][1] > a["bbox"][1] + 3 and b["bbox"][1] < a["bbox"][3] + 65
                         and b["bbox"][0] >= a["bbox"][0] - 4]
            next_body = (a["bbox"][1] > page_view.height * .6
                         and (a["bold"] or a["size"] > body_size * 1.1)
                         and any(_SENTENCE.search(b["text"]) and b["bbox"][0] >= a["bbox"][0] - 4
                                 for b in next_page_atoms if b["text"].strip()))
            if following or next_body:
                emit("HEADING", [a], title=match.group(2).strip(), clause_no=_normal(match.group(1)).replace("．", "."),
                     signals=["NUMBERED_TITLE", "FOLLOWING_BODY" if following else "NEXT_PAGE_BODY"])
                continue
        if evidence.get(a["start"]) in {"section_header", "title", "heading", "section"} and _title(a["text"]):
            if a["bold"] or a["size"] > max((b["size"] for b in atoms if b["start"] != a["start"]), default=0) * 1.1:
                emit("SECTION", [a], title=_clean(a["text"]), signals=["ALIGNED_DOCLING_HEADING", "TYPOGRAPHY"])

    for a in atoms:
        if a["start"] in used:
            continue
        item_risks = []
        if _BARE_NUMBER.fullmatch(a["text"]) and not a["table"]:
            item_risks.append(_risk("AMBIGUOUS_NUMBERED_TEXT", page_number=a["page"], start=a["start"]))
        if not a["aligned"]:
            item_risks.append(_risk("LINE_GEOMETRY_UNRESOLVED", block_id=a["block"].unit_id))
        risks.extend(item_risks)
        emit("UNKNOWN" if item_risks else "BODY", [a],
             signals=["CANONICAL_SOURCE_TEXT"] + (["TABLE_REGION"] if a["table"] else []), risks=item_risks)

    # Row clustering tolerates font ascent differences. Heading groups take the
    # first row's place; suffix fragments do not drag their body backwards.
    rows = []
    for element, group in sorted(elements, key=lambda pair: (min(a["bbox"][1] for a in pair[1]), min(a["start"] for a in pair[1]))):
        top = min(a["bbox"][1] for a in group)
        if not rows or top - rows[-1][0] > 4:
            rows.append((top, []))
        rows[-1][1].append((element, group))
    ordered = []
    for _, entries in rows:
        entries.sort(key=lambda pair: (min(a["bbox"][0] for a in pair[1]), min(a["start"] for a in pair[1])))
        ordered.extend(element for element, _ in entries)
    return ordered, risks


def _validate_coverage(elements, blocks):
    expected = [(b.span.start, b.span.end) for b in blocks]
    spans = sorted((s["start"], s["end"]) for e in elements for s in e["source_spans"])
    index = 0
    for start, end in expected:
        cursor = start
        while index < len(spans) and spans[index][0] < end:
            left, right = spans[index]
            if left != cursor or not left < right <= end:
                raise ValueError("STRUCTURE_COVERAGE_MISMATCH")
            cursor, index = right, index + 1
        if cursor != end:
            raise ValueError("STRUCTURE_COVERAGE_MISMATCH")
    if index != len(spans):
        raise ValueError("STRUCTURE_COVERAGE_MISMATCH")


def _join_cross_page_titles(elements, all_atoms, view):
    """Extend a gutter-supported title only within the matching next-page column."""
    atom_index={(a["start"],a["end"]):a for atoms in all_atoms.values() for a in atoms}
    for page_number in sorted(all_atoms):
        if page_number+1 not in all_atoms:continue
        boundaries=[e for e in elements if e["kind"] in {"HEADING","SECTION","QUESTION","NAVIGATION"}
                    and any(r["page_number"]==page_number for r in e["source_spans"])]
        if not boundaries:continue
        heading=boundaries[-1]
        if heading["kind"]!="HEADING" or "TITLE_COLUMN" not in heading["signals"]:continue
        group=[atom_index[(r["start"],r["end"])] for r in heading["projection_spans"]]
        titles=[a for a in group if not _BARE_NUMBER.fullmatch(a["text"])]
        if not titles or min(a["bbox"][1] for a in titles)<view.pages[page_number-1].height*.8:continue
        tx=titles[0]["bbox"][0];right=max(a["bbox"][2] for a in titles)
        leading=[]
        for element in elements:
            if not any(r["page_number"]==page_number+1 for r in element["source_spans"]):continue
            if element["kind"] not in {"BODY","FURNITURE"}:break
            if element["kind"]=="BODY":leading.append(element)
        continued=[];last_y=None
        first_body_y=min((atom_index[(r["start"],r["end"])]["bbox"][1] for e in leading for r in e["source_spans"]),default=math.inf)
        for element in leading:
            if len(element["source_spans"])!=1:continue
            ref=element["source_spans"][0];atom=atom_index[(ref["start"],ref["end"])];text=_normal(atom["text"])
            if (not re.fullmatch(r"[\u4e00-\u9fffA-Za-z]{1,12}",text) or atom["table"]
                    or abs(atom["bbox"][0]-tx)>4 or atom["bbox"][1]>view.pages[page_number].height*.2
                    or abs(atom["size"]-titles[-1]["size"])>.6 or atom["bold"]!=titles[-1]["bold"]
                    or (last_y is None and atom["bbox"][1]>first_body_y+4)
                    or (last_y is not None and not 4<atom["bbox"][1]-last_y<=24)):
                continue
            if not any(other["bbox"][0]>right+10 and abs(other["bbox"][1]-atom["bbox"][1])<=4
                       and other["text"].strip() for other in all_atoms[page_number+1]):continue
            continued.append((element,atom));last_y=atom["bbox"][1]
        if not continued:continue
        combined=_element("HEADING",group+[a for _,a in continued],view,
                          title=heading["title"]+"".join(_normal(a["text"]) for _,a in continued),
                          clause_no=heading["clause_no"],signals=[*heading["signals"],"NEXT_PAGE_TITLE_CONTINUATION"],
                          risks=[*heading["risks"],*(risk for e,_ in continued for risk in e["risks"])])
        dropped={e["element_id"] for e,_ in continued}
        elements=[combined if e["element_id"]==heading["element_id"] else e
                  for e in elements if e["element_id"] not in dropped]
    return elements


def extract_document_structure(pdf_path, view, *, pages=None, docling_document=None):
    """Return JSON structure with a complete exact partition of selected blocks.

    ``pages`` uses one-based physical page numbers. Optional normalized Docling
    evidence must carry this PDF's raw SHA256; unaligned labels only add risks.
    The default engine uses local PyMuPDF geometry, with OCR explicitly absent.
    """
    if type(view) is not CanonicalTextView:
        raise ValueError("CANONICAL_VIEW_REQUIRED")
    selected = list(range(1, len(view.pages) + 1)) if pages is None else list(pages)
    if len(set(selected)) != len(selected) or any(type(n) is not int or not 1 <= n <= len(view.pages) for n in selected):
        raise ValueError("PAGE_SELECTION_INVALID")
    if sha256(Path(pdf_path).read_bytes()).hexdigest() != view.raw_pdf_sha256:
        raise ValueError("PDF_SOURCE_MISMATCH")
    risks, elements = [], []
    with fitz.open(pdf_path) as pdf:
        if len(pdf) != len(view.pages):
            raise ValueError("PDF_PAGE_COUNT_MISMATCH")
        all_atoms = {n: _atoms(view.pages[n - 1], pdf[n - 1]) for n in sorted(selected)}
        margins = Counter(_normal(a["text"]) for n, atoms in all_atoms.items() for a in atoms
                          if a["text"].strip() and (a["bbox"][1] < view.pages[n - 1].height * .07
                                                   or a["bbox"][3] > view.pages[n - 1].height * .95))
        repeated = {text for text, count in margins.items() if count >= 2}
        evidence = _docling_signals(docling_document, view, [a for atoms in all_atoms.values() for a in atoms], risks)
        for number, atoms in all_atoms.items():
            page_view = view.pages[number - 1]
            page_elements, page_risks = _page_elements(atoms, page_view, view, evidence, repeated,
                                                     all_atoms.get(number + 1, ()))
            elements.extend(page_elements)
            risks.extend(page_risks)
            if atoms and atoms[0]["table_error"]:
                risks.append(_risk("TABLE_GEOMETRY_UNAVAILABLE", page_number=number, error_type=atoms[0]["table_error"]))
            if page_view.image_review_required:
                risks.append(_risk("IMAGE_CONTENT_UNREVIEWED", page_number=number))
            if page_view.coverage_state != "TEXT_EXTRACTED_UNREVIEWED":
                risks.append(_risk(page_view.coverage_state, page_number=number))
    elements=_join_cross_page_titles(elements,all_atoms,view)
    _validate_coverage(elements, [b for b in view.blocks if b.span.page_number in selected])
    return dict(source_id=view.source_id, raw_pdf_sha256=view.raw_pdf_sha256,
                text_revision_id=view.text_revision_id,
                engine="pymupdf-geometry-structure-v1" + ("+docling-evidence" if evidence else ""),
                engine_version=fitz.VersionBind, pages=sorted(selected), elements=elements, risks=risks,
                docling_evidence=dict(scope="ALIGNED_EVIDENCE_ONLY", aligned_lines=len(evidence),
                                      engine=docling_document.get("engine"), version=docling_document.get("version"))
                if isinstance(docling_document, dict) else None,
                coverage=dict(state="COMPLETE_SELECTED_BLOCK_PARTITION", blocks=sum(len(view.pages[n - 1].blocks) for n in selected),
                              characters=sum(b.span.end - b.span.start for b in view.blocks if b.span.page_number in selected)),
                answer_ceiling="SOURCE_ONLY", semantic_complete=False, ocr_enabled=False)
