"""Source-bound business candidates. Structural processing never grants approval."""
from copy import deepcopy
from hashlib import sha256
import json
import re

FLAGS = {"semantic_complete": False, "semantic_review_state": "UNREVIEWED",
         "answer_ceiling": "SOURCE_ONLY", "production_readiness": "NO_GO"}
KINDS = {"DOCUMENT", "CLAUSE_CANDIDATE", "LIST_CANDIDATE", "LIST_INTRO", "LIST_ITEM",
         "TABLE_CANDIDATE", "TABLE_HEADER_CANDIDATE", "TABLE_ROW_CANDIDATE", "FOOTNOTE_CANDIDATE", "UNCLASSIFIED",
         "NAVIGATION", "FURNITURE", "SECTION", "QUESTION_CANDIDATE"}
HEADING = re.compile(r"^\s*(\d+(?:\.\d+){1,5}|第[一二三四五六七八九十百零〇\d]+条)[ \t]*(?:\n[ \t]*)?([^\n]{1,45})")
SECTION = re.compile(r"^(?:计划[一二三四五六七八九十\d]+|第[一二三四五六七八九十百\d]+[章节])$")
ITEM = re.compile(r"[（(](\d+)[)）]")
REFERENCE = re.compile(r"(?:见|根据|按照|参见)\s*(第[一二三四五六七八九十百零〇\d]+条|\d+(?:\.\d+){1,5})")


def _check(ok, code):
    if not ok:
        raise ValueError(code)


def _identity(prefix, value):
    return prefix + sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _risk_codes(values):
    return [v["code"] if isinstance(v,dict) else str(v) for v in values]


def _declared_count(text):
    matches=list(re.finditer(r"(?:以下|下列|如下|上述)\s*(?:共|的|这)?\s*([一二三四五六七八九十两\d]+)\s*[项种个](?:条件|情形|要求)?",text))
    if not matches:return None
    value=matches[-1].group(1)
    if value.isdigit():return int(value)
    digits={c:n for n,c in enumerate("零一二三四五六七八九")};digits["两"]=2
    if value=="十":return 10
    if "十" in value:
        left,right=value.split("十",1)
        return digits.get(left,1)*10+digits.get(right,0)
    return digits.get(value)


def _list_logic(text):
    all_cue=bool(re.search(r"同时|所有|全部|均须",text))
    any_cue=bool(re.search(r"任一|之一|至少一|其一",text))
    return "UNKNOWN" if all_cue==any_cue else "ALL" if all_cue else "ANY"


def _ref(view, block_id, start=None, end=None):
    block = next((b for b in view.blocks if b.unit_id == block_id), None)
    _check(block is not None, "UNKNOWN_SOURCE_BLOCK")
    start = block.span.start if start is None else start
    end = block.span.end if end is None else end
    _check(type(start) is int and type(end) is int and block.span.start <= start < end <= block.span.end, "SOURCE_SPAN_OUTSIDE_BLOCK")
    text = view.canonical_text[start:end]
    return {"block_id": block_id, "page_number": block.span.page_number, "start": start, "end": end,
            "bbox": list(block.span.bbox), "text_revision_id": view.text_revision_id,
            "content_sha256": sha256(text.encode()).hexdigest()}


def _normal_refs(view, refs):
    checked = []
    for proposed in refs:
        actual = _ref(view, proposed["block_id"], proposed.get("start"), proposed.get("end"))
        for key in ("page_number", "text_revision_id", "content_sha256"):
            _check(key not in proposed or proposed[key] == actual[key], "SOURCE_REFERENCE_MISMATCH")
        _check("text" not in proposed or proposed["text"] == view.canonical_text[actual["start"]:actual["end"]], "SOURCE_TEXT_MISMATCH")
        checked.append(actual)
    merged = []
    for ref in sorted(checked, key=lambda r:(r["start"],r["end"])):
        if merged and ref["block_id"] == merged[-1]["block_id"] and ref["start"] <= merged[-1]["end"]:
            merged[-1] = _ref(view, ref["block_id"], merged[-1]["start"], max(ref["end"],merged[-1]["end"]))
        else:
            merged.append(ref)
    return merged


def _projection(view, refs):
    return "\n".join(view.canonical_text[r["start"]:r["end"]] for r in refs)


def _ordered_refs(view, refs):
    """Validate a reading sequence without sorting away its layout information."""
    checked = []
    for proposed in refs:
        actual = _normal_refs(view, [proposed])
        checked.extend(actual)
    return checked


def _validate_projection(view, refs, ordered):
    _check(_normal_refs(view, ordered) == refs and
           sum(r["end"]-r["start"] for r in ordered) == sum(r["end"]-r["start"] for r in refs),
           "PROJECTION_SOURCE_PARTITION_MISMATCH")


def _validate_structure(view, structure, blocks):
    _check(structure.get("source_id") == view.source_id and
           structure.get("raw_pdf_sha256") == view.raw_pdf_sha256 and
           structure.get("text_revision_id") == view.text_revision_id, "STRUCTURE_SOURCE_SCOPE")
    spans, ids = [], set()
    kinds = {"HEADING", "BODY", "NAVIGATION", "FURNITURE", "SECTION", "QUESTION", "UNKNOWN"}
    try:
        for element in structure["elements"]:
            _check(element["element_id"] not in ids and element["kind"] in kinds, "STRUCTURE_ELEMENT_INVALID")
            ids.add(element["element_id"])
            refs = _normal_refs(view, element["source_spans"])
            _check(refs and all(r["block_id"] in blocks for r in refs), "STRUCTURE_SOURCE_SCOPE")
            _validate_projection(view, refs, _ordered_refs(view, element.get("projection_spans", element["source_spans"])))
            spans.extend(refs)
        expected = _normal_refs(view, [_ref(view, b) for b in blocks])
        _check(_normal_refs(view, spans) == expected and
               sum(r["end"]-r["start"] for r in spans) == sum(r["end"]-r["start"] for r in expected),
               "STRUCTURE_COVERAGE_PARTITION_MISMATCH")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("STRUCTURE_INVALID: " + str(error)) from error


def _structure_groups(view, structure, occupied):
    """Business interpretation follows already reconstructed structural boundaries."""
    current, section, last_page, serial = None, "", None, 0
    def emit():
        nonlocal current, serial
        if current is None:
            return None
        serial += 1
        current["key"] = "structure:" + str(serial)
        current["metadata"]["body_projection_text"] = _projection(view, current["metadata"]["body_source_spans"])
        result, current = current, None
        return result
    def start(kind, element, refs):
        title = element.get("title", "") or ("未确认结构" if kind == "UNCLASSIFIED" else "")
        number = element.get("clause_no", "")
        return {"kind": kind, "refs": refs.copy(), "title": title,
                "metadata": {"clause_no": number, "origin": "DOCUMENT_STRUCTURE",
                             "heading_path": [s for s in (section, number, title) if s],
                             "structure_element_ids": [element["element_id"]],
                             "heading_source_spans": refs.copy() if kind in {"CLAUSE_CANDIDATE", "QUESTION_CANDIDATE"} else [],
                             "body_source_spans": refs.copy() if kind == "UNCLASSIFIED" else []},
                "unit_risks": list(_risk_codes(element.get("risks", []))) + (["STRUCTURE_UNRESOLVED"] if kind == "UNCLASSIFIED" else [])}
    for element in structure["elements"]:
        refs = [r for r in _ordered_refs(view, element.get("projection_spans", element["source_spans"]))
                if r["block_id"] not in occupied]
        if not refs:
            if element["kind"] in {"NAVIGATION","SECTION","HEADING","QUESTION","UNKNOWN"}:
                item=emit()
                if item:yield item
                if element["kind"]=="SECTION":section=element.get("title","")
            continue
        kind = element["kind"]
        page = min(r["page_number"] for r in refs)
        if last_page is not None and page > last_page + 1:
            item = emit()
            if item: yield item
        last_page = max(r["page_number"] for r in refs)
        if kind in {"NAVIGATION", "SECTION", "FURNITURE", "UNKNOWN"}:
            # Page furniture is detached without breaking a body/list continuation.
            if kind != "FURNITURE":
                item = emit()
                if item: yield item
            serial += 1
            risk = (["NAVIGATION_REVIEW_REQUIRED"] if kind == "NAVIGATION" else
                    ["FURNITURE_REVIEW_REQUIRED"] if kind == "FURNITURE" else
                    ["STRUCTURE_UNRESOLVED"] if kind == "UNKNOWN" else [])
            yield {"key": "region:" + str(serial), "kind": "UNCLASSIFIED" if kind == "UNKNOWN" else kind,
                   "refs": refs, "title": element.get("title", "") or kind,
                   "metadata": {"origin": "DOCUMENT_STRUCTURE", "structure_element_ids": [element["element_id"]],
                                "heading_path": [section] if section else []},
                   "unit_risks": [*risk, *_risk_codes(element.get("risks", []))]}
            if kind == "SECTION": section = element.get("title", "") or _projection(view, refs)
        elif kind in {"HEADING", "QUESTION"}:
            item = emit()
            if item: yield item
            current = start("QUESTION_CANDIDATE" if kind == "QUESTION" else "CLAUSE_CANDIDATE", element, refs)
        elif current is None:
            current = start("UNCLASSIFIED", element, refs)
        else:
            current["refs"].extend(refs)
            current["metadata"]["body_source_spans"].extend(refs)
            current["metadata"]["structure_element_ids"].append(element["element_id"])
            current["unit_risks"].extend(_risk_codes(element.get("risks", [])))
    item = emit()
    if item: yield item


def _slice(view, refs, start, end):
    """Map a derived joined-text range back to each actual original-text range."""
    result, cursor = [], 0
    for ref in refs:
        length = ref["end"]-ref["start"]
        left, right = max(start,cursor), min(end,cursor+length)
        if left < right:
            result.append(_ref(view,ref["block_id"],ref["start"]+left-cursor,ref["start"]+right-cursor))
        cursor += length+1
    return result


def _reading_bounds(refs, start, end):
    """Exclude join separators at fragment edges; they have no source offsets."""
    cursor, leftmost, rightmost = 0, None, None
    for ref in refs:
        length=ref["end"]-ref["start"]
        left,right=max(start,cursor),min(end,cursor+length)
        if left<right:
            if leftmost is None:leftmost=left
            rightmost=right
        cursor+=length+1
    _check(leftmost is not None,"EMPTY_READING_FRAGMENT")
    return leftmost,rightmost


def _heading(text):
    found = HEADING.match(text)
    if not found:
        return None
    number, title = found.groups()
    title = title.strip()
    # Numbering plus a short heading line are structural cues, never business labels.
    if not re.search(r"[\u4e00-\u9fff]",title) or re.fullmatch(r"[百千万亿元年月日岁%％\d.,， ]+",title):
        return None
    if len(title)>30 or re.search(r"[。；;！？]",title):
        return None
    return number, title


def build_business_document(view, *, product_id, layout=None, clause_records=(), pages=None, structure=None):
    view.__post_init__()
    _check(isinstance(product_id,str) and bool(product_id.strip()), "PRODUCT_SCOPE_REQUIRED")
    selected = sorted({p.page_number for p in view.pages} if pages is None else set(pages))
    _check(selected and all(type(n) is int and 1<=n<=view.physical_page_count for n in selected), "PAGE_SCOPE_INVALID")
    blocks = {b.unit_id:b for b in view.blocks if b.span.page_number in selected}
    if structure is not None:
        _validate_structure(view, structure, blocks)
    nonbody_refs,structure_risks={},{}
    for element in (structure or {}).get("elements",[]):
        codes=_risk_codes(element.get("risks",[]))+(["STRUCTURE_UNRESOLVED"] if element["kind"]=="UNKNOWN" else [])
        for ref in element["source_spans"]:
            if codes:structure_risks.setdefault(ref["block_id"],[]).append((ref,codes))
            if element["kind"] in {"NAVIGATION","FURNITURE"} and view.canonical_text[ref["start"]:ref["end"]].strip():
                nonbody_refs.setdefault(ref["block_id"],[]).append(ref)
    layout = deepcopy(layout or {"objects":[],"references":[],"risks":[]})
    units, edges, risks = [], [], _risk_codes(layout.get("risks",[]))
    unresolved_layout_pages={p for obj in layout.get("objects",[]) if not obj.get("block_ids") for p in obj.get("context_pages",[])}
    layout_objects={o["key"]:o for o in layout.get("objects",[])}
    keys = {}

    def add(key, kind, refs, *, parent=None, title="", metadata=None, shared=(), unit_risks=(), projection_spans=None):
        _check(key not in keys and kind in KINDS, "BUSINESS_UNIT_DUPLICATE")
        refs = _normal_refs(view,refs)
        ordered = _ordered_refs(view, projection_spans) if projection_spans is not None else refs
        _validate_projection(view, refs, ordered)
        _check(all(r["block_id"] in blocks for r in refs), "BUSINESS_UNIT_SCOPE")
        projection=_projection(view,ordered)
        unit_risks=list(unit_risks)
        region_conflict=kind not in {"DOCUMENT","NAVIGATION","FURNITURE","SECTION"} and any(
            min(p["end"],r["end"])>max(p["start"],r["start"])
            for r in refs for p in nonbody_refs.get(r["block_id"],[]))
        if region_conflict:unit_risks.append("STRUCTURE_LAYOUT_REGION_CONFLICT_REVIEW_REQUIRED")
        for r in refs:
            for original,codes in structure_risks.get(r["block_id"],[]):
                if min(original["end"],r["end"])>max(original["start"],r["start"]):unit_risks.extend(codes)
        if structure is not None and any(p.image_review_required and p.page_number in {r["page_number"] for r in refs} for p in view.pages):
            unit_risks.append("IMAGE_CONTENT_REVIEW_REQUIRED")
        if any(r["page_number"] in unresolved_layout_pages for r in refs):unit_risks.append("LAYOUT_SOURCE_UNRESOLVED")
        for risk in layout.get("risks",[]):
            code=_risk_codes([risk])[0]
            if not re.search(r"UNRESOLVED|REVIEW_REQUIRED|AMBIGUOUS|UNCONFIRMED|INCOMPLETE|MISMATCH",code):continue
            if not isinstance(risk,dict):unit_risks.append(code);continue
            affected=set(risk.get("source_block_ids",[]))
            if risk.get('block_id'):affected.add(risk['block_id'])
            for name in ("from_key","to_key"):
                affected.update(layout_objects.get(risk.get(name),{}).get("block_ids",[]))
            risk_pages={risk[name] for name in ("page_number","from_page","to_page") if name in risk}
            if (affected and any(r["block_id"] in affected for r in refs) or
                not affected and risk_pages and any(r["page_number"] in risk_pages for r in refs) or
                not affected and not risk_pages):unit_risks.append(code)
        if kind in {"CLAUSE_CANDIDATE","FOOTNOTE_CANDIDATE","UNCLASSIFIED"}:
            numbered=list(ITEM.finditer(projection))
            if numbered and re.search(r"以下|下列|如下|包括|条件|情形",projection[:numbered[0].start()]):
                values=[int(m.group(1)) for m in numbered]
                if values!=list(range(1,len(values)+1)):unit_risks.append("LIST_SEQUENCE_UNRESOLVED")
                declared=_declared_count(projection[:numbered[0].start()])
                if declared is not None and declared!=len(values):unit_risks.append("LIST_ITEM_COUNT_MISMATCH")
                if _list_logic(projection[:numbered[0].start()])=="UNKNOWN":unit_risks.append("LIST_LOGIC_REVIEW_REQUIRED")
                if projection.rstrip().endswith((";","；",":" ,"：")):unit_risks.append("LIST_CONTINUATION_REVIEW_REQUIRED")
        unit = {"key":key,"kind":kind,"parent_id":parent,"source_spans":refs,"projection_spans":ordered,
                "block_ids":list(dict.fromkeys(r["block_id"] for r in refs)),
                "title":title,"heading_path":list(metadata.get("heading_path",[])) if metadata else [],
                "shared_context_refs":list(shared),"metadata":metadata or {},"risks":sorted(set(_risk_codes(unit_risks))),
                "projection_text":projection,"projection_is_original_contiguous_text":False,
                "product_id":product_id,"text_revision_id":view.text_revision_id,
                "retrieval_eligible":not region_conflict and kind not in {"NAVIGATION","FURNITURE","SECTION","DOCUMENT"},**FLAGS}
        unit["unit_id"] = _identity("business-unit-v1:",unit)
        units.append(unit); keys[key]=unit
        return unit

    root = add("document","DOCUMENT",[_ref(view,b) for b in blocks],title=view.source_id)
    occupied, registered_blocks = set(), set()
    for obj in layout.get("objects",[]):
        _check(obj["kind"] in {"TABLE","FOOTNOTE"}, "LAYOUT_KIND_INVALID")
        meta=obj.get("metadata",{})
        _check(meta.get("product_id",product_id)==product_id,"LAYOUT_PRODUCT_SCOPE")
        _check(meta.get("source_id",view.source_id)==view.source_id and meta.get("text_revision_id",view.text_revision_id)==view.text_revision_id,"LAYOUT_SOURCE_SCOPE")
        ids = obj["block_ids"]
        if not ids and meta.get("provenance")!="REGISTERED":
            risks.append("LAYOUT_SOURCE_UNRESOLVED")
            continue
        _check(ids and all(i in blocks for i in ids),"LAYOUT_SOURCE_SCOPE")
        occupied.update(ids)
        kind = "TABLE_CANDIDATE" if obj["kind"]=="TABLE" else "FOOTNOTE_CANDIDATE"
        table_unit=add(obj["key"],kind,[_ref(view,b) for b in ids],parent=root["unit_id"],title=obj["title"],
            metadata={**obj.get("metadata",{}),"context_pages":obj.get("context_pages",[])},unit_risks=obj.get("risks",[]))
        if kind=="TABLE_CANDIDATE":
            registered_header=None
            for table_index,table in enumerate(obj.get("metadata",{}).get("tables",[])):
                header_refs=table.get("header_source_spans",[]) if table.get("header_rows") else []
                header=None
                if header_refs:
                    header=add(obj["key"]+f":table:{table_index}:header","TABLE_HEADER_CANDIDATE",header_refs,
                        parent=table_unit["unit_id"],title=obj["title"],metadata={"page_number":table["page_number"]},unit_risks=table.get("risks",[]))
                    if obj.get("metadata",{}).get("provenance")=="REGISTERED":registered_header=header
                elif registered_header is not None:
                    header=registered_header
                row_numbers=sorted({r for cell in table.get("cells",[]) if not cell.get("is_header")
                                    for r in range(cell["row"],cell["row"]+cell.get("rowspan",1))})
                for row in row_numbers:
                    cells=[cell for cell in table["cells"] if cell["row"]<=row<cell["row"]+cell.get("rowspan",1)]
                    refs=[ref for cell in cells for ref in cell.get("source_parts",[])]
                    if not refs:continue
                    add(obj["key"]+f":table:{table_index}:row:{row}","TABLE_ROW_CANDIDATE",refs,
                        parent=table_unit["unit_id"],title=obj["title"],shared=[header["unit_id"]] if header else [],
                        metadata={"page_number":table["page_number"],"row":row,"cells":[{k:c.get(k) for k in ("col","rowspan","colspan","bbox")} for c in cells],
                                  "header_inference":"CANDIDATE","heading_path":[obj["title"]]},
                        unit_risks=[*table.get("risks",[]),*obj.get("risks",[])])

    for record in clause_records:
        _check(record["product_id"]==product_id,"REGISTERED_CLAUSE_SCOPE")
        _check(record.get("source_id",view.source_id)==view.source_id and
               record.get("raw_pdf_sha256",view.raw_pdf_sha256)==view.raw_pdf_sha256 and
               record.get("text_revision_id",view.text_revision_id)==view.text_revision_id,"REGISTERED_SOURCE_MISMATCH")
        ids = record["evidence_ids"]
        _check(ids and all(i in blocks for i in ids),"REGISTERED_CLAUSE_SOURCE")
        occupied.update(ids)
        registered_blocks.update(ids)
        refs=[_ref(view,b) for b in ids]
        parent=root
        if record.get("parent_clause_no"):
            candidates=[u for u in units if u["kind"]=="CLAUSE_CANDIDATE" and u["metadata"].get("clause_no")==record["parent_clause_no"]]
            _check(len(candidates)==1 and set(ids)<=set(candidates[0]["block_ids"]),"REGISTERED_PARENT_SCOPE")
            parent=candidates[0]
        add("registered:"+record["record_id"],"CLAUSE_CANDIDATE",refs,parent=parent["unit_id"],
            title=record.get("title",record["clause_no"]),metadata={"clause_no":record["clause_no"],
            "origin":"REGISTERED_SOURCE_RANGES","context_pages":record.get("context_pages",[]),
            "heading_path":[record["clause_no"],record.get("title","")]})

    current, title, number, section, last_page, counter = [], "", "", "", None, 0
    if structure is not None:
        for group in _structure_groups(view, structure, occupied):
            add(**{k:v for k,v in group.items() if k != "refs"}, refs=group["refs"],
                projection_spans=group["refs"], parent=root["unit_id"])
        risks.extend(_risk_codes(structure.get("risks", [])))
    def flush():
        nonlocal current,counter
        if current:
            counter += 1
            add("auto:"+str(counter),"CLAUSE_CANDIDATE" if number else "UNCLASSIFIED",current,parent=root["unit_id"],
                title=title or "未确认结构", metadata={"clause_no":number,"origin":"STRUCTURAL_CANDIDATE",
                "heading_path":[s for s in (section,number,title) if s]},unit_risks=[] if number else ["STRUCTURE_UNRESOLVED"])
            current=[]
    for bid,block in (blocks.items() if structure is None else ()):
        if bid in occupied:
            if bid in registered_blocks:
                flush(); number="";title="";last_page=None
            continue
        text=block.text.strip()
        if last_page is not None and block.span.page_number>last_page+1:
            flush();number="";title="";risks.append("NONADJACENT_PAGE_BOUNDARY")
        starts={0}
        for match in re.finditer(r"(?m)^[ \t]*(?:\d+(?:\.\d+){1,5}|第[一二三四五六七八九十百零〇\d]+条)",block.text):
            if _heading(block.text[match.start():]):starts.add(match.start())
        for match in re.finditer(r"(?m)^[ \t]*(?:计划[一二三四五六七八九十\d]+|第[一二三四五六七八九十百\d]+[章节])[ \t]*$",block.text):
            starts.add(match.start())
        bounds=sorted(starts)+[len(block.text)]
        for left,right in zip(bounds,bounds[1:]):
            part=block.text[left:right];text=part.strip()
            ref=_ref(view,bid,block.span.start+left,block.span.start+right)
            if SECTION.fullmatch(text):
                flush();number="";title="";section=text
                add("section:"+bid+":"+str(left),"UNCLASSIFIED",[ref],parent=root["unit_id"],title=text,
                    metadata={"origin":"SECTION_BOUNDARY","heading_path":[text]})
            else:
                heading = _heading(part)
                if heading:
                    flush();number,title=heading
                current.append(ref)
        last_page=block.span.page_number
    flush()

    # Lists inherit a source-bound lead-in; the complete parent remains available.
    for parent in tuple(units):
        if parent["kind"] not in {"CLAUSE_CANDIDATE","FOOTNOTE_CANDIDATE","UNCLASSIFIED"}:
            continue
        text=parent["projection_text"]; matches=list(ITEM.finditer(text))
        if not matches or not re.search(r"以下|下列|如下|包括|条件|情形",text[:matches[0].start()]):
            continue
        numbers=[int(m.group(1)) for m in matches]
        if numbers!=list(range(1,len(numbers)+1)):
            risks.append("LIST_SEQUENCE_UNRESOLVED")
            continue
        if "LIST_ITEM_COUNT_MISMATCH" in parent["risks"]:continue
        intro_text=text[:matches[0].start()]
        logic=_list_logic(intro_text)
        ordered=parent.get("projection_spans",parent["source_spans"])
        group=add(parent["key"]+":list","LIST_CANDIDATE",parent["source_spans"],projection_spans=ordered,parent=parent["unit_id"],
            title=parent["title"],metadata={"list_logic":logic,"item_count":len(numbers),"declared_count":_declared_count(intro_text),"heading_path":parent["heading_path"]},
            unit_risks=["LIST_COUNT_NOT_DECLARED"] if _declared_count(intro_text) is None else [])
        intro_refs=_slice(view,ordered,0,matches[0].start())
        intro=add(group["key"]+":intro","LIST_INTRO",intro_refs,projection_spans=intro_refs,
            parent=group["unit_id"],title=parent["title"],metadata={"heading_path":parent["heading_path"]})
        for index,match in enumerate(matches):
            end=matches[index+1].start() if index+1<len(matches) else len(text)
            item_refs=_slice(view,ordered,match.start(),end)
            add(group["key"]+":item:"+str(index+1),"LIST_ITEM",item_refs,projection_spans=item_refs,
                parent=group["unit_id"],title=parent["title"],shared=[intro["unit_id"]],
                metadata={"ordinal":index+1,"list_logic":logic,"heading_path":parent["heading_path"]})

    def edge(src,dst,relation,source_refs,target_ref=None):
        value={"from_unit_id":src,"to_unit_id":dst,"relation":relation,"source_spans":source_refs,
               "target_ref":target_ref,"state":"SOURCE_BOUND_CANDIDATE" if dst else "UNRESOLVED",**FLAGS}
        value["dependency_id"]=_identity("business-edge-v1:",value);edges.append(value)
    owners={}
    for unit in units:
        if unit["kind"] in {"CLAUSE_CANDIDATE","QUESTION_CANDIDATE","UNCLASSIFIED","TABLE_CANDIDATE","FOOTNOTE_CANDIDATE"}:
            for bid in unit["block_ids"]: owners.setdefault(bid,[]).append(unit)
    for item in layout.get("references",[]):
        _check(item["from_block_id"] in blocks,"REFERENCE_SOURCE_SCOPE")
        source_ids=item.get("source_block_ids",[item["from_block_id"]])
        _check(source_ids and all(b in blocks for b in source_ids),"REFERENCE_SOURCE_SCOPE")
        refs=_normal_refs(view,item.get("source_spans",[_ref(view,b) for b in source_ids]))
        _check(all(r["block_id"] in source_ids and r["page_number"] in selected for r in refs),"REFERENCE_SOURCE_SCOPE")
        target=keys.get(item["to_key"])
        for owner in owners.get(item["from_block_id"],[]):
            contained=all(any(p["block_id"]==r["block_id"] and p["start"]<=r["start"] and p["end"]>=r["end"] for p in owner["source_spans"]) for r in refs)
            if contained:
                edge(owner["unit_id"],target["unit_id"] if target else None,item.get("reason","LAYOUT_REFERENCE"),refs,item["to_key"])
            elif not item.get("source_spans"):
                edge(owner["unit_id"],None,"REFERENCE_OWNER_UNRESOLVED",owner["source_spans"],item["to_key"])
                risks.append("REFERENCE_OWNER_UNRESOLVED")
    by_number={}
    for unit in units:
        if unit["kind"]=="CLAUSE_CANDIDATE": by_number.setdefault(unit["metadata"].get("clause_no"),[]).append(unit)
    for unit in (u for u in units if u["kind"] in {"CLAUSE_CANDIDATE","QUESTION_CANDIDATE","UNCLASSIFIED","FOOTNOTE_CANDIDATE"}):
        for ref in REFERENCE.finditer(unit["projection_text"]):
            candidates=by_number.get(ref.group(1),[])
            target=candidates[0] if len(candidates)==1 else None
            if target is unit:continue
            edge(unit["unit_id"],target["unit_id"] if target else None,"EXPLICIT_REFERENCE_CANDIDATE",unit["source_spans"],ref.group(1))
            if target is None:risks.append("UNRESOLVED_REFERENCE")
        for obj in layout.get("objects",[]):
            if obj["kind"]!="TABLE":continue
            target=keys.get(obj["key"])
            if target is None:continue
            overlap=set(unit["block_ids"])&set(target["block_ids"])
            mentioned=[m.group() for m in re.finditer(r"附录\s*\d+",unit["projection_text"])]
            appendix_match=any(re.sub(r"\s+","",m) in re.sub(r"\s+","",obj["title"]) for m in mentioned)
            if overlap or appendix_match:
                edge(unit["unit_id"],target["unit_id"],"CONTAINS_LAYOUT_SOURCE" if overlap else "APPENDIX_REFERENCE_CANDIDATE",unit["source_spans"])
    # Identical owner entries are de-duplicated after gathering source-supported links.
    edges=list({e["dependency_id"]:e for e in edges}.values())
    covered=_normal_refs(view,[s for u in units if u["kind"]!="DOCUMENT" for s in u["source_spans"]])
    expected=sum(len(b.text) for b in blocks.values())
    actual=sum(r["end"]-r["start"] for r in covered)
    result={"schema_version":"business-chunks-v1","source_id":view.source_id,"raw_pdf_sha256":view.raw_pdf_sha256,
            "text_revision_id":view.text_revision_id,"product_id":product_id,"selected_pages":selected,
            "units":units,"dependencies":edges,"risks":sorted(set(risks+[r for u in units for r in u["risks"]])),
            "layout_risk_details":layout.get("risks",[]),
            "structure_engine":structure.get("engine") if structure else "LEGACY_BLOCK_HEURISTIC",
            "coverage":{"all_source_characters_mapped":actual==expected,"expected_source_characters":expected,
                        "mapped_source_characters":actual,"source_blocks":len(blocks)},**FLAGS}
    result["document_id"]=_identity("business-document-v1:",result)
    validate_business_document(view,result,product_id=product_id)
    return result


def validate_business_document(view, document, *, product_id):
    _check(document["product_id"]==product_id,"PRODUCT_SCOPE_MISMATCH")
    _check(document["source_id"]==view.source_id and document["text_revision_id"]==view.text_revision_id and document["raw_pdf_sha256"]==view.raw_pdf_sha256,"SOURCE_SCOPE_MISMATCH")
    _check(all(document.get(k)==v for k,v in FLAGS.items()),"BUSINESS_AUTHORITY_ESCALATION")
    _check(document["document_id"]==_identity("business-document-v1:",{k:v for k,v in document.items() if k!="document_id"}),"BUSINESS_DOCUMENT_HASH_MISMATCH")
    by_id={u["unit_id"]:u for u in document["units"]}
    _check(len(by_id)==len(document["units"]),"DUPLICATE_BUSINESS_UNIT")
    for unit in by_id.values():
        _check(unit["kind"] in KINDS and unit["product_id"]==product_id and unit["text_revision_id"]==view.text_revision_id,"UNIT_SCOPE_MISMATCH")
        _check(all(unit.get(k)==v for k,v in FLAGS.items()),"BUSINESS_AUTHORITY_ESCALATION")
        _check(unit["unit_id"]==_identity("business-unit-v1:",{k:v for k,v in unit.items() if k!="unit_id"}),"BUSINESS_UNIT_HASH_MISMATCH")
        _check(unit["source_spans"]==_normal_refs(view,unit["source_spans"]),"UNREGISTERED_SOURCE_SPANS")
        _check(all(s["page_number"] in document["selected_pages"] for s in unit["source_spans"]),"SOURCE_PAGE_SCOPE")
        ordered=unit.get("projection_spans",unit["source_spans"])
        _validate_projection(view,unit["source_spans"],_ordered_refs(view,ordered))
        _check(unit["projection_text"]==_projection(view,ordered),"BUSINESS_PROJECTION_MISMATCH")
        seen=set();current=unit
        while current["parent_id"] is not None:
            _check(current["unit_id"] not in seen,"BUSINESS_PARENT_CYCLE");seen.add(current["unit_id"])
            _check(current["parent_id"] in by_id,"MISSING_BUSINESS_PARENT")
            parent=by_id[current["parent_id"]]
            _check(set(current["block_ids"]).issubset(parent["block_ids"]),"CHILD_OUTSIDE_PARENT")
            _check(all(any(p["block_id"]==s["block_id"] and p["start"]<=s["start"] and p["end"]>=s["end"]
                           for p in parent["source_spans"]) for s in current["source_spans"]),"CHILD_SPAN_OUTSIDE_PARENT")
            current=parent
        _check(current["kind"]=="DOCUMENT","INVALID_BUSINESS_ROOT")
        _check(all(ref in by_id for ref in unit["shared_context_refs"]),"MISSING_SHARED_CONTEXT")
    for edge in document["dependencies"]:
        _check(edge["from_unit_id"] in by_id and (edge["to_unit_id"] is None or edge["to_unit_id"] in by_id),"MISSING_DEPENDENCY_TARGET")
        _check(edge["source_spans"]==_normal_refs(view,edge["source_spans"]),"DEPENDENCY_SOURCE_INVALID")
        _check(all(r["page_number"] in document["selected_pages"] and any(p["block_id"]==r["block_id"] and p["start"]<=r["start"] and p["end"]>=r["end"]
                   for p in by_id[edge["from_unit_id"]]["source_spans"]) for r in edge["source_spans"]),"DEPENDENCY_SOURCE_SCOPE")
        _check(edge["dependency_id"]==_identity("business-edge-v1:",{k:v for k,v in edge.items() if k!="dependency_id"}),"DEPENDENCY_HASH_MISMATCH")
    return document


def build_business_windows(view, document, *, count_tokens, tokenizer_id, max_tokens=256, overlap_chars=32):
    validate_business_document(view,document,product_id=document["product_id"])
    _check(type(max_tokens) is int and max_tokens>0 and type(overlap_chars) is int and overlap_chars>=0,"WINDOW_BUDGET_INVALID")
    _check(isinstance(tokenizer_id,str) and tokenizer_id.strip(),"TOKENIZER_ID_REQUIRED")
    _check(callable(count_tokens),"TOKEN_COUNTER_REQUIRED")
    def count(text):
        value=count_tokens(text)
        _check(type(value) is int and value>=0,"TOKEN_COUNTER_INVALID")
        return value
    parents={u["parent_id"] for u in document["units"] if u["kind"]=="LIST_CANDIDATE"}
    windows=[]
    for unit in document["units"]:
        if unit["kind"] in {"DOCUMENT","LIST_CANDIDATE"} or unit["unit_id"] in parents or not unit["projection_text"]:continue
        prefix=" | ".join(dict.fromkeys([document["product_id"],*unit["heading_path"],unit["title"]]))+"\n"
        if unit["kind"]=="TABLE_ROW_CANDIDATE":
            by_id={u["unit_id"]:u for u in document["units"]}
            prefix+="\n".join(by_id[uid]["projection_text"] for uid in unit["shared_context_refs"])+"\n"
        _check(count(prefix)<max_tokens,"PREFIX_BUDGET_EXCEEDED")
        text=unit["projection_text"];start=0
        while start<len(text):
            end=len(text)
            while end>start:
                refs=_slice(view,unit.get("projection_spans",unit["source_spans"]),start,end)
                body=_projection(view,refs)
                if body and count(prefix+body)<=max_tokens:break
                end=start+(end-start)//2
            _check(end>start,"WINDOW_TOKEN_BUDGET_IMPOSSIBLE")
            boundary="WHOLE_UNIT" if start==0 and end==len(text) else "FRAGMENT"
            if end<len(text):
                candidates=[start+m.end() for m in re.finditer(r"[。；;！？\n]",text[start:end])]
                safe=next((p for p in reversed(candidates) if p>start+(end-start)//2),None)
                if safe:
                    end=safe;refs=_slice(view,unit.get("projection_spans",unit["source_spans"]),start,end);body=_projection(view,refs);boundary="SENTENCE_BOUNDARY_CANDIDATE"
            start,end=_reading_bounds(unit.get("projection_spans",unit["source_spans"]),start,end)
            _check(text[start:end]==body,"WINDOW_READING_OFFSET_MISMATCH")
            window={"unit_id":unit["unit_id"],"parent_unit_id":unit["parent_id"] if unit["kind"] in {"LIST_ITEM","LIST_INTRO"} else unit["unit_id"],
                    "product_id":document["product_id"],"text_revision_id":view.text_revision_id,"source_spans":refs,
                    "body_text":body,"retrieval_prefix":prefix,"search_text":prefix+body,"token_count":count(prefix+body),
                    "tokenizer_id":tokenizer_id,"max_tokens":max_tokens,"boundary":boundary,
                    "shared_context_refs":unit["shared_context_refs"],"start":start,"end":end,
                    "retrieval_eligible":unit.get("retrieval_eligible",True),"unit_kind":unit["kind"],**FLAGS}
            window["window_id"]=_identity("business-window-v1:",window);windows.append(window)
            if end==len(text):break
            start=end-min(overlap_chars,end-start-1)
    return windows


def collect_business_evidence(view, document, unit_ids, *, product_id, max_chars=16000):
    validate_business_document(view,document,product_id=product_id)
    _check(type(max_chars) is int and max_chars>0,"CONTEXT_BUDGET_INVALID")
    by_id={u["unit_id"]:u for u in document["units"]};pending=list(unit_ids);selected={};missing=[]
    while pending:
        uid=pending.pop()
        _check(uid in by_id,"UNKNOWN_BUSINESS_UNIT")
        if uid in selected:continue
        unit=by_id[uid]
        if unit["kind"]=="DOCUMENT":continue
        selected[uid]=unit
        if unit["parent_id"]:pending.append(unit["parent_id"])
        pending.extend(unit["shared_context_refs"])
        for edge in document["dependencies"]:
            if edge["from_unit_id"]==uid:
                if edge["to_unit_id"]:pending.append(edge["to_unit_id"])
                else:missing.append(edge["target_ref"])
    refs=_normal_refs(view,[s for u in selected.values() for s in u["source_spans"]])
    parts=[{**r,"text":view.canonical_text[r["start"]:r["end"]],"source_id":view.source_id,"raw_pdf_sha256":view.raw_pdf_sha256} for r in refs]
    # Maximal selected units carry logical reading order; child/list spans remain
    # available above without repeating the parent's text in model context.
    maximal=[u for u in selected.values() if not any(
        other["unit_id"]!=u["unit_id"] and
        all(any(p["block_id"]==s["block_id"] and p["start"]<=s["start"] and p["end"]>=s["end"]
                for p in other["source_spans"]) for s in u["source_spans"]) and
        (sum(p["end"]-p["start"] for p in other["source_spans"])>sum(p["end"]-p["start"] for p in u["source_spans"]) or
         other["unit_id"]<u["unit_id"])
        for other in selected.values())]
    reading_units=[{"unit_id":u["unit_id"],"title":u["title"],"kind":u["kind"],
                    "projection_text":u["projection_text"],"source_spans":u.get("projection_spans",u["source_spans"])}
                   for u in sorted(maximal,key=lambda u:min((r["start"] for r in u["source_spans"]),default=0))]
    reading_parts=[{**r,"text":view.canonical_text[r["start"]:r["end"]],"source_id":view.source_id,
                    "raw_pdf_sha256":view.raw_pdf_sha256} for u in reading_units for r in u["source_spans"]]
    size=sum(len(p["text"]) for p in parts)
    context_size=sum(len(p["text"]) for p in reading_parts)
    risks=sorted(set(r for u in selected.values() for r in u["risks"]))
    if missing:risks.append("UNRESOLVED_REFERENCE")
    if context_size>max_chars:risks.append("CONTEXT_BUDGET_EXCEEDED")
    uncertain=any(re.search(r"UNRESOLVED|REVIEW_REQUIRED|AMBIGUOUS|UNCONFIRMED|INCOMPLETE|MISMATCH",r) for r in risks)
    usable=context_size<=max_chars and not missing and not uncertain
    unresolved_risks=any("UNRESOLVED" in r for r in risks)
    return {"status":"READY_FOR_SOURCE_REVIEW" if usable else "HUMAN_REVIEW","unit_ids":sorted(selected),
            "source_parts":parts,"reading_units":reading_units,
            "generation_context":reading_parts if usable else [],"source_characters":size,"generation_context_characters":context_size,
            "registered_dependencies_resolved":not missing and not unresolved_risks,"missing_references":sorted(set(missing)),
            "risks":risks,"requires_human_review":True,**FLAGS}
