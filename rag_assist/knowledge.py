"""Build a versioned SQLite evidence index; never alter the legacy project."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from . import ROOT
from chunker import DocMeta, run
from chunker.pdf_parser import parse_pdf


def digest(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def source_specs():
    medical_path = ROOT / "claimMate/chunker/samples/clause_medical.txt"
    liability_path = ROOT / "claimMate/samples_pdf/iachina_示范条款_2020.pdf"
    medical = medical_path.read_text(encoding="utf-8")
    liability, _, _ = parse_pdf(str(liability_path))
    parts=re.split(r"\[p\.(\d+)\]",medical)
    medical_pages=[{"page":parts[i],"text":parts[i+1]} for i in range(1,len(parts),2)]
    import pymupdf
    with pymupdf.open(liability_path) as pdf:
        liability_pages=[{"page":str(i+1),"text":page.get_text("text",sort=True)} for i,page in enumerate(pdf)]
    return {
        "medical": (medical, medical_path, DocMeta("DOC_MEDICAL_DEMO", "MEDICAL_DEMO", "医疗险合成演示条款", "2025", "2025-01-01"), "SYNTHETIC",medical_pages),
        "liability": (liability, liability_path, DocMeta("DOC_IACHINA_2020", "PRD_MAT_001", "重点新材料示范条款", "2020修订版", "2020-11-27"), "PUBLIC_SAMPLE",liability_pages),
    }


def article_of(local_id):
    m = re.match(r"(?:C_|P_ART_)([一二三四五六七八九十百]+)", local_id)
    return m.group(1) if m else ""


def normalized_page(text):
    # Only whitespace and standalone page-footer numerals are removed. No
    # medical terms, negation, punctuation, numbers in prose, or units change.
    return re.sub(r"\s+","","\n".join(line for line in text.splitlines() if not re.fullmatch(r"\s*\d+\s*",line)))


def source_spans(text, pages):
    """Find every literal source line in independently extracted physical pages.

    A line merged across PDF pages gets one span per page, rather than assigning
    the entire sentence to the page before the merge. Coordinates are normalized
    physical-page character offsets, explicitly not PDF bounding boxes.
    """
    parts=[normalized_page(p["text"]) for p in pages]
    combined="".join(parts)
    boundaries=[]
    start=0
    for page,part in zip(pages,parts):
        boundaries.append((page["page"],start,start+len(part)))
        start+=len(part)
    spans=[]
    for line in text.splitlines():
        value=re.sub(r"\s+","",line)
        if not value:
            continue
        pos=combined.find(value)
        if pos<0:
            spans.append({"location_kind":"UNMAPPED_SOURCE_LINE","text":line,"page":""})
            continue
        end=pos+len(value)
        for page,a,b in boundaries:
            if max(a,pos)<min(b,end):
                left,right=max(a,pos),min(b,end)
                spans.append({"location_kind":"PHYSICAL_PAGE_NORMALIZED_CHAR","page":page,
                              "start_char":left-a,"end_char":right-a,"quote_piece":combined[left:right]})
    return spans


def profile_edges(corpus, chunks):
    """Explicit source-article scope policies, independent of evaluation labels.

    These local sample profiles provide related context for checking applicability;
    they do not authorize medical underwriting conclusions from contract clauses.
    """
    by_id = {c.chunk_id: c for c in chunks}
    edges = set()
    def link(src, dst, role):
        if src in by_id and dst in by_id and src != dst:
            edges.add((src, dst, role))
    if corpus == "medical":
        exceptions = [c.chunk_id for c in chunks if article_of(c.chunk_id) == "七"]
        for src in ["C_五_1", "C_五_2", "C_六"]:
            for ex in exceptions:
                link(src, ex, "EXCLUSION")
            for dst in ["C_四", "C_十四", "C_十五"]:
                link(src, dst, "CONDITION" if dst == "C_四" else "DEFINITION")
            for dst in ["C_二","C_八","C_十二","C_十三"]:
                link(src,dst,"CONDITION")
        for src in ["C_五_1", "C_五_2"]:
            link(src, "C_十", "PARAMETER")
            link(src, "C_十六", "DEFINITION")
        link("C_五_1", "C_TBL_10_1", "PARAMETER")
        for src in ["C_十一", "C_TBL_10_1"]:
            link(src, "C_十", "PARAMETER")
            link(src, "C_十七", "DEFINITION")
            link(src, "C_十八", "DEFINITION")
        link("C_十三", "C_十二", "CONDITION")
        terms = {"医院":"C_十四", "住院":"C_十五", "合理且必要":"C_十六",
                 "社会基本医疗保险":"C_十七", "进口耗材":"C_十八"}
        for c in chunks:
            if c.chunk_type != "EXCLUSION":
                for term, target in terms.items():
                    if term in c.text:
                        link(c.chunk_id, target, "DEFINITION")
    else:
        exceptions = [c.chunk_id for c in chunks if article_of(c.chunk_id) in ("七", "八") or c.chunk_id == "C_二十五"]
        for src in ["C_五", "C_六"]:
            for ex in exceptions:
                link(src, ex, "EXCLUSION")
            link(src,"C_四","DEFINITION")
            for dst in ["C_九", "C_十", "C_十一", "C_十六", "C_十七", "C_十九", "C_二十", "C_二十一", "C_二十二", "C_二十三", "C_二十四", "C_二十六", "C_二十七", "C_二十八", "C_二十九"]:
                link(src, dst, "CONDITION")
        link("C_二十六", "C_十一", "PARAMETER")
    return sorted(edges)


DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(corpus TEXT PRIMARY KEY,tenant TEXT NOT NULL,version TEXT NOT NULL,
 effective_from TEXT NOT NULL,classification TEXT NOT NULL,path TEXT NOT NULL,source_hash TEXT NOT NULL,raw_text TEXT NOT NULL,physical_pages TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS units(unit_id TEXT PRIMARY KEY,local_id TEXT NOT NULL,corpus TEXT NOT NULL REFERENCES sources(corpus),
 tenant TEXT NOT NULL,version TEXT NOT NULL,unit_type TEXT NOT NULL,role TEXT NOT NULL,source_text TEXT NOT NULL,
 retrieval_text TEXT NOT NULL,anchor TEXT NOT NULL,spans TEXT NOT NULL,parent_id TEXT NOT NULL,searchable INTEGER NOT NULL,
 content_sha256 TEXT NOT NULL,UNIQUE(corpus,version,local_id));
CREATE TABLE IF NOT EXISTS edges(src TEXT NOT NULL REFERENCES units(unit_id),dst TEXT NOT NULL REFERENCES units(unit_id),
 role TEXT NOT NULL,PRIMARY KEY(src,dst,role));
CREATE TABLE IF NOT EXISTS layout_blocks(block_id TEXT PRIMARY KEY,unit_id TEXT NOT NULL REFERENCES units(unit_id),
 corpus TEXT NOT NULL,tenant TEXT NOT NULL,version TEXT NOT NULL,data_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS graph_entities(entity_id TEXT PRIMARY KEY,corpus TEXT NOT NULL,tenant TEXT NOT NULL,
 version TEXT NOT NULL,label TEXT NOT NULL,aliases_json TEXT NOT NULL,seed_unit_id TEXT NOT NULL REFERENCES units(unit_id));
CREATE TABLE IF NOT EXISTS record_evidence(evidence_id TEXT PRIMARY KEY,tenant TEXT NOT NULL,application_id TEXT NOT NULL,payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,tenant TEXT NOT NULL,bundle_id TEXT NOT NULL,
 kind TEXT NOT NULL,request_json TEXT NOT NULL,response_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs(run_id),tenant TEXT NOT NULL,
 event_type TEXT NOT NULL,payload_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cases(case_id TEXT PRIMARY KEY,tenant TEXT NOT NULL,revision INTEGER NOT NULL,current_run TEXT);
CREATE TABLE IF NOT EXISTS reviews(event_id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs(run_id),tenant TEXT NOT NULL,
 idem_key TEXT NOT NULL,request_hash TEXT NOT NULL,payload_json TEXT NOT NULL,UNIQUE(tenant,idem_key));
"""


def build_index(db_path: Path):
    db_path = Path(db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    specs = source_specs()
    manifest = {k: {"sha256": digest(v[1].read_bytes()), "version": v[2].product_version,"product_id":v[2].product_id,
                    "classification": v[3]} for k,v in specs.items()}
    manifest.update(schema_version="3", builder_version="3", profile_version="sample-context-four-route-v1",
                    recall_code_sha256=digest((Path(__file__).parent/'recall.py').read_bytes()),
                    business_source_sha256=digest((ROOT/'claimMate/claimMate_uw.db').read_bytes()),
                    business_snapshot_as_of='2026-11-01',
                    builder_code_sha256=digest(Path(__file__).read_bytes()))
    bundle = "B_" + digest(json.dumps(manifest, sort_keys=True))[:24]
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(DDL)
        existing = db.execute("SELECT value FROM meta WHERE key='bundle_id'").fetchone()
        if existing and existing[0] != bundle:
            raise ValueError("immutable bundle differs: build a new database and switch pointer")
        if not existing:
            for corpus, (raw, file, meta, classification,pages) in specs.items():
                result = run(raw, meta, apply_context=False, test_set="medical" if corpus == "medical" else "liability")
                db.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)", (corpus,"T_008",meta.product_version,
                           meta.effective_from,classification,str(file.relative_to(ROOT)),digest(file.read_bytes()),raw,json.dumps(pages,ensure_ascii=False)))
                uid = lambda local: corpus + ":" + meta.product_version + ":" + local
                for c in result.chunks + result.parents:
                    source_text = c.text
                    if c.chunk_type == "TABLE":
                        source_text = next(p.text for p in result.parents if p.chunk_id == "P_ART_十一")
                    kind = c.chunk_type
                    if corpus == "medical" and article_of(c.chunk_id) == "五":
                        kind = "LIABILITY"
                    role = "EXCLUSION" if kind == "EXCLUSION" else "MAIN"
                    source_text = re.sub(r"^\[p\.\d+\]\s*$", "", source_text, flags=re.M).strip()
                    retrieval = c.section_path + "\n" + c.clause_name + "\n" + source_text
                    db.execute("INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        uid(c.chunk_id),c.chunk_id,corpus,"T_008",meta.product_version,kind,role,source_text,
                        retrieval,c.page_anchor,json.dumps(source_spans(source_text,pages),ensure_ascii=False),
                        uid(c.parent_chunk_id) if c.parent_chunk_id else "",int(c.is_searchable),digest(source_text)))
                db.executemany("INSERT INTO edges VALUES (?,?,?)", [(uid(a),uid(b),role) for a,b,role in profile_edges(corpus,result.chunks)])
                for c in result.chunks:
                    if c.chunk_type=="TABLE":
                        block={"headers":c.table_headers,"rows":c.table_rows,"unit":c.table_unit,
                               "footnote":c.table_footnote,"caption":c.table_caption,"source_kind":"STRUCTURED_TEXT_TABLE"}
                        db.execute("INSERT INTO layout_blocks VALUES (?,?,?,?,?,?)",(uid('LAYOUT_'+c.chunk_id),uid(c.chunk_id),corpus,'T_008',meta.product_version,json.dumps(block,ensure_ascii=False)))
                aliases={
                    "medical":{'C_五_1':['住院医疗','住院费用','住院报销'], 'C_五_2':['门急诊','门诊'],
                               'C_六':['住院津贴','住院补贴'], 'C_四':['等待期'], 'C_十':['免赔额'],
                               'C_十一':['给付比例','赔付比例'], 'C_十八':['进口耗材'], 'C_十四':['医院'],
                               'C_十五':['住院定义'], 'C_十六':['合理且必要'], 'C_十七':['社会基本医疗保险'],
                               'C_七_9':['既往症','既往疾病'], 'C_九':['续保']},
                    "liability":{'C_五':['质量缺陷','材料损坏'], 'C_六':['法律费用','诉讼费用'],
                                 'C_七_2':['战争'], 'C_七_6':['自然灾害'], 'C_七_9':['投保前已知','投保前已经知道'],
                                 'C_七_10':['存储不当'], 'C_二十九':['重复保险'], 'C_二十四':['赔偿责任认定']}
                }[corpus]
                for local,terms in aliases.items():
                    if any(c.chunk_id==local for c in result.chunks):
                        db.execute("INSERT INTO graph_entities VALUES (?,?,?,?,?,?,?)",(uid('ENTITY_'+local),corpus,'T_008',meta.product_version,terms[0],json.dumps(terms,ensure_ascii=False),uid(local)))
            db.execute("INSERT INTO meta VALUES ('bundle_id',?)", (bundle,))
            db.execute("INSERT INTO meta VALUES ('manifest',?)", (json.dumps(manifest,ensure_ascii=False),))
        count = db.execute("SELECT COUNT(*) FROM units WHERE searchable=1").fetchone()[0]
        total = db.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        unique = db.execute("SELECT COUNT(DISTINCT unit_id) FROM units").fetchone()[0]
    return {"bundle_id":bundle,"searchable_units":count,"total_units":total,"duplicate_ids":total-unique,
            "database":str(db_path),"mode":"LOCAL_DEMO","sources":manifest}
