"""Persistent physical quotes, with no semantic or business approval.

The lexical baseline indexes English words and Chinese character bigrams with
SQLite FTS5. It is not production semantic recall. Full page parents and their
immediate neighbours are retained; no context is silently shortened.
"""
from contextlib import closing
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile

from .public_sources import EXTRACTOR


FORMAT = "public-physical-reference-fts5-v1"
FLAGS = dict(answer_ceiling="SOURCE_ONLY", semantic_complete=False,
             core_signable=False, production_readiness="NO_GO",
             unresolved_dependencies="NOT_ASSESSED")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _terms(text):
    # ponytail: lexical bigrams only; semantic recall needs a separately evaluated retriever.
    tokens = []
    for word in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff\U00020000-\U0003134f]+", text.casefold()):
        if re.fullmatch(r"[a-z0-9_]+", word):
            tokens.append(word)
        else:
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return sorted(set(tokens))


def _require(condition):
    if not condition:
        raise ValueError("SOURCE_INTEGRITY")


def _reference_state(value):
    expected = {**FLAGS, "data_class": "PUBLIC_REFERENCE", "publication_state": "REFERENCE_ONLY",
                "semantic_review_state": "UNREVIEWED", "business_labels": [],
                "approved": False, "business_approved": False, "core_approved": False}
    for key, state in expected.items():
        if key in value and (type(value[key]) is not type(state) or value[key] != state):
            raise ValueError("UNTRUSTED_REFERENCE_STATE")


def _bbox(box):
    _require(isinstance(box, list) and len(box) == 4)
    _require(all(type(n) in (int, float) and math.isfinite(n) for n in box))
    _require(box[2] >= box[0] and box[3] >= box[1])


def _span(item, canonical):
    _require(type(item["start"]) is int and type(item["end"]) is int)
    _require(0 <= item["start"] <= item["end"] <= len(canonical))
    _require(isinstance(item["text"], str))
    _require(canonical[item["start"]:item["end"]] == item["text"])
    _require(item["content_sha256"] == _hash(item["text"]))


def _validate(document):
    """Validate the exact output contract of the unreviewed physical extractor."""
    try:
        _reference_state(document)
        for key in ("data_class", "publication_state", "answer_ceiling", "semantic_complete",
                    "production_readiness"):
            _require(key in document)
        source_id, raw_hash = document["source_id"], document["raw_pdf_sha256"]
        _require(isinstance(source_id, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", source_id))
        _require(isinstance(raw_hash, str) and re.fullmatch(r"[a-f0-9]{64}", raw_hash))
        _require(document["extractor"] == EXTRACTOR)
        _require(type(document["raw_pdf_bytes"]) is int and document["raw_pdf_bytes"] > 0)
        canonical = document["canonical_text"]
        _require(isinstance(canonical, str) and document["canonical_sha256"] == _hash(canonical))
        pages, units = document["pages"], document["units"]
        _require(isinstance(pages, list) and isinstance(units, list))
        _require(type(document["physical_page_count"]) is int and
                 document["physical_page_count"] == len(pages) and len(pages) > 0)
        namespace = f"{source_id}:{raw_hash}:{EXTRACTOR}"
        offset, unit_ids = 0, set()
        by_page = {}
        for unit in units:
            _reference_state(unit)
            _require(unit["source_id"] == source_id and unit["raw_pdf_sha256"] == raw_hash)
            _require(type(unit["page_number"]) is int and 1 <= unit["page_number"] <= len(pages))
            parent = f"{namespace}:p{unit['page_number']}"
            _require(unit["parent_page_id"] == parent)
            _require(isinstance(unit["unit_id"], str) and
                     re.fullmatch(re.escape(parent) + r":b[1-9][0-9]*", unit["unit_id"]))
            _require(unit["unit_id"] not in unit_ids)
            unit_ids.add(unit["unit_id"])
            _require(unit["offset_basis"] == "CANONICAL_TEXT_UNICODE_CODEPOINTS")
            _require(unit["business_labels"] == [] and unit["semantic_review_state"] == "UNREVIEWED")
            _span(unit, canonical)
            _require(bool(unit["text"]))
            _bbox(unit["bbox"])
            _require(isinstance(unit["line_bboxes"], list))
            for box in unit["line_bboxes"]:
                _bbox(box)
            by_page.setdefault(unit["page_number"], []).append(unit)
        for number, page in enumerate(pages, 1):
            _reference_state(page)
            _require(type(page["page_number"]) is int and page["page_number"] == number)
            _require(page["page_id"] == f"{namespace}:p{number}")
            _span(page, canonical)
            _require(page["start"] == offset)
            for dimension in (page["width"], page["height"]):
                _require(type(dimension) in (int, float) and math.isfinite(dimension) and dimension > 0)
            _require(page["coverage_state"] in {"TEXT_EXTRACTED_UNREVIEWED", "OCR_REQUIRED",
                                                "NO_TEXT_REVIEW_REQUIRED"})
            _require(type(page["image_review_required"]) is bool and isinstance(page["image_regions"], list))
            _require(page["image_review_required"] == bool(page["image_regions"]))
            for image in page["image_regions"]:
                _require(image["state"] == "IMAGE_CONTENT_UNREVIEWED")
                _bbox(image["bbox"])
            expected_coverage = ("TEXT_EXTRACTED_UNREVIEWED" if page["text"].strip() else
                                 "OCR_REQUIRED" if page["image_regions"] else "NO_TEXT_REVIEW_REQUIRED")
            _require(page["coverage_state"] == expected_coverage)
            cursor = page["start"]
            page_units = sorted(by_page.get(number, []), key=lambda unit: unit["start"])
            for position, unit in enumerate(page_units):
                _require(unit["start"] == cursor + bool(position))
                _require(unit["end"] <= page["end"])
                cursor = unit["end"]
            _require(cursor == page["end"])
            _require("\n".join(unit["text"] for unit in page_units) == page["text"])
            offset = page["end"] + (number < len(pages))
        _require("\f".join(page["text"] for page in pages) == canonical)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("SOURCE_INTEGRITY") from error


def build_reference_index(documents, path):
    """Atomically replace an index only after all input evidence validates."""
    try:
        documents = json.loads(_json(list(documents)))
    except (TypeError, ValueError) as error:
        raise ValueError("SOURCE_INTEGRITY") from error
    _require(bool(documents))
    seen, hashes = set(), {}
    for document in documents:
        _validate(document)
        source_id = document["source_id"]
        if source_id in seen:
            raise ValueError("DUPLICATE_SOURCE")
        seen.add(source_id)
        hashes[source_id] = _hash(_json(document))
    identity = {"format": FORMAT, "source_hashes": hashes}
    index_id = _hash(_json(identity))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            connection.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE sources (source_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE pages (source_id TEXT NOT NULL, page_number INTEGER NOT NULL,
                    text TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(source_id, page_number));
                CREATE TABLE units (unit_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                    text TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX units_source ON units(source_id);
                CREATE VIRTUAL TABLE unit_fts USING fts5(terms);
            """)
            connection.execute("INSERT INTO metadata VALUES (?, ?)", ("identity", _json(identity)))
            for document in documents:
                source_id = document["source_id"]
                payload = {key: value for key, value in document.items() if key not in {"pages", "units"}}
                connection.execute("INSERT INTO sources VALUES (?, ?)", (source_id, _json(payload)))
                for page in document["pages"]:
                    connection.execute("INSERT INTO pages VALUES (?, ?, ?, ?)",
                                       (source_id, page["page_number"], page["text"], _json(page)))
                for unit in document["units"]:
                    cursor = connection.execute("INSERT INTO units VALUES (?, ?, ?, ?)",
                                                (unit["unit_id"], source_id, unit["text"], _json(unit)))
                    connection.execute("INSERT INTO unit_fts(rowid, terms) VALUES (?, ?)",
                                       (cursor.lastrowid, " ".join(_terms(unit["text"]))))
            connection.commit()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"index_id": index_id, "source_ids": sorted(seen), "format": FORMAT, **FLAGS}


class ReferenceIndex:
    def __init__(self, path):
        self.path = Path(path)

    def search(self, query, *, source_ids=None, limit=5):
        if not isinstance(source_ids, (list, tuple)) or not source_ids:
            raise ValueError("SOURCE_SELECTION_REQUIRED")
        if any(not isinstance(source, str) or not source for source in source_ids):
            raise ValueError("SOURCE_SELECTION_REQUIRED")
        selected = list(dict.fromkeys(source_ids))
        if not isinstance(query, str) or not query.strip():
            raise ValueError("QUERY_REQUIRED")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("INVALID_LIMIT")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
                # One read transaction keeps metadata, physical evidence and ranking in one snapshot.
                connection.execute("BEGIN")
                row = connection.execute("SELECT value FROM metadata WHERE key = ?", ("identity",)).fetchone()
                identity = json.loads(row[0])
                _require(identity["format"] == FORMAT)
                known = identity["source_hashes"]
                if any(source not in known for source in selected):
                    raise ValueError("UNKNOWN_SOURCE")
                documents, by_unit = {}, {}
                for source_id in selected:
                    row = connection.execute("SELECT payload FROM sources WHERE source_id = ?", (source_id,)).fetchone()
                    document = json.loads(row[0])
                    pages = connection.execute("SELECT page_number, text, payload FROM pages WHERE source_id = ? "
                                               "ORDER BY page_number", (source_id,)).fetchall()
                    units = connection.execute("SELECT rowid, unit_id, text, payload FROM units WHERE source_id = ? "
                                               "ORDER BY rowid", (source_id,)).fetchall()
                    document["pages"], document["units"] = [], []
                    for number, text, payload in pages:
                        page = json.loads(payload)
                        _require(page["page_number"] == number and page["text"] == text)
                        document["pages"].append(page)
                    for rowid, unit_id, text, payload in units:
                        unit = json.loads(payload)
                        _require(unit["unit_id"] == unit_id and unit["text"] == text)
                        fts = connection.execute("SELECT terms FROM unit_fts WHERE rowid = ?", (rowid,)).fetchone()
                        _require(fts is not None and fts[0] == " ".join(_terms(text)))
                        document["units"].append(unit)
                        by_unit[rowid] = unit
                    _require(document["source_id"] == source_id)
                    _validate(document)
                    _require(_hash(_json(document)) == known[source_id])
                    documents[source_id] = document
                terms, hits = _terms(query), []
                if terms:
                    match = " OR ".join('"' + term + '"' for term in terms)
                    placeholders = ",".join("?" for _ in selected)
                    rows = connection.execute(
                        "SELECT units.rowid FROM unit_fts JOIN units ON units.rowid = unit_fts.rowid "
                        f"WHERE unit_fts MATCH ? AND units.source_id IN ({placeholders}) "
                        "ORDER BY bm25(unit_fts), units.unit_id LIMIT ?", (match, *selected, limit)).fetchall()
                    for (rowid,) in rows:
                        unit = by_unit[rowid]
                        document = documents[unit["source_id"]]
                        number = unit["page_number"]
                        context = document["pages"][max(0, number - 2):number + 1]
                        hits.append({**unit, "canonical_sha256": document["canonical_sha256"],
                                     "page_context": document["pages"][number - 1]["text"],
                                     "context_pages": [{**page, "source_id": unit["source_id"],
                                                        "raw_pdf_sha256": document["raw_pdf_sha256"],
                                                        "canonical_sha256": document["canonical_sha256"]}
                                                       for page in context], **FLAGS})
                return {"hits": hits, "status": "SOURCE_ONLY" if hits else "NO_MATCH",
                        "index_id": _hash(_json(identity)), "source_ids": selected,
                        "format": FORMAT, "retrieval_method": "LEXICAL_WORDS_AND_CHINESE_BIGRAMS", **FLAGS}
        except ValueError as error:
            if str(error) == "UNKNOWN_SOURCE":
                raise
            raise ValueError("INDEX_INTEGRITY") from error
        except (sqlite3.Error, KeyError, TypeError, IndexError) as error:
            raise ValueError("INDEX_INTEGRITY") from error
