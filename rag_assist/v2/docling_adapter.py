"""Optional Docling evidence; canonical quotations remain owned by the source view.

All public page numbers are physical, one-based. Bboxes use PDF points from the
top-left, not unit-square coordinates. Conversion runs in a private interpreter.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import fitz

DOCLING_VERSION = "2.48.0"
SCHEMA_VERSION = 1
LAYOUT_MODEL_REVISION = "b5b4bd59ad2b69aab715e9b1f1dfd74394c45fd4"
LAYOUT_WEIGHT_SHA256 = "31e60b4709571b613bc8736a9c982fb550d8d7a1809160a68a8282af60c8910b"
OPTIONS = {"do_ocr": False, "do_table_structure": False, "device": "cpu",
           "layout_model": "ds4sd/docling-layout-old", "layout_revision": LAYOUT_MODEL_REVISION,
           "layout_weight_sha256": LAYOUT_WEIGHT_SHA256}
ROOT = Path(__file__).resolve().parents[2]


def _source(pdf_path, pages=None):
    path = Path(pdf_path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with fitz.open(path) as doc:
        sizes = {n + 1: (page.rect.width, page.rect.height) for n, page in enumerate(doc)}
    selected = list(sizes) if pages is None else list(pages)
    if not selected or any(type(p) is not int or p not in sizes for p in selected):
        raise ValueError("pages must contain valid one-based physical PDF page numbers")
    return path, digest, sizes, sorted(set(selected))


def _identity(digest, pages):
    data = {"raw_pdf_sha256": digest, "pages": pages, "options": OPTIONS,
            "version": DOCLING_VERSION, "schema_version": SCHEMA_VERSION}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def cache_identity(pdf_path, *, pages=None):
    _, digest, _, selected = _source(pdf_path, pages)
    return _identity(digest, selected)


def _base(digest, selected):
    return {"raw_pdf_sha256": digest, "engine": "docling", "version": DOCLING_VERSION,
            "schema_version": SCHEMA_VERSION, "pages": selected,
            "coordinate_system": "TOPLEFT_PDF_POINTS", "options": dict(OPTIONS),
            "cache_identity": _identity(digest, selected), "elements": [],
            "risks": ["DOCLING_LABELS_REQUIRE_SOURCE_ALIGNMENT", "OCR_DISABLED",
                      "TABLE_STRUCTURE_DISABLED"]}


def _bbox(values, size):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("Unsupported bounding box")
    coords = [float(v) for v in values]
    l, t, r, b = coords
    width, height = size
    if not all(math.isfinite(v) for v in coords) or not (0 <= l <= r <= width + .5 and 0 <= t <= b <= height + .5):
        raise ValueError("Bounding box outside source page")
    return coords


def normalize_docling_document(document, pdf_path, *, pages=None, page_map=None):
    """Normalize a Docling export; page_map maps sliced pages to original pages.

    Without an explicit map, page numbers must already refer to the original PDF.
    A caller using sliced PDFs is responsible for providing the verified map.
    """
    _, digest, sizes, selected = _source(pdf_path, pages)
    result = _base(digest, selected)
    if not isinstance(document, dict):
        result["risks"].append("DOCLING_UNSUPPORTED_EVIDENCE")
        return result
    for collection in ("texts", "tables", "pictures"):
        items = document.get(collection, [])
        if not isinstance(items, list):
            result["risks"].append("DOCLING_UNSUPPORTED_EVIDENCE")
            continue
        for item in items:
            try:
                label = item["label"]
                text = item.get("text", "")
                if collection == "tables" and not text:
                    text = "\n".join(c["text"] for c in item.get("data", {}).get("table_cells", []))
                if not isinstance(label, str) or not isinstance(text, str) or not item.get("prov"):
                    raise ValueError("Unsupported item")
                for prov in item["prov"]:
                    local_page = prov["page_no"]
                    page = page_map[local_page] if page_map is not None else local_page
                    if type(page) is not int or page not in sizes:
                        raise ValueError("Unknown physical source page")
                    if page not in selected:
                        continue
                    box = prov["bbox"]
                    origin = box.get("coord_origin")
                    coords = [box[k] for k in ("l", "t", "r", "b")]
                    if origin == "BOTTOMLEFT":
                        height = sizes[page][1]
                        coords = [coords[0], height - coords[1], coords[2], height - coords[3]]
                    elif origin != "TOPLEFT":
                        raise ValueError("Unsupported coordinate origin")
                    item_text = text
                    if collection == "texts" and "charspan" in prov:
                        start, end = prov["charspan"]
                        if not (type(start) is int and type(end) is int and 0 <= start <= end <= len(text)):
                            raise ValueError("Invalid text provenance")
                        item_text = text[start:end]
                    element = {"page_number": page, "bbox": _bbox(coords, sizes[page]),
                               "label": label, "text": item_text}
                    if type(item.get("level")) is int and item["level"] >= 0:
                        element["level"] = item["level"]
                    result["elements"].append(element)
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                result["risks"].append("DOCLING_UNSUPPORTED_EVIDENCE")
    result["risks"] = sorted(set(result["risks"]))
    return result


def _valid_cache(data, expected, sizes):
    try:
        for key in ("raw_pdf_sha256", "version", "schema_version", "pages", "coordinate_system", "options", "cache_identity"):
            if data[key] != expected[key]:
                return False
        if data["engine"] != "docling" or not isinstance(data["risks"], list) or not all(isinstance(r, str) for r in data["risks"]):
            return False
        for element in data["elements"]:
            page = element["page_number"]
            if page not in expected["pages"] or not isinstance(element["text"], str) or not isinstance(element["label"], str):
                return False
            _bbox(element["bbox"], sizes[page])
        return isinstance(data["elements"], list)
    except (KeyError, ValueError, TypeError):
        return False


def find_cached_docling_evidence(pdf_path, *, cache_dir, pages=None):
    """Read validated cache only, including explicitly marked partial page scope.

    Returns None when unavailable. It never starts conversion or modifies files.
    A subset cache cannot imply that unconverted pages have Docling evidence.
    """
    _, digest, sizes, selected = _source(pdf_path, pages)
    directory = Path(cache_dir)
    matches = []
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            scoped = data["pages"]
            if not isinstance(scoped, list) or not scoped or not set(scoped).issubset(selected):
                continue
            if any(type(page) is not int for page in scoped):
                continue
            if _valid_cache(data, _base(digest, scoped), sizes):
                matches.append(data)
        except (OSError, KeyError, TypeError, ValueError):
            continue
    if not matches:
        return None
    result = max(matches, key=lambda data: len(data["pages"]))
    if result["pages"] != selected:
        result["risks"] = sorted(set(result["risks"] + ["DOCLING_PARTIAL_PAGE_COVERAGE"]))
    return result


def load_docling_evidence(pdf_path, *, cache_dir, pages=None):
    """Read validated evidence or perform an isolated text-PDF conversion.

    Failure is explicit and returns no evidence. It never disguises native PDF
    extraction as Docling, and callers may retain their native structural path.
    """
    path, digest, sizes, selected = _source(pdf_path, pages)
    result = _base(digest, selected)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / (result["cache_identity"] + ".json")
    if cached.exists():
        try:
            data = json.loads(cached.read_text(encoding="utf-8"))
            if _valid_cache(data, result, sizes):
                return data
        except (OSError, ValueError):
            pass
        result["risks"].append("DOCLING_CACHE_REJECTED")
    interpreter = Path(os.environ.get("DOCLING_STRUCTURE_PYTHON", str(ROOT / "runtime/structure/venv/Scripts/python.exe")))
    result["engine"] = "docling_unavailable"
    if not interpreter.is_file():
        result["risks"].append("DOCLING_RUNTIME_UNAVAILABLE")
        return result
    try:
        with tempfile.TemporaryDirectory(prefix="docling-", dir=directory) as temp:
            output = Path(temp) / "evidence.json"
            command = [str(interpreter), str(ROOT / "tools/run_docling_structure.py"),
                       str(path), "--worker", "--pages", ",".join(map(str, selected)), "--output", str(output)]
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=600)
            if process.returncode:
                raise RuntimeError(process.stderr[-4000:] or process.stdout[-4000:])
            data = json.loads(output.read_text(encoding="utf-8"))
            if not _valid_cache(data, _base(digest, selected), sizes):
                raise ValueError("Worker returned evidence with invalid source/cache identity")
            data["risks"] = sorted(set(data["risks"] + result["risks"]))
            cached.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        result["risks"].append("DOCLING_CONVERSION_FAILED")
        result["error"] = str(error)
        return result
