"""OCR support for scanned PDFs: tesseract via pytesseract + PyMuPDF rendering.

The tesseract/pymupdf imports are lazy: the module imports cleanly without
the [ocr] dependency group installed, so the whole app keeps working without
OCR. Cache is a JSON file next to the manifest (persist_dir), keyed by the
full sha256 of the file — re-ingest of an unchanged file never re-OCRs.
"""
import hashlib
import json
import tempfile
from importlib import import_module
from pathlib import Path

from .. import vector_store
from ..config import RAG_OCR_LANGS

CACHE_FILE = "ocr_cache.json"
CACHE_KEY = "ocr_cache_v1"

# Lazy module handles: set by _load_engine(); None = engine not installed.
fitz = None
pytesseract = None


def _load_engine() -> bool:
    """Import the OCR stack; False (and keep handles None) when unavailable."""
    global fitz, pytesseract
    if fitz is not None and pytesseract is not None:
        return True
    try:
        fitz = import_module("fitz")
        pytesseract = import_module("pytesseract")
        pytesseract.get_tesseract_version()  # raises when the binary is missing
        return True
    except Exception:
        fitz = None
        pytesseract = None
        return False


def ocr_available() -> bool:
    return _load_engine()


def ocr_page(path: str, page_index: int, langs: str) -> str:
    """Render one PDF page to PNG (300 dpi) and OCR it with tesseract."""
    _load_engine()
    doc = fitz.open(path)
    try:
        pix = doc[page_index].get_pixmap(dpi=300)
        png = pix.tobytes("png")
    finally:
        doc.close()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png)
        tmp_path = tmp.name
    try:
        return (pytesseract.image_to_string(tmp_path, lang=langs) or "").strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _cache_path() -> Path:
    return Path(vector_store.persist_dir()) / CACHE_FILE


def _load_cache() -> dict:
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("key") == CACHE_KEY else {}
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict):
    _cache_path().parent.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(json.dumps(cache), encoding="utf-8")


def ocr_pages(path: str, page_indices: list[int], langs: str) -> dict[int, str]:
    """OCR the given pages, reusing cached text for unchanged files.

    Cache key is the full file sha256: a changed file gets a new key, an
    unchanged re-ingest is a pure cache hit. `langs` is part of the entry,
    so a changed language recomputes. Only pages present in `page_indices`
    are OCR'd; the returned dict maps page index -> text.
    """
    file_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cache = _load_cache()
    files = cache.get("files") or {}
    entry = files.get(file_sha) or {}
    if entry.get("langs") != langs:
        entry = {}
    pages = {int(k): v for k, v in (entry.get("pages") or {}).items()}
    for i in page_indices:
        if i not in pages:
            pages[i] = ocr_page(path, i, langs)
    files[file_sha] = {"langs": langs, "pages": {str(k): v for k, v in pages.items()}}
    cache["key"] = CACHE_KEY
    cache["files"] = files
    _save_cache(cache)
    return pages
