from pypdf import PdfReader

from .base import LOW_YIELD_PAGE_CHARS, OCR_INSTALL_CMD, ParseResult
from . import ocr as ocr_module

OCR_INSTALL_HINT = f"OCR requested but not available — install with: {OCR_INSTALL_CMD}"


def parse_pdf(path: str, ocr: str = "auto") -> ParseResult:
    """Parse a PDF. Pages whose text layer yields less than
    LOW_YIELD_PAGE_CHARS are considered scanned and — when OCR is enabled —
    are re-extracted with tesseract. Pages with a good text layer are never
    OCR'd. `ocr`: 'on' (error if engine missing), 'off', 'auto' (on iff the
    engine is installed).
    """
    reader = PdfReader(path)
    mode = ocr
    if mode == "auto":
        mode = "on" if ocr_module.ocr_available() else "off"
    if mode == "on" and not ocr_module.ocr_available():
        raise ValueError(OCR_INSTALL_HINT)

    raw = []
    low_yield = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        raw.append(text)
        if mode == "on" and len(text.strip()) < LOW_YIELD_PAGE_CHARS:
            low_yield.append(i)
    ocr_map = {}
    if low_yield:
        ocr_map = ocr_module.ocr_pages(path, low_yield, ocr_module.RAG_OCR_LANGS)

    pages = []
    section_chars = []
    for i, text in enumerate(raw):
        ocr_text = ocr_map.get(i, "").strip()
        if ocr_text:
            text = ocr_text
        section_chars.append(len(text.strip()))
        pages.append(f"\n\n--- Page {i+1} ---\n\n{text}")
    return ParseResult(text="".join(pages), section_unit="page", section_chars=section_chars)
