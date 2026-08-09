from .base import ParseResult as ParseResult
from .base import as_result as as_result
from .pdf import parse_pdf
from .epub import parse_epub
from .markdown import parse_markdown
from .docx import parse_docx
from .pptx import parse_pptx

PARSERS = {
    ".pdf": parse_pdf,
    ".epub": parse_epub,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
}

def pick_parser(path: str):
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return PARSERS[ext]

def parse_file(path: str, ocr: str = "auto"):
    """Dispatch to the right parser; only PDFs take the OCR mode."""
    parser = pick_parser(path)
    if parser is parse_pdf:
        return parse_pdf(path, ocr=ocr)
    return parser(path)
