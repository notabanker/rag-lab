from .base import ParseResult as ParseResult
from .base import as_result as as_result
from .pdf import parse_pdf
from .epub import parse_epub
from .markdown import parse_markdown
from .docx import parse_docx

PARSERS = {
    ".pdf": parse_pdf,
    ".epub": parse_epub,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".docx": parse_docx,
}

def pick_parser(path: str):
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return PARSERS[ext]
