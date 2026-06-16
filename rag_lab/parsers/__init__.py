from .pdf import parse_pdf
from .epub import parse_epub
from .markdown import parse_markdown

PARSERS = {
    ".pdf": parse_pdf,
    ".epub": parse_epub,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
}

def pick_parser(path: str):
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return PARSERS[ext]
