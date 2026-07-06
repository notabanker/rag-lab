from pypdf import PdfReader

from .base import ParseResult

def parse_pdf(path: str) -> ParseResult:
    reader = PdfReader(path)
    pages = []
    section_chars = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        section_chars.append(len(text.strip()))
        pages.append(f"\n\n--- Page {i+1} ---\n\n{text}")
    return ParseResult(text="".join(pages), section_unit="page", section_chars=section_chars)
