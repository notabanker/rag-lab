import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from .base import ParseResult

def parse_epub(path: str) -> ParseResult:
    book = epub.read_epub(path)
    chapters = []
    section_chars = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        section_chars.append(len(text))
        if text:
            chapters.append(f"\n\n--- {item.get_name()} ---\n\n{text}")
    return ParseResult(text="".join(chapters), section_unit="chapter", section_chars=section_chars)
