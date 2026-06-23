import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub


def _get_parser(item) -> str:
    media_type = item.get_type() if hasattr(item, 'get_type') else getattr(item, 'media_type', '')
    if isinstance(media_type, bytes):
        media_type = media_type.decode("utf-8", errors="replace")
    if "xhtml" in str(media_type).lower() or "xml" in str(media_type).lower():
        return "xml"
    return "html.parser"


def parse_epub(path: str) -> str:
    book = epub.read_epub(path)
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        parser = _get_parser(item)
        soup = BeautifulSoup(content, parser)
        text = soup.get_text(separator="\n", strip=True)
        if text:
            chapters.append(f"\n\n--- {item.get_name()} ---\n\n{text}")
    return "".join(chapters)
