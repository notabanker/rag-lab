import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def parse_epub(path: str) -> str:
    book = epub.read_epub(path)
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text:
            chapters.append(f"\n\n--- {item.get_name()} ---\n\n{text}")
    return "".join(chapters)
