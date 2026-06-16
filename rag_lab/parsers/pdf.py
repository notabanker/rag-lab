from pypdf import PdfReader

def parse_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"\n\n--- Page {i+1} ---\n\n{text}")
    return "".join(pages)
