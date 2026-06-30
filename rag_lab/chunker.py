import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start: int
    end: int


def chunk_fixed(text: str, size: int = 512, overlap: int = 64) -> list[Chunk]:
    if size <= overlap:
        raise ValueError(f"size ({size}) must be greater than overlap ({overlap})")
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append(Chunk(text=text[i:end], start=i, end=end))
        if end == n:
            break
        i += size - overlap
    return chunks


def chunk_sentence(text: str, target_size: int = 512, overlap: int = 1) -> list[Chunk]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current: list[str] = []
    current_len = 0
    start = 0
    pos = 0
    for s in sentences:
        sep_len = 1 if current else 0
        if current_len + len(s) + sep_len > target_size and current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(text=chunk_text, start=start, end=start + len(chunk_text)))
            current = current[-overlap:] if overlap else []
            current_len = sum(len(x) for x in current) + max(0, len(current) - 1)
            start = pos - current_len
        current.append(s)
        current_len += len(s) + 1
        pos += len(s) + 1
    if current:
        chunk_text = " ".join(current)
        chunks.append(Chunk(text=chunk_text, start=start, end=start + len(chunk_text)))
    return chunks


def chunk_document(text: str) -> list[Chunk]:
    if not text.strip():
        return []
    stripped = text.strip()
    return [Chunk(text=stripped, start=text.index(stripped), end=text.index(stripped) + len(stripped))]


def chunk_paragraph(text: str, max_chars: int = 1000, overlap_paragraphs: int = 1) -> list[Chunk]:
    if not text.strip():
        return []
    chunks: list[Chunk] = []
    buffer: list[tuple[str, int]] = []
    buffer_chars = 0
    chunk_start = 0
    cursor = 0

    for m in re.finditer(r'[^\n]+(?:\n[^\n]+)*', text):
        para_text = m.group()
        para_start = m.start()
        para_len = len(para_text)
        cursor = para_start + para_len

        sep = 2 if buffer else 0
        if buffer and buffer_chars + len(para_text) + sep > max_chars:
            chunk_text = "\n\n".join(p[0] for p in buffer)
            chunks.append(Chunk(text=chunk_text, start=chunk_start, end=chunk_start + len(chunk_text)))
            if overlap_paragraphs > 0:
                buffer = buffer[-overlap_paragraphs:]
            else:
                buffer = []
            buffer_chars = sum(len(p[0]) + 2 for p in buffer) - 2 if buffer else 0
            chunk_start = buffer[0][1] if buffer else cursor

        buffer.append((para_text, para_start))
        buffer_chars += len(para_text) + sep

    if buffer:
        chunk_text = "\n\n".join(p[0] for p in buffer)
        chunks.append(Chunk(text=chunk_text, start=chunk_start, end=chunk_start + len(chunk_text)))
    return chunks


STRATEGIES = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "document": chunk_document,
    "paragraph": chunk_paragraph,
}


def chunk(text: str, strategy: str = "sentence", **kwargs) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Use one of {list(STRATEGIES)}")
    return STRATEGIES[strategy](text, **kwargs)
