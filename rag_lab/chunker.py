from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    start: int  # char offset in source
    end: int

def chunk_fixed(text: str, size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Fixed-size chunks. Fast but breaks mid-sentence."""
    if size <= 0:
        raise ValueError(f"chunk size must be > 0, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be >= 0 and < size ({size}), got {overlap}")
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
    """Sentence-aware chunks. Slower but preserves semantics."""
    import re
    if target_size <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")
    # Sentence boundaries as (start, end) offsets into the source text
    spans = []
    start = 0
    for m in re.finditer(r'(?<=[.!?])\s+', text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))

    chunks = []
    current: list[tuple[int, int]] = []
    current_len = 0
    for span in spans:
        s_len = span[1] - span[0]
        if current and current_len + s_len > target_size:
            c_start, c_end = current[0][0], current[-1][1]
            chunks.append(Chunk(text=text[c_start:c_end], start=c_start, end=c_end))
            # Keep last N sentences as overlap
            current = current[-overlap:] if overlap else []
            current_len = sum(e - s for s, e in current)
        current.append(span)
        current_len += s_len
    if current:
        c_start, c_end = current[0][0], current[-1][1]
        chunks.append(Chunk(text=text[c_start:c_end], start=c_start, end=c_end))
    return chunks
