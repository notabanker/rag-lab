from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    start: int  # char offset in source
    end: int

def chunk_fixed(text: str, size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Fixed-size chunks. Fast but breaks mid-sentence."""
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
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0
    start = 0
    pos = 0
    for s in sentences:
        if current_len + len(s) > target_size and current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(text=chunk_text, start=start, end=start + len(chunk_text)))
            # Keep last N sentences as overlap
            current = current[-overlap:] if overlap else []
            current_len = sum(len(x) for x in current)
            start = pos - current_len
        current.append(s)
        current_len += len(s) + 1
        pos += len(s) + 1
    if current:
        chunk_text = " ".join(current)
        chunks.append(Chunk(text=chunk_text, start=start, end=start + len(chunk_text)))
    return chunks

STRATEGIES = {"fixed": chunk_fixed, "sentence": chunk_sentence}

def chunk(text: str, strategy: str = "sentence", **kwargs) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Use one of {list(STRATEGIES)}")
    return STRATEGIES[strategy](text, **kwargs)
