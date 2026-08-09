"""Shared ingestion metadata helpers."""
from pathlib import Path
import hashlib
import re

from . import chunker, embedder, manifest, vector_store
from .config import CHUNKING_VERSION, EMBEDDING_MODEL, INDEX_VERSION

# Label branch: starts with a non-hyphen, may contain single hyphens
# ("State-of-the-art") but never crosses a "---" or a newline.
_MARKER_RE = re.compile(r"---\s*(Page\s+\d+|[^-\n](?:(?!---)[^\n])*?)\s*---", re.IGNORECASE)

def content_sha(content: bytes) -> str:
    # Full digest: 40-bit truncation made collisions material (~1M docs) and a
    # collision silently replaced an unrelated document. Note: chunk ids change,
    # so an existing index must be re-baselined (rag rebuild).
    return hashlib.sha256(content).hexdigest()

def make_chunks(text: str, strategy: str, chunk_size: int, overlap: int) -> list[chunker.Chunk]:
    if strategy == "fixed":
        return chunker.chunk_fixed(text, size=chunk_size, overlap=overlap)
    if strategy == "sentence":
        return chunker.chunk_sentence(text, target_size=chunk_size, overlap=max(1, overlap // 64))
    raise ValueError(f"Unknown strategy: {strategy}. Use one of ['fixed', 'sentence']")

def _markers(text: str) -> list[tuple[int, str]]:
    return [(m.start(), " ".join(m.group(1).split())) for m in _MARKER_RE.finditer(text)]

def _marker_for(markers: list[tuple[int, str]], offset: int) -> str | None:
    current = None
    for pos, label in markers:
        if pos > offset:
            break
        current = label
    return current

def _citation(source: str, marker: str | None, chunk_idx: int) -> str:
    name = Path(source).name
    if marker:
        page = re.search(r"page\s+(\d+)", marker, re.IGNORECASE)
        if page:
            return f"{name} p.{page.group(1)}"
        return f"{name} §{marker}"
    return f"{name} chunk {chunk_idx + 1}"

def build_metadatas(
    source: str,
    text: str,
    chunks: list[chunker.Chunk],
    file_sha: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    parent_size: int = 4,
) -> list[dict]:
    markers = _markers(text)
    parent_size = max(1, parent_size)
    metadatas = []
    for i, c in enumerate(chunks):
        parent_idx = i // parent_size
        marker = _marker_for(markers, c.start)
        metadatas.append({
            "source": source,
            "doc_id": file_sha,
            "file_sha": file_sha,
            "chunk_idx": i,
            "chunk_start": c.start,
            "chunk_end": c.end,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "parent_id": f"{file_sha}-p{parent_idx}",
            "parent_idx": parent_idx,
            "parent_size": parent_size,
            "citation": _citation(source, marker, i),
            "source_marker": marker or "",
            "embedding_model": EMBEDDING_MODEL,
            "chunking_version": CHUNKING_VERSION,
            "index_version": INDEX_VERSION,
        })
    return metadatas

def ingest_text(
    source: str,
    content: bytes,
    text: str,
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
    parent_size: int = 4,
    force_model_mismatch: bool = False,
    parse_quality: dict | None = None,
) -> dict:
    warnings = (parse_quality or {}).get("warnings", [])
    chunks = make_chunks(text, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        # Record the known-empty document so it shows up in docs list
        # (and future syncs can skip it by SHA) without touching the index.
        file_sha = content_sha(content)
        manifest.log_document(source, file_sha, 0, parse_report=parse_quality)
        return {"chunks": 0, "file_sha": file_sha, "warnings": warnings}
    file_sha = content_sha(content)
    metadatas = build_metadatas(source, text, chunks, file_sha, strategy, chunk_size, overlap, parent_size)
    ids = [f"{file_sha}-{i}" for i in range(len(chunks))]
    vector_store.ensure_collection_compatible(force=force_model_mismatch)
    vecs = embedder.embed([c.text for c in chunks], input_type="document")
    # Atomic swap: upsert the new chunks first, then drop only stale ones with
    # this sha. A failure between the two leaves the previous version intact —
    # the old delete-before-write destroyed it silently on any mid-ingest error.
    vector_store.upsert(
        chunks,
        vecs,
        metadatas,
        ids,
        embedding_model=EMBEDDING_MODEL,
        chunking_version=CHUNKING_VERSION,
        force_model_mismatch=force_model_mismatch,
    )
    vector_store.delete_stale_chunks(file_sha, set(ids))
    manifest.log_document(source, file_sha, len(chunks), parse_report=parse_quality)
    return {"chunks": len(chunks), "file_sha": file_sha, "warnings": warnings}

def ingest_file(
    path: str,
    ocr: str = "auto",
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
    parent_size: int = 4,
    force_model_mismatch: bool = False,
    allow_empty: bool = False,
) -> dict:
    """Parse + quality-check + chunk + embed + store one file (silent).

    Shared by the CLI and `rag sync`. Returns {'chunks', 'file_sha',
    'warnings', 'quality'}; raises ValueError on unusable input.
    """
    from .parsers import as_result, parse_file
    parsed = as_result(parse_file(path, ocr))
    quality = parsed.quality()
    if quality["total_chars"] == 0 and not allow_empty:
        raise ValueError(
            "No text extracted — scanned PDF? Enable OCR with: "
            "brew install tesseract tesseract-lang && uv sync --group ocr. "
            "Use --allow-empty to record the file in the manifest anyway."
        )
    content = Path(path).read_bytes()
    result = ingest_text(
        path, content, parsed.effective_text,
        strategy=strategy, chunk_size=chunk_size, overlap=overlap,
        parent_size=parent_size, force_model_mismatch=force_model_mismatch,
        parse_quality=quality,
    )
    result["quality"] = quality
    return result
