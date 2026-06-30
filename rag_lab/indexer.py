import hashlib
import logging
from pathlib import Path

from . import chunker, embedder, vector_store
from .metadata import extract_frontmatter, frontmatter_to_metadata
from .vault_config import load_exclude_patterns, resolve_layer, should_index

logger = logging.getLogger("rag_lab.indexer")


def _index_single_file(file_path: str, vault_root: str, embed_model: str = None) -> dict | None:
    fp = Path(file_path)
    try:
        rel = str(fp.relative_to(vault_root))
    except ValueError:
        logger.warning(f"File outside vault root, skipping: {file_path}")
        return None

    collection = resolve_layer(file_path, vault_root)
    if collection is None:
        logger.debug(f"Cannot resolve layer for {file_path}, skipping")
        return None

    try:
        raw = fp.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return None

    fm, body = extract_frontmatter(raw)

    if len(body) <= 1000:
        chunks = chunker.chunk(body, strategy="document")
    else:
        chunks = chunker.chunk(body, strategy="paragraph", max_chars=1000, overlap_paragraphs=1)

    if not chunks:
        return None

    file_sha = hashlib.sha256(rel.encode()).hexdigest()[:12]
    meta_base = frontmatter_to_metadata(fm, rel, collection)

    metadatas = []
    ids = []
    for i, ch in enumerate(chunks):
        m = dict(meta_base)
        m["chunk_idx"] = i
        m["chunk_count"] = len(chunks)
        metadatas.append(m)
        ids.append(f"{file_sha}-{i}")

    vector_store.delete_by_source(file_sha, collection=collection)
    vecs = embedder.embed([c.text for c in chunks], model_name=embed_model)
    vector_store.upsert(chunks, vecs, metadatas, ids, collection=collection)

    logger.info(f"Indexed {rel} → {collection} ({len(chunks)} chunks)")
    return {"file": rel, "collection": collection, "chunks": len(chunks)}


def index_vault(
    vault_root: str,
    collections: list[str] | None = None,
    force: bool = False,
    embed_model: str = None,
) -> dict:
    vault = Path(vault_root).resolve()
    if not vault.exists():
        raise FileNotFoundError(f"Vault root does not exist: {vault}")

    exclude_patterns = load_exclude_patterns(str(vault))
    logger.info(f"Indexing vault: {vault} with {len(exclude_patterns)} exclude patterns")

    files_indexed = 0
    chunks_total = 0
    by_collection: dict[str, int] = {}

    for md_file in sorted(vault.rglob("*.md")):
        path_str = str(md_file)
        if not should_index(path_str, str(vault), exclude_patterns):
            logger.debug(f"Excluded: {md_file.relative_to(vault)}")
            continue

        result = _index_single_file(path_str, str(vault), embed_model=embed_model)
        if result is None:
            continue

        files_indexed += 1
        chunks_total += result["chunks"]
        coll = result["collection"]
        by_collection[coll] = by_collection.get(coll, 0) + result["chunks"]

    return {
        "files_indexed": files_indexed,
        "chunks_total": chunks_total,
        "by_collection": by_collection,
    }
