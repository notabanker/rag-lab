import re
from typing import Iterator

import chromadb
from chromadb.config import Settings

from .config import CHUNKING_VERSION, DEFAULT_COLLECTION, EMBEDDING_MODEL, INDEX_VERSION, default_db_path

_PERSIST_DIR = None  # resolved lazily so RAG_DB_PATH / tests take effect
_CLIENT = None
_COLLECTIONS = {}
_DEFAULT_NAME = DEFAULT_COLLECTION


def init_store(persist_dir: str):
    global _PERSIST_DIR, _CLIENT, _COLLECTIONS
    _PERSIST_DIR = persist_dir
    _CLIENT = None
    _COLLECTIONS = {}


def persist_dir() -> str:
    global _PERSIST_DIR
    if _PERSIST_DIR is None:
        _PERSIST_DIR = default_db_path()
    return _PERSIST_DIR


def set_default_collection(name: str):
    global _DEFAULT_NAME
    _DEFAULT_NAME = name


def default_collection_name() -> str:
    return _DEFAULT_NAME


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=persist_dir(), settings=Settings(anonymized_telemetry=False))
    return _CLIENT


def get_collection(name: str = None):
    name = name or _DEFAULT_NAME
    if name not in _COLLECTIONS:
        _COLLECTIONS[name] = _client().get_or_create_collection(
            name=name,
            metadata=_base_metadata()
        )
    return _COLLECTIONS[name]


def _base_metadata(
    embedding_model: str = None,
    chunking_version: str = None,
) -> dict:
    return {
        "hnsw:space": "cosine",
        "index_version": INDEX_VERSION,
        "embedding_model": embedding_model or EMBEDDING_MODEL,
        "chunking_version": chunking_version or CHUNKING_VERSION,
    }


def collection_metadata(name: str = None) -> dict:
    return dict(get_collection(name).metadata or {})


def _set_collection_metadata(name: str = None, **updates):
    coll = get_collection(name)
    merged = {**(coll.metadata or {}), **{k: v for k, v in updates.items() if v is not None}}
    merged.setdefault("hnsw:space", "cosine")
    coll.modify(metadata=merged)


def ensure_collection_compatible(
    name: str = None,
    embedding_model: str = None,
    chunking_version: str = None,
    force: bool = False,
):
    """Guard against mixing incompatible embeddings in one Chroma collection."""
    name = name or _DEFAULT_NAME
    model = embedding_model or EMBEDDING_MODEL
    chunking = chunking_version or CHUNKING_VERSION
    coll = get_collection(name)
    existing_model = (coll.metadata or {}).get("embedding_model")
    if not existing_model and coll.count() > 0 and not force:
        raise ValueError(
            f"collection '{name}' has {coll.count()} chunks but no embedding_model metadata; "
            "rebuild it with V3 or pass force=True if this legacy index is intentional"
        )
    if existing_model and existing_model != model and not force:
        raise ValueError(
            f"collection '{name}' was built with embedding_model={existing_model!r}, "
            f"but current EMBEDDING_MODEL is {model!r}; use another collection or force the mismatch"
        )
    if not existing_model or force:
        _set_collection_metadata(name, embedding_model=model, chunking_version=chunking, index_version=INDEX_VERSION)


def _invalidate_lexical():
    from . import lexical
    lexical.invalidate(_DEFAULT_NAME)


def upsert(
    chunks: list,
    embeddings: list,
    metadatas: list,
    ids: list,
    embedding_model: str = None,
    chunking_version: str = None,
    force_model_mismatch: bool = False,
):
    ensure_collection_compatible(
        embedding_model=embedding_model,
        chunking_version=chunking_version,
        force=force_model_mismatch,
    )
    coll = get_collection()
    coll.upsert(
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=metadatas,
        ids=ids,
    )
    _invalidate_lexical()


def iter_records(collection: str = None, include_documents: bool = False) -> Iterator[dict]:
    """Yield every chunk in a collection as {id, metadata[, text]} (paged)."""
    coll = get_collection(collection)
    include = ["metadatas"] + (["documents"] if include_documents else [])
    offset = 0
    while True:
        results = coll.get(limit=500, offset=offset, include=include)
        if not results["ids"]:
            return
        for i, id_ in enumerate(results["ids"]):
            rec = {"id": id_, "metadata": results["metadatas"][i] or {}}
            if include_documents:
                rec["text"] = results["documents"][i]
            yield rec
        offset += 500


def query(query_embedding: list[float], top_k: int = 20) -> list[dict]:
    ensure_collection_compatible()
    coll = get_collection()
    n = coll.count()
    if n == 0:
        return []
    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, n),
    )
    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "id": results["ids"][0][i],
        })
    return hits


def keyword_search(pattern: str, limit: int = 200) -> list[dict]:
    # Literal substring match: a user-supplied "pattern" must never become a
    # regex (catastrophic backtracking on a large corpus is a ReDoS).
    compiled = re.compile(re.escape(pattern), re.IGNORECASE)
    hits = []
    for rec in iter_records(include_documents=True):
        if compiled.search(rec["text"]):
            hits.append({**rec, "distance": 0})
            if len(hits) >= limit:
                break
    return hits


def delete_stale_chunks(file_sha: str, keep_ids: set[str]):
    """Remove chunks of this file NOT in keep_ids. Called AFTER upsert so a
    failure mid-ingest never destroys the previous good version (atomic swap)."""
    coll = get_collection()
    existing = coll.get(where={"file_sha": file_sha}, include=["metadatas"])["ids"]
    stale = [i for i in existing if i not in keep_ids]
    if stale:
        coll.delete(ids=stale)
        _invalidate_lexical()


def list_documents() -> list[dict]:
    docs: dict[str, dict] = {}
    for rec in iter_records():
        meta = rec["metadata"]
        source = meta.get("source") or "unknown"
        key = meta.get("doc_id") or meta.get("file_sha") or source
        row = docs.setdefault(key, {
            "doc_id": meta.get("doc_id") or meta.get("file_sha") or key,
            "file_sha": meta.get("file_sha"),
            "source": source,
            "basename": source.split("/")[-1],
            "chunks": 0,
            "strategy": meta.get("strategy"),
            "embedding_model": meta.get("embedding_model"),
            "chunking_version": meta.get("chunking_version"),
        })
        row["chunks"] += 1
    return sorted(docs.values(), key=lambda d: (d["source"], d["doc_id"]))


def find_document(identifier: str) -> dict | None:
    """Locate a document by doc_id, file_sha, source, or source basename.

    Returns {"manifest": dict|None, "indexed": dict|None}; None when neither
    the manifest nor the index knows the identifier."""
    from . import manifest
    m = manifest.get_document(identifier)
    indexed = next((
        d for d in list_documents()
        if identifier in {d.get("doc_id"), d.get("file_sha"), d.get("source"), d.get("basename")}
    ), None)
    if m is None and indexed is None:
        return None
    return {"manifest": m, "indexed": indexed}


def document_source(identifier: str) -> str | None:
    """Stored source path for a document matched by any identifier, or None."""
    found = find_document(identifier)
    if found is None:
        return None
    return (found["manifest"] or found["indexed"] or {}).get("source")


def delete_document(identifier: str) -> int:
    """Delete by file_sha/doc_id, exact source, or source basename. Returns removed chunk count."""
    coll = get_collection()
    before = coll.count()
    matches = [
        d for d in list_documents()
        if identifier in {d.get("file_sha"), d.get("doc_id"), d.get("source"), d.get("basename")}
    ]
    if not matches:
        return 0
    for d in matches:
        for key in ("doc_id", "file_sha"):
            if d.get(key):
                coll.delete(where={key: d[key]})
        if not d.get("doc_id") and not d.get("file_sha"):
            coll.delete(where={"source": d["source"]})
    try:
        from . import manifest
        manifest.delete_document(identifier)
    except Exception:
        pass
    _invalidate_lexical()
    return before - coll.count()


def count() -> int:
    return get_collection().count()


def distinct_sources() -> int:
    return len({r["metadata"].get("source") for r in iter_records() if r["metadata"].get("source")})


def list_collections() -> list[dict]:
    out = []
    for item in _client().list_collections():
        name = item if isinstance(item, str) else item.name
        coll = get_collection(name)
        out.append({"name": name, "count": coll.count(), "metadata": dict(coll.metadata or {})})
    return sorted(out, key=lambda c: c["name"])


def delete_collection(name: str) -> bool:
    if name not in {c["name"] for c in list_collections()}:
        return False
    _client().delete_collection(name)
    _COLLECTIONS.pop(name, None)
    from . import lexical
    lexical.invalidate(name)
    return True


def get_by_parent_id(parent_id: str) -> list[dict]:
    results = get_collection().get(where={"parent_id": parent_id}, include=["documents", "metadatas"])
    hits = []
    for i, id_ in enumerate(results["ids"]):
        hits.append({
            "id": id_,
            "text": results["documents"][i],
            "metadata": results["metadatas"][i] or {},
            "distance": None,
        })
    return sorted(hits, key=lambda h: (h["metadata"].get("chunk_idx", 0), h["id"]))
