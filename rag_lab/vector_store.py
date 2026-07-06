import re
import chromadb
from chromadb.config import Settings

from .config import CHUNKING_VERSION, DEFAULT_COLLECTION, EMBEDDING_MODEL, INDEX_VERSION, default_db_path

_PERSIST_DIR = default_db_path()
_CLIENT = None
_COLLECTIONS = {}
_DEFAULT_NAME = DEFAULT_COLLECTION

def init_store(persist_dir: str):
    global _PERSIST_DIR, _CLIENT, _COLLECTIONS
    _PERSIST_DIR = persist_dir
    _CLIENT = None
    _COLLECTIONS = {}

def set_default_collection(name: str):
    global _DEFAULT_NAME
    _DEFAULT_NAME = name

def default_collection_name() -> str:
    return _DEFAULT_NAME

def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
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
    if "hnsw:space" not in merged:
        merged["hnsw:space"] = "cosine"
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
    meta = dict(coll.metadata or {})
    existing_model = meta.get("embedding_model")
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
    coll = get_collection()
    total = coll.count()
    hits = []
    seen = set()
    compiled = re.compile(pattern, re.IGNORECASE)
    offset = 0
    batch = 500
    while offset < total and len(hits) < limit:
        results = coll.get(limit=batch, offset=offset, include=["documents", "metadatas"])
        if not results["ids"]:
            break
        for i, doc in enumerate(results["documents"]):
            if compiled.search(doc) and results["ids"][i] not in seen:
                seen.add(results["ids"][i])
                hits.append({
                    "text": doc,
                    "metadata": results["metadatas"][i],
                    "distance": 0,
                    "id": results["ids"][i],
                })
                if len(hits) >= limit:
                    break
        offset += batch
    return hits

def delete_by_sha(file_sha: str):
    """Remove all chunks previously ingested from the file with this sha."""
    coll = get_collection()
    coll.delete(where={"file_sha": file_sha})
    _invalidate_lexical()

def delete_document(identifier: str) -> int:
    """Delete by file_sha/doc_id, exact source, or source basename. Returns removed chunk count."""
    coll = get_collection()
    before = coll.count()
    docs = list_documents()
    matches = [
        d for d in docs
        if identifier in {d.get("file_sha"), d.get("doc_id"), d.get("source"), d.get("basename")}
    ]
    if not matches:
        return 0
    for d in matches:
        doc_id = d.get("doc_id") or d.get("file_sha")
        if d.get("doc_id"):
            coll.delete(where={"doc_id": doc_id})
        if d.get("file_sha"):
            coll.delete(where={"file_sha": d["file_sha"]})
        if not d.get("doc_id") and not d.get("file_sha"):
            coll.delete(where={"source": d["source"]})
        try:
            from . import manifest
            manifest.delete_document(d.get("doc_id") or d.get("file_sha") or d.get("source"))
        except Exception:
            pass
    _invalidate_lexical()
    return before - coll.count()

def count() -> int:
    return get_collection().count()

def distinct_sources() -> int:
    coll = get_collection()
    total = coll.count()
    sources = set()
    offset = 0
    while offset < total:
        results = coll.get(limit=500, offset=offset, include=["metadatas"])
        if not results["ids"]:
            break
        for m in results["metadatas"]:
            src = (m or {}).get("source")
            if src:
                sources.add(src)
        offset += 500
    return len(sources)

def _all_records(include_documents: bool = False) -> list[dict]:
    coll = get_collection()
    total = coll.count()
    records = []
    offset = 0
    include = ["metadatas"]
    if include_documents:
        include.append("documents")
    while offset < total:
        results = coll.get(limit=500, offset=offset, include=include)
        if not results["ids"]:
            break
        for i, id_ in enumerate(results["ids"]):
            rec = {"id": id_, "metadata": (results["metadatas"][i] or {})}
            if include_documents:
                rec["text"] = results["documents"][i]
            records.append(rec)
        offset += 500
    return records

def list_documents() -> list[dict]:
    docs: dict[str, dict] = {}
    for rec in _all_records():
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
    coll = get_collection()
    results = coll.get(where={"parent_id": parent_id}, include=["documents", "metadatas"])
    hits = []
    for i, id_ in enumerate(results["ids"]):
        hits.append({
            "id": id_,
            "text": results["documents"][i],
            "metadata": results["metadatas"][i] or {},
            "distance": None,
        })
    return sorted(hits, key=lambda h: (h["metadata"].get("chunk_idx", 0), h["id"]))
