import threading
import chromadb
from chromadb.config import Settings

from .chunker import Chunk

_PERSIST_DIR = "./chroma_db"
_CLIENT = None
_COLLECTIONS: dict[str, object] = {}
_lock = threading.Lock()


def init_store(persist_dir: str):
    global _PERSIST_DIR, _CLIENT, _COLLECTIONS
    with _lock:
        if _CLIENT is not None:
            try:
                _CLIENT.clear_system_cache()
            except Exception:
                pass
        _PERSIST_DIR = persist_dir
        _CLIENT = None
        _COLLECTIONS = {}


def _init_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))


def get_collection(name: str = "rag_lab"):
    global _COLLECTIONS
    if name not in _COLLECTIONS:
        with _lock:
            if name not in _COLLECTIONS:
                _init_client()
                _COLLECTIONS[name] = _CLIENT.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )
    return _COLLECTIONS[name]


def upsert(chunks: list[Chunk], embeddings: list, metadatas: list, ids: list, collection: str = "rag_lab"):
    coll = get_collection(collection)
    coll.upsert(
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=metadatas,
        ids=ids,
    )


def query(query_embedding: list[float], top_k: int = 20, collection: str = "rag_lab") -> list[dict]:
    coll = get_collection(collection)
    try:
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
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
    except KeyError as e:
        raise RuntimeError(f"ChromaDB query returned unexpected shape: missing key {e}")


def query_multi(query_embedding: list[float], collections: list[str], top_k: int = 20) -> list[dict]:
    all_hits = []
    for coll_name in collections:
        hits = query(query_embedding, top_k=top_k, collection=coll_name)
        for h in hits:
            h["collection"] = coll_name
        all_hits.extend(hits)
    all_hits.sort(key=lambda h: h.get("distance", float("inf")))
    return all_hits[:top_k]


def count(collection: str = "rag_lab") -> int:
    return get_collection(collection).count()


def delete_by_source(file_sha: str, collection: str = "rag_lab") -> int:
    coll = get_collection(collection)
    try:
        results = coll.get(where={"file_sha": file_sha})
    except Exception:
        return 0
    ids = results.get("ids", [])
    if ids:
        coll.delete(ids=ids)
    return len(ids)


def shutdown():
    global _CLIENT, _COLLECTIONS
    with _lock:
        if _CLIENT is not None:
            try:
                _CLIENT.clear_system_cache()
            except Exception:
                pass
        _CLIENT = None
        _COLLECTIONS = {}
