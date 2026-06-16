import chromadb
from chromadb.config import Settings

_PERSIST_DIR = "./chroma_db"
_CLIENT = None
_COLLECTION = None

def init_store(persist_dir: str):
    global _PERSIST_DIR, _CLIENT, _COLLECTION
    _PERSIST_DIR = persist_dir
    _CLIENT = None
    _COLLECTION = None

def get_collection(name: str = "rag_lab"):
    global _CLIENT, _COLLECTION
    if _COLLECTION is None:
        _CLIENT = chromadb.PersistentClient(path=_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
        _COLLECTION = _CLIENT.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    return _COLLECTION

def upsert(chunks: list, embeddings: list, metadatas: list, ids: list):
    coll = get_collection()
    coll.upsert(
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=metadatas,
        ids=ids,
    )

def query(query_embedding: list[float], top_k: int = 20) -> list[dict]:
    coll = get_collection()
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

def count() -> int:
    return get_collection().count()
