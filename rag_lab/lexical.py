"""BM25 lexical index over the chunk store.

Built lazily from all documents in a collection and cached in memory.
Invalidated explicitly on writes (upsert/delete) and implicitly when the
collection's chunk count changes.
"""
import re

from rank_bm25 import BM25Okapi

from . import vector_store

# Keeps letters+digits together so codes like DLBFMWFT1 stay one token;
# includes German umlauts/ß.
_TOKEN_RE = re.compile(r"[a-z0-9äöüß]+")

_CACHE: dict[str, dict] = {}  # collection name -> {count, bm25, ids, docs, metas}

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

def invalidate(collection_name: str = None):
    if collection_name is None:
        _CACHE.clear()
    else:
        _CACHE.pop(collection_name, None)

def _build_index(name: str) -> dict | None:
    coll = vector_store.get_collection(name)
    total = coll.count()
    if total == 0:
        return None
    ids, docs, metas = [], [], []
    offset = 0
    while offset < total:
        batch = coll.get(limit=500, offset=offset, include=["documents", "metadatas"])
        if not batch["ids"]:
            break
        ids.extend(batch["ids"])
        docs.extend(batch["documents"])
        metas.extend(batch["metadatas"])
        offset += 500
    corpus = [tokenize(d) for d in docs]
    return {
        "count": total,
        "bm25": BM25Okapi(corpus),
        "ids": ids,
        "docs": docs,
        "metas": metas,
    }

def _get_index(name: str = None) -> dict | None:
    name = name or vector_store.default_collection_name()
    cached = _CACHE.get(name)
    current = vector_store.get_collection(name).count()
    if cached is None or cached["count"] != current:
        cached = _build_index(name)
        if cached is None:
            _CACHE.pop(name, None)
            return None
        _CACHE[name] = cached
    return cached

def rrf_fuse(hit_lists: dict[str, list[dict]], k: int = 60, limit: int = None) -> list[dict]:
    """Reciprocal rank fusion across named hit lists (e.g. {"vector": [...], "lexical": [...]}).

    score(chunk) = Σ over lists of 1/(k + rank). Deduped by chunk id; each fused hit
    carries rrf_score and source_ranks. Deterministic tie-break by id.
    """
    fused: dict[str, dict] = {}
    for source, hits in hit_lists.items():
        for rank, h in enumerate(hits, 1):
            entry = fused.get(h["id"])
            if entry is None:
                entry = fused[h["id"]] = {**h, "rrf_score": 0.0, "source_ranks": {}}
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["source_ranks"][source] = rank
            if "bm25_score" in h:
                entry["bm25_score"] = h["bm25_score"]
            if h.get("distance") is not None:
                entry["distance"] = h["distance"]
    ranked = sorted(fused.values(), key=lambda e: (-e["rrf_score"], e["id"]))
    return ranked[:limit] if limit is not None else ranked

def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """Hits in the same shape as vector_store.query; score is BM25 (higher = better)."""
    idx = _get_index()
    if idx is None:
        return []
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = idx["bm25"].get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits = []
    for i in order:
        if scores[i] <= 0:
            break
        hits.append({
            "id": idx["ids"][i],
            "text": idx["docs"][i],
            "metadata": idx["metas"][i],
            "distance": None,
            "bm25_score": float(scores[i]),
        })
    return hits
