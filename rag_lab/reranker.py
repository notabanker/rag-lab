"""Cross-encoder reranker. Lazy-loaded, CPU, model configurable via RERANKER_MODEL."""
from .config import RERANKER_BATCH_SIZE, RERANKER_MODEL

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder
        _MODEL = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _MODEL

def rerank(query: str, hits: list[dict], top_n: int = None, batch_size: int = None) -> list[dict]:
    """Score (query, chunk) pairs with the cross-encoder and reorder best-first.

    Annotates each hit with 'rerank_score'. Returns all hits unless top_n is given.
    """
    if not hits:
        return []
    pairs = [(query, h["text"]) for h in hits]
    try:
        scores = get_model().predict(
            pairs,
            batch_size=batch_size or RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )
    except TypeError:
        scores = get_model().predict(pairs, show_progress_bar=False)
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    ordered = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ordered if top_n is None else ordered[:top_n]
