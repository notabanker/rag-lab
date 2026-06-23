import logging
import httpx
from .config import get_provider
from . import embedder, vector_store
from .verifier import verify

logger = logging.getLogger("rag_lab.retriever")

GENERATOR_SYSTEM = """You answer questions using ONLY the provided CONTEXT chunks.

Rules:
- Use ONLY information from the CONTEXT below.
- Cite chunk IDs in square brackets, e.g. [chunk-abc-1].
- If CONTEXT does not contain the answer, reply exactly: "I don't know from the provided documents."
- Do not use outside knowledge."""

_HTTP_CLIENT: httpx.Client | None = None


def _get_client(timeout: float = 180.0) -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.Client(timeout=httpx.Timeout(timeout))
    return _HTTP_CLIENT


def _generate(prompt: str, provider_name: str = None, model: str = None, max_tokens: int = 600) -> str:
    prov = get_provider(provider_name)
    model = model or prov.model
    chat_url = f"{prov.base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if prov.api_key:
        headers["Authorization"] = f"Bearer {prov.api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    try:
        r = _get_client().post(chat_url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {e}")


def _format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{c['id']}]\n{c['text']}" for c in chunks
    )


def _refine_query(question: str, issues: list[str]) -> str:
    if not issues:
        return question
    suffix = f"\n\n(Previous attempt was weak because: {'; '.join(issues[:3])}. Be more specific.)"
    combined = f"{question}{suffix}"
    if len(combined) > 2000:
        max_q = 2000 - len(suffix) - 3
        question = question[:max_q] + "..."
        combined = f"{question}{suffix}"
    return combined


def retrieve(
    question: str,
    top_k: int = 20,
    rerank_top: int = 5,
    max_iters: int = 3,
    min_score: int = 8,
    provider: str = None,
    model: str = None,
    embed_model: str = None,
) -> dict:
    """The /goal retrieval loop. Returns answer + full trace."""
    trace = []
    current_q = question
    answer = ""
    verdict = {"score": 0, "grounded": False, "issues": [], "verdict": "UNGROUNDED"}
    chunks = []

    if max_iters < 1:
        return {
            "answer": "max_iters must be >= 1",
            "chunks": [], "verifier": verdict,
            "iterations": 0, "trace": trace,
        }

    for i in range(max_iters):
        logger.debug(f"Iter {i+1}/{max_iters}: embedding query...")
        q_vec = embedder.embed([current_q], model_name=embed_model)[0]
        chunks = vector_store.query(q_vec, top_k=top_k)[:rerank_top]
        logger.debug(f"Iter {i+1}: retrieved {len(chunks)} chunks")
        if not chunks:
            return {
                "answer": "I don't know from the provided documents (no chunks retrieved).",
                "chunks": [],
                "verifier": {"score": 0, "grounded": False, "issues": ["empty retrieval"], "verdict": "UNGROUNDED"},
                "iterations": i + 1,
                "trace": trace,
            }
        context = _format_context(chunks)
        prompt = f"CONTEXT:\n{context}\n\nQUESTION: {current_q}\n\nANSWER:"
        try:
            answer = _generate(prompt, provider_name=provider, model=model)
        except RuntimeError as e:
            verdict = {"score": 0, "grounded": False, "issues": [str(e)], "verdict": "ERROR"}
            trace.append({
                "iter": i + 1, "query": current_q, "answer": str(e),
                "verifier_score": 0, "issues": [str(e)],
            })
            return {
                "answer": f"LLM generation error: {e}",
                "chunks": chunks, "verifier": verdict,
                "iterations": i + 1, "trace": trace, "partial": True,
            }
        verdict = verify(current_q, answer, chunks, provider=provider, model=model)
        score = verdict.get("score", 0)
        try:
            score = int(score)
        except (ValueError, TypeError):
            score = 0
        trace.append({
            "iter": i + 1, "query": current_q, "answer": answer,
            "verifier_score": score, "issues": verdict.get("issues", []),
        })
        if score >= min_score:
            return {"answer": answer, "chunks": chunks, "verifier": verdict, "iterations": i + 1, "trace": trace}
        current_q = _refine_query(current_q, verdict.get("issues", []))

    return {
        "answer": answer, "chunks": chunks, "verifier": verdict,
        "iterations": max_iters, "trace": trace, "partial": True,
    }


def shutdown():
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        _HTTP_CLIENT.close()
        _HTTP_CLIENT = None
