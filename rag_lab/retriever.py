import httpx
from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from . import embedder, vector_store
from .verifier import verify

CHAT_URL = f"{DEEPSEEK_BASE_URL}/chat/completions"

GENERATOR_SYSTEM = """You answer questions using ONLY the provided CONTEXT chunks.

Rules:
- Use ONLY information from the CONTEXT below.
- Cite chunk IDs in square brackets, e.g. [chunk-abc-1].
- If CONTEXT does not contain the answer, reply exactly: "I don't know from the provided documents."
- Do not use outside knowledge."""

def _generate(prompt: str, model: str = None, max_tokens: int = 600) -> str:
    model = model or DEEPSEEK_MODEL
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
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
        with httpx.Client(timeout=180.0) as client:
            r = client.post(CHAT_URL, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {e}")

def _format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{c['id']}]\n{c['text']}" for c in chunks
    )

def _refine_query(question: str, issues: list[str]) -> str:
    """Simple refinement: append the issues as additional constraint."""
    if not issues:
        return question
    return f"{question}\n\n(Previous attempt was weak because: {'; '.join(issues[:3])}. Be more specific.)"

def retrieve(
    question: str,
    top_k: int = 20,
    rerank_top: int = 5,
    max_iters: int = 3,
    min_score: int = 8,
    model: str = None,
) -> dict:
    """The /goal retrieval loop. Returns answer + full trace."""
    model = model or DEEPSEEK_MODEL
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
        q_vec = embedder.embed([current_q])[0]
        chunks = vector_store.query(q_vec, top_k=top_k)[:rerank_top]
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
            answer = _generate(prompt, model=model)
        except RuntimeError as e:
            trace.append({
                "iter": i + 1, "query": current_q, "answer": str(e),
                "verifier_score": 0, "issues": [str(e)],
            })
            return {
                "answer": f"LLM generation error: {e}",
                "chunks": chunks, "verifier": verdict,
                "iterations": i + 1, "trace": trace, "partial": True,
            }
        verdict = verify(current_q, answer, chunks, model=model)
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
