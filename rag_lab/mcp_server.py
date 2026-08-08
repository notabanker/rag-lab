"""MCP tools for rag-lab.

The helper functions are intentionally importable without the MCP SDK so tests
and CLI code can exercise behavior directly. `build_server()` wires them to
FastMCP when the optional runtime is available.
"""
from pathlib import Path

from . import evaluation, ingestion, vector_store
from .parsers import as_result, pick_parser
from .retriever import RetrievalConfig, retrieve, retrieve_hits, select_context_chunks


def _cfg(
    mode: str = "hybrid",
    top_k: int = 50,
    rerank_top: int = 5,
    use_reranker: bool = True,
    small_to_big: bool = True,
    parent_top_k: int = 5,
    max_context_chars: int = 12000,
) -> RetrievalConfig:
    # Caps: an MCP client must not be able to balloon a query into a
    # multi-thousand-chunk, unbounded-cost run.
    top_k = min(max(top_k, 1), 200)
    rerank_top = min(max(rerank_top, 1), 50)
    parent_top_k = min(max(parent_top_k, 1), 50)
    max_context_chars = min(max(max_context_chars, 1), 100000)
    return RetrievalConfig(
        mode=mode,
        top_k=top_k,
        rerank_top=rerank_top,
        use_reranker=use_reranker,
        small_to_big=small_to_big,
        parent_top_k=parent_top_k,
        max_context_chars=max_context_chars,
    )


def _compact_chunk(chunk: dict, include_text: bool = True) -> dict:
    meta = chunk.get("metadata") or {}
    out = {
        "id": chunk.get("id"),
        "citation": chunk.get("citation") or meta.get("citation"),
        "source": meta.get("source"),
        "distance": chunk.get("distance"),
        "rerank_score": chunk.get("rerank_score"),
        "rrf_score": chunk.get("rrf_score"),
    }
    if include_text:
        text = chunk.get("text") or ""
        out["text"] = text[:2000]
    return out


def rag_search(
    question: str,
    mode: str = "hybrid",
    top_k: int = 50,
    rerank_top: int = 5,
    use_reranker: bool = True,
    small_to_big: bool = True,
) -> dict:
    cfg = _cfg(mode=mode, top_k=top_k, rerank_top=rerank_top, use_reranker=use_reranker, small_to_big=small_to_big)
    hits = retrieve_hits(question, cfg)
    context = select_context_chunks(hits, cfg)
    return {
        "question": question,
        "candidates": [_compact_chunk(h, include_text=False) for h in hits[:rerank_top]],
        "context": [_compact_chunk(c) for c in context],
    }


def rag_answer(
    question: str,
    mode: str = "hybrid",
    top_k: int = 50,
    rerank_top: int = 5,
    use_reranker: bool = True,
    small_to_big: bool = True,
) -> dict:
    cfg = _cfg(mode=mode, top_k=top_k, rerank_top=rerank_top, use_reranker=use_reranker, small_to_big=small_to_big)
    result = retrieve(question, cfg)
    return {
        "run_id": result.get("run_id"),
        "answer": result.get("answer"),
        "verifier": result.get("verifier"),
        "citation_validation": result.get("citation_validation"),
        "iterations": result.get("iterations"),
        "partial": result.get("partial", False),
        "chunks": [_compact_chunk(c) for c in result.get("chunks", [])],
    }


def rag_ingest(
    path: str,
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
    parent_size: int = 4,
) -> dict:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {path}")
    parsed = as_result(pick_parser(str(p))(str(p)))
    quality = parsed.quality()
    content = p.read_bytes()
    result = ingestion.ingest_text(
        str(p),
        content,
        parsed.effective_text,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        parent_size=parent_size,
        parse_quality=quality,
    )
    return {"status": "ok", "source": str(p), **result}


def rag_delete(identifier: str, confirm: bool = False) -> dict:
    if not confirm:
        return {
            "status": "confirmation_required",
            "message": "Set confirm=true to delete by exact doc_id, file_sha, source, or filename.",
            "deleted_chunks": 0,
        }
    removed = vector_store.delete_document(identifier)
    return {"status": "ok" if removed else "not_found", "identifier": identifier, "deleted_chunks": removed}


def rag_docs_list() -> dict:
    from . import manifest
    return {"documents": vector_store.list_documents(), "manifest": manifest.list_documents()}


def rag_reingest(
    identifier: str,
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
    parent_size: int = 4,
) -> dict:
    from . import manifest
    doc = manifest.get_document(identifier)
    indexed = next((
        d for d in vector_store.list_documents()
        if identifier in {d.get("doc_id"), d.get("file_sha"), d.get("source"), d.get("basename")}
    ), None)
    source = (doc or indexed or {}).get("source")
    if not source:
        return {"status": "not_found", "identifier": identifier}
    return rag_ingest(
        source,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        parent_size=parent_size,
    )


def rag_collections_list() -> dict:
    return {"collections": vector_store.list_collections()}


def rag_runs_show(run_id: int) -> dict:
    from . import runs
    run = runs.get_run(run_id)
    if run is None:
        return {"status": "not_found", "run_id": run_id}
    return {"status": "ok", "run": run}


def rag_eval_run(
    questions_file: str = "eval/questions.yaml",
    retrieval_only: bool = True,
    mode: str = "hybrid",
    use_reranker: bool = True,
    small_to_big: bool = True,
) -> dict:
    from . import runs
    questions = evaluation.load_questions(evaluation.secure_questions_path(questions_file))
    if len(questions) > evaluation.MAX_EVAL_QUESTIONS:
        return {"status": "error", "message": f"too many questions ({len(questions)} > {evaluation.MAX_EVAL_QUESTIONS})"}
    cfg = _cfg(mode=mode, use_reranker=use_reranker, small_to_big=small_to_big)
    report = evaluation.evaluate(questions, cfg, retrieval_only=retrieval_only)
    report["config"]["collection"] = vector_store.default_collection_name()
    eval_id = runs.log_eval("mcp", report["config"], report["summary"], report["per_question"])
    return {"status": "ok", "eval_id": eval_id, "summary": report["summary"], "config": report["config"]}


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise RuntimeError("Install the 'mcp' package to run the rag-lab MCP server") from e

    server = FastMCP("rag-lab")
    server.tool()(rag_search)
    server.tool()(rag_answer)
    server.tool()(rag_ingest)
    server.tool()(rag_delete)
    server.tool()(rag_docs_list)
    server.tool()(rag_reingest)
    server.tool()(rag_collections_list)
    server.tool()(rag_runs_show)
    server.tool()(rag_eval_run)
    return server


def main():
    build_server().run()


if __name__ == "__main__":
    main()
