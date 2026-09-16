"""MCP tools for rag-lab.

The tool functions are intentionally importable without the MCP SDK so tests
and CLI code can exercise behavior directly. `build_server()` wires them to
FastMCP when the optional runtime is available.
"""
from . import evaluation, ingestion, vector_store
from .retriever import RetrievalConfig, compact_chunk, retrieve, retrieve_hits, select_context_chunks


def _cfg(**kwargs) -> RetrievalConfig:
    """RetrievalConfig with API/MCP safety clamps applied."""
    return RetrievalConfig(**kwargs).clamped()


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
        "candidates": [compact_chunk(h, include_text=False) for h in hits[:rerank_top]],
        "context": [compact_chunk(c) for c in context],
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
        "chunks": [compact_chunk(c) for c in result.get("chunks", [])],
    }


def rag_ingest(
    path: str,
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
    parent_size: int = 4,
) -> dict:
    try:
        result = ingestion.ingest_file(
            path, strategy=strategy, chunk_size=chunk_size,
            overlap=overlap, parent_size=parent_size,
        )
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    return {"status": "ok", "source": path, **result}


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
    source = vector_store.document_source(identifier)
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
    try:
        report = evaluation.run_eval(
            questions_file,
            _cfg(mode=mode, use_reranker=use_reranker, small_to_big=small_to_big),
            retrieval_only=retrieval_only,
            variant="mcp",
        )
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    return {"status": "ok", "eval_id": report["eval_id"], "summary": report["summary"], "config": report["config"]}


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise RuntimeError("Install the 'mcp' package to run the rag-lab MCP server") from e

    server = FastMCP("rag-lab")
    for tool in (
        rag_search, rag_answer, rag_ingest, rag_delete, rag_docs_list,
        rag_reingest, rag_collections_list, rag_runs_show, rag_eval_run,
    ):
        server.tool()(tool)
    return server


def main():
    build_server().run()


if __name__ == "__main__":
    main()
