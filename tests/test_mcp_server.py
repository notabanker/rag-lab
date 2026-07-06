from rag_lab import mcp_server, vector_store
from rag_lab.chunker import Chunk


def test_mcp_delete_requires_confirmation():
    result = mcp_server.rag_delete("doc")
    assert result["status"] == "confirmation_required"
    assert result["deleted_chunks"] == 0


def test_mcp_delete_confirmed(tmp_path):
    vector_store.init_store(str(tmp_path / "db"))
    chunks = [Chunk(text="alpha", start=0, end=5)]
    metas = [{"source": "a.md", "doc_id": "doc", "file_sha": "doc", "chunk_idx": 0}]
    vector_store.upsert(chunks, [[1.0, 0.0]], metas, ["c1"])
    result = mcp_server.rag_delete("doc", confirm=True)
    assert result["status"] == "ok"
    assert result["deleted_chunks"] == 1


def test_mcp_search_uses_shared_retrieval(monkeypatch):
    hit = {"id": "c1", "text": "child", "metadata": {"source": "a.md", "citation": "a.md chunk 1"}}
    ctx = {"id": "p1", "text": "parent", "citation": "a.md chunk 1", "metadata": {"source": "a.md"}}
    monkeypatch.setattr(mcp_server, "retrieve_hits", lambda question, cfg: [hit])
    monkeypatch.setattr(mcp_server, "select_context_chunks", lambda hits, cfg: [ctx])
    result = mcp_server.rag_search("q", use_reranker=False)
    assert result["candidates"][0]["id"] == "c1"
    assert result["context"][0]["citation"] == "a.md chunk 1"


def test_mcp_answer_returns_validation(monkeypatch):
    monkeypatch.setattr(mcp_server, "retrieve", lambda question, cfg: {
        "run_id": 7,
        "answer": "A [a.md chunk 1]",
        "verifier": {"score": 9},
        "citation_validation": {"citation_valid": True},
        "iterations": 1,
        "chunks": [{"id": "c1", "text": "ctx", "citation": "a.md chunk 1", "metadata": {"source": "a.md"}}],
    })
    result = mcp_server.rag_answer("q", use_reranker=False)
    assert result["run_id"] == 7
    assert result["citation_validation"]["citation_valid"] is True


def test_mcp_ingest_uses_shared_ingestion(tmp_path, monkeypatch):
    f = tmp_path / "note.md"
    f.write_text("# Note\n\nHello")
    monkeypatch.setattr(mcp_server, "pick_parser", lambda path: lambda p: "Hello")
    monkeypatch.setattr(mcp_server.ingestion, "ingest_text", lambda source, content, text, **kwargs: {
        "chunks": 1,
        "file_sha": "abc",
    })
    result = mcp_server.rag_ingest(str(f))
    assert result["status"] == "ok"
    assert result["chunks"] == 1
    assert result["file_sha"] == "abc"


def test_mcp_reingest_uses_manifest_source(monkeypatch):
    from rag_lab import manifest

    monkeypatch.setattr(manifest, "get_document", lambda identifier: {"source": "note.md"})
    monkeypatch.setattr(mcp_server, "rag_ingest", lambda path, **kwargs: {
        "status": "ok",
        "source": path,
        "chunks": 1,
    })

    result = mcp_server.rag_reingest("doc")

    assert result["status"] == "ok"
    assert result["source"] == "note.md"


def test_mcp_reingest_falls_back_to_indexed_source(monkeypatch):
    from rag_lab import manifest

    monkeypatch.setattr(manifest, "get_document", lambda identifier: None)
    monkeypatch.setattr(mcp_server.vector_store, "list_documents", lambda: [{
        "doc_id": "doc",
        "file_sha": "doc",
        "source": "indexed.md",
        "basename": "indexed.md",
    }])
    monkeypatch.setattr(mcp_server, "rag_ingest", lambda path, **kwargs: {
        "status": "ok",
        "source": path,
        "chunks": 1,
    })

    result = mcp_server.rag_reingest("doc")

    assert result["status"] == "ok"
    assert result["source"] == "indexed.md"
