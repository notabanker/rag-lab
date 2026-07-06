from fastapi.testclient import TestClient

from rag_lab import manifest, vector_store, web
from rag_lab.web import app


def test_dashboard_config_and_stats_endpoints():
    client = TestClient(app)
    cfg = client.get("/api/config")
    assert cfg.status_code == 200
    assert cfg.json()["llm_model"] == "qwen/qwen3.7-plus"

    stats = client.get("/api/stats")
    assert stats.status_code == 200
    assert "collection" in stats.json()


def test_dashboard_collections_docs_runs_evals_endpoints():
    client = TestClient(app)
    assert client.get("/api/collections").status_code == 200
    assert client.get("/api/docs").status_code == 200
    assert client.get("/api/runs").status_code == 200
    assert client.get("/api/evals").status_code == 200


def test_dashboard_doc_detail_includes_manifest_and_index(monkeypatch):
    monkeypatch.setattr(manifest, "get_document", lambda doc_id: {
        "doc_id": "doc1",
        "source": "notes/doc.md",
        "file_sha": "doc1",
        "chunk_count": 2,
    })
    monkeypatch.setattr(vector_store, "list_documents", lambda: [{
        "doc_id": "doc1",
        "file_sha": "doc1",
        "source": "notes/doc.md",
        "basename": "doc.md",
        "chunks": 2,
    }])

    resp = TestClient(app).get("/api/docs/doc1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["manifest"]["doc_id"] == "doc1"
    assert body["indexed"]["chunks"] == 2


def test_dashboard_reingest_uses_manifest_source(tmp_path, monkeypatch):
    source = tmp_path / "doc.md"
    source.write_text("# Doc\n\nHello")
    monkeypatch.setattr(manifest, "get_document", lambda doc_id: {
        "doc_id": "doc1",
        "source": str(source),
        "file_sha": "doc1",
        "chunk_count": 1,
    })
    monkeypatch.setattr(web, "pick_parser", lambda path: lambda p: "Hello")
    monkeypatch.setattr(web.ingestion, "ingest_text", lambda source, content, text, **kwargs: {
        "chunks": 1,
        "file_sha": "doc1",
    })

    resp = TestClient(app).post("/api/docs/doc1/reingest", json={"strategy": "sentence"})

    assert resp.status_code == 200
    assert resp.json()["chunks"] == 1


def test_dashboard_reingest_falls_back_to_indexed_source(tmp_path, monkeypatch):
    source = tmp_path / "indexed.md"
    source.write_text("# Doc\n\nHello")
    monkeypatch.setattr(manifest, "get_document", lambda doc_id: None)
    monkeypatch.setattr(vector_store, "list_documents", lambda: [{
        "doc_id": "doc1",
        "file_sha": "doc1",
        "source": str(source),
        "basename": "indexed.md",
        "chunks": 1,
    }])
    monkeypatch.setattr(web, "pick_parser", lambda path: lambda p: "Hello")
    monkeypatch.setattr(web.ingestion, "ingest_text", lambda source, content, text, **kwargs: {
        "chunks": 1,
        "file_sha": "doc1",
    })

    resp = TestClient(app).post("/api/docs/doc1/reingest", json={"strategy": "sentence"})

    assert resp.status_code == 200
    assert resp.json()["source"] == str(source)


def test_dashboard_delete_collection_refuses_active(monkeypatch):
    monkeypatch.setattr(vector_store, "default_collection_name", lambda: "active")

    resp = TestClient(app).delete("/api/collections/active")

    assert resp.status_code == 400


def test_dashboard_delete_collection(monkeypatch):
    monkeypatch.setattr(vector_store, "default_collection_name", lambda: "active")
    monkeypatch.setattr(vector_store, "delete_collection", lambda name: name == "old")

    resp = TestClient(app).delete("/api/collections/old")

    assert resp.status_code == 200
    assert resp.json()["collection"] == "old"
