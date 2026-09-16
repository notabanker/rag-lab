import pytest
from fastapi.testclient import TestClient

from rag_lab import config, manifest, vector_store, web
from rag_lab.web import app


@pytest.fixture(autouse=True)
def _isolated_store(store, monkeypatch):
    """Every test runs against a throwaway DB, never the real default store,
    and with the API token off so tests are hermetic."""
    monkeypatch.setattr(web, "get_api_token", lambda: "")


@pytest.fixture
def client():
    # base_url sends a loopback Host header; the default "testserver" would
    # be rejected by the DNS-rebinding middleware.
    return TestClient(app, base_url="http://127.0.0.1")


def test_dashboard_config_and_stats_endpoints(client):
    cfg = client.get("/api/config")
    assert cfg.status_code == 200
    assert cfg.json()["llm_model"] == config.LLM_MODEL

    stats = client.get("/api/stats")
    assert stats.status_code == 200
    assert "collection" in stats.json()


def test_dashboard_collections_docs_runs_evals_endpoints(client):
    assert client.get("/api/collections").status_code == 200
    assert client.get("/api/docs").status_code == 200
    assert client.get("/api/runs").status_code == 200
    assert client.get("/api/evals").status_code == 200


def test_dashboard_doc_detail_includes_manifest_and_index(client, monkeypatch):
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

    resp = client.get("/api/docs/doc1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["manifest"]["doc_id"] == "doc1"
    assert body["indexed"]["chunks"] == 2


def test_dashboard_reingest_uses_manifest_source(client, tmp_path, monkeypatch):
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

    resp = client.post("/api/docs/doc1/reingest", json={"strategy": "sentence"})

    assert resp.status_code == 200
    assert resp.json()["chunks"] == 1


def test_dashboard_reingest_falls_back_to_indexed_source(client, tmp_path, monkeypatch):
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

    resp = client.post("/api/docs/doc1/reingest", json={"strategy": "sentence"})

    assert resp.status_code == 200
    assert resp.json()["source"] == str(source)


def test_dashboard_delete_collection_refuses_active(client, monkeypatch):
    monkeypatch.setattr(vector_store, "default_collection_name", lambda: "active")

    resp = client.delete("/api/collections/active")

    assert resp.status_code == 400


def test_dashboard_delete_collection(client, monkeypatch):
    monkeypatch.setattr(vector_store, "default_collection_name", lambda: "active")
    monkeypatch.setattr(vector_store, "delete_collection", lambda name: name == "old")

    resp = client.delete("/api/collections/old")

    assert resp.status_code == 200
    assert resp.json()["collection"] == "old"


def test_health_reports_ok(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_token_required_when_configured(monkeypatch):
    monkeypatch.setattr(web, "get_api_token", lambda: "secret")
    c = TestClient(app)

    assert c.get("/api/config").status_code == 401
    assert c.get("/api/config", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = c.get("/api/config", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_rebinded_host_rejected_in_no_token_mode():
    """DNS-rebinding defense: in no-token loopback mode a request whose Host
    header is not a loopback hostname must be refused."""
    c = TestClient(app, base_url="http://127.0.0.1")

    assert c.get("/api/config", headers={"Host": "evil.example.com"}).status_code == 403
    assert c.get("/api/config", headers={"Host": ""}).status_code == 403


def test_loopback_host_allowed_in_no_token_mode():
    c = TestClient(app, base_url="http://127.0.0.1")

    for host in ("127.0.0.1", "127.0.0.1:8000", "localhost", "localhost:8000", "[::1]", "[::1]:8000"):
        assert c.get("/api/config", headers={"Host": host}).status_code == 200


def test_host_header_ignored_when_token_configured(monkeypatch):
    """Token mode may serve LAN hostnames; the bearer gate is the protection."""
    monkeypatch.setattr(web, "get_api_token", lambda: "secret")
    c = TestClient(app, base_url="http://127.0.0.1")
    headers = {"Host": "192.168.1.5:8000", "Authorization": "Bearer secret"}

    assert c.get("/api/config", headers=headers).status_code == 200


def test_ingest_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 16)
    resp = client.post(
        "/api/ingest",
        files={"file": ("big.md", b"x" * 64, "text/markdown")},
    )

    assert resp.status_code == 413


def test_reingest_reports_clean_error_not_traceback(client, tmp_path, monkeypatch):
    """B1: a parser raising on a real file must surface a 400/500 detail,
    not a raw 500 traceback."""
    import rag_lab.parsers as parsers_mod
    source = tmp_path / "doc.md"
    source.write_text("x")
    monkeypatch.setattr(manifest, "get_document", lambda doc_id: {
        "doc_id": "doc1",
        "source": str(source),
        "file_sha": "doc1",
        "chunk_count": 1,
    })
    monkeypatch.setattr(parsers_mod, "pick_parser", lambda path: (_ for _ in ()).throw(ValueError("boom")))

    resp = client.post("/api/docs/doc1/reingest", json={"strategy": "sentence"})

    assert resp.status_code == 400
    assert "boom" in resp.json()["detail"]
