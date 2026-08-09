from rag_lab import config


def test_db_path_env_override(monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", "/tmp/custom_rag_db")
    assert config.default_db_path() == "/tmp/custom_rag_db"


def test_db_path_env_expands_home(monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", "~/somewhere/db")
    resolved = config.default_db_path()
    assert not resolved.startswith("~")
    assert resolved.endswith("somewhere/db")


def test_db_path_default_is_user_data_dir(monkeypatch):
    monkeypatch.delenv("RAG_DB_PATH", raising=False)
    resolved = config.default_db_path()
    assert not resolved.startswith(".")
    assert resolved.endswith(".local/share/rag-lab/chroma_db")


def test_db_path_blank_env_falls_through(monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", "   ")
    assert config.default_db_path().endswith(".local/share/rag-lab/chroma_db")


def test_is_loopback_host_rejects_0_0_0_0():
    """0.0.0.0 bindet ALLE Interfaces — es ist KEIN Loopback. Sonst wäre
    `rag serve --host 0.0.0.0` ohne Token netzweit auth-frei."""
    assert config.is_loopback_host("0.0.0.0") is False


def test_is_loopback_host_accepts_real_loopbacks():
    for host in ("127.0.0.1", "localhost", "::1", "127.99.1.1"):
        assert config.is_loopback_host(host) is True
