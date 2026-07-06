import pytest

from rag_lab import reranker
from rag_lab import vector_store
from rag_lab.chunker import Chunk
from rag_lab.retriever import RetrievalConfig, select_context_chunks


def test_config_defaults():
    cfg = RetrievalConfig()
    assert cfg.mode == "hybrid"
    assert cfg.top_k == 50
    assert cfg.rerank_top == 5
    assert cfg.use_reranker is True
    assert cfg.small_to_big is True


def test_config_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        RetrievalConfig(mode="quantum")


def test_config_rejects_bad_context_limits():
    with pytest.raises(ValueError, match="must be >= 1"):
        RetrievalConfig(parent_top_k=0)


class _FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        # score = length of chunk text -> longest text wins
        return [float(len(text)) for _, text in pairs]


def test_rerank_orders_and_annotates(monkeypatch):
    monkeypatch.setattr(reranker, "get_model", lambda: _FakeCrossEncoder())
    hits = [
        {"id": "short", "text": "ab"},
        {"id": "long", "text": "abcdefghij"},
        {"id": "mid", "text": "abcde"},
    ]
    out = reranker.rerank("q", hits)
    assert [h["id"] for h in out] == ["long", "mid", "short"]
    assert all("rerank_score" in h for h in out)
    assert reranker.rerank("q", hits, top_n=2)[1]["id"] == "mid"


def test_rerank_empty():
    assert reranker.rerank("q", []) == []


def test_select_context_chunks_expands_parent(tmp_path):
    vector_store.init_store(str(tmp_path / "db"))
    chunks = [
        Chunk(text="first child", start=0, end=11),
        Chunk(text="second child", start=12, end=24),
    ]
    metas = [
        {"source": "a.md", "doc_id": "doc", "file_sha": "doc", "chunk_idx": 0, "parent_id": "doc-p0", "citation": "a.md chunk 1"},
        {"source": "a.md", "doc_id": "doc", "file_sha": "doc", "chunk_idx": 1, "parent_id": "doc-p0", "citation": "a.md chunk 1"},
    ]
    vector_store.upsert(chunks, [[1.0, 0.0], [0.9, 0.1]], metas, ["c0", "c1"])
    hits = [{"id": "c1", "text": "second child", "metadata": metas[1], "distance": 0.1}]
    out = select_context_chunks(hits, RetrievalConfig(rerank_top=1, parent_top_k=1))
    assert len(out) == 1
    assert out[0]["id"] == "doc-p0"
    assert "first child" in out[0]["text"]
    assert "second child" in out[0]["text"]
    assert out[0]["metadata"]["child_ids"] == "c0,c1"
