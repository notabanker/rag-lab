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


def test_verify_receives_original_question_not_refined(monkeypatch):
    """Regression: verify() must receive the ORIGINAL question, not the
    refined (noisy) question that accumulates meta-instructions like
    'Previous attempt was weak because: ...'.

    Bug: _retrieve passed current_q (which _refine_query appends noise to)
    to verify() instead of the original question parameter.
    """
    from rag_lab.retriever import _retrieve, RetrievalConfig

    fake_hit = {
        "id": "test-0",
        "text": "Some content about LCR and Basel III.",
        "metadata": {"source": "test.md", "citation": "test.md chunk 1"},
        "distance": 0.1,
    }
    monkeypatch.setattr("rag_lab.retriever.retrieve_hits", lambda q, cfg: [fake_hit])
    monkeypatch.setattr(
        "rag_lab.retriever._generate",
        lambda prompt, model=None, max_tokens=600: ("fake answer", {"prompt_tokens": 10, "completion_tokens": 5}),
    )

    captured = []
    def fake_verify(question, answer, chunks, model=None, chunk_cap=12000):
        captured.append(question)
        return {"score": 0, "grounded": False, "issues": ["fake low score"], "verdict": "UNGROUNDED", "_usage": {}}
    monkeypatch.setattr("rag_lab.retriever.verify", fake_verify)

    original_question = "What is LCR?"
    cfg = RetrievalConfig(max_iters=3, min_score=999)
    result = _retrieve(original_question, cfg)

    # The UNGROUNDED gate stops the loop after the first audit — the refined
    # question never gets built, so exactly one verify call must happen.
    assert len(captured) == 1, f"Expected 1 verify call, got {len(captured)}"
    assert captured[0] == original_question, (
        f"verify received {captured[0]!r}, expected {original_question!r}\n"
        "Bug: refined question with noise was passed to verify() instead of original"
    )
    # Fact-Forcing gate: a final ungrounded answer must be withheld.
    assert result["answer"] == "I don't know from the provided documents."
    assert result.get("partial") is True
    assert result["trace"][-1].get("withheld") is True


def test_keyword_mode_includes_all_matches(monkeypatch):
    """Keyword mode is documented as exhaustive enumeration: every match
    (up to top_k and the context budget) must reach the LLM — the old
    limit=rerank_top silently dropped matches 6+."""
    from rag_lab.retriever import _retrieve, RetrievalConfig
    from rag_lab import vector_store as vs

    hits = [
        {"id": f"m{i}", "text": f"match {i} about LCR.", "metadata": {"source": "a.md", "citation": f"a.md chunk {i+1}"}}
        for i in range(7)
    ]
    seen = {}
    def fake_keyword(pattern, limit=200):
        seen["limit"] = limit
        return hits
    monkeypatch.setattr(vs, "keyword_search", fake_keyword)
    monkeypatch.setattr(
        "rag_lab.retriever._generate",
        lambda prompt, model=None, max_tokens=600: ("fake answer", {"prompt_tokens": 10, "completion_tokens": 5}),
    )
    monkeypatch.setattr(
        "rag_lab.retriever.verify",
        lambda question, answer, chunks, model=None, chunk_cap=12000: {
            "score": 9, "grounded": True, "issues": [], "verdict": "GROUNDED", "_usage": {}
        },
    )

    result = _retrieve("LCR", RetrievalConfig(keyword="LCR", top_k=50))

    assert seen["limit"] == 50, f"keyword_search limit was {seen['limit']}, expected top_k (50)"
    assert len(result["chunks"]) == 7, f"got {len(result['chunks'])} chunks, expected all 7 matches"
