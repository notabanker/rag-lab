"""Verifier unit tests: schema validation, chunk budget, graceful failure."""
from rag_lab import verifier


def _chunks():
    return [{"id": "c0", "text": "alpha", "metadata": {"source": "a.md"}}]


def test_verify_parses_valid_response(monkeypatch):
    body = {
        "choices": [{"message": {"content": '{"score": 8.5, "grounded": true, "issues": [], "verdict": "GROUNDED"}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    monkeypatch.setattr(verifier.llm, "chat", lambda payload, **kw: body)
    out = verifier.verify("q", "a", _chunks())
    assert out["verdict"] == "GROUNDED"
    assert out["score"] == 8.5  # string scores must stay numeric, not int()-mangled
    assert out["grounded"] is True
    assert out["_usage"]["prompt_tokens"] == 10


def test_verify_accepts_fenced_json(monkeypatch):
    body = {"choices": [{"message": {"content": "```json\n{\"score\": 5, \"verdict\": \"PARTIAL\"}\n```"}}], "usage": {}}
    monkeypatch.setattr(verifier.llm, "chat", lambda payload, **kw: body)
    out = verifier.verify("q", "a", _chunks())
    assert out["verdict"] == "PARTIAL"


def test_verify_rejects_invalid_verdict(monkeypatch):
    body = {"choices": [{"message": {"content": '{"score": 9, "verdict": "MAYBE"}'}}], "usage": {}}
    monkeypatch.setattr(verifier.llm, "chat", lambda payload, **kw: body)
    out = verifier.verify("q", "a", _chunks())
    assert out["verdict"] == "ERROR"  # a bogus verdict must never reach the gate


def test_verify_rejects_non_numeric_score(monkeypatch):
    body = {"choices": [{"message": {"content": '{"score": "high", "verdict": "GROUNDED"}'}}], "usage": {}}
    monkeypatch.setattr(verifier.llm, "chat", lambda payload, **kw: body)
    out = verifier.verify("q", "a", _chunks())
    assert out["verdict"] == "ERROR"


def test_verify_budgets_context_to_chunk_cap(monkeypatch):
    captured = {}
    def fake_chat(payload, **kw):
        captured["user"] = payload["messages"][1]["content"]
        return {"choices": [{"message": {"content": '{"score": 3, "verdict": "UNGROUNDED"}'}}], "usage": {}}
    monkeypatch.setattr(verifier.llm, "chat", fake_chat)
    chunks = [{"id": "c0", "text": "x" * 3000, "metadata": {"source": "a.md"}},
              {"id": "c1", "text": "y" * 3000, "metadata": {"source": "b.md"}}]
    verifier.verify("q", "a", chunks, chunk_cap=4000)
    # Budget is shared across chunks: c1 is truncated to exactly 1000 chars.
    # (Count inside the document body — the prompt text itself contains 'y'.)
    doc2 = captured["user"].split('source="b.md">\n', 1)[1].split("\n</document>")[0]
    assert doc2 == "y" * 1000


def test_verify_returns_error_on_network_failure(monkeypatch):
    monkeypatch.setattr(verifier.llm, "chat", lambda payload, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    out = verifier.verify("q", "a", _chunks())
    assert out["verdict"] == "ERROR"
    assert "boom" not in " ".join(out["issues"])  # no exception text leakage
