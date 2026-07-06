import pytest

from rag_lab import gates


def test_evaluate_gates_passes_retrieval_and_skips_answer():
    loaded = {
        "retrieval": {"hit_rate": 0.9},
        "answer": {"citation_validity_rate": 0.95},
    }
    result = gates.evaluate_gates({"hit_rate": 1.0}, loaded, include_answer=False)
    assert result["passed"] is True
    assert result["checked"][0]["metric"] == "hit_rate"
    assert result["skipped"] == [{"section": "answer", "metric": "citation_validity_rate"}]


def test_evaluate_gates_fails_below_threshold():
    result = gates.evaluate_gates({"hit_rate": 0.8}, {"retrieval": {"hit_rate": 0.9}})
    assert result["passed"] is False
    assert result["failures"][0]["reason"] == "below threshold"


def test_evaluate_gates_fails_missing_metric():
    result = gates.evaluate_gates({}, {"retrieval": {"hit_rate": 0.9}})
    assert result["passed"] is False
    assert result["failures"][0]["reason"] == "metric missing"


def test_load_gates_valid(tmp_path):
    f = tmp_path / "gates.yaml"
    f.write_text("retrieval:\n  hit_rate: 0.95\nanswer:\n  verifier_mean: 8\n")
    loaded = gates.load_gates(str(f))
    assert loaded["retrieval"]["hit_rate"] == 0.95
    assert loaded["answer"]["verifier_mean"] == 8


def test_load_gates_rejects_unknown_section(tmp_path):
    f = tmp_path / "gates.yaml"
    f.write_text("speed:\n  latency: 1\n")
    with pytest.raises(ValueError, match="unknown gate section"):
        gates.load_gates(str(f))


def test_load_gates_rejects_non_numeric_threshold(tmp_path):
    f = tmp_path / "gates.yaml"
    f.write_text("retrieval:\n  hit_rate: high\n")
    with pytest.raises(ValueError, match="must be numeric"):
        gates.load_gates(str(f))
