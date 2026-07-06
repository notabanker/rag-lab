import pytest

from rag_lab.evaluation import (
    fragment_match,
    is_refusal,
    load_questions,
    load_variants,
    retrieval_metrics,
    summarize,
)


def _hit(source):
    return {"id": "x", "text": "t", "metadata": {"source": source}, "distance": 0.1}


def test_retrieval_metrics_hit_and_mrr():
    hits = [_hit("/abs/path/other.pdf"), _hit("data/markdowns/target.md"), _hit("noise.md")]
    m = retrieval_metrics(hits, ["target.md"], k=5)
    assert m["hit_at_k"] is True
    assert m["hit_at_1"] is False
    assert m["hit_at_3"] is True
    assert m["mrr"] == 0.5
    assert m["first_rank"] == 2


def test_retrieval_metrics_miss_within_k():
    hits = [_hit("a.md"), _hit("b.md"), _hit("target.md")]
    m = retrieval_metrics(hits, ["target.md"], k=2)
    assert m["hit_at_k"] is False
    assert m["mrr"] == pytest.approx(1 / 3)


def test_retrieval_metrics_no_match():
    m = retrieval_metrics([_hit("a.md")], ["target.md"], k=5)
    assert m["hit_at_k"] is False
    assert m["mrr"] == 0.0
    assert m["first_rank"] is None


def test_retrieval_metrics_any_expected_source_counts():
    hits = [_hit("second-choice.epub")]
    m = retrieval_metrics(hits, ["first-choice.pdf", "second-choice.epub"], k=1)
    assert m["hit_at_k"] is True


def test_is_refusal():
    assert is_refusal("I don't know from the provided documents.")
    assert not is_refusal("The answer is 42 [chunk-1].")


def test_fragment_match():
    assert fragment_match("The LCR and NSFR were introduced.", ["lcr", "xyz"]) is True
    assert fragment_match("Nothing relevant.", ["lcr"]) is False
    assert fragment_match("anything", []) is None


def test_summarize_mixed():
    per = [
        {"expect_refusal": False, "hit_at_k": True, "mrr": 1.0, "fragment_matched": True,
         "verifier_score": 9, "citation_valid": True, "citation_count": 2,
         "latency_ms": 100, "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        {"expect_refusal": False, "hit_at_k": False, "mrr": 0.0, "fragment_matched": None,
         "verifier_score": 4, "citation_valid": False, "citation_count": 1,
         "latency_ms": 300, "usage": {"prompt_tokens": 20, "completion_tokens": 5}},
        {"expect_refusal": True, "refusal_correct": True, "latency_ms": 200,
         "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ]
    s = summarize(per)
    assert s["questions"] == 3
    assert s["hit_rate"] == 0.5
    assert s["hit_at_1"] is None
    assert s["mrr"] == 0.5
    assert s["fragment_rate"] == 1.0  # None entries excluded
    assert s["verifier_mean"] == 6.5
    assert s["refusal_accuracy"] == 1.0
    assert s["citation_validity_rate"] == 0.5
    assert s["citation_count_mean"] == 1.5
    assert s["prompt_tokens"] == 35
    assert s["completion_tokens"] == 12


def test_summarize_retrieval_only():
    per = [{
        "expect_refusal": False, "hit_at_k": True, "hit_at_1": True,
        "hit_at_3": True, "hit_at_5": True, "hit_at_10": True,
        "mrr": 1.0, "context_hit_at_k": True, "context_chars": 123,
    }]
    s = summarize(per)
    assert s["hit_rate"] == 1.0
    assert s["hit_at_1"] == 1.0
    assert s["context_hit_rate"] == 1.0
    assert s["context_chars_mean"] == 123
    assert s["verifier_mean"] is None
    assert s["refusal_accuracy"] is None


def test_load_questions_valid(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text(
        "questions:\n"
        "  - id: q1\n"
        "    question: What is X?\n"
        "    expected_sources: [a.md]\n"
        "  - id: q2\n"
        "    question: What is Y?\n"
        "    expect_refusal: true\n"
    )
    qs = load_questions(str(f))
    assert len(qs) == 2
    assert qs[0].expected_sources == ["a.md"]
    assert qs[1].expect_refusal is True


@pytest.mark.parametrize("content,error_part", [
    ("questions: {}", "expected a top-level"),
    ("questions:\n  - question: no id\n    expected_sources: [a.md]\n", "needs both"),
    ("questions:\n  - id: q1\n    question: no sources\n", "expected_sources"),
    ("questions:\n  - id: q1\n    question: a\n    expected_sources: [x]\n"
     "  - id: q1\n    question: b\n    expected_sources: [x]\n", "duplicate"),
])
def test_load_questions_invalid(tmp_path, content, error_part):
    f = tmp_path / "q.yaml"
    f.write_text(content)
    with pytest.raises(ValueError, match=error_part):
        load_questions(str(f))


def test_load_variants(tmp_path):
    f = tmp_path / "v.yaml"
    f.write_text(
        "variants:\n"
        "  - name: a\n"
        "    top_k: 30\n"
        "    small_to_big: false\n"
        "  - name: b\n"
        "    collection: other\n"
    )
    vs = load_variants(str(f))
    assert vs[0].top_k == 30
    assert vs[0].small_to_big is False
    assert vs[0].collection is None
    assert vs[1].collection == "other"


def test_load_variants_unknown_key(tmp_path):
    f = tmp_path / "v.yaml"
    f.write_text("variants:\n  - name: a\n    reranker: true\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_variants(str(f))
