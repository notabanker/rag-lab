from rag_lab import citations


def _chunk(label):
    return {"id": label, "text": "ctx", "citation": label, "metadata": {"citation": label}}


def test_extract_exact_context_citations():
    labels = ["ba_fintech.pdf p.42", "risk.md chunk 1"]
    assert citations.extract("Answer [ba_fintech.pdf p.42].", labels) == ["ba_fintech.pdf p.42"]


def test_validate_valid_citation():
    v = citations.validate("Answer [ba_fintech.pdf p.42].", [_chunk("ba_fintech.pdf p.42")])
    assert v["citation_valid"] is True
    assert v["citation_count"] == 1
    assert v["citation_validity_rate"] == 1.0
    assert v["citation_errors"] == []


def test_validate_fake_citation_fails():
    v = citations.validate("Answer [fake.pdf p.9].", [_chunk("ba_fintech.pdf p.42")])
    assert v["citation_valid"] is False
    assert "citation not in context: fake.pdf p.9" in v["citation_errors"]


def test_validate_missing_citation_fails_for_non_refusal():
    v = citations.validate("Answer without source.", [_chunk("ba_fintech.pdf p.42")])
    assert v["citation_valid"] is False
    assert "answer has no citations" in v["citation_errors"]


def test_validate_refusal_does_not_require_citation():
    v = citations.validate("I don't know from the provided documents.", [_chunk("ba_fintech.pdf p.42")])
    assert v["citation_valid"] is True
    assert v["citation_count"] == 0


def test_validate_mixed_citations_reports_exact_errors():
    chunks = [_chunk("a.pdf p.1"), _chunk("b.pdf p.2")]
    v = citations.validate("Use [a.pdf p.1] and [missing.pdf p.3].", chunks)
    assert v["citation_valid"] is False
    assert v["citation_validity_rate"] == 0.5
    assert v["citations"] == ["a.pdf p.1", "missing.pdf p.3"]
