from rag_lab import ingestion
from rag_lab.chunker import Chunk


def test_build_metadatas_adds_parent_ids_and_pdf_citations():
    text = "\n\n--- Page 7 ---\n\nAlpha beta. Gamma delta."
    chunks = [Chunk(text="Alpha beta.", start=text.index("Alpha"), end=text.index("Alpha") + 11)]
    metas = ingestion.build_metadatas(
        "report.pdf",
        text,
        chunks,
        file_sha="abc123",
        strategy="sentence",
        chunk_size=512,
        overlap=64,
        parent_size=4,
    )
    assert metas[0]["doc_id"] == "abc123"
    assert metas[0]["parent_id"] == "abc123-p0"
    assert metas[0]["citation"] == "report.pdf p.7"
    assert metas[0]["embedding_model"]


def test_make_chunks_rejects_bad_strategy():
    try:
        ingestion.make_chunks("text", strategy="weird", chunk_size=10, overlap=1)
    except ValueError as e:
        assert "Unknown strategy" in str(e)
    else:
        raise AssertionError("expected ValueError")
