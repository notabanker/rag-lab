import pytest
from pypdf import PdfWriter

from rag_lab import ingestion, manifest, vector_store
from rag_lab.cli import _ingest_one
from rag_lab.parsers import ParseResult, as_result, parse_markdown, parse_pdf


def _blank_pdf(path):
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        writer.write(f)


def test_quality_zero_text_warns():
    q = ParseResult(text="", section_unit="page", section_chars=[0, 0]).quality()
    assert q["total_chars"] == 0
    assert q["empty_sections"] == 2
    assert any("no text extracted" in w for w in q["warnings"])


def test_quality_low_yield_pages_warn():
    q = ParseResult(text="x" * 60, section_unit="page", section_chars=[20, 20, 20]).quality()
    assert any("scanned" in w for w in q["warnings"])


def test_quality_mostly_empty_pages_warn():
    q = ParseResult(text="x", section_unit="page", section_chars=[300, 0, 0, 0]).quality()
    assert any("contain no text" in w for w in q["warnings"])


def test_quality_healthy_page_doc_is_clean():
    q = ParseResult(text="x", section_unit="page", section_chars=[900, 1100, 850]).quality()
    assert q["warnings"] == []


def test_quality_document_unit_skips_page_heuristics():
    q = ParseResult(text="short", section_unit="document", section_chars=[5]).quality()
    assert q["warnings"] == []


def test_as_result_coerces_plain_str():
    r = as_result("Hello world")
    assert isinstance(r, ParseResult)
    assert r.text == "Hello world"
    assert r.section_chars == [11]


def test_parse_pdf_blank_page_reports_empty(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _blank_pdf(pdf)
    result = parse_pdf(str(pdf))
    q = result.quality()
    assert q["sections"] == 1
    assert q["total_chars"] == 0
    assert any("no text extracted" in w for w in q["warnings"])


def test_parse_markdown_returns_result(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Title\n\nSome real content here.")
    result = parse_markdown(str(md))
    assert isinstance(result, ParseResult)
    assert "Some real content" in result.text
    assert result.quality()["warnings"] == []


def test_manifest_round_trips_parse_report(tmp_path):
    vector_store.init_store(str(tmp_path / "db"))
    report = {"section_unit": "page", "sections": 3, "warnings": ["low text yield"]}
    manifest.log_document("a.pdf", "sha1", 5, parse_report=report)
    docs = manifest.list_documents()
    assert docs[0]["parse_report"]["warnings"] == ["low text yield"]


def test_ingest_text_empty_logs_manifest_entry(tmp_path):
    vector_store.init_store(str(tmp_path / "db"))
    quality = ParseResult(text="", section_unit="page", section_chars=[0]).quality()
    result = ingestion.ingest_text("scan.pdf", b"raw", "", parse_quality=quality)
    assert result["chunks"] == 0
    assert result["warnings"]
    doc = manifest.get_document("scan.pdf")
    assert doc["chunk_count"] == 0
    assert doc["parse_report"]["total_chars"] == 0


def test_cli_refuses_empty_pdf_by_default(tmp_path):
    vector_store.init_store(str(tmp_path / "db"))
    pdf = tmp_path / "scan.pdf"
    _blank_pdf(pdf)
    with pytest.raises(ValueError, match="No text extracted"):
        _ingest_one(str(pdf), "sentence", 512, 64, 4, False)


def test_cli_allow_empty_records_document(tmp_path):
    vector_store.init_store(str(tmp_path / "db"))
    pdf = tmp_path / "scan.pdf"
    _blank_pdf(pdf)
    result = _ingest_one(str(pdf), "sentence", 512, 64, 4, False, allow_empty=True)
    assert result["chunks"] == 0
    assert manifest.get_document("scan.pdf")["chunk_count"] == 0
