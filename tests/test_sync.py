
import pytest

from rag_lab import ingestion, manifest, vector_store
from rag_lab.sync import apply_plan, build_plan


@pytest.fixture
def corpus(tmp_path):
    """Ein kleines Verzeichnis mit 2 Markdown-Dateien."""
    d = tmp_path / "notes"
    (d / "sub").mkdir(parents=True)
    (d / "a.md").write_text("# A\n\nAlpha content about LCR.")
    (d / "sub" / "b.md").write_text("# B\n\nBeta content about VaR.")
    (d / "ignore.txt").write_text("not supported")
    return d


def _init(tmp_path):
    vector_store.init_store(str(tmp_path / "db"))


def test_build_plan_classifies(tmp_path, corpus):
    docs = []
    plan = build_plan([str(corpus)], docs)
    assert set(plan.added) == {str(corpus / "a.md"), str(corpus / "sub" / "b.md")}
    assert plan.unchanged == []
    assert plan.failed == []


def test_apply_plan_ingests_added_and_skips_unchanged(tmp_path, corpus):
    _init(tmp_path)
    ingestion.ingest_file(str(corpus / "a.md"), ocr="off", chunk_size=256)
    docs = manifest.list_documents()
    plan = build_plan([str(corpus)], docs)
    counts = apply_plan(plan, ocr="off", chunk_size=256)
    assert counts["added"] == 1          # nur b.md ist neu
    assert counts["unchanged"] == 1      # a.md unverändert
    assert counts["pruned"] == 0
    assert len(manifest.list_documents()) == 2


def test_apply_plan_reingests_changed_file(tmp_path, corpus):
    _init(tmp_path)
    ingestion.ingest_file(str(corpus / "a.md"), ocr="off", chunk_size=256)
    (corpus / "a.md").write_text("# A\n\nCompletely rewritten content about LCR.")
    plan = build_plan([str(corpus)], manifest.list_documents())
    counts = apply_plan(plan, ocr="off", chunk_size=256)
    assert counts["updated"] == 1
    assert counts["added"] == 1
    shas = {d["file_sha"] for d in manifest.list_documents()}
    assert len(shas) == 2  # alte sha ersetzt, keine Orphan-Chunks


def test_apply_plan_prunes_missing_sources(tmp_path, corpus):
    _init(tmp_path)
    ingestion.ingest_file(str(corpus / "a.md"), ocr="off", chunk_size=256)
    ingestion.ingest_file(str(corpus / "sub" / "b.md"), ocr="off", chunk_size=256)
    (corpus / "a.md").unlink()
    plan = build_plan([str(corpus)], manifest.list_documents())
    assert plan.pruned == [str(corpus / "a.md")]
    counts = apply_plan(plan, ocr="off", chunk_size=256)
    assert counts["pruned"] == 1
    remaining = {d["source"] for d in manifest.list_documents()}
    assert remaining == {str(corpus / "sub" / "b.md")}


def test_apply_plan_dry_run_changes_nothing(tmp_path, corpus):
    _init(tmp_path)
    ingestion.ingest_file(str(corpus / "a.md"), ocr="off", chunk_size=256)
    plan = build_plan([str(corpus)], manifest.list_documents())
    counts = apply_plan(plan, dry_run=True, ocr="off", chunk_size=256)
    assert counts["added"] == 1 and counts["pruned"] == 0
    assert len(manifest.list_documents()) == 1  # nichts passiert


def test_build_plan_rejects_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="Not a directory"):
        build_plan([str(tmp_path / "nope")], [])
