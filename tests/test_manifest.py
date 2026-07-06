from rag_lab import manifest, vector_store


def test_manifest_log_lookup_and_delete(tmp_path):
    vector_store.init_store(str(tmp_path / "db"))

    manifest.log_document("notes/module.md", "abc123", 7)

    docs = manifest.list_documents()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "abc123"
    assert docs[0]["chunk_count"] == 7
    assert manifest.get_document("abc123")["source"] == "notes/module.md"
    assert manifest.get_document("module.md")["file_sha"] == "abc123"

    manifest.delete_document("module.md")

    assert manifest.list_documents() == []
