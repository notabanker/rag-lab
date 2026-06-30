import pytest

from rag_lab.indexer import _index_single_file, index_vault
from rag_lab.vector_store import count, init_store, shutdown


@pytest.fixture(autouse=True)
def _clean_store(tmp_path):
    db = tmp_path / "chroma_db"
    db.mkdir()
    init_store(str(db))
    yield
    shutdown()


def _make_vault(tmp_path, extra_dirs=None):
    vault = tmp_path / "vault"
    (vault / "L1-working").mkdir(parents=True)
    (vault / "L2-episodic" / "agent-logs").mkdir(parents=True)
    (vault / "L3-semantic" / "entities").mkdir(parents=True)
    for d in (extra_dirs or []):
        (vault / d).mkdir(parents=True, exist_ok=True)
    return vault


class TestIndexSingleFile:
    def test_small_file_document_strategy(self, tmp_path):
        vault = _make_vault(tmp_path)
        feat = vault / "L3-semantic" / "entities" / "concept.md"
        feat.write_text("Short content.")
        result = _index_single_file(str(feat), str(vault))
        assert result is not None
        assert result["file"] == "L3-semantic/entities/concept.md"
        assert result["collection"] == "l3_semantic"
        assert result["chunks"] == 1

    def test_file_with_frontmatter(self, tmp_path):
        vault = _make_vault(tmp_path)
        feat = vault / "L3-semantic" / "entities" / "agent.md"
        feat.write_text("---\ntype: agent\naliases: [test]\n---\n\n# Agent\nSome body text.")
        result = _index_single_file(str(feat), str(vault))
        assert result is not None
        assert result["chunks"] >= 1

    def test_empty_file_skipped(self, tmp_path):
        vault = _make_vault(tmp_path)
        feat = vault / "L3-semantic" / "entities" / "empty.md"
        feat.write_text("")
        result = _index_single_file(str(feat), str(vault))
        assert result is None

    def test_whitespace_only_skipped(self, tmp_path):
        vault = _make_vault(tmp_path)
        feat = vault / "L3-semantic" / "entities" / "blank.md"
        feat.write_text("   \n\n  ")
        result = _index_single_file(str(feat), str(vault))
        assert result is None

    def test_outside_vault_skipped(self, tmp_path):
        vault = _make_vault(tmp_path)
        result = _index_single_file("/tmp/nonexistent.md", str(vault))
        assert result is None

    def test_unresolved_layer_skipped(self, tmp_path):
        vault = _make_vault(tmp_path)
        (vault / "misc").mkdir()
        feat = vault / "misc" / "file.md"
        feat.write_text("Content.")
        result = _index_single_file(str(feat), str(vault))
        assert result is None

    def test_low_importance_skipped(self, tmp_path):
        vault = _make_vault(tmp_path)
        feat = vault / "L3-semantic" / "entities" / "low.md"
        feat.write_text("---\nimportance: low\n---\n\n# Low prio\nShould be skipped.")
        result = _index_single_file(str(feat), str(vault))
        assert result is None

    def test_idempotent_reindex(self, tmp_path):
        vault = _make_vault(tmp_path)
        feat = vault / "L3-semantic" / "entities" / "concept.md"
        feat.write_text("Content.")
        _index_single_file(str(feat), str(vault))
        n1 = count("l3_semantic")
        _index_single_file(str(feat), str(vault))
        n2 = count("l3_semantic")
        assert n1 == n2


class TestIndexVault:
    def test_indexes_all_layers(self, tmp_path):
        vault = _make_vault(tmp_path)
        (vault / "L1-working" / "today.md").write_text("Today's notes.")
        (vault / "L2-episodic" / "agent-logs" / "session.md").write_text("Session log.")
        (vault / "L3-semantic" / "entities" / "concept.md").write_text("Concept body.")
        result = index_vault(str(vault))
        assert result["files_indexed"] == 2
        assert result["chunks_total"] >= 2
        assert "l2_episodic" in result["by_collection"]
        assert "l3_semantic" in result["by_collection"]

    def test_skips_excluded(self, tmp_path):
        vault = _make_vault(tmp_path)
        (vault / "personal").mkdir()
        (vault / "personal" / "secret.md").write_text("Secret.")
        (vault / "L3-semantic" / "entities" / "ok.md").write_text("Public.")
        result = index_vault(str(vault))
        assert result["files_indexed"] == 1

    def test_skips_ds_store(self, tmp_path):
        vault = _make_vault(tmp_path)
        (vault / "L3-semantic" / "entities" / "ok.md").write_text("OK.")
        (vault / ".DS_Store").write_text("")
        result = index_vault(str(vault))
        assert result["files_indexed"] == 1

    def test_empty_vault(self, tmp_path):
        vault = _make_vault(tmp_path)
        result = index_vault(str(vault))
        assert result["files_indexed"] == 0
        assert result["chunks_total"] == 0

    def test_missing_vault_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            index_vault(str(tmp_path / "nonexistent"))

    def test_metadata_stored(self, tmp_path):
        vault = _make_vault(tmp_path)
        feat = vault / "L3-semantic" / "entities" / "test.md"
        feat.write_text("---\ntype: concept\naliases: [alpha, beta]\n---\n\n# Test\nBody content.")
        index_vault(str(vault))

        from rag_lab.vector_store import query
        hits = query([0.1] * 384, top_k=1, collection="l3_semantic")
        assert len(hits) > 0
        assert hits[0]["metadata"]["node_type"] == "concept"
        assert hits[0]["metadata"]["file_name"] == "test.md"
