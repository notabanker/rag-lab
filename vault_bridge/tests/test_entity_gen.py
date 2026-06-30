import pytest

from vault_bridge.entity_gen import generate_entity_file, regenerate_all_entities, regenerate_entity
from vault_bridge.hippocampus import init_db, upsert_edge, upsert_node


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "L3-semantic" / "entities").mkdir(parents=True)
    monkeypatch.setenv("STOA_VAULT_ROOT", str(vault))

    db_dir = tmp_path / ".stoa"
    db_dir.mkdir()
    monkeypatch.setattr("vault_bridge.hippocampus._get_db_path", lambda: db_dir / "memory_graph.db")
    init_db()


class TestGenerateEntityFile:
    def test_basic(self):
        node = {
            "id": "node:abc", "name": "hermes", "type": "agent",
            "properties": {"emotional_tone": "affirmative"}, "layer": "L3",
        }
        edges = [
            {"direction": "outbound", "relation_type": "uses", "target_name": "obscura", "confidence": 0.95, "created_at": "2026-04-25"},
            {"direction": "inbound", "relation_type": "part_of", "source_name": "Francesca", "confidence": 0.90, "created_at": "2026-04-26"},
        ]
        content = generate_entity_file(node, edges)
        assert "canonical_id: node:abc" in content
        assert "type: agent" in content
        assert "# hermes" in content
        assert "emotional_tone" in content
        assert "uses ->" not in content
        assert "part_of <-" in content

    def test_no_properties(self):
        node = {"id": "node:x", "name": "empty", "type": "concept", "properties": {}, "layer": "L3"}
        content = generate_entity_file(node, [])
        assert "no properties" in content.lower() or "*(no properties)*" in content
        assert "no relationships" in content.lower() or "*(no relationships)*" in content


class TestRegenerateAll:
    def test_writes_files(self):
        hermes = upsert_node("hermes", "agent")
        rag = upsert_node("rag-lab", "project")
        upsert_edge(hermes, rag, "uses", confidence=0.95)

        result = regenerate_all_entities()
        assert result["files_written"] >= 2
        assert "agent" in result["by_type"]
        assert "project" in result["by_type"]

    def test_regenerate_single(self):
        upsert_node("bitcoin", "concept")
        path = regenerate_entity("bitcoin")
        assert path is not None
        from pathlib import Path
        content = Path(path).read_text()
        assert "bitcoin" in content.lower()

    def test_regenerate_unknown_returns_none(self):
        assert regenerate_entity("nonexistent_entity_xyz") is None
