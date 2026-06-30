import pytest

from rag_lab.embedder import embed
from rag_lab.indexer import index_vault
from rag_lab.vector_store import init_store, query_multi, shutdown


def _setup_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    for d in [
        "L1-working",
        "L2-episodic/agent-logs",
        "L2-episodic/decisions",
        "L2-episodic/daily",
        "L3-semantic/entities",
        "L3-semantic/knowledge",
        "L3-semantic/identity",
    ]:
        (vault / d).mkdir(parents=True)

    db = tmp_path / "chroma_db"
    db.mkdir()
    init_store(str(db))
    monkeypatch.setenv("STOA_VAULT_ROOT", str(vault))

    (vault / ".rag-ignore").write_text("# test\n")

    return vault


class TestFullPipeline:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        self.vault = _setup_vault(tmp_path, monkeypatch)
        yield
        shutdown()

    def test_write_then_index_then_search(self):
        """End-to-end: write vault_bridge → index → search"""
        from vault_bridge.writer import append_l1_today, write_session_l2

        write_session_l2(
            "2026-06-30",
            [
                {"role": "user", "content": "What is the risk regime?"},
                {"role": "assistant", "content": "GREEN. VIX at 14.2. No tail risk signals. Maintain risk-on bias."}
            ],
            "hermes",
            task="risk check",
            decisions="Maintain risk-on bias",
            lessons="VIX data was stale by 2h",
        )
        append_l1_today("Hermes session: risk check completed")

        entity = self.vault / "L3-semantic" / "entities" / "bitcoin-decision.md"
        entity.write_text("""---
type: decision
aliases: [bitcoin 150k]
---
# Bitcoin Hold Decision
The strategic decision was to hold Bitcoin until 150k by June 2026.
Based on polymarket activity and macro regime analysis showing green risk conditions.
""")

        result = index_vault(str(self.vault))
        assert result["files_indexed"] >= 2
        assert "l3_semantic" in result["by_collection"]

        q_vec = embed(["Bitcoin 150k decision strategy"])[0]
        hits = query_multi(q_vec, ["l3_semantic"], top_k=3)
        assert len(hits) > 0
        assert any("Bitcoin" in h["text"] for h in hits)

        q_vec2 = embed(["risk regime VIX tail risk"])[0]
        hits2 = query_multi(q_vec2, ["l2_episodic", "l1_working"], top_k=5)
        assert len(hits2) > 0
        assert any("GREEN" in h["text"] for h in hits2)

    def test_entity_metadata(self):
        """Verify frontmatter flows through into ChromaDB metadata"""
        entity = self.vault / "L3-semantic" / "entities" / "test-agent.md"
        entity.write_text("""---
type: agent
canonical_id: node:abc123
aliases: [test, example]
last_synced: 2026-06-30T00:00:00Z
node_count_inbound: 3
---
# Test Agent
This is a test agent entity.
""")

        index_vault(str(self.vault))
        q_vec = embed(["test agent entity"])[0]
        hits = query_multi(q_vec, ["l3_semantic"], top_k=1)

        assert len(hits) > 0
        meta = hits[0]["metadata"]
        assert meta["file_name"] == "test-agent.md"
        assert meta["node_type"] == "agent"
        assert meta["canonical_id"] == "node:abc123"
        assert meta["layer"] == "l3_semantic"

    def test_exclusion_works_end_to_end(self):
        """Files in excluded dirs should not appear in search"""
        personal_dir = self.vault / "personal"
        personal_dir.mkdir()
        (personal_dir / "secret.md").write_text("Super secret content about Bitcoin 150k.")

        entity = self.vault / "L3-semantic" / "entities" / "public.md"
        entity.write_text("Public content about Ethereum.")

        index_vault(str(self.vault))
        q_vec = embed(["secret Bitcoin"])[0]
        hits = query_multi(q_vec, ["l3_semantic"], top_k=5)

        assert not any("secret" in h["text"].lower() for h in hits)

    def test_search_api_structure(self):
        """Verify search result shape matches API contract"""
        from rag_lab.web import app
        from fastapi.testclient import TestClient

        entity = self.vault / "L3-semantic" / "entities" / "search-test.md"
        entity.write_text("The Monte Carlo simulation showed 95% confidence in risk estimates.")

        index_vault(str(self.vault))

        client = TestClient(app)
        response = client.post("/api/search", json={
            "query": "Monte Carlo risk simulation",
            "layers": ["l3_semantic"],
            "top_k": 3,
        })

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert "total_chunks_searched" in data
        assert len(data["results"]) >= 1
        assert "text" in data["results"][0]
        assert "metadata" in data["results"][0]
        assert "distance" in data["results"][0]
        assert "collection" in data["results"][0]

    def test_search_empty_query_rejected(self):
        from rag_lab.web import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post("/api/search", json={"query": "", "layers": ["l3_semantic"]})
        assert response.status_code == 400


class TestPhase2Pipeline:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        self.vault = _setup_vault(tmp_path, monkeypatch)
        yield
        shutdown()

    def test_write_compress_index_search(self, monkeypatch):
        """End-to-end Phase 2: session → compress → index → search"""
        from unittest import mock
        from vault_bridge.session_writer import write_session
        from vault_bridge.compressor import run as run_compressor

        write_session(
            "2026-06-30", "hermes",
            [
                {"role": "user", "content": "What is the risk regime?"},
                {"role": "assistant", "content": "GREEN. VIX at 14.2. No tail risk signals."}
            ],
            task="risk check",
            decisions=["Maintain risk-on bias"],
            lessons=["VIX data was stale by 2h"],
        )

        mock_response = {
            "daily_summary": "## Overview\nRisk check completed. Regime GREEN.\n\n## Decisions Made\n- Maintain risk-on bias",
            "new_entities": [],
            "updated_entities": [],
            "decisions_captured": [{"decision": "Maintain risk-on bias", "rationale": "GREEN regime", "context": "Risk check"}],
            "lessons_learned": [{"lesson": "VIX data was stale by 2h", "source": "risk session"}],
        }
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")

        with mock.patch("vault_bridge.compressor._llm_synthesize", return_value=mock_response):
            comp_result = run_compressor("2026-06-30")

        assert "error" not in comp_result
        assert comp_result["summary_chars"] > 0

        index_vault(str(self.vault))

        q_vec = embed(["risk regime GREEN VIX tail"])[0]
        hits_l2 = query_multi(q_vec, ["l2_episodic"], top_k=5)
        assert len(hits_l2) > 0
        assert any("GREEN" in h["text"] for h in hits_l2)

        q_vec2 = embed(["risk-on bias decision regime"])[0]
        hits_all = query_multi(q_vec2, ["l2_episodic", "l1_working"], top_k=5)
        assert len(hits_all) > 0

    def test_compressor_output_indexed(self, monkeypatch):
        """Verify compressed output is searchable after indexing"""
        from unittest import mock
        from vault_bridge.compressor import run as run_compressor

        (self.vault / "L1-working" / "today.md").write_text("# Today\nStrategic decision: increase gold allocation by 5%.\n")
        log = self.vault / "L2-episodic" / "agent-logs" / "agent-session-2026-06-30-abc.md"
        log.write_text("# Agent Session\nDiscussed gold allocation strategy.\n")

        mock_response = {
            "daily_summary": "## Overview\nDecision: increase gold allocation by 5% based on inflation outlook.",
            "new_entities": [],
            "updated_entities": [],
            "decisions_captured": [{"decision": "Increase gold allocation 5%", "rationale": "Inflation hedge", "context": "Strategy review"}],
            "lessons_learned": [],
        }
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")

        with mock.patch("vault_bridge.compressor._llm_synthesize", return_value=mock_response):
            run_compressor("2026-06-30")

        index_vault(str(self.vault))

        q_vec = embed(["gold allocation increase strategy"])[0]
        hits = query_multi(q_vec, ["l2_episodic"], top_k=5)
        assert len(hits) > 0
        assert any("gold" in h["text"].lower() for h in hits)
