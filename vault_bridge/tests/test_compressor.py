import json
from pathlib import Path
from unittest import mock

import pytest

from vault_bridge.compressor import _llm_synthesize, run


@pytest.fixture(autouse=True)
def _setup_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("STOA_VAULT_ROOT", str(vault))
    for d in [
        "L1-working",
        "L2-episodic/agent-logs",
        "L2-episodic/daily",
        "L3-semantic/knowledge",
    ]:
        (vault / d).mkdir(parents=True, exist_ok=True)


class TestLLMSynthesize:
    def test_missing_api_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY not set"):
                _llm_synthesize("test prompt")


class TestCompressorRun:
    def test_empty_l1_returns_error(self):
        result = run("2026-06-30")
        assert "error" in result

    def test_writes_output_files(self, monkeypatch):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])

        (vault / "L1-working" / "today.md").write_text("# Today\nMorning session completed.\n")

        log = vault / "L2-episodic" / "agent-logs" / "agent-session-2099-06-30-abc.md"
        log.write_text("# Agent Session\nUser asked about risk.\n")

        mock_response = {
            "importance": 8,
            "daily_summary": "## Overview\nToday was productive.\n\n## Key Events\n- Risk check completed",
            "new_entities": [{"name": "concept_x", "type": "concept", "description": "A new concept"}],
            "updated_entities": [],
            "decisions_captured": [{"decision": "Hold position", "rationale": "Market stable", "context": "Risk check"}],
            "lessons_learned": [{"lesson": "Check data freshness", "source": "risk session"}],
        }

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")

        with mock.patch("vault_bridge.compressor._llm_synthesize", return_value=mock_response):
            result = run("2099-06-30")

        assert "error" not in result
        assert result["summary_chars"] > 0
        assert result["entities_extracted"] == 1
        assert result["importance"] == 8
        assert result["indexed"] is True
        assert "daily" in result["l2_path"]

        l2_path = Path(result["l2_path"])
        assert l2_path.exists()
        l2_content = l2_path.read_text()
        assert "Daily Summary" in l2_content
        assert "productive" in l2_content

        l3_path = Path(result["l3_path"])
        assert l3_path.exists()
        l3_data = json.loads(l3_path.read_text())
        assert l3_data["date"] == "2099-06-30"
        assert len(l3_data["decisions_captured"]) == 1

    def test_low_importance_goes_to_drafts(self, monkeypatch):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])
        (vault / "L1-working" / "today.md").write_text("# Today\nSlow day.\n")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")

        mock_response = {
            "importance": 3,
            "daily_summary": "Nothing happened.",
            "new_entities": [],
            "updated_entities": [],
            "decisions_captured": [],
            "lessons_learned": [],
        }
        with mock.patch("vault_bridge.compressor._llm_synthesize", return_value=mock_response):
            result = run("2099-06-30")

        assert not result["indexed"]
        assert "drafts" in result["l2_path"]

    def test_daily_summary_has_frontmatter(self, monkeypatch):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])
        (vault / "L1-working" / "today.md").write_text("# Today\nRisk meeting.\n")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")

        mock_response = {
            "importance": 9,
            "daily_summary": "Risk meeting outcomes.",
            "new_entities": [],
            "updated_entities": [],
            "decisions_captured": [],
            "lessons_learned": [],
        }
        with mock.patch("vault_bridge.compressor._llm_synthesize", return_value=mock_response):
            result = run("2099-06-30")

        content = Path(result["l2_path"]).read_text()
        assert "importance: 9" in content
        assert "date: 2099-06-30" in content
        assert "Daily Summary" in content

    def test_llm_failure_returns_error(self, monkeypatch):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])
        (vault / "L1-working" / "today.md").write_text("# Today\nContent.\n")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")

        with mock.patch("vault_bridge.compressor._llm_synthesize", side_effect=RuntimeError("API down")):
            result = run("2099-06-30")

        assert "error" in result
        assert "API down" in result["error"]
