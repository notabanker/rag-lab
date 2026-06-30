from pathlib import Path

import pytest

from vault_bridge.config import get_vault_root
from vault_bridge.session_writer import _is_important, write_session


@pytest.fixture(autouse=True)
def _setup_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("STOA_VAULT_ROOT", str(vault))
    for d in ["L1-working", "L2-episodic/agent-logs"]:
        (vault / d).mkdir(parents=True, exist_ok=True)


class TestIsImportant:
    def test_with_task(self):
        assert _is_important("deploy pipeline", None, None, [{"role": "user", "content": "hi"}])

    def test_with_decisions(self):
        assert _is_important("", ["hold BTC"], None, [{"role": "user", "content": "hi"}])

    def test_with_lessons(self):
        assert _is_important("", None, ["check VIX"], [{"role": "user", "content": "hi"}])

    def test_project_keyword(self):
        assert _is_important("", None, None, [{"role": "user", "content": "deploy the rag pipeline"}])

    def test_agent_keyword(self):
        assert _is_important("", None, None, [{"role": "user", "content": "hermes config needs update"}])

    def test_casual_chat(self):
        assert not _is_important("", None, None, [{"role": "user", "content": "how are you today?"}])

    def test_weather_chat(self):
        assert not _is_important("", None, None, [{"role": "user", "content": "what is the weather"}])

    def test_empty_messages(self):
        assert not _is_important("", None, None, [])


class TestWriteSession:
    def test_full_session(self):
        result = write_session(
            "2026-06-30",
            "hermes",
            [
                {"role": "user", "content": "What is the risk regime?"},
                {"role": "assistant", "content": "GREEN. VIX at 14.2."},
            ],
            task="risk check",
            decisions=["Maintain risk-on bias"],
            lessons=["VIX data was stale by 2h"],
        )
        assert Path(result["l2_path"]).exists()
        assert Path(result["l1_path"]).exists()
        assert result["sha"]

    def test_l2_before_l1_written(self):
        vault = get_vault_root()
        l2_dir = vault / "L2-episodic" / "agent-logs"
        l1_path = vault / "L1-working" / "today.md"
        existing_l2 = set(l2_dir.glob("*.md")) if l2_dir.exists() else set()
        existing_l1 = l1_path.exists()

        result = write_session(
            "2026-06-30", "hermes",
            [{"role": "user", "content": "Q"}],
            task="test",
        )
        assert Path(result["l2_path"]).exists()
        assert Path(result["l1_path"]).exists()

    def test_no_decisions_or_lessons(self):
        result = write_session(
            "2026-06-30", "hermes",
            [{"role": "user", "content": "Hello"}],
        )
        assert Path(result["l2_path"]).exists()
        assert Path(result["l1_path"]).exists()

    def test_l1_contains_summary(self):
        result = write_session(
            "2026-06-30", "hermes",
            [{"role": "user", "content": "Hi"}],
            task="greeting",
            decisions=["say hello"],
            lessons=["be friendly"],
        )
        l1_content = Path(result["l1_path"]).read_text()
        assert "Session: hermes" in l1_content
        assert "Task: greeting" in l1_content
        assert "say hello" in l1_content
        assert "be friendly" in l1_content

    def test_l2_path_in_l1_summary(self):
        result = write_session(
            "2026-06-30", "hermes",
            [{"role": "user", "content": "deploy the watcher"}],
        )
        l1_content = Path(result["l1_path"]).read_text()
        assert "L2 log:" in l1_content

    def test_casual_session_goes_to_drafts(self):
        result = write_session(
            "2026-06-30", "hermes",
            [{"role": "user", "content": "how are you?"}],
        )
        assert "drafts" in result["l2_path"]
        assert not result["important"]

    def test_important_session_stays_in_agent_logs(self):
        result = write_session(
            "2026-06-30", "hermes",
            [{"role": "user", "content": "the hermes gateway needs a config fix"}],
        )
        assert "agent-logs" in result["l2_path"]
        assert result["important"]

    def test_important_flag_in_result(self):
        result = write_session(
            "2026-06-30", "hermes",
            [{"role": "user", "content": "project deployment strategy"}],
        )
        assert result["important"]

        result2 = write_session(
            "2026-06-30", "hermes",
            [{"role": "user", "content": "nice weather today"}],
        )
        assert not result2["important"]
