import os
from pathlib import Path

import pytest

from vault_bridge.config import get_vault_root
from vault_bridge.writer import (
    append_l1_today,
    write_decision,
    write_lesson,
    write_session_l2,
)


@pytest.fixture(autouse=True)
def _set_vault_root(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("STOA_VAULT_ROOT", str(vault))
    # create layer directories
    for d in ["L1-working", "L2-episodic/agent-logs", "L2-episodic/decisions", "L2-episodic/lessons"]:
        (vault / d).mkdir(parents=True, exist_ok=True)
    yield


class TestWriteSessionL2:
    def test_creates_file(self):
        result = write_session_l2("2026-06-30", [{"role": "user", "content": "Hello"}], "test-agent")
        assert Path(result["path"]).exists()

    def test_format_has_header(self):
        result = write_session_l2("2026-06-30", [{"role": "user", "content": "Hello"}], "test-agent")
        content = Path(result["path"]).read_text()
        assert "# Agent Session" in content
        assert "## Messages" in content
        assert "agent: test-agent" in content

    def test_sha_receipt_saved(self):
        result = write_session_l2("2026-06-30", [{"role": "user", "content": "Hi"}], "agent")
        receipt = Path(result["receipt_path"])
        assert receipt.exists()
        assert result["sha"] in receipt.read_text()

    def test_with_task_decisions_lessons(self):
        result = write_session_l2(
            "2026-06-30",
            [{"role": "user", "content": "Q"}],
            "agent",
            task="risk check",
            decisions="hold BTC",
            lessons="VIX was stale",
        )
        content = Path(result["path"]).read_text()
        assert "task: risk check" in content
        assert "decisions: hold BTC" in content
        assert "lessons: VIX was stale" in content

    def test_different_sessions_have_unique_paths(self):
        msgs = [{"role": "user", "content": "Q"}]
        r1 = write_session_l2("2026-06-30", msgs, "agent")
        r2 = write_session_l2("2026-06-30", [{"role": "user", "content": "Different"}], "agent")
        assert r1["path"] != r2["path"]


class TestAppendL1Today:
    def test_creates_new(self):
        path = append_l1_today("Session completed")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Session completed" in content

    def test_appends_to_existing(self):
        append_l1_today("First entry")
        append_l1_today("Second entry")
        content = Path(get_vault_root() / "L1-working" / "today.md").read_text()
        assert "First entry" in content
        assert "Second entry" in content

    def test_has_timestamp(self):
        append_l1_today("Test")
        content = Path(get_vault_root() / "L1-working" / "today.md").read_text()
        assert "##" in content


class TestWriteDecision:
    def test_creates_file(self):
        path = write_decision("2026-06-30", "Hold position", context="Market is bullish", agent="hermes")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "# Decision" in content
        assert "Hold position" in content
        assert "Market is bullish" in content


class TestWriteLesson:
    def test_creates_file(self):
        path = write_lesson("2026-06-30", "Always check VIX before opening", source="risk session")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "# Lesson" in content
        assert "Always check VIX" in content
        assert "risk session" in content
