import json
from pathlib import Path

import pytest

from vault_bridge.reader import (
    read_identity,
    read_l3_deltas,
    read_layer,
    read_recent_sessions,
    read_today,
)


@pytest.fixture(autouse=True)
def _set_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("STOA_VAULT_ROOT", str(vault))
    for d in [
        "L1-working",
        "L2-episodic/agent-logs",
        "L2-episodic/daily",
        "L2-episodic/decisions",
        "L2-episodic/lessons",
        "L3-semantic/identity",
        "L3-semantic/knowledge",
        "L3-semantic/entities",
    ]:
        (vault / d).mkdir(parents=True, exist_ok=True)
    yield


class TestReadToday:
    def test_returns_content(self):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])
        (vault / "L1-working" / "today.md").write_text("Today's content")
        assert read_today() == "Today's content"

    def test_empty_when_missing(self):
        assert read_today() == ""


class TestReadLayer:
    def test_returns_paths(self):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])
        (vault / "L3-semantic" / "entities" / "a.md").write_text("a")
        (vault / "L3-semantic" / "entities" / "b.md").write_text("b")
        paths = read_layer("L3", "entities")
        assert len(paths) == 2

    def test_empty_for_missing_layer(self):
        paths = read_layer("L99")
        assert paths == []


class TestReadRecentSessions:
    def test_returns_recent(self):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])
        log = vault / "L2-episodic" / "agent-logs" / "agent-session-2099-06-30-abc.md"
        log.write_text("Future session")
        sessions = read_recent_sessions(days=99999)
        assert len(sessions) == 1
        assert sessions[0]["date"] == "2099-06-30"


class TestReadIdentity:
    def test_returns_dict(self):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])
        (vault / "L3-semantic" / "identity" / "profile.md").write_text("Name: Test")
        result = read_identity()
        assert "profile" in result
        assert result["profile"] == "Name: Test"


class TestReadL3Deltas:
    def test_reads_json_deltas(self):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])
        delta = vault / "L3-semantic" / "knowledge" / "l3_delta_2099-06-30.json"
        delta.write_text(json.dumps({"date": "2099-06-30", "new_entities": []}))
        deltas = read_l3_deltas(days=99999)
        assert len(deltas) == 1
        assert deltas[0]["date"] == "2099-06-30"

    def test_skips_broken_json(self):
        vault = Path(__import__("os").environ["STOA_VAULT_ROOT"])
        (vault / "L3-semantic" / "knowledge" / "l3_delta_2099-07-01.json").write_text("not json")
        deltas = read_l3_deltas(days=99999)
        assert deltas == []
