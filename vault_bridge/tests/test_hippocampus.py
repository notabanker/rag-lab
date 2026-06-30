import json
from pathlib import Path

import pytest

from vault_bridge.hippocampus import (
    get_all_nodes,
    get_edges_for_node,
    get_node,
    init_db,
    process_delta,
    stats,
    upsert_edge,
    upsert_node,
)


@pytest.fixture(autouse=True)
def _clean_db(tmp_path, monkeypatch):
    db_dir = tmp_path / ".stoa"
    db_dir.mkdir()
    monkeypatch.setattr("vault_bridge.hippocampus._get_db_path", lambda: db_dir / "memory_graph.db")
    init_db()
    yield


class TestUpsertNode:
    def test_create(self):
        nid = upsert_node("hermes", "agent")
        assert nid.startswith("node:")
        node = get_node(nid)
        assert node["name"] == "hermes"
        assert node["type"] == "agent"

    def test_update(self):
        upsert_node("bitcoin", "concept", properties={"price": "85k"})
        upsert_node("bitcoin", "concept", properties={"price": "90k"})
        node = get_node("node:" + _sha("bitcoin"))
        props = node["properties"]
        assert props["price"] == "90k"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid node type"):
            upsert_node("test", "invalid_type")

    def test_properties_persisted(self):
        nid = upsert_node("concept_x", "concept", properties={"desc": "test", "count": 5})
        node = get_node(nid)
        assert node["properties"]["desc"] == "test"
        assert node["properties"]["count"] == 5


class TestUpsertEdge:
    def test_create(self):
        src = upsert_node("hermes", "agent")
        tgt = upsert_node("rag-lab", "project")
        eid = upsert_edge(src, tgt, "uses", confidence=0.95)
        assert eid > 0

    def test_duplicate_replaced(self):
        src = upsert_node("a", "concept")
        tgt = upsert_node("b", "concept")
        upsert_edge(src, tgt, "similar_to", confidence=0.5)
        upsert_edge(src, tgt, "similar_to", confidence=0.9)
        edges = get_edges_for_node(src)
        similar = [e for e in edges if e["relation_type"] == "similar_to"]
        assert len(similar) == 1
        assert similar[0]["confidence"] == 0.9

    def test_invalid_relation(self):
        src = upsert_node("x", "concept")
        tgt = upsert_node("y", "concept")
        with pytest.raises(ValueError, match="Invalid relation type"):
            upsert_edge(src, tgt, "invalid_rel")


class TestGetEdges:
    def test_outbound_and_inbound(self):
        a = upsert_node("alpha", "agent")
        b = upsert_node("beta", "project")
        upsert_edge(a, b, "worked_on", confidence=0.8)

        a_edges = get_edges_for_node(a)
        out = [e for e in a_edges if e["direction"] == "outbound"]
        assert len(out) == 1
        assert out[0]["target_name"] == "beta"

        b_edges = get_edges_for_node(b)
        inc = [e for e in b_edges if e["direction"] == "inbound"]
        assert len(inc) == 1
        assert inc[0]["source_name"] == "alpha"


class TestProcessDelta:
    def test_new_entities(self):
        delta = {
            "date": "2026-06-30",
            "new_entities": [
                {"name": "quant_signals", "type": "concept", "description": "ML factor signals"},
            ],
            "updated_entities": [],
            "decisions_captured": [],
            "lessons_learned": [],
        }
        result = process_delta(delta)
        assert result["created_nodes"] == 1
        node = get_node("node:" + _sha("quant_signals"))
        assert node is not None

    def test_decisions_and_lessons(self):
        delta = {
            "date": "2026-06-30",
            "new_entities": [],
            "updated_entities": [],
            "decisions_captured": [
                {"decision": "Increase gold 5%", "rationale": "Inflation hedge", "context": "Strategy review"},
            ],
            "lessons_learned": [
                {"lesson": "Check data freshness", "source": "risk session"},
            ],
        }
        result = process_delta(delta)
        assert result["created_nodes"] == 2

    def test_update_existing(self):
        upsert_node("bitcoin", "concept", properties={"price": "85k"})
        delta = {
            "date": "2026-06-30",
            "new_entities": [],
            "updated_entities": [
                {"name": "bitcoin", "field": "price", "new_value": "90k"},
            ],
            "decisions_captured": [],
            "lessons_learned": [],
        }
        result = process_delta(delta)
        assert result["updated_nodes"] == 1
        node = get_node("node:" + _sha("bitcoin"))
        assert node["properties"]["price"] == "90k"


class TestStats:
    def test_empty(self):
        s = stats()
        assert s["nodes"] == 0
        assert s["edges"] == 0

    def test_with_data(self):
        upsert_node("a", "concept")
        upsert_node("b", "concept")
        s = stats()
        assert s["nodes"] == 2


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:20]
