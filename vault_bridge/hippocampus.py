import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import get_vault_root

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    layer TEXT NOT NULL DEFAULT 'L3',
    properties TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES nodes(id),
    target_id TEXT NOT NULL REFERENCES nodes(id),
    relation_type TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(source_id, target_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
"""

_VALID_TYPES = {"agent", "project", "topic", "concept", "file", "user", "api", "decision", "lesson"}
_VALID_RELATIONS = {"uses", "part_of", "worked_on", "depends_on", "authored", "serves", "similar_to", "discussed"}

_lock = threading.Lock()


def _get_db_path() -> Path:
    return Path.home() / ".stoa" / "memory_graph.db"


def _get_conn() -> sqlite3.Connection:
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def init_db():
    with _lock:
        conn = _get_conn()
        conn.close()


def upsert_node(name: str, node_type: str, layer: str = "L3", properties: dict | None = None) -> str:
    if node_type not in _VALID_TYPES:
        raise ValueError(f"Invalid node type: {node_type}. Must be one of {_VALID_TYPES}")

    node_id = f"node:{hashlib_sha(name)}"
    now = datetime.now(timezone.utc).isoformat()
    props = json.dumps(properties or {}, default=str)

    with _lock:
        conn = _get_conn()
        try:
            existing = conn.execute("SELECT id FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE nodes SET name=?, type=?, layer=?, properties=?, updated_at=? WHERE id=?",
                    (name, node_type, layer, props, now, node_id),
                )
            else:
                conn.execute(
                    "INSERT INTO nodes (id, name, type, layer, properties, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                    (node_id, name, node_type, layer, props, now, now),
                )
            conn.commit()
        finally:
            conn.close()
    return node_id


def upsert_edge(source_id: str, target_id: str, relation_type: str, confidence: float = 0.5, source: str = "") -> int:
    if relation_type not in _VALID_RELATIONS:
        raise ValueError(f"Invalid relation type: {relation_type}. Must be one of {_VALID_RELATIONS}")

    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO edges (source_id, target_id, relation_type, confidence, source, created_at) VALUES (?,?,?,?,?,?)",
                (source_id, target_id, relation_type, confidence, source, now),
            )
            conn.commit()
            edge_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        finally:
            conn.close()
    return edge_id


def get_node(node_id: str) -> dict | None:
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT id, name, type, layer, properties, created_at, updated_at FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if not row:
                return None
            return {
                "id": row[0], "name": row[1], "type": row[2], "layer": row[3],
                "properties": json.loads(row[4]), "created_at": row[5], "updated_at": row[6],
            }
        finally:
            conn.close()


def get_edges_for_node(node_id: str) -> list[dict]:
    with _lock:
        conn = _get_conn()
        try:
            out = conn.execute(
                "SELECT e.id, e.target_id, n.name, e.relation_type, e.confidence, e.source, e.created_at FROM edges e JOIN nodes n ON e.target_id = n.id WHERE e.source_id = ?",
                (node_id,),
            ).fetchall()
            incoming = conn.execute(
                "SELECT e.id, e.source_id, n.name, e.relation_type, e.confidence, e.source, e.created_at FROM edges e JOIN nodes n ON e.source_id = n.id WHERE e.target_id = ?",
                (node_id,),
            ).fetchall()
            results = []
            for row in out:
                results.append({"id": row[0], "target_id": row[1], "target_name": row[2], "relation_type": row[3], "confidence": row[4], "source": row[5], "created_at": row[6], "direction": "outbound"})
            for row in incoming:
                results.append({"id": row[0], "source_id": row[1], "source_name": row[2], "relation_type": row[3], "confidence": row[4], "source": row[5], "created_at": row[6], "direction": "inbound"})
            return results
        finally:
            conn.close()


def get_all_nodes() -> list[dict]:
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute("SELECT id, name, type, layer, properties, created_at, updated_at FROM nodes ORDER BY name").fetchall()
            return [
                {"id": r[0], "name": r[1], "type": r[2], "layer": r[3],
                 "properties": json.loads(r[4]), "created_at": r[5], "updated_at": r[6]}
                for r in rows
            ]
        finally:
            conn.close()


def stats() -> dict:
    with _lock:
        conn = _get_conn()
        try:
            nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            return {"nodes": nodes, "edges": edges}
        finally:
            conn.close()


def process_delta(delta: dict) -> dict:
    """
    Process an L3 delta JSON into graph operations.
    Creates/updates nodes and edges from new_entities, decisions, lessons.
    Returns counts of created/updated items.
    """
    created_nodes = 0
    updated_nodes = 0
    created_edges = 0

    for entity in delta.get("new_entities", []):
        name = entity.get("name", "")
        etype = entity.get("type", "concept")
        if not name:
            continue
        props = {"description": entity.get("description", "")}
        node_id = upsert_node(name, etype, properties=props)
        created_nodes += 1

    for entity in delta.get("updated_entities", []):
        name = entity.get("name", "")
        if not name:
            continue
        node_id = f"node:{hashlib_sha(name)}"
        existing = get_node(node_id)
        if existing:
            props = existing.get("properties", {})
            field = entity.get("field", "")
            value = entity.get("new_value", "")
            if field:
                props[field] = value
            upsert_node(name, existing["type"], properties=props)
            updated_nodes += 1

    for decision in delta.get("decisions_captured", []):
        dname = decision.get("decision", "")[:80]
        if not dname:
            continue
        props = {"rationale": decision.get("rationale", ""), "context": decision.get("context", "")}
        node_id = upsert_node(dname, "decision", properties=props)
        created_nodes += 1

    for lesson in delta.get("lessons_learned", []):
        lname = lesson.get("lesson", "")[:80]
        if not lname:
            continue
        props = {"source": lesson.get("source", "")}
        node_id = upsert_node(lname, "lesson", properties=props)
        created_nodes += 1

    return {
        "created_nodes": created_nodes,
        "updated_nodes": updated_nodes,
        "created_edges": created_edges,
    }


def _hashlib_sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:20]


hashlib_sha = _hashlib_sha
