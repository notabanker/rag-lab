"""Persistent document manifest for corpus management."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import vector_store
from .config import CHUNKING_VERSION, EMBEDDING_MODEL

PARSER_VERSION = "parser-v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    file_sha TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    chunking_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
"""


def _db_path() -> Path:
    d = Path(vector_store._PERSIST_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d / "runs.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_document(source: str, file_sha: str, chunk_count: int):
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, source, file_sha, parser_version, chunking_version, embedding_model, chunk_count, ingested_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                file_sha,
                source,
                file_sha,
                PARSER_VERSION,
                CHUNKING_VERSION,
                EMBEDDING_MODEL,
                chunk_count,
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_documents() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM documents ORDER BY source").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_document(identifier: str) -> dict | None:
    docs = list_documents()
    for d in docs:
        if identifier in {d["doc_id"], d["file_sha"], d["source"], Path(d["source"]).name}:
            return d
    return None


def delete_document(identifier: str):
    doc = get_document(identifier)
    if doc is None:
        return
    conn = _connect()
    try:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc["doc_id"],))
        conn.commit()
    finally:
        conn.close()
