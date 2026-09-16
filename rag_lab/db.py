"""Shared SQLite plumbing for the runs database (runs.sqlite3 in the persist dir).

One DB file, one schema, one connect path: used by runs.py (runs/evals) and
manifest.py (documents). WAL + busy_timeout keep concurrent CLI/web writers
from hitting "database is locked".
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import vector_store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    config TEXT NOT NULL,
    answer TEXT,
    verifier TEXT,
    iterations INTEGER,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    trace TEXT,
    citation_validation TEXT,
    partial INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS run_chunks (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    chunk_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    distance REAL,
    source TEXT
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    variant TEXT NOT NULL,
    config TEXT NOT NULL,
    summary TEXT NOT NULL,
    per_question TEXT NOT NULL
);
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

# Additive migrations: (table, column) pairs appended over time.
_MIGRATIONS = [
    ("runs", "citation_validation"),
    ("documents", "parse_report"),
]


def db_path() -> Path:
    d = Path(vector_store.persist_dir())
    d.mkdir(parents=True, exist_ok=True)
    return d / "runs.sqlite3"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    for table, column in _MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
            conn.commit()
    return conn


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
