"""SQLite persistence for retrieval runs and eval runs.

The DB lives inside the ChromaDB persist directory so it travels with the index.
"""
import json
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
"""

def _db_path() -> Path:
    d = Path(vector_store.persist_dir())
    d.mkdir(parents=True, exist_ok=True)
    return d / "runs.sqlite3"

def _connect() -> sqlite3.Connection:
    # WAL + busy_timeout: the web API and CLI touch this DB concurrently
    # (run logging, health checks, manifest writes).
    conn = sqlite3.connect(_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn

def _migrate(conn: sqlite3.Connection):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "citation_validation" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN citation_validation TEXT")
        conn.commit()

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def log_run(question: str, config: dict, result: dict, latency_ms: int) -> int:
    usage = result.get("usage") or {}
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO runs (timestamp, question, config, answer, verifier, iterations,"
            " latency_ms, prompt_tokens, completion_tokens, trace, citation_validation, partial)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                _now(), question, json.dumps(config),
                result.get("answer"), json.dumps(result.get("verifier")),
                result.get("iterations"), latency_ms,
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                json.dumps(result.get("trace", [])),
                json.dumps(result.get("citation_validation") or {}),
                int(bool(result.get("partial"))),
            ),
        )
        run_id = cur.lastrowid
        for rank, c in enumerate(result.get("chunks", []), 1):
            conn.execute(
                "INSERT INTO run_chunks (run_id, chunk_id, rank, distance, source) VALUES (?,?,?,?,?)",
                (run_id, c.get("id"), rank, c.get("distance"), (c.get("metadata") or {}).get("source")),
            )
        conn.commit()
        return run_id
    finally:
        conn.close()

def list_runs(limit: int = 20) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, timestamp, question, config, verifier, iterations, latency_ms,"
            " prompt_tokens, completion_tokens, citation_validation, partial FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d["config"] or "{}")
            d["verifier"] = json.loads(d["verifier"] or "{}")
            d["citation_validation"] = json.loads(d["citation_validation"] or "{}")
            out.append(d)
        return out
    finally:
        conn.close()

def get_run(run_id: int) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for key in ("config", "verifier", "trace", "citation_validation"):
            d[key] = json.loads(d[key] or "null")
        d["chunks"] = [
            dict(c) for c in conn.execute(
                "SELECT chunk_id, rank, distance, source FROM run_chunks WHERE run_id = ? ORDER BY rank",
                (run_id,),
            ).fetchall()
        ]
        return d
    finally:
        conn.close()

def log_eval(variant: str, config: dict, summary: dict, per_question: list) -> int:
    conn = _connect()
    try:
        # Dedup: re-running the same variant+config+summary shouldn't spam the
        # history with byte-identical rows. per_question is deliberately left
        # out of the key — same summary means same outcome.
        # ponytail: summary equality as the identity, per-question diff if
        # identical-summary-but-different-questions ever matters.
        existing = conn.execute(
            "SELECT id FROM eval_runs WHERE variant = ? AND config = ? AND summary = ?"
            " ORDER BY id DESC LIMIT 1",
            (variant, json.dumps(config, sort_keys=True), json.dumps(summary, sort_keys=True)),
        ).fetchone()
        if existing:
            return existing["id"]
        cur = conn.execute(
            "INSERT INTO eval_runs (timestamp, variant, config, summary, per_question) VALUES (?,?,?,?,?)",
            (_now(), variant, json.dumps(config), json.dumps(summary), json.dumps(per_question)),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()

def list_evals(limit: int = 20) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, timestamp, variant, config, summary FROM eval_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d["config"])
            d["summary"] = json.loads(d["summary"])
            out.append(d)
        return out
    finally:
        conn.close()

def get_eval(eval_id: int) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (eval_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for key in ("config", "summary", "per_question"):
            d[key] = json.loads(d[key])
        return d
    finally:
        conn.close()
