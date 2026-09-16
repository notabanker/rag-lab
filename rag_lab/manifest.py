"""Persistent document manifest for corpus management (shares the runs DB)."""
import json
from pathlib import Path

from . import db
from .config import CHUNKING_VERSION, EMBEDDING_MODEL

PARSER_VERSION = "parser-v1"


def log_document(source: str, file_sha: str, chunk_count: int, parse_report: dict | None = None):
    conn = db.connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, source, file_sha, parser_version, chunking_version, embedding_model, chunk_count, ingested_at, parse_report) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                file_sha,
                source,
                file_sha,
                PARSER_VERSION,
                CHUNKING_VERSION,
                EMBEDDING_MODEL,
                chunk_count,
                db.now(),
                json.dumps(parse_report) if parse_report else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_documents() -> list[dict]:
    conn = db.connect()
    try:
        rows = conn.execute("SELECT * FROM documents ORDER BY source").fetchall()
        docs = [dict(r) for r in rows]
        for d in docs:
            d["parse_report"] = json.loads(d["parse_report"]) if d.get("parse_report") else None
        return docs
    finally:
        conn.close()


def get_document(identifier: str) -> dict | None:
    for d in list_documents():
        if identifier in {d["doc_id"], d["file_sha"], d["source"], Path(d["source"]).name}:
            return d
    return None


def delete_document(identifier: str):
    doc = get_document(identifier)
    if doc is None:
        return
    conn = db.connect()
    try:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc["doc_id"],))
        conn.commit()
    finally:
        conn.close()
