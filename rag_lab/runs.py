"""SQLite persistence for retrieval runs and eval runs."""
import json

from . import db


def log_run(question: str, config: dict, result: dict, latency_ms: int) -> int:
    usage = result.get("usage") or {}
    conn = db.connect()
    try:
        cur = conn.execute(
            "INSERT INTO runs (timestamp, question, config, answer, verifier, iterations,"
            " latency_ms, prompt_tokens, completion_tokens, trace, citation_validation, partial)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                db.now(), question, json.dumps(config),
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
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT id, timestamp, question, config, verifier, iterations, latency_ms,"
            " prompt_tokens, completion_tokens, citation_validation, partial FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for key in ("config", "verifier", "citation_validation"):
                d[key] = json.loads(d[key] or "{}")
            out.append(d)
        return out
    finally:
        conn.close()


def get_run(run_id: int) -> dict | None:
    conn = db.connect()
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
    conn = db.connect()
    try:
        # Dedup: re-running the same variant+config+summary shouldn't spam the
        # history with byte-identical rows.
        existing = conn.execute(
            "SELECT id FROM eval_runs WHERE variant = ? AND config = ? AND summary = ?"
            " ORDER BY id DESC LIMIT 1",
            (variant, json.dumps(config, sort_keys=True), json.dumps(summary, sort_keys=True)),
        ).fetchone()
        if existing:
            return existing["id"]
        cur = conn.execute(
            "INSERT INTO eval_runs (timestamp, variant, config, summary, per_question) VALUES (?,?,?,?,?)",
            (db.now(), variant, json.dumps(config), json.dumps(summary), json.dumps(per_question)),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_evals(limit: int = 20) -> list[dict]:
    conn = db.connect()
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
    conn = db.connect()
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
