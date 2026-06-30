import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .config import get_vault_root


def _atomic_write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return sha


def write_session_l2(
    date: str,
    messages: list[dict],
    agent: str,
    task: str = "",
    decisions: str = "",
    lessons: str = "",
) -> dict:
    now = datetime.now()
    ts = now.strftime("%H:%M:%S")
    date_str = date or now.strftime("%Y-%m-%d")

    lines = [
        f"# Agent Session — {date_str}",
        f"> agent: {agent}",
        f"> timestamp: {now.isoformat()}",
    ]
    if task:
        lines.append(f"> task: {task}")
    if decisions:
        lines.append(f"> decisions: {decisions}")
    if lessons:
        lines.append(f"> lessons: {lessons}")
    lines.append("")
    lines.append("## Messages")
    lines.append("")

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        ts_label = msg.get("timestamp", "")
        if ts_label:
            lines.append(f"### {role.title()} ({ts_label})")
        else:
            lines.append(f"### {role.title()}")
        lines.append(content)
        lines.append("")

    body = "\n".join(lines)

    vault = get_vault_root()
    l2_dir = vault / "L2-episodic" / "agent-logs"
    filename = f"agent-session-{date_str}-{hashlib.sha256(body.encode()).hexdigest()[:16]}.md"
    filepath = l2_dir / filename

    content_sha = _atomic_write(filepath, body)

    receipt_dir = Path.home() / ".stoa" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{filename}.sha256"
    _atomic_write(receipt_path, content_sha)

    return {
        "path": str(filepath),
        "sha": content_sha,
        "receipt_path": str(receipt_path),
    }


def append_l1_today(content: str, timestamp: str = None) -> str:
    vault = get_vault_root()
    today_path = vault / "L1-working" / "today.md"
    ts = timestamp or datetime.now().strftime("%H:%M:%S")
    date_header = datetime.now().strftime("%Y-%m-%d")

    entry = f"\n## {date_header} {ts}\n\n{content}\n"

    if today_path.exists():
        with open(today_path, "a", encoding="utf-8") as f:
            f.write(entry)
    else:
        _atomic_write(today_path, f"# Today — {date_header}\n{entry}")

    return str(today_path)


def write_decision(date: str, decision: str, context: str = "", agent: str = "") -> str:
    ts = datetime.now().isoformat()
    body = f"# Decision — {date}\n\n> timestamp: {ts}\n> agent: {agent}\n\n## Context\n\n{context}\n\n## Decision\n\n{decision}\n"
    vault = get_vault_root()
    sha = hashlib.sha256(body.encode()).hexdigest()[:12]
    path = vault / "L2-episodic" / "decisions" / f"decision-{date}-{sha}.md"
    _atomic_write(path, body)
    return str(path)


def write_lesson(date: str, lesson: str, source: str = "") -> str:
    ts = datetime.now().isoformat()
    body = f"# Lesson — {date}\n\n> timestamp: {ts}\n> source: {source}\n\n{lesson}\n"
    vault = get_vault_root()
    sha = hashlib.sha256(body.encode()).hexdigest()[:12]
    path = vault / "L2-episodic" / "lessons" / f"lesson-{date}-{sha}.md"
    _atomic_write(path, body)
    return str(path)
