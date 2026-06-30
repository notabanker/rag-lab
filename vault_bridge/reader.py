import json
from datetime import datetime, timedelta
from pathlib import Path

from .config import get_vault_root


def read_today() -> str:
    vault = get_vault_root()
    path = vault / "L1-working" / "today.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_layer(layer: str, subdir: str | None = None) -> list[Path]:
    vault = get_vault_root()
    pattern = f"{layer}-*/"
    layer_dirs = list(vault.glob(pattern))
    if not layer_dirs:
        return []
    search_dir = layer_dirs[0]
    if subdir:
        search_dir = search_dir / subdir
    return sorted(search_dir.rglob("*.md"))


def read_recent_sessions(days: int = 7) -> list[dict]:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    files = read_layer("L2", "agent-logs")
    sessions = []
    for f in files:
        parts = f.name.split("-")
        if len(parts) >= 4:
            file_date = f"{parts[2]}-{parts[3]}-{parts[4][:2]}"
        else:
            file_date = f.name[:10]
        if file_date >= cutoff:
            sessions.append({
                "path": str(f),
                "content": f.read_text(encoding="utf-8"),
                "date": file_date,
            })
    return sessions


def read_identity() -> dict:
    files = read_layer("L3", "identity")
    result = {}
    for f in files:
        result[f.stem] = f.read_text(encoding="utf-8")
    return result


def read_l3_deltas(days: int = 30) -> list[dict]:
    vault = get_vault_root()
    knowledge_dir = vault / "L3-semantic" / "knowledge"
    if not knowledge_dir.exists():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    deltas = []
    for f in sorted(knowledge_dir.glob("l3_delta_*.json")):
        if f.name >= cutoff:
            try:
                deltas.append(json.loads(f.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
    return deltas
