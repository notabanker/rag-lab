import os
from pathlib import Path


def get_vault_root() -> Path:
    env = os.environ.get("STOA_VAULT_ROOT", "")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "vault"


def ensure_layer_dirs(vault_root: Path | None = None):
    root = vault_root or get_vault_root()
    dirs = [
        root / "L1-working" / "tasks",
        root / "L2-episodic" / "daily",
        root / "L2-episodic" / "agent-logs",
        root / "L2-episodic" / "decisions",
        root / "L2-episodic" / "lessons",
        root / "L3-semantic" / "identity",
        root / "L3-semantic" / "knowledge",
        root / "L3-semantic" / "entities",
        root / "L3-semantic" / "architecture",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
