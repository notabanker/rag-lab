import fnmatch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LayerConfig:
    key: str
    collection: str
    vault_subdir: str
    description: str


DEFAULT_LAYERS = [
    LayerConfig("l2", "l2_episodic", "L2-episodic", "Episodic memory"),
    LayerConfig("l3", "l3_semantic", "L3-semantic", "Semantic memory"),
]

LAYER_KEY_MAP = {lc.vault_subdir: lc for lc in DEFAULT_LAYERS}
COLLECTION_MAP = {lc.collection: lc for lc in DEFAULT_LAYERS}

DEFAULT_EXCLUDE_PATTERNS = [
    ".git/",
    ".obsidian/",
    ".DS_Store",
    ".rag-ignore",
    ".trash/",
    "personal/",
    "50_Research/",
    "60_Sessions/",
    "L2-episodic/drafts/",
    "L2-episodic/scribe/",
    "L2-episodic/fad-debates/",
]


def resolve_layer(file_path: str, vault_root: str) -> str | None:
    try:
        rel = Path(file_path).relative_to(vault_root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    first_dir = rel.parts[0]
    lc = LAYER_KEY_MAP.get(first_dir)
    return lc.collection if lc else None


def load_exclude_patterns(vault_root: str) -> list[str]:
    ignore_path = Path(vault_root) / ".rag-ignore"
    defaults = DEFAULT_EXCLUDE_PATTERNS.copy()
    if not ignore_path.exists():
        return defaults
    with open(ignore_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                defaults.append(line)
    return defaults


def should_index(file_path: str, vault_root: str, exclude_patterns: list[str]) -> bool:
    try:
        rel = Path(file_path).relative_to(vault_root)
    except ValueError:
        return False
    rel_str = str(rel)
    for pattern in exclude_patterns:
        clean = pattern.rstrip("/")
        if fnmatch.fnmatch(rel_str, clean) or fnmatch.fnmatch(rel_str, clean + "/*"):
            return False
        if any(fnmatch.fnmatch(part, clean) for part in rel.parts):
            return False
    return True
