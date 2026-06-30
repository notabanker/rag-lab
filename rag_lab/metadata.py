import json
import re
from datetime import date, datetime
from pathlib import Path

import yaml

_YAML_BLOCK_RE = re.compile(r"^---\s*\n(.*?)\n?---\s*\n", re.DOTALL)

_FM_FIELD_MAP = {
    "canonical_id": "canonical_id",
    "type": "node_type",
    "aliases": "aliases",
    "last_synced": "last_synced",
}


def extract_frontmatter(text: str) -> tuple[dict, str]:
    match = _YAML_BLOCK_RE.match(text)
    if not match:
        return {}, text
    try:
        fm = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, text
    if fm is None:
        fm = {}
    if not isinstance(fm, dict):
        return {}, text
    body = text[match.end():].lstrip("\n")
    return fm, body


def frontmatter_to_metadata(fm: dict, file_path: str, layer: str) -> dict:
    meta: dict = {
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "layer": layer,
    }
    for fm_key, meta_key in _FM_FIELD_MAP.items():
        if fm_key in fm:
            val = fm[fm_key]
            if isinstance(val, list):
                val = json.dumps(val)
            elif isinstance(val, (datetime, date)):
                val = val.isoformat()
            meta[meta_key] = val
    extra = {k: v for k, v in fm.items() if k not in _FM_FIELD_MAP}
    if extra:
        meta["frontmatter"] = json.dumps(extra, default=str)
    return meta
