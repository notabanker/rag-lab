import json
from datetime import datetime, timezone

from .config import get_vault_root
from .hippocampus import get_all_nodes, get_edges_for_node, stats as graph_stats
from .writer import _atomic_write

ENTITY_TEMPLATE = """---
canonical_id: {node_id}
type: {node_type}
aliases: {aliases}
last_synced: {last_synced}
node_count_inbound: {inbound_count}
node_count_outbound: {outbound_count}
---

# {name}

> Auto-generated from `~/.stoa/memory_graph.db`. Do not edit by hand.

## Current Properties
{properties_block}

## Relationships
{relationships_block}
"""


def _generate_aliases(name: str) -> str:
    import re
    parts = re.split(r'[_\-\s]+', name.lower())
    aliases = [name.lower().replace("_", " ")]
    if len(parts) > 1:
        aliases.append("-".join(parts))
    return json.dumps(aliases)


def generate_entity_file(node: dict, edges: list[dict]) -> str:
    props = node.get("properties", {})
    props_lines = []
    for k, v in sorted(props.items()):
        if isinstance(v, str) and len(v) > 200:
            v = v[:200] + "..."
        props_lines.append(f"- **{k}**: {v} *(conf 1.00)*" if isinstance(v, (str, int, float, bool)) else f"- **{k}**: {v}")

    outbound = [e for e in edges if e["direction"] == "outbound"]
    inbound = [e for e in edges if e["direction"] == "inbound"]

    rel_lines = []
    for e in inbound:
        rel_lines.append(f"- {e['relation_type']} <- [[{e['source_name']}]] *({e.get('created_at', '')[:10]}, conf {e['confidence']:.2f})*")

    last_synced = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return ENTITY_TEMPLATE.format(
        node_id=node["id"],
        node_type=node["type"],
        aliases=_generate_aliases(node["name"]),
        last_synced=last_synced,
        inbound_count=len(inbound),
        outbound_count=len(outbound),
        name=node["name"],
        properties_block="\n".join(props_lines) if props_lines else "- *(no properties)*",
        relationships_block="\n".join(rel_lines) if rel_lines else "- *(no relationships)*",
    )


def regenerate_all_entities() -> dict:
    """
    Regenerate all entity markdown files from graph state.
    Returns {files_written, by_type: {type: count}}.
    """
    vault = get_vault_root()
    entities_dir = vault / "L3-semantic" / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)

    nodes = get_all_nodes()
    files_written = 0
    by_type: dict[str, int] = {}

    for node in nodes:
        edges = get_edges_for_node(node["id"])
        content = generate_entity_file(node, edges)
        safe_name = node["name"].lower().replace(" ", "-").replace("/", "-")[:80]
        filepath = entities_dir / f"{safe_name}.md"
        _atomic_write(filepath, content)
        files_written += 1
        by_type[node["type"]] = by_type.get(node["type"], 0) + 1

    return {"files_written": files_written, "by_type": by_type, "graph_stats": graph_stats()}


def regenerate_entity(name: str) -> str | None:
    """Regenerate a single entity file by name."""
    from .hippocampus import get_all_nodes
    nodes = get_all_nodes()
    node = next((n for n in nodes if n["name"].lower() == name.lower()), None)
    if node is None:
        return None
    edges = get_edges_for_node(node["id"])
    content = generate_entity_file(node, edges)
    safe_name = node["name"].lower().replace(" ", "-").replace("/", "-")[:80]
    vault = get_vault_root()
    filepath = vault / "L3-semantic" / "entities" / f"{safe_name}.md"
    _atomic_write(filepath, content)
    return str(filepath)


