# Stoa Rebuild — Architecture

> Status: Draft | Date: 2026-06-30 | Context: Hetzner server lost, rebuilding on MacBook → Lenovo

---

## 1. System Overview

```
                        Telegram
                           │
               ┌───────────▼───────────┐
               │   Hermes Gateway      │  port 8642
               │   (user builds)       │
               └──────┬──────┬─────────┘
                      │      │
          HTTP        │      │  filesystem (vault_bridge lib)
                      │      │
          ┌───────────▼──┐   ├──────────────────────────────┐
          │  rag-lab API  │   │                              │
          │  port 8001    │   ▼                              ▼
          │               │  ┌──────────────┐    ┌──────────────────┐
          │  search       │  │ vault_bridge │    │  vault/          │
          │  stats        │  │ (Python lib) │    │  ├── L1-working/ │
          │  management   │  └──────┬───────┘    │  ├── L2-episodic/ │
          └──────┬────────┘         │            │  ├── L3-semantic/ │☇ surviving
                 │                  │            │  └── L4-index/    │
                 │                  │            └──────────────────┘
          ┌──────▼────────┐  ┌──────▼───────┐
          │  ChromaDB     │  │ session_writer│  cron: on-demand
          │  (persistent) │  │    .py        │
          └──────▲────────┘  └──────────────┘
                 │
          ┌──────┴────────┐
          │  file_watcher  │  watchdog / inotify
          │  (auto-index)  │
          └────────────────┘
```

---

## 2. Layer Definitions

### L1 — Working Memory

| Property | Value |
|---|---|
| Path | `vault/L1-working/` |
| Files | `today.md`, `YYYY-MM-DD.md`, `tasks/*.md`, `sessions/*.md` |
| Writer | `session_writer.py` (called after every Hermes session) |
| Content | Today's interactions, active tasks, current context |
| Lifespan | ~1 day, then compressed into L2 |
| RAG collection | `l1_working` |

### L2 — Episodic Memory

| Property | Value |
|---|---|
| Path | `vault/L2-episodic/` |
| Subdirs | `daily/`, `agent-logs/`, `decisions/`, `lessons/` |
| Writer | `daily_compressor.py` (nightly cron), `session_writer.py` (agent-logs) |
| Content | Compressed daily summaries, agent session logs (SHA-256 receipt), decisions, lessons |
| Lifespan | Permanent — historical truth |
| RAG collection | `l2_episodic` |

### L3 — Semantic Memory

| Property | Value |
|---|---|
| Path | `vault/L3-semantic/` |
| Subdirs | `identity/`, `knowledge/`, `entities/`, `architecture/` |
| Writer | `daily_compressor.py` (L3 deltas), `memory_hippocampus.py` (entity extraction) |
| Content | Stable domain knowledge, entity files, architectural docs, user identity |
| Lifespan | Permanent — evolves slowly |
| RAG collection | `l3_semantic` |
| Status | ☇ Surviving from Hetzner — 505 files already present |

### L4 — Semantic Index

| Property | Value |
|---|---|
| Path | `vault/L4-index/` (was `icarus/index/`) |
| Backend | rag-lab + ChromaDB |
| Content | Vector embeddings + metadata for all layers |
| Builder | File watcher (continuous) + `indexer.py` (batch) |
| RAG collection | N/A — L4 IS the RAG layer |

---

## 3. Microservice Boundaries

| Service | Port | Transport | Lifecycle |
|---|---|---|---|
| Hermes Gateway | 8642 | HTTP + filesystem | `systemctl --user` |
| rag-lab API | 8001 | HTTP REST | `systemctl --user` |
| rag-lab Watcher | — | inotify → ChromaDB | `systemctl --user` |
| vault_bridge | — | Python import | Library (not a service) |
| session_writer | — | Cron / on-demand | Called by Hermes |
| daily_compressor | — | Cron | `systemd timer` |

Every service is independently startable. No shared state beyond filesystem + ChromaDB.

---

## 4. Port Map

```
8642  Hermes Gateway       (Telegram ↔ LLM)
8001  rag-lab REST API     (search, stats, management)
```

ChromaDB is embedded — no port. Graph DB is SQLite — no port. No overlap with any existing services.

---

## 5. Data Flow

### 5.1 Session Write

```
Hermes session ends
  │
  ▼
session_writer.write_session(session_messages, date)
  │
  ├── L2 agent-logs/agent-session-{date}-{sha}.md  (historical truth, written first)
  │      └── SHA-256 receipt saved
  │
  └── L1 today.md  (appended)
  │
  ▼
File watcher detects new/changed files
  │
  ▼
rag-lab indexes into ChromaDB (l2_episodic + l1_working collections)
```

### 5.2 Daily Compression

```
cron triggers (e.g. 00:05)
  │
  ▼
daily_compressor.run()
  │
  ├── Read L1 today.md
  ├── Read today's L2 agent-logs
  ├── deepseek_heavy() → synthesize
  │     ├── L2 daily/YYYY-MM-DD.md  (compressed day summary)
  │     └── L3 knowledge/l3_delta_YYYY-MM-DD.json  (knowledge deltas)
  │
  ▼
File watcher detects changes → rag-lab re-indexes
```

### 5.3 Query

```
User → Telegram → Hermes
  │
  ▼
Hermes needs context → POST /api/search
  {
    "query": "What was decided about Bitcoin position?",
    "layers": ["l3_semantic", "l2_episodic"],
    "top_k": 5
  }
  │
  ▼
rag-lab embeds query → cosine similarity → top-K chunks
  │
  ▼
Returns: [{text, metadata, distance, file_path, chunk_id}, ...]
  │
  ▼
Hermes injects into LLM system prompt + answers user
```

---

## 6. rag-lab Modifications

| Feature | Status | Description |
|---|---|---|
| Whole-file chunking | New | Strategy `document`: one chunk = one markdown file (entire file content) |
| Layer collections | New | `l1_working`, `l2_episodic`, `l3_semantic` collections in ChromaDB |
| YAML metadata | New | Parse `---\n...\n---` frontmatter → ChromaDB metadata |
| Exclusion patterns | New | `.rag-ignore` file with glob patterns (skip `personal/`, `50_Research/`, `60_Sessions/`) |
| File watcher | New | `rag watch` command: watchdog-based auto-indexing |
| Search endpoint | New | `POST /api/search` — lightweight top-K retrieval (no verifier loop) |
| Layer-scoped query | New | Query one or multiple collections |
| Incremental indexing | Exists | SHA-based chunk IDs already deduplicate |

### Chunking Strategy

Two modes depending on file type:

**`document`** (default for vault) — entire markdown file as one chunk. Metadata: `{file_path, layer, canonical_id, file_type, aliases, last_synced, sha}`. Works for the 26-line entity stubs up to the 443-line system-architecture doc. Within embedding context windows (~512 tokens for all-MiniLM-L6-v2 input, but sentence-transformers truncates to the model's max_seq_length which is 256 tokens = ~400 words = ~2000 chars). Files longer than ~2000 chars need **paragraph chunking**.

**`paragraph`** (fallback for long files) — split on `\n\n` boundaries. Each paragraph is a chunk. Metadata links back to parent file.

### YAML Frontmatter → Metadata Mapping

```
canonical_id → canonical_id
type         → node_type    (agent, project, topic, concept, file, user, api)
aliases      → aliases      (list, for alternative names)
last_synced  → last_synced
```

Other fields preserved as-is in a `frontmatter` metadata blob.

---

## 7. vault_bridge Design

A Python library that all agents import. Provides:

```python
# writer.py
def write_session_l2(date: str, messages: list, agent: str) -> str
    # Writes to L2-episodic/agent-logs/ with SHA-256 receipt
    # Returns file path

def append_l1_today(content: str)
    # Appends to L1-working/today.md with timestamp

def write_decision(date: str, decision: str, context: str) -> str
    # Writes to L2-episodic/decisions/

def write_lesson(date: str, lesson: str, source: str) -> str
    # Writes to L2-episodic/lessons/

# reader.py
def read_today() -> str
    # Returns L1-working/today.md contents

def read_layer(layer: str, subdir: str = None) -> list[Path]
    # Returns file paths in a layer

def read_recent_sessions(days: int = 7) -> list[str]
    # Returns recent L2 agent-log contents

def read_identity() -> dict
    # Parses all files in L3/identity/
```

### Invariants (from the old system, preserved)
1. **L2 before L1** — historical truth written first, scratchpad updated second
2. **Atomic writes** — `.tmp` + `os.replace()`, never partial files visible
3. **SHA-256 receipts** — every L2 file has verifiable provenance
4. **Truncation detection** — never produce truncated output

---

## 8. session_writer Design

Called after every Hermes session ends. Triggered by Hermes itself (not cron).

```python
# session_writer.py

def write_session(
    date: str,           # ISO date string, called once at entry
    agent: str,          # "hermes", "orchestrator", "market_analyst", etc.
    messages: list[dict], # [{role, content}, ...]
    metadata: dict = None # {task, decisions_made, lessons_extracted, ...}
) -> dict:
    """
    1. Compute SHA-256 of serialized messages
    2. Write L2 agent-logs/agent-session-{date}-{sha}.md (historical truth)
    3. Save SHA receipt to ~/.stoa/receipts/
    4. Append summary to L1 today.md
    5. Return {l2_path, sha, receipt_path}
    """
```

### L2 Session Log Format

```markdown
# Agent Session — {date}
> agent: hermes
> sha: a1b2c3d4e5
> task: {task description}
> decisions: {key decisions}
> lessons: {extracted lessons}

## Messages

### User (14:32:01)
What's the current Bitcoin outlook?

### Assistant (14:32:15)
Based on market data, Bitcoin is trading at...
```

---

## 9. daily_compressor Design

Runs nightly via cron/systemd timer.

```python
# daily_compressor.py

def run(date: str = None, llm_model: str = "deepseek-v4-pro") -> dict:
    """
    1. Read L1 today.md
    2. Read all L2 agent-logs from {date}
    3. deepseek_heavy() synthesize:
       a. L2 daily/YYYY-MM-DD.md — compressed day summary
       b. L3 knowledge/l3_delta_YYYY-MM-DD.json — knowledge deltas
    4. Optionally: extract entities → memory_hippocampus
    5. Return {l2_path, l3_path, summary_chars, entities_extracted}
    """
```

### L2 Daily Summary Format

```markdown
# Daily Summary — 2026-06-30

## Overview
...

## Key Events
- ...

## Decisions Made
- ...

## Market Context
- Regime: GOLDILOCKS
- VIX: 14.2
- ...

## Agent Activity
| Agent | Tasks | Output |
|---|---|---|
| hermes | 3 sessions | — |
| market_analyst | daily brief | sector rotation note |
```

### L3 Delta Format (JSON)

```json
{
  "date": "2026-06-30",
  "new_entities": [
    {"name": "some_new_concept", "type": "concept", "context": "..."}
  ],
  "updated_entities": [
    {"name": "bitcoin", "field": "current_status", "value": "..."}
  ],
  "new_relationships": [
    {"from": "hermes", "to": "rag-lab", "type": "uses", "confidence": 0.95}
  ],
  "decisions_captured": [
    {"decision": "...", "rationale": "...", "context": "..."}
  ],
  "lessons_learned": [
    {"lesson": "...", "source": "..."}
  ]
}
```

---

## 10. Graph DB (memory_hippocampus)

Deferred to Phase 3. Simple SQLite schema:

```sql
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,        -- canonical_id
    name TEXT NOT NULL,         -- display name
    type TEXT NOT NULL,         -- agent, concept, project, file, topic, decision
    layer TEXT,                 -- L2, L3
    properties TEXT,            -- JSON blob
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE edges (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES nodes(id),
    target_id TEXT NOT NULL REFERENCES nodes(id),
    relation_type TEXT NOT NULL,  -- uses, part_of, worked_on, depends_on, authored, serves
    confidence REAL DEFAULT 0.5,
    source TEXT,                  -- which agent/compressor created this edge
    created_at TEXT
);

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE VIRTUAL TABLE edges_fts USING fts5(source_id, target_id, relation_type);
```

Entity files in Obsidian (`entities/*.md`) are materialized views of this graph. The hippocampus processes L3 deltas → creates/updates nodes and edges → regenerates affected entity markdown files.

---

## 11. Directory Structure

```
~/
├── vault/                          # Obsidian vault (4-layer memory)
│   ├── L1-working/
│   │   ├── today.md
│   │   └── 2026-06-30.md
│   ├── L2-episodic/
│   │   ├── daily/
│   │   │   └── 2026-06-29.md
│   │   ├── agent-logs/
│   │   │   └── agent-session-2026-06-30-a1b2c3.md
│   │   ├── decisions/
│   │   └── lessons/
│   ├── L3-semantic/                # ☇ Surviving from Hetzner
│   │   ├── identity/
│   │   ├── knowledge/
│   │   ├── entities/
│   │   └── architecture/
│   ├── L4-index/                   # ChromaDB persist dir
│   └── .rag-ignore                 # Exclusion patterns
│
├── hermes/                         # Hermes Gateway (user builds)
│
├── rag-lab/                        # RAG system (this repo)
│   ├── rag_lab/
│   │   ├── cli.py                  # + watch command
│   │   ├── web.py                  # + /api/search endpoint
│   │   ├── watcher.py             # NEW: watchdog integration
│   │   ├── indexer.py             # NEW: batch indexing
│   │   └── ...
│   └── chroma_db/                  # ChromaDB persist (L4)
│
├── vault_bridge/                   # Write-side library (NEW repo)
│   ├── vault_bridge/
│   │   ├── __init__.py
│   │   ├── writer.py               # session_writer, decision_writer, lesson_writer
│   │   ├── reader.py               # vault file reader
│   │   ├── compressor.py           # daily_compressor → L2+L3 synthesis
│   │   └── hippocampus.py          # graph DB (Phase 3)
│   └── pyproject.toml
│
└── .stoa/
    ├── config.yaml                 # Global config (vault path, LLM keys, ports)
    └── receipts/                   # SHA-256 session receipts
```

---

## 12. Phase Plan

### Phase 1 — Foundation (this sprint)

| # | Task | Repo | Effort |
|---|---|---|---|
| 1.1 | rag-lab: whole-file chunking strategy | rag-lab | S |
| 1.2 | rag-lab: YAML frontmatter extraction | rag-lab | S |
| 1.3 | rag-lab: layer-aware collections (`l1_working`, `l2_episodic`, `l3_semantic`) | rag-lab | M |
| 1.4 | rag-lab: exclusion patterns (`.rag-ignore`) | rag-lab | S |
| 1.5 | rag-lab: `POST /api/search` — lightweight top-K retrieval | rag-lab | M |
| 1.6 | rag-lab: `rag watch` — filesystem watcher | rag-lab | M |
| 1.7 | rag-lab: `rag index` — batch index vault directory | rag-lab | M |
| 1.8 | vault_bridge: writer.py + reader.py (no LLM) | vault_bridge | M |
| 1.9 | Integrate: watcher auto-indexes when vault_bridge writes | both | S |

### Phase 2 — Memory Pipeline

| # | Task | Repo | Effort |
|---|---|---|---|
| 2.1 | vault_bridge: session_writer.py (L1+L2 writes) | vault_bridge | M |
| 2.2 | vault_bridge: daily_compressor.py (L2+L3 synthesis via LLM) | vault_bridge | L |
| 2.3 | Cron/systemd timers for compressor | infra | S |
| 2.4 | Test: full L1→L2→L3 write → watcher → index → search | all | M |

### Phase 3 — Entity Graph

| # | Task | Repo | Effort |
|---|---|---|---|
| 3.1 | vault_bridge: hippocampus.py (SQLite graph) | vault_bridge | L |
| 3.2 | Entity extraction from L3 deltas | vault_bridge | M |
| 3.3 | Obsidian entity file generation from graph | vault_bridge | M |

---

## 13. Configuration

### `~/.stoa/config.yaml`

```yaml
vault:
  path: /Users/notabanker/vault
  layers:
    l1: L1-working
    l2: L2-episodic
    l3: L3-semantic
  exclude:
    - personal/
    - 50_Research/
    - 60_Sessions/
    - .git/

llm:
  provider: deepseek_direct
  model: deepseek-v4-pro
  api_key_env: DEEPSEEK_API_KEY

rag:
  port: 8001
  chroma_path: /Users/notabanker/vault/L4-index
  embed_model: all-MiniLM-L6-v2

hermes:
  port: 8642

receipts:
  path: /Users/notabanker/.stoa/receipts
```

---

## 14. Open Questions

1. **Graph DB in Phase 1?** The old system's 503 entity files were auto-generated from the graph. If we start with flat markdown and no graph, entity files lose their relationships. But we can rebuild the graph later and regenerate.

2. **Whole-file vs paragraph chunking?** Default to `document` (whole-file). Fall back to `paragraph` only when file exceeds ~2000 chars. Configurable per layer.

3. **ChromaDB single-collection vs multi-collection?** Multi-collection (`l1_working`, `l2_episodic`, `l3_semantic`). Enables layer-scoped queries, different retention policies per layer, and independent re-indexing.

4. **Existing L3 semantic vault — re-index or preserve?** The 505 L3 files are valuable knowledge. We index them into `l3_semantic` collection on first run. No content changes needed.

5. **Search endpoint: verifier loop or not?** For agent-to-agent queries, we want raw top-K retrieval without the verifier/goal loop. The verifier loop stays on the human-facing `POST /api/query` endpoint. The new `POST /api/search` is machines-only.
