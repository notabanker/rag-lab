# Stoa Deployment Plan

> Status: Ready | Date: 2026-06-30 | Machine: MacBook → Lenovo

---

## 1. Where Everything Lives

```
~/
├── vault/                              # Obsidian vault (existing, in-place)
│   ├── .rag-ignore                     # ← NEW: created by us
│   ├── L1-working/                     # working memory (41 files)
│   ├── L2-episodic/                    # episodic memory (1000+ files)
│   ├── L3-semantic/                    # semantic memory (589 .md files)
│   ├── L4-index/                       # ← NEW: ChromaDB persist dir
│   │   └── chroma.sqlite3              # auto-created on first index
│   └── ...
│
├── projects/
│   ├── hermes/                         # Hermes Gateway (you build)
│   │   └── ...
│   │
│   ├── rag-lab/                        # RAG system (already built)
│   │   ├── rag_lab/
│   │   │   ├── cli.py                  # rag index, rag watch, rag serve
│   │   │   ├── web.py                  # POST /api/search
│   │   │   ├── indexer.py              # batch indexer
│   │   │   ├── watcher.py              # filesystem watcher
│   │   │   └── ...
│   │   └── pyproject.toml
│   │
│   └── vault_bridge/                   # Memory pipeline (already built)
│       ├── vault_bridge/
│       │   ├── session_writer.py       # write_session()
│       │   ├── compressor.py           # run_compressor()
│       │   ├── hippocampus.py          # graph DB
│       │   ├── entity_gen.py           # Obsidian entity files
│       │   ├── writer.py               # low-level writes
│       │   └── reader.py               # vault reads
│       └── pyproject.toml
│
├── .stoa/
│   ├── config.yaml                     # ← NEW: global config
│   ├── receipts/                       # SHA-256 session receipts
│   └── memory_graph.db                 # ← NEW: entity graph (auto-created)
│
└── .hermes/
    ├── auth.json                       # API keys
    └── config.yaml                     # Hermes gateway config
```

---

## 2. Vault — Use In-Place (No Copy Needed)

The vault at `/Users/notabanker/vault` already has the correct structure:

| Layer | Directory | Files | Status |
|---|---|---|---|
| L1 | `L1-working/` | 41 files | Ready |
| L2 | `L2-episodic/` | 1000+ files | Ready |
| L3 | `L3-semantic/` | 589 .md files | Ready |
| L4 | `L4-index/` | — | Created on first `rag index` |

`.rag-ignore` already created at vault root. It excludes:
- `personal/`, `50_Research/`, `60_Sessions/` (private/noise)
- `.json`, `.bak`, `.canvas`, `.zip`, `.ics`, `.py`, `.sh`, `.txt` files
- `L2-episodic/audits/`, `sessions/`, `scribe/`, `dedup/` (raw machine output, not useful for search)
- `memory_graph.db` (the database itself, not its content)
- `icarus/` (old vector index, replaced by rag-lab)

---

## 3. Configuration

### 3.1 `~/.stoa/config.yaml`

```yaml
vault:
  path: /Users/notabanker/vault

llm:
  provider: deepseek_direct
  model: deepseek-chat
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

### 3.2 Environment variables (in `~/.zshrc` or `.env`)

```bash
export STOA_VAULT_ROOT=/Users/notabanker/vault
export DEEPSEEK_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-v1-...
```

### 3.3 vault_bridge in Hermes code

Hermes imports vault_bridge as a Python library:

```python
from vault_bridge import write_session, read_recent_sessions

# After every user conversation:
result = write_session(
    date="2026-06-30",
    agent="hermes",
    messages=session_messages,
    task="risk check",
    decisions=["Maintain risk-on bias"],
    lessons=["VIX data was stale by 2h"],
)
# result = {l2_path, l1_path, sha}

# When Hermes needs memory context:
recent = read_recent_sessions(days=3)
# recent = [{path, content, date}, ...]
```

---

## 4. Startup Sequence (MacBook)

### Step 1 — Set env vars

```bash
export STOA_VAULT_ROOT=/Users/notabanker/vault
export DEEPSEEK_API_KEY=sk-...
```

### Step 2 — Create directories that don't exist yet

```bash
mkdir -p ~/vault/L4-index
mkdir -p ~/.stoa/receipts
```

### Step 3 — First-time vault index (one-time, takes a few minutes)

```bash
cd ~/projects/rag-lab
source .venv/bin/activate

# Batch-index all existing markdown files
python -m rag_lab.cli index --vault ~/vault

# Expected output:
# Indexed X files, Y chunks
#   l1_working: N chunks
#   l2_episodic: N chunks
#   l3_semantic: N chunks
```

### Step 4 — Start the watcher (continuous)

```bash
# Terminal 1: file watcher
python -m rag_lab.cli watch --vault ~/vault
# Keeps running. Auto-indexes new/changed files.
```

### Step 5 — Start the search API

```bash
# Terminal 2: search API
python -m rag_lab.cli serve --port 8001
# → http://127.0.0.1:8001
# Verify:
curl -s -X POST http://127.0.0.1:8001/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "Bitcoin 150k", "layers": ["l3_semantic"], "top_k": 3}' | python -m json.tool
```

### Step 6 — Start Hermes (you)

```bash
# Terminal 3: Hermes Gateway
cd ~/projects/hermes
python -m hermes_cli.main gateway run --replace
# → port 8642, connects to Telegram
```

---

## 5. Verifying It Works

### Test 1: Search existing vault

```bash
curl -s -X POST http://127.0.0.1:8001/api/search \
  -d '{"query": "risk regime monitoring", "layers": ["l3_semantic"], "top_k": 3}' \
  | python -c "import json,sys; d=json.load(sys.stdin); [print(r['text'][:100], r['distance']) for r in d['results']]"
```

### Test 2: Write a session manually

```bash
python -c "
from vault_bridge import write_session
result = write_session('2026-06-30', 'test',
    [{'role': 'user', 'content': 'What is the macro regime?'},
     {'role': 'assistant', 'content': 'GOLDILOCKS. Low VIX, tight spreads.'}],
    task='regime check')
print('Wrote:', result['l2_path'])
"
# Wait 3 seconds for watcher to index, then search:
curl -s -X POST http://127.0.0.1:8001/api/search \
  -d '{"query": "GOLDILOCKS regime", "layers": ["l2_episodic"], "top_k": 3}' \
  | python -c "import json,sys; d=json.load(sys.stdin); print(len(d['results']), 'results')"
```

### Test 3: How Hermes calls search

From inside Hermes code:

```python
import httpx

def search_memory(query: str, layers: list = None, top_k: int = 5) -> list[dict]:
    """Hermes calls this when he needs context from memory."""
    r = httpx.post("http://127.0.0.1:8001/api/search", json={
        "query": query,
        "layers": layers or ["l3_semantic", "l2_episodic"],
        "top_k": top_k,
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["results"]
```

---

## 6. Cron / Automation

### Nightly compressor (runs at midnight)

```bash
# Add to crontab: crontab -e
5 0 * * * cd ~/projects/vault_bridge && ~/projects/rag-lab/.venv/bin/python -c "
from vault_bridge.compressor import run
result = run()
print(f'Compressed: {result.get(\"summary_chars\", 0)} chars')
"
```

### Entity regeneration (after compressor)

```bash
# Add to crontab: crontab -e
15 0 * * * cd ~/projects/vault_bridge && ~/projects/rag-lab/.venv/bin/python -c "
from vault_bridge.hippocampus import process_delta, stats
from vault_bridge.reader import read_l3_deltas
from vault_bridge.entity_gen import regenerate_all_entities

# Process latest delta
deltas = read_l3_deltas(days=1)
for d in deltas:
    process_delta(d)

# Regenerate entity files
result = regenerate_all_entities()
print(f'Entities: {result[\"files_written\"]} files, graph: {result[\"graph_stats\"]}')
"
```

### Full nightly pipeline

```bash
5 0 * * *  cd ~/projects/vault_bridge && ~/projects/rag-lab/.venv/bin/python -c "from vault_bridge.compressor import run; print(run())"
15 0 * * * cd ~/projects/vault_bridge && ~/projects/rag-lab/.venv/bin/python -c "from vault_bridge.hippocampus import process_delta; from vault_bridge.reader import read_l3_deltas; from vault_bridge.entity_gen import regenerate_all_entities; [process_delta(d) for d in read_l3_deltas(1)]; print(regenerate_all_entities())"
```

---

## 7. Moving to Lenovo

When you're ready to deploy to the Lenovo:

1. **Copy vault**: `rsync -av ~/vault/ lenovo:~/vault/` (exclude `.git/`, `L4-index/`, `.DS_Store`)
2. **Copy repos**: `rsync -av ~/projects/rag-lab/ lenovo:~/projects/rag-lab/` and same for vault_bridge
3. **Copy config**: `rsync -av ~/.stoa/ lenovo:~/.stoa/`
4. **Recreate venv**: `cd ~/projects/rag-lab && uv sync`
5. **Install vault_bridge**: `uv pip install -e ~/projects/vault_bridge`
6. **Re-index**: `python -m rag_lab.cli index --vault ~/vault` (L4-index doesn't transfer — rebuild on new machine)
7. **Start services**: watcher + API + Hermes

---

## 8. What Hermes Needs to Integrate

When you build Hermes, he needs two things from our stack:

### After every session:

```python
from vault_bridge import write_session

write_session(
    date=datetime.now().strftime("%Y-%m-%d"),
    agent="hermes",
    messages=conversation_history,
    task=extracted_task,
    decisions=extracted_decisions,
    lessons=extracted_lessons,
)
```

### When he needs memory context:

```python
import httpx

response = httpx.post("http://127.0.0.1:8001/api/search", json={
    "query": user_question,
    "layers": ["l3_semantic", "l2_episodic", "l1_working"],
    "top_k": 5,
})

for hit in response.json()["results"]:
    # hit["text"] → the matching paragraph
    # hit["metadata"]["file_path"] → which vault file it came from
    # hit["distance"] → similarity score (lower = better match)
    # hit["collection"] → which layer (l1_working, l2_episodic, l3_semantic)
```

That's it. Two calls. The rest is automated.
