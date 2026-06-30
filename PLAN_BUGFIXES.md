# Bug Fix Plan — Prioritized by Severity

## P0 — Breaks Core Functionality (fix first, nothing works correctly without these)

### P0.1 — `_index_single_file` duplication + `delete_by_source` failure

**Files:** `indexer.py:12-62`, `watcher.py:16-65`, `vector_store.py:90-99`

**Problem:** Two identical implementations. Both call `delete_by_source(file_sha, collection)` which queries `coll.get(where={"file_sha": ...})`. The key `file_sha` is never stored in ChromaDB metadata. The indexer metadata dict has `file_path`, `file_name`, `layer`, `canonical_id`, `node_type`, `aliases`, `last_synced`, `frontmatter`. No `file_sha`. The function silently returns 0 every call. Re-indexing a file leaves old chunks orphaned.

**Fix:**
1. Extract `_index_single_file` to `rag_lab/indexer.py` as the single source of truth
2. Have `watcher.py` import it: `from .indexer import _index_single_file`
3. Fix `delete_by_source`: use ChromaDB's `coll.get(ids=[...])` by generating the expected chunk IDs from the file path hash, rather than filtering by metadata:

```python
def delete_chunks_for_file(file_path: str, vault_root: str, collection: str) -> int:
    rel = str(Path(file_path).relative_to(vault_root))
    file_sha = hashlib.sha256(rel.encode()).hexdigest()[:12]
    coll = get_collection(collection)
    try:
        existing = coll.get()
        matching = [i for i, mid in enumerate(existing["ids"]) if mid.startswith(file_sha + "-")]
        if matching:
            coll.delete(ids=[existing["ids"][i] for i in matching])
        return len(matching)
    except Exception:
        return 0
```

**Verification:** Index a file → modify it → re-index. Old chunks should be 0 (deleted), new chunks present. Check `count()` before/after.

---

### P0.2 — Retriever queries wrong collection

**File:** `retriever.py:100`

**Problem:** `vector_store.query(q_vec, top_k=top_k)` defaults to collection `"rag_lab"`. Vault indexer puts data in `l2_episodic` and `l3_semantic`. CLI `rag query` against vault content finds nothing.

**Fix:** Add `collections` parameter to `retrieve()`. Default to `["l2_episodic", "l3_semantic"]`. Use `query_multi` instead of `query`. Fall back to `["rag_lab"]` if those collections are empty (backward compatibility).

```python
def retrieve(
    question: str,
    top_k: int = 20,
    collections: list[str] | None = None,
    ...
) -> dict:
    if collections is None:
        if vector_store.count("l2_episodic") > 0 or vector_store.count("l3_semantic") > 0:
            collections = ["l2_episodic", "l3_semantic"]
        else:
            collections = ["rag_lab"]
    ...
    chunks = vector_store.query_multi(q_vec, collections=collections, top_k=top_k)[:rerank_top]
```

**Verification:** `rag index --vault ~/vault` → `rag query "Bitcoin decision"` returns vault content.

---

### P0.3 — Config import-time env reads

**File:** `config.py:15-40`

**Problem:** `PROVIDERS` dict evaluates `os.environ.get(...)` at module import. If API key is set after import, config dict has empty string. Silent 401 errors.

**Fix:** Convert `PROVIDERS` to store factory functions, not evaluated configs. Read env at call time.

```python
def _make_provider(name, base_url, model, key_env):
    return ProviderConfig(
        name=name,
        api_key=os.environ.get(key_env, "").strip(),
        base_url=base_url,
        model=model,
    )

def get_provider(name: Optional[str] = None) -> ProviderConfig:
    name = name or DEFAULT_PROVIDER
    if name not in _PROVIDER_FACTORIES:
        raise ValueError(...)
    key_env = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "tokenrouter": "TOKENROUTER_API_KEY",
        "ollama": "",
    }[name]
    return ProviderConfig(
        name=name,
        api_key=os.environ.get(key_env, "").strip() if key_env else "",
        base_url=_PROVIDER_BASE_URLS[name],
        model=_PROVIDER_MODELS[name],
    )
```

**Verification:** Start process, THEN set env var, call `get_provider()`. API key should reflect current env.

---

## P1 — Operational Hazards (works but breaks under stress or over time)

### P1.1 — `_atomic_write` file descriptor leak

**File:** `writer.py:13-20`

**Problem:** `os.fdopen(fd, "w")` can fail. If it does, `fd` (integer file descriptor) is leaked. Python eventually GCs it at process exit but under heavy use, fd exhaustion crashes the process.

**Fix:** Wrap fdopen in try/except and close fd on failure.

```python
def _atomic_write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, str(path))
    except Exception:
        os.close(fd)
        raise
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return sha
```

**Verification:** Same as existing atomic write tests — output file exists, SHA matches, no `.tmp` residue.

---

### P1.2 — Hippocampus never creates edges

**File:** `hippocampus.py:172-226`

**Problem:** `process_delta` declares `created_edges = 0` but never increments it. Entities and decisions are created as isolated nodes. The graph has no connections. Entity files show "*(no relationships)*" for every node.

**Fix:** After creating entity/decision/lesson nodes, create edges between them and relevant context nodes. At minimum, link decisions to their parent agent, and entities to the compressor that created them.

```python
# In process_delta, after each upsert_node call:
if agent_node_id:
    upsert_edge(node_id, agent_node_id, "created_by", confidence=0.9, source="compressor")

# Link new entities to existing related concepts by name similarity
# (optional, more complex — defer to P2)
```

Simplest working fix: create a "daily_compressor" agent node once, link every new node to it.

```python
compressor_id = upsert_node("daily_compressor", "agent", properties={"role": "memory compression"})
for entity in delta.get("new_entities", []):
    node_id = upsert_node(name, etype, properties=props)
    upsert_edge(node_id, compressor_id, "created_by", confidence=1.0, source="compressor")
    upsert_edge(compressor_id, node_id, "authored", confidence=1.0, source="compressor")
    created_nodes += 1
    created_edges += 2
```

**Verification:** Run `process_delta` with a delta containing new entities. Check `stats()` — edges > 0. Check entity file output — relationships section not empty.

---

### P1.3 — `append_l1_today` non-atomic

**File:** `writer.py:91-95`

**Problem:** Uses `open(path, "a")` — not atomic. If the process crashes mid-write, `today.md` is corrupted with a partial entry. Other writers use `_atomic_write`, this one doesn't.

**Fix:** Read existing content, append, write atomically.

```python
def append_l1_today(content: str, timestamp: str = None) -> str:
    vault = get_vault_root()
    today_path = vault / "L1-working" / "today.md"
    ts = timestamp or datetime.now().strftime("%H:%M:%S")
    date_header = datetime.now().strftime("%Y-%m-%d")
    entry = f"\n## {date_header} {ts}\n\n{content}\n"
    existing = today_path.read_text(encoding="utf-8") if today_path.exists() else f"# Today — {date_header}\n"
    _atomic_write(today_path, existing + entry)
    return str(today_path)
```

**Verification:** Write to today.md, kill process, check file is intact (no partial lines).

---

## P2 — Correctness Issues (produces wrong results in edge cases)

### P2.1 — Entity gen only renders inbound relationships

**File:** `entity_gen.py:46-51`

**Problem:** Outbound edges are computed but never written. Entity files show `uses <- [[hermes]]` but never `uses -> [[rag-lab]]`. Half the relationship graph is invisible.

**Fix:** Render both directions in the template.

```python
# Before (only inbound):
rel_lines = []
for e in inbound:
    rel_lines.append(f"- {e['relation_type']} <- [[{e['source_name']}]] ...")

# After (both directions):
rel_lines = []
for e in outbound:
    rel_lines.append(f"- {e['relation_type']} -> [[{e['target_name']}]] *({e.get('created_at', '')[:10]}, conf {e['confidence']:.2f})*")
for e in inbound:
    rel_lines.append(f"- {e['relation_type']} <- [[{e['source_name']}]] *({e.get('created_at', '')[:10]}, conf {e['confidence']:.2f})*")
```

**Verification:** Create two nodes with an edge between them. Generate entity file. Both directions present.

---

### P2.2 — Compressor JSON extraction fragile against backticks

**File:** `compressor.py:56-64`

**Problem:** If the LLM output contains triple backticks inside a JSON field value, the fence-stripping logic removes too much or too little. `json.loads` crashes.

**Fix:** Use regex to find the outermost `{...}` instead of manual stripping. Already done in `verifier.py:_extract_json`. Reuse that approach.

```python
import re
_JSON_RE = re.compile(r'\{[\s\S]*\}', re.DOTALL)

def _llm_synthesize(prompt: str) -> dict:
    ...
    raw = r.json()["choices"][0]["message"]["content"].strip()
    match = _JSON_RE.search(raw)
    if match:
        raw = match.group(0)
    return json.loads(raw)
```

**Verification:** Test with mock LLM response containing `"description": "use ```code``` here"`. Should parse correctly.

---

### P2.3 — CLI `index` fallback path is broken

**File:** `cli.py:104`

**Problem:** `vault_root = vault_path or vector_store._PERSIST_DIR`. Without `--vault`, defaults to `./chroma_db`. This directory contains a SQLite database, not markdown files. `rglob("*.md")` finds nothing. Result: zero files indexed, zero error.

**Fix:** Remove fallback. Require `--vault` explicitly.

```python
vault_path: str = typer.Option(..., "--vault", help="Path to Obsidian vault root"),
```

Or validate that the path actually contains markdown files and warn.

**Verification:** Run `rag index` without `--vault`. Should show error, not silent success.

---

### P2.4 — CLI `query` missing `--rerank` control

**File:** `cli.py:67-83`

**Problem:** `retrieve()` has `rerank_top: int = 5` but CLI doesn't expose it. User cannot control how many chunks reach the LLM.

**Fix:** Add `--rerank` option.

```python
rerank: int = typer.Option(5, "--rerank", help="Chunks passed to LLM after retrieval"),
```

Pass to `retrieve(question, top_k=top_k, rerank_top=rerank, ...)`.

**Verification:** `rag query "test" --rerank 10 --trace` shows 10 chunks in trace output.

---

### P2.5 — CLI `stats` shows wrong collection info

**File:** `cli.py:137-150`

**Problem:** Shows `"rag_lab"` as collection name and `count()` for default collection only. Vault indexer uses `l2_episodic` and `l3_semantic`. User sees 0 chunks and thinks nothing was indexed.

**Fix:** Show all collections.

```python
@app.command()
def stats():
    """Show vector store stats."""
    prov_cfg = get_provider()
    t = Table(title="Vector store")
    t.add_column("Metric")
    t.add_column("Value")
    t.add_row("DB path", vector_store._PERSIST_DIR)
    t.add_row("Embedder", DEFAULT_EMBED_MODEL)
    t.add_row("LLM", f"{prov_cfg.name} {prov_cfg.model}")
    t.add_row("---", "---")
    for coll_name in ["l2_episodic", "l3_semantic", "rag_lab"]:
        n = vector_store.count(coll_name)
        if n > 0:
            t.add_row(f"Collection: {coll_name}", str(n))
    console.print(t)
```

**Verification:** `rag stats` after vault index shows `l2_episodic: X chunks, l3_semantic: Y chunks`.

---

### P2.6 — `embedder.py` lock not released on exception

**File:** `embedder.py:23-25`

**Problem:** If `SentenceTransformer(model_name, device="cpu")` raises (network error, disk full), `with _lock:` exits but `get_model` returns nothing. Actually, `with _lock:` correctly releases the lock on exception — Python context managers guarantee this. **This finding is wrong. The lock IS released.** Remove from plan.

**Status:** Won't fix — the audit finding was incorrect. `with _lock:` is a context manager that releases on exception.

---

### P2.7 — `session_writer.py` inline imports

**File:** `session_writer.py:44-46`

**Problem:** `import shutil`, `from pathlib import Path`, `from .config import get_vault_root` inside function body. Adds overhead per call. `Path` is already imported at module level in writer.py (which this imports from). `shutil` is only used in the `not important` branch.

**Fix:** Move `import shutil` to top of file. The `Path` import already exists via `from .writer import ...` which imports Path internally, but that's not accessible from this module's namespace. The current code re-imports it. Just move to module level.

```python
import shutil
from pathlib import Path

from .config import get_vault_root
from .writer import append_l1_today, write_session_l2
```

**Verification:** Import still works, tests pass.

---

## P3 — Low Priority (cosmetic / unlikely to matter)

### P3.1 — `shutdown()` never called

**File:** `__init__.py:12-15`

**Problem:** `shutdown()` orchestrates cleanup but nothing calls it. No atexit hook, no signal handler.

**Fix:** Register atexit in `__init__.py`.

```python
import atexit
atexit.register(shutdown)
```

Or better: don't. ChromaDB's `PersistentClient` handles its own cleanup. The HTTP client's sockets close on process exit. This is only relevant for long-running processes that create/destroy many clients. **Defer.**

### P3.2 — `chunk_paragraph` Windows line endings

**File:** `chunker.py:67`

**Problem:** `re.finditer(r'[^\n]+(?:\n[^\n]+)*', text)` fails on `\r\n`.

**Fix:** Normalize line endings before processing.

```python
text = text.replace('\r\n', '\n').replace('\r', '\n')
```

**Verification:** Test with `"line1\r\n\r\nline2"` input. Should produce 2 paragraphs.

---

## Execution Order

| Order | Task | Est. Time | Depends On |
|---|---|---|---|
| 1 | P0.1 — Unify _index_single_file + fix delete | 45 min | — |
| 2 | P0.2 — Fix retriever collection routing | 30 min | — |
| 3 | P0.3 — Fix config lazy env reads | 30 min | — |
| 4 | P1.1 — Fix fd leak in _atomic_write | 15 min | — |
| 5 | P1.2 — Hippocampus edge creation | 30 min | — |
| 6 | P1.3 — Atomic append L1 | 15 min | — |
| 7 | P2.1 — Entity gen both directions | 15 min | P1.2 |
| 8 | P2.2 — Compressor JSON extraction | 15 min | — |
| 9 | P2.3 — CLI index require --vault | 10 min | — |
| 10 | P2.4 — CLI query --rerank | 10 min | P0.2 |
| 11 | P2.5 — CLI stats multi-collection | 15 min | — |
| 12 | P2.7 — Clean up inline imports | 5 min | — |
| 13 | P3.2 — Fix Windows line endings | 10 min | — |

**Total:** ~3.5 hours. P0 first (blocks everything else), P1 next (stability), P2 last (correctness).
