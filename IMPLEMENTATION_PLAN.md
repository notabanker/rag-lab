# Stoa Rebuild — Implementation Plan

> Status: Ready | Date: 2026-06-30 | Machine: MacBook → Lenovo

---

## Prerequisites & Constraints

- **Python:** 3.11+ (CWD: `/Users/notabanker/Projects/rag-lab`)
- **Existing repo:** `rag-lab` — already works, 30/30 tests passing, ruff clean
- **Vault:** `/Users/notabanker/vault/L3-semantic/` — 589 markdown files
  - 503 entity stubs (100% YAML frontmatter)
  - 53 knowledge files (some frontmatter, some raw)
  - 12 identity files (mixed frontmatter)
  - 1 architecture file
  - 589 files total across all subdirs
  - File sizes: 531 < 2KB, 21 2–5KB, 7 5–10KB, 30 > 10KB
- **Embed model:** `all-MiniLM-L6-v2` — max_seq_length 256 tokens (~1000 chars before truncation)
- **LLM:** DeepSeek V4 Pro Direct API
- **ChromaDB:** PersistentClient, cosine distance

### Chunking threshold

After measurement: `all-MiniLM-L6-v2` truncates input at 256 tokens (~1000–1200 chars). Files ≤ 1000 chars → whole-file chunk. Files > 1000 chars → paragraph chunking. This means 531 files are one-chunk-per-file; 58 files need paragraph splitting.

---

## Phase 1 — Foundation (rag-lab + vault_bridge)

### T1 — Whole-file chunking strategy

**File:** `rag_lab/chunker.py`

**Subtask 1.1 — Add `chunk_document` function**

```python
def chunk_document(text: str) -> list[Chunk]:
    """Whole-file chunk — one Chunk per document."""
    if not text.strip():
        return []
    return [Chunk(text=text.strip(), start=0, end=len(text))]
```

- Returns exactly one `Chunk` for non-empty text
- Returns empty list for empty/whitespace-only text
- Register as `"document"` in `STRATEGIES` dict

**Subtask 1.2 — Add `chunk_paragraph` function**

```python
def chunk_paragraph(text: str, max_chars: int = 1000, overlap_paragraphs: int = 1) -> list[Chunk]:
    """
    Split text on \\n\\n boundaries, group paragraphs into chunks ≤ max_chars.
    overlap_paragraphs: number of previous paragraphs to carry forward as context.
    """
```

Algorithm:
1. Split raw text on `\n\n+` into paragraphs
2. Walk paragraphs, accumulating into a buffer
3. When buffer length exceeds `max_chars`, emit a Chunk
4. Carry forward `overlap_paragraphs` previous paragraphs as context overlap
5. Final partial buffer emits as last Chunk
6. Track `start`/`end` offsets in original text (scan with `str.find`)

- Returns empty list for empty text
- Register as `"paragraph"` in `STRATEGIES` dict

**Subtask 1.3 — Update `chunk()` dispatch**

Add `"document"` and `"paragraph"` to `STRATEGIES`. No signature changes needed — `**kwargs` already passes through.

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_document_basic` | `"Hello world\n\nFoo bar"` | 1 Chunk, text = full trimmed text |
| `test_document_empty` | `""` | 0 Chunks |
| `test_document_whitespace_only` | `"   \n  "` | 0 Chunks |
| `test_paragraph_basic` | `"A.\n\nB.\n\nC."` max=10, overlap=0 | 2+ Chunks, no paragraph split across chunks |
| `test_paragraph_overlap` | `"P1.\n\nP2.\n\nP3."` max=6, overlap=1 | P1 appears in chunk[1] as context |
| `test_paragraph_single` | `"Only one paragraph."` max=1000 | 1 Chunk |
| `test_paragraph_empty` | `""` | 0 Chunks |
| `test_paragraph_offsets_monotonic` | 3-paragraph text, max=10 | start/end values non-decreasing |
| `test_dispatch_document` | `chunk("text", strategy="document")` | returns list of Chunk |
| `test_dispatch_paragraph` | `chunk("A.\n\nB.", strategy="paragraph")` | returns list of Chunk |

**Verification:**
```bash
python -m pytest tests/test_chunker.py -v -k "document or paragraph"
```

---

### T2 — YAML frontmatter extraction

**Files:** `rag_lab/parsers/markdown.py`, new file `rag_lab/metadata.py`

**Subtask 2.1 — Extract YAML frontmatter as dict**

Create `rag_lab/metadata.py`:

```python
import re
import yaml  # already in deps via other packages? Check. If not, use regex-only parser.

_YAML_BLOCK_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

def extract_frontmatter(text: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from markdown text.
    Returns (frontmatter_dict, body_text).
    If no frontmatter found, returns ({}, text).
    """
    match = _YAML_BLOCK_RE.match(text)
    if not match:
        return {}, text
    try:
        fm = yaml.safe_load(match.group(1))
        body = text[match.end():].lstrip("\n")
        return fm if isinstance(fm, dict) else {}, body
    except yaml.YAMLError:
        return {}, text
```

**Subtask 2.2 — Map frontmatter to ChromaDB metadata fields**

```python
_FM_FIELD_MAP = {
    "canonical_id": "canonical_id",
    "type": "node_type",
    "aliases": "aliases",
    "last_synced": "last_synced",
}

def frontmatter_to_metadata(fm: dict, file_path: str, layer: str) -> dict:
    """
    Convert YAML frontmatter dict to ChromaDB-compatible metadata.
    Always includes: file_path, file_name, layer.
    Maps known fields from _FM_FIELD_MAP.
    Preserves all other frontmatter fields in a 'frontmatter' JSON string.
    """
    meta = {
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "layer": layer,
    }
    for fm_key, meta_key in _FM_FIELD_MAP.items():
        if fm_key in fm:
            val = fm[fm_key]
            # ChromaDB requires primitive types. Convert lists to strings.
            if isinstance(val, list):
                val = json.dumps(val)
            meta[meta_key] = val
    # Store everything else
    extra = {k: v for k, v in fm.items() if k not in _FM_FIELD_MAP}
    if extra:
        meta["frontmatter"] = json.dumps(extra, default=str)
    return meta
```

**Subtask 2.3 — Wire into markdown parser**

Update `parse_markdown()` in `rag_lab/parsers/markdown.py` to return both frontmatter and body, OR create a new function `parse_markdown_with_metadata()` that returns `(text, frontmatter_dict)`. Do NOT break the existing `parse_markdown()` signature — it's used by `pick_parser()` and the ingest pipeline.

**Decision:** Add `extract_frontmatter` to `metadata.py`. Call it AFTER parsing in the ingestion/indexing pipeline. Keep `parse_markdown()` unchanged.

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_extract_with_frontmatter` | `"---\nkey: val\n---\n\n# Body"` | `{"key": "val"}, "# Body"` |
| `test_extract_list_aliases` | `"---\naliases: [a, b]\n---\n\nBody"` | `{"aliases": ["a", "b"]}, "Body"` |
| `test_extract_no_frontmatter` | `"# Just a heading\n\nBody"` | `{}, "# Just a heading\n\nBody"` |
| `test_extract_invalid_yaml` | `"---\nbad: [unclosed\n---\n\nBody"` | `{}, original text` |
| `test_extract_not_dict` | `"---\n- list item\n---\n\nBody"` | `{}, original text` |
| `test_extract_empty_frontmatter` | `"---\n---\n\nBody"` | `{}, "Body"` |
| `test_map_canonical_id` | fm with `canonical_id: node:abc` | meta `{"canonical_id": "node:abc", ...}` |
| `test_map_node_type` | fm with `type: agent` | meta `{"node_type": "agent", ...}` |
| `test_map_aliases_list` | fm with `aliases: [hermes, gateway]` | meta `{"aliases": '["hermes", "gateway"]', ...}` |
| `test_map_extra_fields` | fm with `node_count_inbound: 5` | meta has `"frontmatter": '{"node_count_inbound": 5, ...}'` |
| `test_map_no_frontmatter` | empty dict | meta has `file_path`, `file_name`, `layer` only |

**Verification:**
```bash
python -m pytest tests/test_metadata.py -v
```

---

### T3 — Layer-aware collections

**Files:** `rag_lab/vector_store.py`, `rag_lab/config.py`, new `rag_lab/vault_config.py`

**Subtask 3.1 — Multi-collection support**

Currently `vector_store.py` has a single `_COLLECTION` and `get_collection()`. Refactor:

```python
_COLLECTIONS: dict[str, object] = {}  # name → ChromaDB Collection

def get_collection(name: str = "rag_lab") -> object:
    """Get or create a named collection."""
    if name not in _COLLECTIONS:
        with _lock:
            if name not in _COLLECTIONS:
                if _CLIENT is None:
                    init_client()
                _COLLECTIONS[name] = _CLIENT.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )
    return _COLLECTIONS[name]
```

**Subtask 3.2 — Layer ↔ Collection mapping**

```python
# vault_config.py

from pathlib import Path

@dataclass
class LayerConfig:
    name: str              # "l1_working"
    collection: str         # ChromaDB collection name
    vault_subdir: str       # "L1-working"
    description: str

DEFAULT_LAYERS = [
    LayerConfig("l1", "l1_working", "L1-working", "Working memory"),
    LayerConfig("l2", "l2_episodic", "L2-episodic", "Episodic memory"),
    LayerConfig("l3", "l3_semantic", "L3-semantic", "Semantic memory"),
]

def resolve_layer(file_path: str, vault_root: str) -> str | None:
    """Given a file path inside vault, return layer key (l1/l2/l3) or None."""
    rel = Path(file_path).relative_to(vault_root)
    first_dir = rel.parts[0] if rel.parts else ""
    for layer in DEFAULT_LAYERS:
        if first_dir == layer.vault_subdir:
            return layer.collection
    return None
```

**Subtask 3.3 — Update upsert to accept collection name**

```python
def upsert(chunks, embeddings, metadatas, ids, collection: str = "rag_lab"):
    coll = get_collection(collection)
    coll.upsert(embeddings=embeddings, documents=[c.text for c in chunks],
                metadatas=metadatas, ids=ids)

def count(collection: str = "rag_lab") -> int:
    return get_collection(collection).count()

def query(query_embedding, top_k=20, collection: str = "rag_lab") -> list[dict]:
    # use get_collection(collection)

def delete_by_source(file_sha, collection: str = "rag_lab") -> int:
    # use get_collection(collection)
```

Accept `collection` as optional parameter everywhere with default `"rag_lab"` — preserves backward compatibility with existing CLI.

**Subtask 3.4 — Multi-collection query**

New function:

```python
def query_multi(query_embedding, collections: list[str], top_k: int = 20) -> list[dict]:
    """
    Query multiple collections, merge and sort by distance.
    Returns top_k results across all collections.
    """
    all_hits = []
    for coll_name in collections:
        hits = query(query_embedding, top_k=top_k, collection=coll_name)
        for h in hits:
            h["collection"] = coll_name
        all_hits.extend(hits)
    all_hits.sort(key=lambda h: h.get("distance", float("inf")))
    return all_hits[:top_k]
```

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_get_collection_creates` | `get_collection("test_coll")` | Returns collection, count=0 |
| `test_get_collection_reuses` | Two calls with same name | Returns same object |
| `test_upsert_to_named_collection` | upsert to "test_coll" | count("test_coll") > 0, count() unchanged |
| `test_query_named_collection` | query "test_coll" after upsert | Returns hits |
| `test_query_multi_collections` | query_multi across 2 collections | Results from both, sorted by distance |
| `test_resolve_layer_l1` | file "vault/L1-working/foo.md" | Returns "l1_working" |
| `test_resolve_layer_l3` | file "vault/L3-semantic/entities/x.md" | Returns "l3_semantic" |
| `test_resolve_layer_none` | file "vault/other/foo.md" | Returns None |

**Verification:**
```bash
python -m pytest tests/test_vector_store.py -v
```

---

### T4 — Exclusion patterns

**File:** `rag_lab/vault_config.py`

**Subtask 4.1 — `.rag-ignore` parser**

```python
def load_exclude_patterns(vault_root: str) -> list[str]:
    """
    Read .rag-ignore from vault root.
    Returns list of glob patterns. Lines starting with # are comments.
    Default patterns if file doesn't exist.
    """
    ignore_path = Path(vault_root) / ".rag-ignore"
    defaults = [".git/", ".obsidian/", ".DS_Store", ".rag-ignore"]
    if not ignore_path.exists():
        return defaults
    patterns = defaults.copy()
    with open(ignore_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns
```

**Subtask 4.2 — File filter**

```python
def should_index(file_path: str, vault_root: str, exclude_patterns: list[str]) -> bool:
    """Return True if file should be indexed."""
    try:
        rel = Path(file_path).relative_to(vault_root)
    except ValueError:
        return False
    rel_str = str(rel)
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(rel_str, pattern + "/*"):
            return False
    return True
```

**Subtask 4.3 — Write default `.rag-ignore`**

```
# Stoa RAG — exclusion patterns
# Lines starting with # are comments
# Glob patterns relative to vault root

.git/
.obsidian/
.DS_Store
.rag-ignore
.trash/
personal/
50_Research/
60_Sessions/
```

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_load_defaults_no_file` | No .rag-ignore | Returns default patterns |
| `test_load_custom_patterns` | .rag-ignore with `personal/` | defaults + `personal/` |
| `test_load_comment_skip` | .rag-ignore with `# comment` | Comment line excluded |
| `test_should_index_markdown` | `vault/L3-semantic/entities/x.md` | True |
| `test_should_index_excluded` | `vault/personal/secret.md` | False |
| `test_should_index_git` | `vault/.git/config` | False |
| `test_should_index_ds_store` | `vault/.DS_Store` | False |

---

### T5 — Batch indexer (`rag index`)

**File:** `rag_lab/indexer.py` (new), update `rag_lab/cli.py`

**Subtask 5.1 — `index_vault` function**

```python
def index_vault(
    vault_root: str,
    collections: list[str] | None = None,
    chunk_strategy: str = "auto",  # "auto" → document if ≤1000 chars else paragraph
    force: bool = False,
    embed_model: str = None,
) -> dict:
    """
    Walk vault directory, index all markdown files into ChromaDB.
    
    1. Load exclude patterns from .rag-ignore
    2. Walk vault for *.md files
    3. For each file:
       a. Check should_index()
       b. Resolve layer → collection
       c. Read file, parse frontmatter
       d. Choose chunk strategy (document vs paragraph based on text length)
       e. Chunk text
       f. Generate metadata per chunk
       g. Generate chunk IDs: {sha256_filepath}_{chunk_idx}
       h. Embed chunks
       i. Upsert into layer collection
    4. Return {"files_indexed": N, "chunks_total": N, "by_collection": {...}}
    """
```

**Subtask 5.2 — Chunk ID scheme for vault files**

Use SHA-256 of relative file path (not file content — file paths are stable, content changes):

```python
file_sha = hashlib.sha256(str(rel_path).encode()).hexdigest()[:12]
chunk_ids = [f"{file_sha}-{i}" for i in range(len(chunks))]
```

Using file path hash means: re-indexing replaces chunks for that file. Same file at same path = same IDs = upsert replaces old.

**Subtask 5.3 — CLI command**

```python
@app.command()
def index(
    vault_path: str = typer.Option(None, "--vault", help="Path to Obsidian vault root"),
    collections: str = typer.Option(None, "--collections", help="Comma-separated collections (default: all)"),
    strategy: str = typer.Option("auto", "--strategy", help="auto | document | paragraph"),
    force: bool = typer.Option(False, "--force", help="Re-index all files even if unchanged"),
    embed_model: str = typer.Option(None, "--embed-model"),
):
    """Batch-index all markdown files from a vault directory."""
    vault_root = vault_path or vector_store._PERSIST_DIR
    # ... call index_vault()
```

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_index_vault_creates_collections` | Vault with 3 layers | l1_working, l2_episodic, l3_semantic exist |
| `test_index_vault_skips_excluded` | Vault with personal/ | Files in personal/ not indexed |
| `test_index_vault_small_file` | 200-char file | 1 chunk, strategy=document |
| `test_index_vault_large_file` | 5000-char file | Multiple chunks, strategy=paragraph |
| `test_index_vault_metadata` | File with frontmatter | canonical_id, node_type in ChromaDB metadata |
| `test_index_vault_no_frontmatter` | File without frontmatter | file_path, file_name, layer in metadata |
| `test_index_vault_idempotent` | Index same vault twice | Same chunk count, no duplicates |
| `test_index_vault_empty_dir` | Empty vault | 0 files, 0 chunks, no error |

**Verification:**
```bash
# Create test vault
mkdir -p /tmp/test_vault/L3-semantic/entities
echo '---\ntype: agent\n---\n\n# Test' > /tmp/test_vault/L3-semantic/entities/test.md
python -m rag_lab.cli index --vault /tmp/test_vault
python -m rag_lab.cli stats
```

---

### T6 — Filesystem watcher (`rag watch`)

**File:** `rag_lab/watcher.py` (new), dependency: `watchdog` (add to pyproject.toml)

**Subtask 6.1 — Watchdog event handler**

```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class VaultIndexHandler(FileSystemEventHandler):
    def __init__(self, vault_root: str, embed_model: str = None, debounce_sec: float = 2.0):
        self.vault_root = Path(vault_root)
        self.embed_model = embed_model
        self.debounce_sec = debounce_sec
        self._pending: dict[str, float] = {}  # path → first_seen_time
        self._exclude = load_exclude_patterns(vault_root)
    
    def on_created(self, event):
        self._schedule(event.src_path)
    
    def on_modified(self, event):
        self._schedule(event.src_path)
    
    def _schedule(self, path: str):
        if not path.endswith('.md'):
            return
        if not should_index(path, str(self.vault_root), self._exclude):
            return
        self._pending[path] = time.time()
    
    def process_pending(self):
        """Called on a timer — indexes files that have been stable for debounce_sec."""
        now = time.time()
        ready = []
        for path, first_seen in list(self._pending.items()):
            if now - first_seen >= self.debounce_sec:
                ready.append(path)
                del self._pending[path]
        for path in ready:
            try:
                _index_single_file(path, str(self.vault_root), self.embed_model)
            except Exception as e:
                logger.error(f"Failed to index {path}: {e}")
```

**Subtask 6.2 — Single file indexing**

```python
def _index_single_file(file_path: str, vault_root: str, embed_model: str = None):
    """
    Index one markdown file into the appropriate collection.
    Handles: read → frontmatter → chunk → embed → upsert.
    Removes old chunks for this file first.
    """
    fp = Path(file_path)
    rel = fp.relative_to(vault_root)
    raw = fp.read_text(encoding="utf-8")
    
    # Resolve collection
    collection = resolve_layer(file_path, vault_root)
    if collection is None:
        logger.warning(f"Cannot resolve layer for {file_path}, skipping")
        return
    
    # Parse frontmatter
    fm, body = extract_frontmatter(raw)
    
    # Choose strategy
    if len(body) <= 1000:
        chunks = chunker.chunk(body, strategy="document")
    else:
        chunks = chunker.chunk(body, strategy="paragraph", max_chars=1000, overlap_paragraphs=1)
    
    if not chunks:
        return
    
    # Metadata
    file_sha = hashlib.sha256(str(rel).encode()).hexdigest()[:12]
    meta_base = frontmatter_to_metadata(fm, str(rel), collection)
    
    metadatas = []
    ids = []
    for i, ch in enumerate(chunks):
        m = dict(meta_base)
        m["chunk_idx"] = i
        m["chunk_count"] = len(chunks)
        metadatas.append(m)
        ids.append(f"{file_sha}-{i}")
    
    # Delete old, upsert new
    vector_store.delete_by_source(file_sha, collection=collection)
    vecs = embedder.embed([c.text for c in chunks], model_name=embed_model)
    vector_store.upsert(chunks, vecs, metadatas, ids, collection=collection)
    
    logger.info(f"Indexed {file_path} → {collection} ({len(chunks)} chunks)")
```

**Subtask 6.3 — `_index_single_file` should handle file deletion**

Add `on_deleted` to handler:

```python
def on_deleted(self, event):
    if not event.src_path.endswith('.md'):
        return
    fp = Path(event.src_path)
    rel = str(fp.relative_to(self.vault_root))
    file_sha = hashlib.sha256(rel.encode()).hexdigest()[:12]
    collection = resolve_layer(event.src_path, str(self.vault_root))
    if collection:
        n = vector_store.delete_by_source(file_sha, collection=collection)
        logger.info(f"Removed {n} chunks for deleted file {event.src_path}")
```

**Subtask 6.4 — CLI command**

```python
@app.command()
def watch(
    vault_path: str = typer.Option(..., "--vault", help="Path to Obsidian vault root"),
    debounce: float = typer.Option(2.0, "--debounce", help="Seconds to wait before indexing changed file"),
    embed_model: str = typer.Option(None, "--embed-model"),
):
    """Watch vault for changes and auto-index markdown files."""
    from .watcher import VaultIndexHandler
    import time
    from watchdog.observers import Observer
    
    vault_root = str(Path(vault_path).resolve())
    handler = VaultIndexHandler(vault_root, embed_model=embed_model, debounce_sec=debounce)
    observer = Observer()
    observer.schedule(handler, vault_root, recursive=True)
    observer.start()
    
    console.print(f"[green]Watching {vault_root} for changes...[/green]")
    try:
        while True:
            time.sleep(1)
            handler.process_pending()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

**Subtask 6.5 — Add `watchdog` to dependencies**

Add to `pyproject.toml`:
```toml
"watchdog>=6.0",
```

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_single_file_index_new` | Write new .md file | File appears in ChromaDB |
| `test_single_file_index_modified` | Modify existing .md file | Chunks updated in ChromaDB |
| `test_single_file_index_deleted` | Delete .md file | Chunks removed from ChromaDB |
| `test_single_file_index_excluded` | Write file in personal/ | NOT in ChromaDB |
| `test_handler_debounce` | Rapid modify events | File indexed once (not N times) |
| `test_single_file_small` | 200-char file | 1 chunk, document strategy |
| `test_single_file_large` | 5000-char file | Multiple chunks, paragraph strategy |
| `test_single_file_non_md` | Write .json file | NOT indexed |
| `test_handler_process_pending` | Multiple files changed | All indexed after debounce |

**Verification:**
```bash
# Terminal 1
python -m rag_lab.cli watch --vault /tmp/test_vault

# Terminal 2
echo '---\ntype: concept\n---\n\n# Test concept' > /tmp/test_vault/L3-semantic/entities/concept-test.md
# Wait 3 seconds
python -m rag_lab.cli stats  # should show new chunk
```

---

### T7 — Search API endpoint

**File:** `rag_lab/web.py`

**Subtask 7.1 — `POST /api/search`**

```python
@app.post("/api/search")
async def api_search(data: dict):
    """
    Lightweight semantic search — no verifier loop.
    
    Request:
    {
        "query": "What was decided about Bitcoin?",
        "layers": ["l3_semantic", "l2_episodic"],  # optional, default all
        "top_k": 5                                    # optional, default 20
    }
    
    Response:
    {
        "results": [
            {
                "text": "chunk content...",
                "metadata": {...},
                "distance": 0.23,
                "collection": "l3_semantic",
                "chunk_id": "abc123-0"
            }
        ],
        "query": "original query",
        "total_chunks_searched": 1234
    }
    """
    query_text = data.get("query", "")
    if not query_text.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    
    layers = data.get("layers")
    top_k = data.get("top_k", 20)
    embed_model = data.get("embed_model")  # optional override
    
    # Resolve collections
    if layers:
        collections = layers
    else:
        collections = [lc.collection for lc in DEFAULT_LAYERS]
    
    # Embed query
    q_vec = embedder.embed([query_text], model_name=embed_model)[0]
    
    # Multi-collection search
    results = vector_store.query_multi(q_vec, collections=collections, top_k=top_k)
    
    # Count total chunks
    total = sum(vector_store.count(c) for c in collections)
    
    return {
        "results": results,
        "query": query_text,
        "total_chunks_searched": total,
    }
```

**Subtask 7.2 — Keep existing `/api/query` unchanged**

The existing human-facing `/api/query` (with verifier loop) stays as-is. The new `/api/search` is machines-only — raw retrieval, no LLM generation, no verification.

**Subtask 7.3 — Add OpenAPI metadata**

Document the endpoint in FastAPI's auto-generated docs (automatic with the function signature + docstring).

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_search_returns_results` | Valid query | results list, distance < 1.0 |
| `test_search_top_k` | top_k=3 | exactly 3 results |
| `test_search_layer_filter` | layers=["l3_semantic"] | All results from l3_semantic |
| `test_search_empty_query` | query="" | 400 error |
| `test_search_all_layers` | No layers specified | Results from all collections |
| `test_search_collection_in_result` | Any query | Each result has "collection" field |
| `test_search_response_structure` | Valid query | Has results, query, total_chunks_searched |

**Verification:**
```bash
# Start server
python -m rag_lab.cli serve --port 8001 &

# Index test vault
python -m rag_lab.cli index --vault /tmp/test_vault

# Search
curl -s -X POST http://127.0.0.1:8001/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "test concept", "layers": ["l3_semantic"], "top_k": 3}' | python -m json.tool
```

---

### T8 — vault_bridge library

**Repo:** New — `~/vault_bridge/`

**Subtask 8.1 — Package scaffold**

```
vault_bridge/
├── pyproject.toml
├── vault_bridge/
│   ├── __init__.py
│   ├── writer.py
│   ├── reader.py
│   └── config.py
└── tests/
    ├── __init__.py
    ├── test_writer.py
    └── test_reader.py
```

`pyproject.toml`:
```toml
[project]
name = "vault_bridge"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pyyaml>=6.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Subtask 8.2 — Config**

```python
# vault_bridge/config.py
import os
from pathlib import Path

def get_vault_root() -> Path:
    """Resolve vault root from env var or default."""
    env = os.environ.get("STOA_VAULT_ROOT", "")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "vault"

def ensure_layer_dirs(vault_root: Path):
    """Create all layer directories if they don't exist."""
    dirs = [
        vault_root / "L1-working" / "tasks",
        vault_root / "L2-episodic" / "daily",
        vault_root / "L2-episodic" / "agent-logs",
        vault_root / "L2-episodic" / "decisions",
        vault_root / "L2-episodic" / "lessons",
        vault_root / "L3-semantic" / "identity",
        vault_root / "L3-semantic" / "knowledge",
        vault_root / "L3-semantic" / "entities",
        vault_root / "L3-semantic" / "architecture",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
```

**Subtask 8.3 — Writer**

```python
# vault_bridge/writer.py
import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .config import get_vault_root

def _atomic_write(path: Path, content: str) -> str:
    """
    Write content atomically: temp file → os.replace.
    Returns SHA-256 of content.
    """
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

def write_session_l2(date: str, messages: list[dict], agent: str,
                     task: str = "", decisions: str = "", lessons: str = "") -> dict:
    """
    Write agent session to L2-episodic/agent-logs/.
    Returns {path, sha, receipt_path}.
    """
    now = datetime.now()
    ts = now.strftime("%H:%M:%S")
    date_str = date or now.strftime("%Y-%m-%d")
    
    # Format session log
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
        lines.append(f"### {role.title()} ({ts})" if role == list({m.get("role") for m in messages[:1]}) else f"### {role.title()}")
        lines.append(content)
        lines.append("")
    
    body = "\n".join(lines)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    
    vault = get_vault_root()
    l2_dir = vault / "L2-episodic" / "agent-logs"
    filename = f"agent-session-{date_str}-{sha}.md"
    filepath = l2_dir / filename
    
    content_sha = _atomic_write(filepath, body)
    
    # Save receipt
    receipt_dir = Path.home() / ".stoa" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{filename}.sha256"
    receipt_path.write_text(content_sha)
    
    return {
        "path": str(filepath),
        "sha": content_sha,
        "receipt_path": str(receipt_path),
    }

def append_l1_today(content: str, timestamp: str = None) -> str:
    """Append entry to L1-working/today.md with timestamp header."""
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
    """Write a decision to L2-episodic/decisions/."""
    ts = datetime.now().isoformat()
    body = f"# Decision — {date}\n\n> timestamp: {ts}\n> agent: {agent}\n\n## Context\n\n{context}\n\n## Decision\n\n{decision}\n"
    vault = get_vault_root()
    sha = hashlib.sha256(body.encode()).hexdigest()[:12]
    path = vault / "L2-episodic" / "decisions" / f"decision-{date}-{sha}.md"
    _atomic_write(path, body)
    return str(path)

def write_lesson(date: str, lesson: str, source: str = "") -> str:
    """Write a lesson to L2-episodic/lessons/."""
    ts = datetime.now().isoformat()
    body = f"# Lesson — {date}\n\n> timestamp: {ts}\n> source: {source}\n\n{lesson}\n"
    vault = get_vault_root()
    sha = hashlib.sha256(body.encode()).hexdigest()[:12]
    path = vault / "L2-episodic" / "lessons" / f"lesson-{date}-{sha}.md"
    _atomic_write(path, body)
    return str(path)
```

**Subtask 8.4 — Reader**

```python
# vault_bridge/reader.py
from pathlib import Path
from datetime import datetime, timedelta
import json

from .config import get_vault_root

def read_today() -> str:
    vault = get_vault_root()
    path = vault / "L1-working" / "today.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")

def read_layer(layer: str, subdir: str = None) -> list[Path]:
    """
    Read all .md file paths from a layer.
    layer: "L1", "L2", "L3"
    subdir: optional subdirectory within layer
    """
    vault = get_vault_root()
    pattern = f"{layer}-*/"
    base_dir = vault
    layer_dirs = list(base_dir.glob(pattern))
    if not layer_dirs:
        return []
    search_dir = layer_dirs[0]
    if subdir:
        search_dir = search_dir / subdir
    return sorted(search_dir.rglob("*.md"))

def read_recent_sessions(days: int = 7) -> list[dict]:
    """Read L2 agent-logs from the last N days."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    files = read_layer("L2", "agent-logs")
    sessions = []
    for f in files:
        if f.stem >= cutoff:  # filename starts with date
            sessions.append({
                "path": str(f),
                "content": f.read_text(encoding="utf-8"),
                "date": f.stem[:10],
            })
    return sessions

def read_identity() -> dict:
    """Read all files from L3/identity/ and parse as key-value pairs."""
    files = read_layer("L3", "identity")
    result = {}
    for f in files:
        content = f.read_text(encoding="utf-8")
        result[f.stem] = content
    return result

def read_l3_deltas(days: int = 30) -> list[dict]:
    """Read L3 delta JSON files from the last N days."""
    vault = get_vault_root()
    knowledge_dir = vault / "L3-semantic" / "knowledge"
    if not knowledge_dir.exists():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    deltas = []
    for f in sorted(knowledge_dir.glob("l3_delta_*.json")):
        if f.stem >= cutoff:
            try:
                deltas.append(json.loads(f.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
    return deltas
```

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_write_session_creates_file` | write_session_l2(...) | File exists at returned path |
| `test_write_session_format` | write_session_l2 with messages | Has # Agent Session header, ## Messages |
| `test_write_session_sha_receipt` | write_session_l2(...) | Receipt file exists with matching SHA |
| `test_append_l1_today_new` | append to non-existent today.md | File created with content |
| `test_append_l1_today_existing` | append to existing today.md | Content appended, old content preserved |
| `test_append_l1_today_has_timestamp` | append_l1_today("test") | File contains timestamp header |
| `test_write_decision` | write_decision(...) | File created, has Context + Decision sections |
| `test_write_lesson` | write_lesson(...) | File created, has Lesson section |
| `test_atomic_write_no_partial` | Kill during write (mock) | No .tmp files visible |
| `test_read_today_exists` | read_today() | Returns content |
| `test_read_today_missing` | read_today() on empty vault | Returns "" |
| `test_read_layer_l3` | read_layer("L3") | Returns list of Paths |
| `test_read_recent_sessions` | read_recent_sessions(7) | Returns sessions from last 7 days |
| `test_read_identity` | read_identity() | Returns dict with file_stem → content |
| `test_read_l3_deltas` | read_l3_deltas(30) | Returns list of parsed JSON dicts |

**Verification:**
```bash
cd ~/vault_bridge
python -m pytest tests/ -v

# Integration: write then read
python -c "
from vault_bridge.writer import write_session_l2, append_l1_today
from vault_bridge.reader import read_today
result = write_session_l2('2026-06-30', [{'role': 'user', 'content': 'Test'}], 'test-agent')
print('Wrote:', result['path'])
append_l1_today('Test session completed')
print('Today:', read_today()[:200])
"
```

---

### T9 — Integration test (end-to-end)

**Subtask 9.1 — Full pipeline test**

Create a test that exercises the complete write → watch → search flow:

```python
# tests/test_integration.py

def test_full_pipeline(tmp_path):
    """
    1. Create vault structure with L1/L2/L3 dirs
    2. Write a session via vault_bridge → L2 agent-log appears
    3. Append to L1 today.md
    4. Run indexer on vault
    5. Search for content from the session
    6. Verify result contains the session text
    """
    vault_root = tmp_path / "vault"
    # Create L1, L2, L3 directories
    (vault_root / "L1-working").mkdir(parents=True)
    (vault_root / "L2-episodic" / "agent-logs").mkdir(parents=True)
    (vault_root / "L3-semantic" / "entities").mkdir(parents=True)
    
    # Write test content
    test_content = "The strategic Bitcoin decision was to hold until 150k by June 2026"
    entity_path = vault_root / "L3-semantic" / "entities" / "test-bitcoin.md"
    entity_path.write_text("""---
type: decision
aliases: [bitcoin decision]
---
# Bitcoin Hold Decision
The strategic Bitcoin decision was to hold until 150k by June 2026.
Based on polymarket activity and macro regime analysis.
""")
    
    # Index
    from rag_lab.indexer import index_vault
    result = index_vault(str(vault_root))
    assert result["files_indexed"] >= 1
    
    # Search
    from rag_lab.embedder import embed
    from rag_lab.vector_store import query_multi
    q_vec = embed(["Bitcoin 150k decision"])[0]
    hits = query_multi(q_vec, ["l3_semantic"], top_k=3)
    
    assert len(hits) > 0
    assert any("Bitcoin" in h["text"] for h in hits)
```

**Verification:**
```bash
python -m pytest tests/test_integration.py -v
```

---

## Phase 2 — Memory Pipeline

### T10 — session_writer (Hermes integration)

**File:** `vault_bridge/vault_bridge/session_writer.py`

Already partially built in T8 (writer.py). This task makes it a single-call entry point for Hermes.

```python
# session_writer.py

def write_session(
    date: str,
    agent: str,
    messages: list[dict],
    task: str = "",
    decisions: list[str] | None = None,
    lessons: list[str] | None = None,
) -> dict:
    """
    The one function Hermes calls after every session.
    
    1. Write L2 agent-log (historical truth, first)
    2. Save SHA-256 receipt
    3. Append summary to L1 today.md
    
    Returns {l2_path, l1_path, sha}
    """
    dec_str = "; ".join(decisions) if decisions else ""
    les_str = "; ".join(lessons) if lessons else ""
    
    l2_result = write_session_l2(date, messages, agent, task=task, decisions=dec_str, lessons=les_str)
    
    # Build L1 summary
    summary_parts = [
        f"Session: {agent}",
    ]
    if task:
        summary_parts.append(f"Task: {task}")
    if decisions:
        summary_parts.append(f"Decisions: {dec_str}")
    if lessons:
        summary_parts.append(f"Lessons: {les_str}")
    summary_parts.append(f"Messages: {len(messages)}")
    summary_parts.append(f"L2 log: {l2_result['path']}")
    
    l1_path = append_l1_today("\n".join(summary_parts))
    
    return {
        "l2_path": l2_result["path"],
        "l1_path": l1_path,
        "sha": l2_result["sha"],
    }
```

**How Hermes calls it:**
```python
# In Hermes gateway code:
from vault_bridge.session_writer import write_session

# After each session:
write_session(
    date=datetime.now().strftime("%Y-%m-%d"),
    agent="hermes",
    messages=session_messages,
    task=current_task,
    decisions=extracted_decisions,
    lessons=extracted_lessons,
)
```

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_write_session_full` | Full session with decisions | L2 file exists, L1 updated, SHA receipt |
| `test_write_session_no_decisions` | Session without decisions | L2 written, L1 has empty decisions |
| `test_write_session_l2_before_l1` | Mock L2 failure | L1 NOT written (invariant) |

---

### T11 — daily_compressor

**File:** `vault_bridge/vault_bridge/compressor.py`

**Subtask 11.1 — LLM integration**

```python
import httpx
import json
import os
from datetime import datetime
from .reader import read_today, read_recent_sessions, read_l3_deltas
from .writer import _atomic_write
from .config import get_vault_root

COMPRESSOR_SYSTEM = """You are a memory compression agent for the Stoa AI system.

You receive today's working memory and session logs.
Produce two outputs:

1. A daily summary (markdown) — condensed overview of the day's events, decisions, and context.
2. Knowledge deltas (JSON) — what new entities, relationships, decisions, and lessons were created today.

Be concise. Focus on what matters for long-term memory. Drop transient chatter."""

def _llm_synthesize(prompt: str) -> dict:
    """Call DeepSeek to synthesize. Returns parsed JSON."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": COMPRESSOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.1,
        "stream": False,
        "response_format": {"type": "json_object"},  # DeepSeek supports JSON mode
    }
    
    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
```

**Subtask 11.2 — Compressor run**

```python
def run(date: str = None) -> dict:
    """
    Nightly memory compression.
    
    1. Read L1 today.md
    2. Read today's L2 agent-logs
    3. Read recent L3 deltas (for context)
    4. LLM synthesis → L2 daily summary + L3 deltas
    5. Atomic writes for both outputs
    
    Returns {l2_path, l3_path, summary_chars, entities_extracted}
    """
    date = date or datetime.now().strftime("%Y-%m-%d")
    
    # Gather inputs
    today = read_today()
    sessions = read_recent_sessions(days=1)  # today's sessions only
    recent_deltas = read_l3_deltas(days=3)   # last 3 days for context
    
    # Build prompt
    sessions_text = "\n---\n".join(
        f"### {s['date']}\n{s['content'][:2000]}"  # truncate long sessions
        for s in sessions
    )
    
    prompt = f"""DATE: {date}

TODAY'S WORKING MEMORY:
{today[:3000]}

TODAY'S SESSIONS:
{sessions_text}

RECENT KNOWLEDGE DELTAS (for context):
{json.dumps(recent_deltas, indent=2)[:2000]}

Produce JSON with these keys:
- daily_summary: string (markdown, the day's condensed summary)
- new_entities: list of {{name, type, description}}
- updated_entities: list of {{name, field, new_value}}
- decisions_captured: list of {{decision, rationale, context}}
- lessons_learned: list of {{lesson, source}}
"""
    
    try:
        result = _llm_synthesize(prompt)
    except Exception as e:
        return {"error": str(e), "date": date}
    
    vault = get_vault_root()
    
    # Write L2 daily summary
    l2_dir = vault / "L2-episodic" / "daily"
    l2_path = l2_dir / f"{date}.md"
    daily_md = result.get("daily_summary", f"# Daily Summary — {date}\n\nCompression failed.")
    _atomic_write(l2_path, f"# Daily Summary — {date}\n\n{daily_md}")
    
    # Write L3 delta
    l3_dir = vault / "L3-semantic" / "knowledge"
    l3_path = l3_dir / f"l3_delta_{date}.json"
    delta = {
        "date": date,
        "new_entities": result.get("new_entities", []),
        "updated_entities": result.get("updated_entities", []),
        "decisions_captured": result.get("decisions_captured", []),
        "lessons_learned": result.get("lessons_learned", []),
    }
    _atomic_write(l3_path, json.dumps(delta, indent=2))
    
    return {
        "l2_path": str(l2_path),
        "l3_path": str(l3_path),
        "summary_chars": len(daily_md),
        "entities_extracted": len(delta.get("new_entities", [])) + len(delta.get("updated_entities", [])),
    }
```

**Subtask 11.3 — CLI entry point**

```python
# In vault_bridge __init__.py or separate CLI
# Can be called as: python -m vault_bridge.compressor --date 2026-06-30

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None)
    args = p.parse_args()
    result = run(args.date)
    print(json.dumps(result, indent=2))
```

**Subtask 11.4 — Cron/systemd timer config**

```
# ~/.config/systemd/user/daily-compressor.service
[Unit]
Description=Stoa daily memory compressor

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 -m vault_bridge.compressor
Environment=STOA_VAULT_ROOT=/Users/notabanker/vault
Environment=DEEPSEEK_API_KEY=%h/.stoa/auth.json  # or direct

# ~/.config/systemd/user/daily-compressor.timer
[Unit]
Description=Run daily compressor at midnight

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

**Tests:**

| Test | Input | Expected |
|---|---|---|
| `test_compressor_run` | Mock vault with L1 + L2 data | L2 daily file created, L3 delta JSON created |
| `test_compressor_empty_vault` | Empty vault | Returns gracefully, no files created |
| `test_compressor_llm_failure` | No API key | Returns error dict, doesn't crash |
| `test_compressor_output_format` | After run | L2 daily has heading, L3 delta has required keys |

**Verification:**
```bash
# Manual run
python -m vault_bridge.compressor --date 2026-06-30

# Check outputs
cat ~/vault/L2-episodic/daily/2026-06-30.md
cat ~/vault/L3-semantic/knowledge/l3_delta_2026-06-30.json

# Cron simulation
systemctl --user start daily-compressor.service
systemctl --user status daily-compressor.service
```

---

### T12 — End-to-end Phase 2 test

```python
def test_write_compress_search(tmp_path):
    """
    1. Create vault
    2. Write session via session_writer
    3. Run compressor
    4. Verify L1, L2, L3 files created
    5. Index vault
    6. Search for session content
    7. Search for compressed summary
    """
    vault_root = tmp_path / "vault"
    # ... setup vault structure with L1/L2/L3 dirs
    
    # Set env for vault path
    import os
    os.environ["STOA_VAULT_ROOT"] = str(vault_root)
    
    from vault_bridge.session_writer import write_session
    write_session("2026-06-30", "test-agent", [
        {"role": "user", "content": "What's the risk regime?"},
        {"role": "assistant", "content": "GREEN. VIX at 14.2, spreads tight. No tail risk signals."}
    ], task="risk check", decisions=["Maintain risk-on bias"], lessons=["VIX data was stale by 2h"])
    
    # Verify L2 written
    agent_logs = list((vault_root / "L2-episodic" / "agent-logs").glob("*.md"))
    assert len(agent_logs) >= 1
    
    # Run compressor (if LLM key available)
    # ... 
    
    # Index
    from rag_lab.indexer import index_vault
    index_vault(str(vault_root))
    
    # Search
    from rag_lab.embedder import embed
    from rag_lab.vector_store import query_multi
    q_vec = embed(["risk regime GREEN VIX"])[0]
    hits = query_multi(q_vec, ["l1_working", "l2_episodic"], top_k=5)
    
    assert len(hits) > 0
    assert any("GREEN" in h["text"] for h in hits)
```

---

## Phase 3 — Entity Graph (deferred)

### T13 — hippocampus (graph DB)

**File:** `vault_bridge/vault_bridge/hippocampus.py`

### T14 — Entity file generation

**File:** `vault_bridge/vault_bridge/entity_gen.py`

(Detailed subtasks to be specified when Phase 3 begins. Rough scope: SQLite schema from ARCHITECTURE.md §10, entity extraction from L3 deltas, Obsidian markdown file regeneration with [[wikilinks]].)

---

## Task Dependency Graph

```
T1 (chunking) ──────────────────────────────────────────────────┐
T2 (metadata) ──────────────────────────────────────────────────┤
T3 (collections) ───────────────────────────────────────────────┤
T4 (exclusion) ─────────────────────────────────────────────────┤
                                                                 ├──> T9 (integration test P1)
T5 (indexer) ── depends on: T1, T2, T3, T4 ─────────────────────┤
T6 (watcher) ── depends on: T1, T2, T3, T4 ─────────────────────┤
T7 (search API) ── depends on: T3 ──────────────────────────────┘

T8 (vault_bridge library) ── independent ──┐
                                             ├──> T10 (session_writer)
                                             │        depends on: T8
                                             │
                                             └──> T11 (compressor)
                                                      depends on: T8

T10 + T11 + T5 ──> T12 (integration test P2)
```

T1–T4 can be parallelized. T8 is independent. T5, T6, T7, T10, T11 can start after their dependencies.

---

## Verification Checklist

After each task, run:

```bash
# Unit tests
python -m pytest tests/ -v

# Linting
python -m ruff check rag_lab/ tests/

# Type checking (if configured)
python -m mypy rag_lab/ --ignore-missing-imports

# Integration smoke test
python -m rag_lab.cli index --vault /tmp/test_vault
python -m rag_lab.cli stats
curl -s http://127.0.0.1:8001/api/search -d '{"query":"test"}' | python -m json.tool
```

---

## File Manifest (new & modified)

### rag-lab (modified existing)
| File | Action | Tasks |
|---|---|---|
| `rag_lab/chunker.py` | Modify | T1 — add chunk_document, chunk_paragraph |
| `rag_lab/vector_store.py` | Modify | T3 — multi-collection support, query_multi |
| `rag_lab/web.py` | Modify | T7 — POST /api/search endpoint |
| `rag_lab/cli.py` | Modify | T5, T6 — index + watch commands |
| `pyproject.toml` | Modify | T6 — add watchdog dependency |
| `tests/test_chunker.py` | Modify | T1 — document + paragraph tests |
| `tests/test_verifier.py` | Unchanged | — |
| `tests/test_config.py` | Unchanged | — |
| `tests/test_parsers.py` | Unchanged | — |

### rag-lab (new files)
| File | Action | Tasks |
|---|---|---|
| `rag_lab/metadata.py` | Create | T2 — frontmatter extraction |
| `rag_lab/vault_config.py` | Create | T3, T4 — layer config, exclusion patterns |
| `rag_lab/indexer.py` | Create | T5 — batch indexing |
| `rag_lab/watcher.py` | Create | T6 — watchdog integration |
| `tests/test_metadata.py` | Create | T2 |
| `tests/test_vector_store.py` | Create | T3 |
| `tests/test_vault_config.py` | Create | T4 |
| `tests/test_integration.py` | Create | T9, T12 |

### vault_bridge (new repo)
| File | Action | Tasks |
|---|---|---|
| `pyproject.toml` | Create | T8 |
| `vault_bridge/__init__.py` | Create | T8 |
| `vault_bridge/config.py` | Create | T8 |
| `vault_bridge/writer.py` | Create | T8, T10 |
| `vault_bridge/reader.py` | Create | T8 |
| `vault_bridge/session_writer.py` | Create | T10 |
| `vault_bridge/compressor.py` | Create | T11 |
| `tests/__init__.py` | Create | T8 |
| `tests/test_writer.py` | Create | T8 |
| `tests/test_reader.py` | Create | T8 |

### vault (new files at root)
| File | Action | Tasks |
|---|---|---|
| `.rag-ignore` | Create | T4 |

---

## Effort Estimates

| Task | Effort | Files |
|---|---|---|
| T1 — Chunking strategies | 1–2h | 2 files |
| T2 — Metadata extraction | 1h | 2 files |
| T3 — Layer collections | 2h | 3 files |
| T4 — Exclusion patterns | 0.5h | 1 file |
| T5 — Batch indexer | 3h | 3 files |
| T6 — File watcher | 2h | 3 files |
| T7 — Search API | 1h | 1 file |
| T8 — vault_bridge library | 3h | 8 files |
| T9 — Integration test P1 | 1h | 1 file |
| T10 — session_writer | 0.5h | 1 file |
| T11 — daily_compressor | 2h | 1 file |
| T12 — Integration test P2 | 1h | 1 file |
| **Total Phase 1+2** | **~18h** | **27 files** |

---

## Open Decisions

1. **Chunking threshold: 1000 chars?** Based on `all-MiniLM-L6-v2` max_seq_length=256 (~1000 chars). Files ≤ 1000 chars → whole-file. Files > 1000 chars → paragraph chunking. This handles 531/589 files as single-chunk. Can adjust threshold after testing retrieval quality.

2. **Chunk ID scheme: file-path-hash or file-content-hash?** File-path-hash means same path = same IDs = upsert always replaces. File-content-hash means same content = same IDs (dedup across renames). **Decision: file-path-hash** — vault paths are stable, content changes.

3. **watcher vs indexer interaction:** After `index_vault()` batch run, watcher handles incremental changes. Watcher should NOT re-index files already handled by a recent batch index. **Decision: watcher uses debounce, batch indexer uses --force flag. No coordination needed** — ChromaDB upsert is idempotent.

4. **`.rag-ignore` location:** Vault root. Pattern: glob relative to vault root. Defaults if file doesn't exist.

5. **Search endpoint name:** `POST /api/search` (not `/api/query` — that's the human-facing verifier endpoint). Hermes calls `/api/search` for raw top-K retrieval.
