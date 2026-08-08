# rag-lab — Technical Architecture Overview

**Audience:** Another LLM or designer producing architecture diagrams.  
**Project:** Personal/local RAG lab (Python 3.11+, version 3.1.0).  
**Purpose:** Ingest multi-format documents into a versioned vector index; answer questions with hybrid retrieval, cited generation, and a verifier loop; expose CLI, HTTP API, React dashboard, and MCP tools; evaluate quality with gated metrics.

---

## 1. System context

```
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│ Human / CLI │   │ React SPA    │   │ MCP client      │
│ (Typer)     │   │ (Vite+React) │   │ (stdio agent)   │
└──────┬──────┘   └──────┬───────┘   └────────┬────────┘
       │                 │ HTTPS/HTTP         │ JSON-RPC stdio
       │                 ▼                    │
       │          ┌──────────────┐            │
       └─────────►│ FastAPI app  │◄───────────┘
                  │ (rag serve)  │
                  └──────┬───────┘
                         │ same Python package
                         ▼
                  ┌──────────────────────────────────────┐
                  │           rag_lab core library        │
                  │  parsers → ingest → store → retrieve │
                  │  → generate → verify → evaluate      │
                  └──────┬───────────────┬───────────────┘
                         │               │
           ┌─────────────┼───────┐       │ OpenAI-compatible
           ▼             ▼       ▼       ▼ HTTP
      ┌─────────┐  ┌──────────┐ ┌────────────────┐
      │ Chroma  │  │ SQLite   │ │ LLM provider   │
      │ (disk)  │  │ runs.db  │ │ (OpenRouter…)  │
      └─────────┘  └──────────┘ └────────────────┘
           ▲
           │ local models (CPU)
      ┌────┴─────────────────────────────┐
      │ sentence-transformers             │
      │  • multilingual-e5-small (embed) │
      │  • mmarco cross-encoder (rerank) │
      └──────────────────────────────────┘
```

**Boundaries**

| Boundary | Technology | Notes |
|----------|------------|-------|
| Local compute | sentence-transformers, rank-bm25, pypdf, ebooklib, python-docx | No cloud for embed/rerank |
| Remote LLM | OpenAI-compatible `/chat/completions` | Generator + verifier only |
| Persistence | Chroma PersistentClient + SQLite | Same data dir |
| Surfaces | CLI, FastAPI REST, MCP stdio, React UI | All call the same library |

---

## 2. Repository / module map

```
rag-lab/
├── rag_lab/                 # Python package (core)
│   ├── cli.py               # Typer entry: `rag`
│   ├── web.py               # FastAPI + optional embedded HTML
│   ├── mcp_server.py        # MCP tool server
│   ├── config.py            # Env config, collection fingerprint
│   ├── parsers/             # Format adapters
│   │   ├── base.py          # ParseResult + quality heuristics
│   │   ├── pdf.py, epub.py, markdown.py, docx.py
│   │   └── __init__.py      # PARSERS registry by extension
│   ├── chunker.py           # fixed / sentence Chunk(start,end,text)
│   ├── ingestion.py         # SHA, markers, metadatas, ingest_text
│   ├── embedder.py          # E5 document/query prefixes
│   ├── vector_store.py      # Chroma CRUD + query + keyword
│   ├── lexical.py           # BM25 index + RRF fusion
│   ├── reranker.py          # Cross-encoder rescoring
│   ├── retriever.py         # RetrievalConfig + retrieve() loop
│   ├── verifier.py          # Grounding auditor LLM call
│   ├── citations.py         # Extract/validate [labels]
│   ├── manifest.py          # documents table (corpus catalog)
│   ├── runs.py              # runs + eval_runs tables
│   ├── evaluation.py        # Question set, metrics, variants
│   └── gates.py             # Pass/fail thresholds
├── dashboard/               # React 19 + Vite + TanStack Query
├── data/                    # Sample corpus (pdf/epub/md)
├── eval/                    # questions.yaml, variants.yaml, gates.yaml
└── tests/
```

**Dependency direction (no import cycles):**  
`cli/web/mcp` → `retriever/ingestion/evaluation` → `vector_store/lexical/embedder/reranker` → `config`  
Parsers are leaves under `ingestion` / CLI.

---

## 3. Configuration model

All config is **process environment** (no `.env` loader).

| Variable | Default | Role |
|----------|---------|------|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Chat API base |
| `LLM_MODEL` | `qwen/qwen3.7-plus` | Generator |
| `LLM_VERIFIER_MODEL` | same as generator | Verifier |
| `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` | empty | Auth for LLM |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Dense vectors |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Rerank |
| `RAG_DB_PATH` | `~/.local/share/rag-lab/chroma_db` | Persist root |
| `RAG_COLLECTION` | `rag_lab_{fingerprint}` | Active collection |

**Collection fingerprint**

```
INDEX_VERSION = "v3"
CHUNKING_VERSION = "sentence-v2-parent"
fingerprint = f"{INDEX_VERSION}-{slug(embedding_model)}-{slug(chunking_version)}"
# e.g. rag_lab_v3-intfloat-multilingual-e5-small-sentence-v2-parent
```

Collections store metadata: `hnsw:space=cosine`, `embedding_model`, `chunking_version`, `index_version`. Ingest refuses model mismatch unless forced.

---

## 4. Pipeline A — Ingestion (write path)

### 4.1 Flow

```
Path → pick_parser(ext) → parse_*() → ParseResult
         → quality()  [warnings: empty / low-yield scanned PDF]
         → effective_text  [blank if zero real content]
         → make_chunks(strategy, size, overlap)
         → build_metadatas()  [markers → citations, parent_id]
         → embed(document-prefixed)
         → delete_by_sha(file_sha)  [idempotent re-ingest]
         → chroma.upsert(ids, embeddings, documents, metadatas)
         → manifest.log_document(...)
```

### 4.2 Parser contract

```python
# parsers/base.py
@dataclass
class ParseResult:
    text: str
    section_unit: str   # "page" | "chapter" | "section" | "slide" | "document"
    section_chars: list[int]  # per-section stripped char counts

    def quality() -> dict:
        # sections, empty_sections, total_chars, avg_section_chars, warnings[]

    @property
    def effective_text(self) -> str:
        # "" if total section chars == 0 (don't chunk marker scaffolding alone)
```

**Registry** (`PARSERS`):

| Ext | Module | Section unit | Marker form in text |
|-----|--------|--------------|---------------------|
| `.pdf` | `parse_pdf` | page | `--- Page N ---` |
| `.epub` | `parse_epub` | chapter | `--- chapter.xhtml ---` |
| `.md` / `.markdown` | `parse_markdown` | document | (varies) |
| `.docx` | `parse_docx` | section | `--- Heading ---` (WIP) |

Markers are embedded in the concatenated text so chunk offsets can map to citations.

### 4.3 Chunking

```python
@dataclass
class Chunk:
    text: str
    start: int  # char offset in full source text
    end: int
```

| Strategy | Behavior |
|----------|----------|
| `sentence` (default) | Split on `(?<=[.!?])\s+`; pack to `target_size` (~512); sentence overlap |
| `fixed` | Sliding window `size` with `overlap` |

**Small-to-big parents:** every `parent_size` (default 4) consecutive child chunks share:

```
parent_id = f"{file_sha}-p{parent_idx}"
```

### 4.4 Content identity & chunk IDs

```
file_sha = sha256(file_bytes)[:10]
chunk_id = f"{file_sha}-{i}"   # i = 0..n-1
```

Re-ingest deletes all chunks with that `file_sha` then upserts.

### 4.5 Chunk metadata (Chroma)

Each vector row carries:

```
source, doc_id, file_sha, chunk_idx, chunk_start, chunk_end,
strategy, chunk_size, overlap,
parent_id, parent_idx, parent_size,
citation,           # e.g. "report.pdf p.12" or "doc.docx §Methods" or "file.md chunk 3"
source_marker,      # raw marker label or ""
embedding_model, chunking_version, index_version
```

**Citation derivation:** scan text for `_MARKER_RE` → last marker with `position ≤ chunk.start` → if Page N → `{name} p.N`; else if label → `{name} §{label}`; else `{name} chunk {i+1}`.

### 4.6 Embeddings

- Model: multilingual E5 (default).
- Document embed: prefix `passage: ` (E5 convention via `input_type="document"`).
- Query embed: prefix `query: `.
- Space: cosine (Chroma HNSW).

---

## 5. Pipeline B — Retrieval + answer (read path)

### 5.1 `RetrievalConfig`

```python
@dataclass
class RetrievalConfig:
    mode: str = "hybrid"       # vector | lexical | hybrid
    top_k: int = 50            # candidates per source before fuse
    rerank_top: int = 5        # children kept after rerank
    use_reranker: bool = True
    small_to_big: bool = True  # expand children → parents
    parent_top_k: int = 5
    max_context_chars: int = 12000
    min_score: int = 8         # verifier threshold
    max_iters: int = 3
    max_tokens: int = 600
    model: str | None = None
    keyword: str | None = None # regex mode: bypass vector/BM25
```

### 5.2 Candidate retrieval (`retrieve_hits`)

```
if mode ∈ {vector, hybrid}:
    q_vec = embed_query(question)
    vector_hits = chroma.query(q_vec, top_k)
if mode ∈ {lexical, hybrid}:
    lexical_hits = bm25_search(question, top_k)
if hybrid:
    hits = rrf_fuse({"vector": ..., "lexical": ...}, k=60, limit=top_k)
if use_reranker:
    hits = cross_encoder.rerank(question, hits)  # higher score = better
return hits  # ordered candidate list
```

**RRF:** reciprocal rank fusion across named lists; produces fused score + optional `source_ranks`.

**BM25:** lazy in-memory index over full collection documents; token regex includes German umlauts; invalidated on corpus mutation.

### 5.3 Context selection (`select_context_chunks`)

```
selected = hits[:rerank_top]
if small_to_big:
    for each child hit:
        siblings = get_by_parent_id(parent_id)
        parent_text = join sibling texts
        dedupe by parent_id; keep parent_top_k
else:
    attach citation labels to children
apply max_context_chars budget (greedy by order)
```

### 5.4 Keyword mode (bypass)

If `keyword` is set: regex scan all chunks → take matches → still run parent expansion / budget → single generate+verify (no multi-iter retrieval refine).

### 5.5 Generation + verification loop (`retrieve` / `_retrieve`)

```
current_q = question
for i in 1..max_iters:
    hits = retrieve_hits(current_q)
    chunks = select_context_chunks(hits)
    if empty → refuse ("no chunks")
    context = format as:
        [citation_label]
        text
        ---
        ...
    answer = LLM(system=GENERATOR_SYSTEM, user=CONTEXT+QUESTION)
    verdict = verifier(question, answer, chunks)  # score 1-10, grounded, issues, verdict
    if score >= min_score → accept
    else current_q = question + refinement(issues)
log run to SQLite; attach citation_validation
```

**Generator rules (system prompt):** answer only from CONTEXT; cite with `[label]`; if unknown reply exactly: `I don't know from the provided documents.`

**Verifier:** separate LLM call; returns JSON-like structure with score, grounded, issues, verdict (`GROUNDED` / `UNGROUNDED` style).

**Citation validation (`citations.validate`):** extract `[...]` from answer; check membership against context labels; detect refusal marker; report valid/invalid/missing.

### 5.6 Result shape (logical)

```json
{
  "answer": "...",
  "chunks": [{"id", "text", "metadata", "citation", "rerank_score", ...}],
  "verifier": {"score": 9, "grounded": true, "issues": [], "verdict": "..."},
  "iterations": 1,
  "trace": [...],
  "usage": {"prompt_tokens": N, "completion_tokens": M},
  "latency_ms": 1234,
  "citation_validation": {"ok": true, "citations": [...], ...},
  "run_id": 42
}
```

---

## 6. Persistence architecture

**Root:** `RAG_DB_PATH` (default `~/.local/share/rag-lab/chroma_db/`)

```
chroma_db/
├── chroma.sqlite3          # Chroma internal
├── <uuid>/                 # segment data
└── runs.sqlite3            # app SQLite (manifest + runs + evals)
```

### 6.1 Chroma collections

- Multiple collections allowed (version experiments).
- Active collection: `DEFAULT_COLLECTION` / CLI `--collection`.
- Operations: upsert, delete by sha/doc, query vectors, keyword regex, get by parent_id, list collections, delete collection.

### 6.2 SQLite schema (`runs.sqlite3`)

**`documents` (manifest)**

| Column | Meaning |
|--------|---------|
| `doc_id` PK | = `file_sha` |
| `source` | original path |
| `file_sha` | content hash prefix |
| `parser_version`, `chunking_version`, `embedding_model` | lineage |
| `chunk_count`, `ingested_at` | stats |
| `parse_report` JSON | quality report |

**`runs`**

| Column | Meaning |
|--------|---------|
| `id`, `timestamp`, `question` | identity |
| `config` JSON | RetrievalConfig + model names |
| `answer`, `verifier` JSON, `iterations` | outcome |
| `latency_ms`, token counts | cost/perf |
| `trace` JSON, `citation_validation` JSON | debug |
| `partial` | incomplete flag |

**`run_chunks`:** `(run_id, chunk_id, rank, distance, source)` — which chunks were used.

**`eval_runs`:** variant name, config, summary metrics, per_question detail.

---

## 7. Evaluation subsystem

```
eval/questions.yaml  → EvalQuestion[]  (id, question, expected sources, optional fragments)
eval/variants.yaml   → Variant[]       (named RetrievalConfig presets)
eval/gates.yaml      → thresholds
```

**Retrieval metrics:** hit@k, MRR, chunk-level hit/MRR against expected source basenames.

**Answer metrics (full eval):** fragment match, refusal accuracy, citation validity rate, mean verifier score.

**Gates example:**

```yaml
retrieval:
  hit_rate: 0.95
  mrr: 0.90
  chunk_hit_rate: 0.85
  context_hit_rate: 0.95
answer:
  fragment_rate: 0.80
  refusal_accuracy: 0.90
  citation_validity_rate: 0.95
  verifier_mean: 8.0
```

`scripts/check.sh` = `pytest` + retrieval-only eval with gates (green bar for commits).

---

## 8. Interface surfaces

### 8.1 CLI (`rag` → `rag_lab.cli:app`)

| Command | Function |
|---------|----------|
| `ingest` / `rebuild` | Write path |
| `query` | Full retrieve loop |
| `stats`, `config show` | Introspection |
| `docs list/show/reingest/delete` | Corpus mgmt |
| `collections list/delete` | Index mgmt |
| `eval`, `compare` | Evaluation |
| `runs list/show` | History |
| `serve` | Start FastAPI |
| `mcp serve` | Start MCP |

Global: `--db-path`, `--collection`.

### 8.2 HTTP API (FastAPI)

| Method | Path | Role |
|--------|------|------|
| GET | `/` | HTML shell / SPA host |
| POST | `/api/ingest` | Multipart file upload |
| POST | `/api/query` | Ask (body = RetrievalConfig fields + question) |
| GET | `/api/config` | Runtime config |
| GET/DELETE | `/api/collections`, `/api/collections/{name}` | Collections |
| GET/DELETE | `/api/docs`, `/api/docs/{id}` | Documents |
| POST | `/api/docs/{id}/reingest` | Reingest |
| GET | `/api/runs`, `/api/runs/{id}` | Query history |
| GET | `/api/evals`, `/api/evals/{id}` | Eval history |
| POST | `/api/eval` | Run evaluation |
| GET | `/api/stats` | Counts / models |

CORS enabled for local dashboard (`127.0.0.1:5173` → API `:8000` or proxied).

### 8.3 Dashboard (React)

Tabs: **Ask | Corpus | Runs | Eval | Settings**.  
Talks only to REST (`fetch`). Shows answers, traces, citations, parse warnings, eval tables, config.

### 8.4 MCP tools

`rag_search`, `rag_answer`, `rag_ingest`, `rag_delete` (confirm required), `rag_docs_list`, `rag_reingest`, `rag_collections_list`, `rag_runs_show`, `rag_eval_run`.

---

## 9. Data-flow diagrams (for visualizers)

### 9.1 Ingest (sequence)

```
Client → CLI/API → pick_parser → Parser
Parser → ParseResult.quality → (optional refuse if empty)
ParseResult → chunker → [Chunk]
[Chunk] + markers → build_metadatas → metadatas
[Chunk].text → embedder → vectors
vectors + metadatas → vector_store.upsert → Chroma
file_sha + quality → manifest → SQLite documents
```

### 9.2 Query (sequence)

```
Client → retriever.retrieve(question, cfg)
  loop:
    embedder.embed_query → vector_store.query  ─┐
    lexical.bm25_search                        ─┼→ rrf_fuse → reranker
    select_context_chunks (parent expand) ←────┘
    LLM generate
    verifier.verify
    if score < min_score: refine query
  citations.validate
  runs.log_run → SQLite
Client ← result
```

### 9.3 Component graph (boxes to draw)

**Layers top → bottom:**

1. **Presentation:** CLI, React Dashboard, MCP Client  
2. **API adapters:** Typer commands, FastAPI routes, MCP tool handlers  
3. **Application services:** `ingestion`, `retriever`, `evaluation`, `gates`  
4. **Domain helpers:** `chunker`, `citations`, `parsers/*`  
5. **Infrastructure:** `embedder`, `reranker`, `lexical`, `vector_store`, `manifest`, `runs`, `config`  
6. **External systems:** Chroma files, SQLite, HuggingFace models (local), LLM HTTP API  

---

## 10. Runtime characteristics (for annotations)

| Concern | Behavior |
|---------|----------|
| Offline-capable | Ingest + retrieval + BM25 + rerank work offline |
| Needs network | Generation, verification, full eval, first model download |
| Default DB location | User data dir (not CWD) to avoid empty-corpus footgun |
| Idempotent ingest | Same file bytes → same `file_sha` → replace chunks |
| Multilingual | E5 multiling + German-aware BM25 tokenize |
| Latency hotspots | First embedder/reranker load; LLM round-trips × iters |
| Scale | Personal corpora (thousands of chunks); BM25 holds all docs in memory |

---

## 11. Known architectural limitations (label honestly on diagrams)

1. **No OCR path yet** — image-only PDFs produce zero text and are refused (or `--allow-empty`).  
2. **Citation marker vs chunk offset edge case** — if `chunk.start` is before the first `---` marker position, citation falls back to `chunk N` (affects first chunk of some docs / DOCX WIP).  
3. **Manifest can drift** from Chroma if docs were ingested before manifest logging.  
4. **Single-process local design** — no multi-tenant auth, no queue, no distributed vector DB.  
5. **LLM provider is pluggable** only via OpenAI-compatible HTTP; no native Anthropic/Google SDKs.  
6. **PPTX not in registry yet** (roadmap Phase A).

---

## 12. Suggested diagram set for the other LLM

Produce these views:

1. **C4 Context** — users, rag-lab, LLM API, local model cache, disk.  
2. **C4 Container** — CLI / Dashboard / FastAPI / MCP / core lib / Chroma / SQLite / ST models.  
3. **Ingest pipeline** (linear flowchart with ParseResult & metadata schema callouts).  
4. **Query pipeline** (hybrid RRF + rerank + small-to-big + verifier loop with decision diamond).  
5. **Data model ER** — Chroma collection row, documents, runs, run_chunks, eval_runs.  
6. **Deployment** — single machine; ports 8000 (API) + 5173 (Vite dev); env vars listed.

---

## 13. One-paragraph elevator (for titles/captions)

> rag-lab is a local-first RAG system: multi-format parsers emit marker-annotated text; a sentence-aware chunker builds parent-grouped child chunks; multilingual E5 embeddings land in version-fingerprinted Chroma collections; queries fuse dense retrieval with BM25 via RRF, optionally cross-encoder-rerank, expand to parent context, and drive an OpenAI-compatible generator–verifier loop with citation validation—all orchestrated from a shared Python library exposed through Typer CLI, FastAPI, a React dashboard, and MCP, with SQLite run/eval history and YAML-gated regression metrics.
