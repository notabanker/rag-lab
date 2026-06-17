# rag-lab Documentation

Standalone RAG (Retrieval-Augmented Generation) learning project.  
Ingest PDF, EPUB, Markdown → embed → ChromaDB → query with verifier-governed retrieval loop.

---

## Architecture

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│  Parser   │───▶│  Chunker  │───▶│  Embedder  │───▶│  ChromaDB │
│ pdf/epub/md│    │ fixed/   │    │ all-MiniLM │    │ Persistent │
│            │    │ sentence │    │  L6-v2 CPU │    │  Client    │
└──────────┘    └──────────┘    └───────────┘    └──────────┘
                                                       │
                    ┌──────────────────────────────────┘
                    ▼
              ┌──────────┐    ┌───────────┐    ┌──────────┐
              │ Retriever │───▶│  Generator │───▶│ Verifier  │
              │ top-k=20  │    │  (LLM)     │    │ (LLM)     │
              │ rerank=5  │    │            │    │ score 1-10│
              └──────────┘    └───────────┘    └──────────┘
                    │                │               │
                    └────────────────┴───────────────┘
                                     │
                              score < 8? refine query, retry (max 3)
                              score >= 8? return answer with citations
```

**Two retrieval modes:**

| Mode | Trigger | How it works | Best for |
|---|---|---|---|
| Vector (default) | `rag query "..."` | Embed query → cosine similarity → top-K chunks | Factoid questions |
| Keyword | `rag query "..." --keyword "regex"` | Regex scan all chunks → matched chunks → LLM | Enumeration, structured extraction |

---

## Setup

### 1. Install dependencies

```bash
cd ~/code/rag-lab
uv sync
```

If `uv sync` pulls CUDA torch (unnecessary on CPU), install CPU torch first:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv sync
```

### 2. Configure LLM

Copy your API key into `/home/flo/RAG-Project/.env` (or export it):

```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3.7-plus
LLM_VERIFIER_MODEL=qwen/qwen3.7-plus
```

Any OpenAI-compatible provider works. Set `LLM_BASE_URL` and `LLM_MODEL` accordingly.

### 3. Verify

```bash
uv run rag stats
```

Should show chunk count 0 on first run, or the count from your existing DB.

---

## CLI Reference

### `rag ingest` — add documents to the vector store

```bash
uv run rag ingest <file> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--strategy` | `sentence` | `sentence` (semantic) or `fixed` (character-window) |
| `--size` | `512` | Target chunk size in characters |
| `--overlap` | `64` | Overlap between chunks in characters |
| `--db-path` | `./chroma_db` | ChromaDB persist directory (global option) |

**Examples:**

```bash
# Ingest a PDF with sentence-aware chunking
uv run rag ingest ~/Documents/handbook.pdf

# Ingest with fixed-size chunks for comparison
uv run rag ingest ~/Documents/handbook.pdf --strategy fixed --size 256 --overlap 32

# Ingest to a specific database
uv run rag --db-path ~/my_chroma_db ingest report.md
```

**Supported formats:** `.pdf`, `.epub`, `.md`, `.markdown`

**What happens:**
1. Parser extracts raw text from the file
2. Chunker splits text into overlapping chunks (sentence-aware or fixed-size)
3. Embedder converts each chunk to a 384-dimensional vector (all-MiniLM-L6-v2, CPU)
4. ChromaDB upserts chunks with embeddings, metadata, and unique IDs (SHA-256 prefix)

---

### `rag query` — ask questions against ingested documents

```bash
uv run rag query "<question>" [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--top-k` | `20` | Chunks fetched from ChromaDB for vector search |
| `--rerank` | `5` | Chunks actually passed to the LLM |
| `--min-score` | `8` | Minimum verifier score (1-10) to accept answer |
| `--max-tokens` | `600` | Maximum output tokens from the LLM |
| `--keyword` | — | Regex pattern to enable keyword retrieval mode |
| `--trace` | — | Show full iteration trace (query, answer, score per round) |
| `--db-path` | `./chroma_db` | ChromaDB persist directory (global option) |

**Examples:**

```bash
# Vector search — factoid question
uv run rag query "How many credit points is the AI in Business program?"

# With trace to see the refinement loop
uv run rag query "What is the Monte Carlo method?" --trace

# Lower verifier threshold, more chunks
uv run rag query "Describe the risk models" --min-score 6 --rerank 10

# Keyword mode — enumerate all modules matching a pattern
uv run rag query "Liste alle Module" --keyword "Modulcode|Modultitel" --rerank 100 --max-tokens 3000

# Keyword mode — find all chunks mentioning a specific code
uv run rag query "What is DLBFMWFT1 about?" --keyword "DLBFMWFT1"
```

**The /goal retrieval loop:**
1. Embed the question → query ChromaDB → get top-K chunks → slice to rerank-top
2. Send chunks + question to the LLM (generator) → get answer
3. Send answer + chunks + question to the LLM (verifier) → get {score, grounded, issues, verdict}
4. If score >= min_score: return answer
5. If score < min_score: append issues to question, retry (max 3 iterations)

---

### `rag stats` — show vector store statistics

```bash
uv run rag stats [--db-path PATH]
```

Displays: collection name, chunk count, database path, embedder model, LLM model.

---

### `rag serve` — launch web test console

```bash
uv run rag serve [--host 127.0.0.1] [--port 8000]
```

Opens a web UI at `http://127.0.0.1:8000` with three panels:

- **Ingest** — drag & drop file upload with strategy/size/overlap controls
- **Query** — question input with top-K, min-score, trace viewer
- **Stats** — live chunk count and database path

API endpoints:
- `POST /api/ingest` — multipart file upload
- `POST /api/query` — JSON `{question, top_k, min_score}`
- `GET /api/stats` — JSON `{chunk_count, db_path}`

---

## Parsers

### PDF (`pypdf`)
Extracts text from all pages. Handles text-based PDFs. Scanned/image-only PDFs produce empty output — no OCR.

### EPUB (`ebooklib` + `BeautifulSoup`)
Extracts text from all HTML documents in the EPUB container. Strips HTML tags, preserves structure via separators.

### Markdown (`markdown-it-py`)
Strips YAML frontmatter, renders markdown to plain-ish text. Handles headings, paragraphs, inline text, code blocks. Lists, tables, and blockquotes are simplified.

---

## Chunking Strategies

### `sentence` (default)
Splits on sentence boundaries (`[.!?]\s+`). Groups sentences into chunks of ~target_size characters. Keeps `overlap` previous sentences as context overlap. Preserves semantic boundaries.

### `fixed`
Character-window sliding. Size `N`, overlap `M`. Fast, but may split mid-word or mid-sentence.

**Comparing strategies:**
```bash
uv run rag ingest book.pdf --strategy sentence
uv run rag query "some question"
# Then re-ingest and compare:
uv run rag ingest book.pdf --strategy fixed --size 300
uv run rag query "some question"
```

---

## Embedder

`all-MiniLM-L6-v2` from sentence-transformers. 384-dimensional embeddings. Runs on CPU only (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`). Model is cached locally after first download (~80 MB).

---

## Vector Store

ChromaDB in persistent mode (`PersistentClient`). Cosine distance metric. Data stored in `./chroma_db/` (configurable via `--db-path`).

**Warning:** The database is CWD-relative by default. Running from different directories creates separate databases. Use `--db-path` for consistency.

---

## LLM Configuration

All LLM settings are read from environment variables / `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter API key (preferred) |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (fallback) |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `deepseek-chat` | Model for answer generation |
| `LLM_VERIFIER_MODEL` | (same as `LLM_MODEL`) | Model for verification (can differ) |

**Tested models (all $0/$1 per M tokens via OpenRouter):**

| Model | Generation | Verification | Notes |
|---|---|---|---|
| `qwen/qwen3.7-plus` | 10/10 reliable | 10/10 reliable | Recommended |
| `deepseek/deepseek-v4-pro` | Good, no hallucination | Works, intermittent null | Solid backup |
| `minimax/minimax-m3` | Excellent when available | Returns null (don't use) | Unreliable on free tier |
| `xiaomi/mimo-v2.5-pro` | Excellent when available | Returns null (don't use) | Unreliable on free tier |

---

## Verifier

The verifier is a separate LLM call with a different prompt. It receives the question, the retrieved chunks, and the generator's answer, and outputs structured JSON:

```json
{
  "score": 8,
  "grounded": true,
  "issues": ["Missing detail about X"],
  "verdict": "GROUNDED"
}
```

Verdicts: `GROUNDED` (all claims supported), `PARTIAL` (some gaps), `UNGROUNDED` (claims not in context), `ERROR` (verifier itself failed).

If score < `--min-score`, the system refines the query by appending the verifier's issues and retries (up to 3 times).

---

## Keyword Retrieval

When `--keyword` is supplied, vector search is bypassed entirely. The system:

1. Paginates through ALL chunks in ChromaDB (batches of 500)
2. Regex-matches each chunk's text against the pattern
3. Deduplicates by chunk ID
4. Collects up to `--rerank` matching chunks
5. Feeds them directly to the LLM in one prompt

**Use cases:**
- Enumerate all items matching a pattern (module codes, dates, names)
- Find specific identifiers regardless of semantic similarity
- Extract structured lists from semi-structured documents

**Pattern tips:**
```bash
# Module codes: DLB followed by uppercase letters/numbers
--keyword "DLB[A-Z0-9]+"

# Module headers in the document (language-specific)
--keyword "Modulcode|Modultitel|Kurscode"

# Find mentions of a specific person or term
--keyword "Kerron Samaroo"

# German exam types
--keyword "Klausur|Hausarbeit|Seminararbeit|Written Assessment"
```

---

## Test Plan

### T1 — Single-document roundtrip
```bash
uv run rag ingest document.pdf
uv run rag query "Key question about the document" --trace
```
Expect: cited answer with score >= 8, trace showing iterations.

### T2 — Multi-format retrieval
```bash
uv run rag ingest book.pdf
uv run rag ingest notes.md
uv run rag ingest supplement.epub
uv run rag query "question spanning all three"
```
Expect: chunks from multiple sources in the answer citations.

### T3 — Chunking comparison
```bash
uv run rag ingest book.pdf --strategy sentence
uv run rag ingest book.pdf --strategy fixed --size 300
# Compare answer quality for the same question
```

### T4 — Hallucination refusal
```bash
uv run rag query "How do I bake a chocolate soufflé?"
```
Expect: "I don't know from the provided documents." with score 10/10.

### T5 — Verifier-governed refinement
```bash
uv run rag query "vague or ambiguous question" --trace
```
Expect: trace shows multiple iterations, score improves across rounds, or max_iters reached with `partial: true`.

### T6 — Keyword enumeration
```bash
uv run rag query "List all modules" --keyword "Modulcode|Modultitel" --rerank 100 --max-tokens 3000
```
Expect: comprehensive list with citations.

---

## File Structure

```
rag-lab/
├── pyproject.toml          # Project metadata, dependencies, CLI entry point
├── README.md               # Quick start
├── documentation.md        # This file
├── .gitignore
├── data/                   # Sample test documents
│   ├── pdfs/
│   ├── epubs/
│   └── markdowns/
├── chroma_db/              # ChromaDB persist directory (gitignored)
├── rag_lab/
│   ├── __init__.py
│   ├── config.py           # LLM configuration (env vars)
│   ├── cli.py              # Typer CLI (ingest, query, stats, serve)
│   ├── chunker.py          # Fixed and sentence-aware chunking
│   ├── embedder.py         # SentenceTransformer wrapper (CPU)
│   ├── vector_store.py     # ChromaDB client (upsert, query, keyword_search)
│   ├── retriever.py        # Retrieval loop, generator, refinement
│   ├── verifier.py         # Grounding auditor (separate LLM call)
│   ├── web.py              # FastAPI web test console
│   └── parsers/
│       ├── __init__.py     # Parser registry, pick_parser()
│       ├── pdf.py          # pypdf text extraction
│       ├── epub.py         # ebooklib + BeautifulSoup
│       └── markdown.py     # markdown-it-py + frontmatter stripping
└── tests/                  # (reserved for future tests)
```

---

## Known Limitations

- **Vector retrieval ceiling**: Only `rerank` chunks reach the LLM. Exhaustive enumeration requires keyword mode.
- **No OCR**: Scanned/image PDFs produce empty text. Only text-layer PDFs work.
- **Single-collection**: One ChromaDB collection. No multi-tenancy or document-level deletion.
- **No incremental updates**: Re-ingesting a file with the same content replaces chunks by ID, but there's no "remove document X" command.
- **No streaming**: LLM responses are fully buffered. No token-by-token output.
- **CWD-dependent DB**: Default `./chroma_db` is relative to working directory. Use `--db-path` for absolute paths.
- **No authentication on web UI**: `rag serve` binds to 127.0.0.1 by default. Changing `--host` exposes it without auth.
- **Cross-language retrieval**: `all-MiniLM-L6-v2` is multilingual but weaker at cross-language matching than dedicated multilingual models.
