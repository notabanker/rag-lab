# rag-lab Documentation

Standalone RAG (Retrieval-Augmented Generation) learning project.  
Ingest PDF, EPUB, Markdown → embed → ChromaDB → hybrid retrieval → cited answer with verifier → eval gates, dashboard, and MCP tools.

---

## Architecture

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│  Parser   │───▶│  Chunker  │───▶│  Embedder  │───▶│  ChromaDB │
│ pdf/epub/md│    │ fixed/   │    │ multilingual│   │ Persistent │
│            │    │ sentence │    │   E5 CPU   │    │  Client    │
└──────────┘    └──────────┘    └───────────┘    └──────────┘
                                                       │
                    ┌──────────────────────────────────┘
                    ▼
              ┌──────────┐    ┌───────────┐    ┌──────────┐
              │ Retriever │───▶│  Generator │───▶│ Verifier  │
              │ top-k=50  │    │  (LLM)     │    │ (LLM)     │
              │ small→big │    │            │    │ score 1-10│
              └──────────┘    └───────────┘    └──────────┘
                    │                │               │
                    └────────────────┴───────────────┘
                                     │
                              score < 8? refine query, retry (max 3)
                              score >= 8? return answer with citations
```

**Retrieval modes:**

| Mode | Trigger | How it works | Best for |
|---|---|---|---|
| Hybrid (default) | `rag query "..."` | Vector + BM25, fused with reciprocal rank fusion, then cross-encoder rerank | Almost everything |
| Vector | `--mode vector` | Embed query → cosine similarity → top-K chunks | Paraphrased/semantic questions |
| Lexical | `--mode lexical` | BM25 over all chunks | Exact terms, codes, names |
| Keyword | `--keyword "regex"` | Regex scan all chunks → matched chunks → LLM | Exhaustive enumeration |

The V3 default pipeline is: fetch `top_k` child candidates per source → RRF fusion → multilingual cross-encoder rerank (`--no-rerank` to skip) → expand the best child chunks into parent context (`--flat` to skip) → send cited context to the LLM.

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

Export the variables in your shell (nothing loads a `.env` file automatically — `config.py` reads only the process environment):

```bash
export OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_MODEL=qwen/qwen3.7-plus
export LLM_VERIFIER_MODEL=qwen/qwen3.7-plus
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
| `--parent-size` | `4` | Child chunks grouped into one parent context for small-to-big retrieval |
| `--force-model-mismatch` | off | Allow ingest into a legacy or mismatched collection |
| `--allow-empty` | off | Record a zero-text file in the manifest instead of failing |
| `--db-path` | `$RAG_DB_PATH` or `~/.local/share/rag-lab/chroma_db` | ChromaDB persist directory (global option) |

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
1. Parser extracts raw text from the file and reports per-section (page/chapter) character counts
2. Chunker splits text into overlapping chunks (sentence-aware or fixed-size)
3. Embedder converts each child chunk with `EMBEDDING_MODEL` (default `intfloat/multilingual-e5-small`, CPU)
4. Metadata records doc ID, parent ID, citation label, chunking version, and embedding model
5. Any chunks previously ingested from the same file content (matched by SHA-256 prefix) are deleted
6. ChromaDB upserts chunks with embeddings, metadata, and unique IDs (SHA-256 prefix + index)

**Parse-quality report:** every ingest prints sections, extracted characters, and warnings, and persists the report to the manifest (visible via `rag docs show` and the dashboard Corpus view). Heuristics: zero extracted text (scanned/image-only file), average under 50 chars/page, or more than half the pages empty — all signs of a scanned PDF. Zero-text files are refused by default; `--allow-empty` records them in the manifest (0 chunks) so they show up as known-bad instead of disappearing silently. The web API returns the warnings in the `POST /api/ingest` response.

### `rag rebuild` — batch-ingest a corpus

```bash
uv run rag rebuild data/markdowns/*.md data/pdfs/*.pdf data/epubs/*.epub
```

`rebuild` uses the same options as `ingest` and writes to the active collection. V3's default collection is versioned from the embedding model and chunking version, e.g. `rag_lab_v3-intfloat-multilingual-e5-small-sentence-v2-parent`, so V3 indexes do not silently mix with legacy M2 indexes.

---

### `rag query` — ask questions against ingested documents

```bash
uv run rag query "<question>" [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--mode` | `hybrid` | Retrieval mode: `vector`, `lexical` (BM25), or `hybrid` (RRF fusion) |
| `--top-k` | `50` | Candidates fetched per source before reranking |
| `--rerank` | `5` | Child candidates selected after fusion/rerank |
| `--no-rerank` | off | Skip the cross-encoder reranker |
| `--small-to-big / --flat` | small-to-big | Expand winning child chunks into parent context |
| `--parent-top-k` | `5` | Parent contexts passed to the LLM |
| `--max-context-chars` | `12000` | Character budget for context sent to the LLM |
| `--min-score` | `8` | Minimum verifier score (1-10) to accept answer |
| `--max-tokens` | `600` | Maximum output tokens from the LLM |
| `--keyword` | — | Regex pattern to enable keyword retrieval mode |
| `--trace` | — | Show full iteration trace (query, answer, score per round) |
| `--db-path` | `$RAG_DB_PATH` or `~/.local/share/rag-lab/chroma_db` | ChromaDB persist directory (global option) |
| `--collection` | versioned V3 name | ChromaDB collection name (global option) |

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
1. Embed the question → query ChromaDB and/or BM25 → get top-K child candidates
2. Fuse/rerank child candidates → optionally expand into parent contexts
3. Send cited context + question to the LLM (generator) → get answer
4. Send answer + context + question to the LLM (verifier) → get {score, grounded, issues, verdict}
5. If score >= min_score: return answer
6. If score < min_score: append issues to question, retry (max 3 iterations)

---

### `rag stats` — show vector store statistics

```bash
uv run rag stats [--db-path PATH]
```

Displays: collection name, chunk count, document count, database path, configured embedder, collection embedder, reranker, and LLM models.

---

### `rag config` — show runtime configuration

```bash
uv run rag config show
```

Displays the active LLM endpoint/model, verifier model, API-key presence, local embedder/reranker, default collection, active collection, and DB path. This is the fastest way to confirm OpenRouter/Qwen is actually configured before running a full eval.

---

### `rag collections` and `rag docs` — manage the corpus

```bash
uv run rag collections list
uv run rag collections delete <collection-name>
uv run rag docs list
uv run rag docs show <doc_id-or-source-or-filename>
uv run rag docs reingest <doc_id-or-source-or-filename>
uv run rag docs delete <doc_id-or-source>
```

Collections show chunk count plus index metadata. Deleting the active collection is refused; switch `--collection` first if you really want to remove another index.

Documents are grouped by `doc_id`/file SHA and can be inspected, reingested, or deleted without reaching into Chroma manually. Ingest writes a SQLite document manifest to `<db-path>/runs.sqlite3`, so `docs show` can display both the original source path and the currently indexed state. Delete accepts exact `doc_id`, `file_sha`, source path, or source filename.

---

### `rag eval` — run the golden question set

```bash
uv run rag eval [--questions eval/questions.yaml] [--retrieval-only] [--json]
```

| Option | Default | Description |
|---|---|---|
| `--questions` | `eval/questions.yaml` | Golden question set (see Evaluation section) |
| `--retrieval-only` | off | Skip LLM calls entirely; retrieval metrics only (free, fast) |
| `--mode` | `hybrid` | Retrieval mode for the eval run |
| `--top-k` | `50` | Child candidates fetched before fusion/reranking |
| `--rerank` | `5` | Child candidates selected — also the k in hit@k |
| `--no-rerank` | off | Skip the cross-encoder reranker |
| `--small-to-big / --flat` | small-to-big | Expand child hits into parent context |
| `--parent-top-k` | `5` | Parent contexts selected after child ranking |
| `--min-score` | `8` | Verifier acceptance threshold |
| `--variant` | `default` | Label stored with this eval run |
| `--gate` | — | YAML metric gate file; exits with code 2 if checked gates fail |
| `--json` | off | Print the full report as JSON |

Metrics: **hit@1/3/k/10**, **MRR**, **chunk hit/MRR**, **context hit/MRR** after small-to-big expansion, **fragment rate** (answer contains an expected substring), **verifier mean**, **refusal accuracy**, **citation validity**, retrieval latency, context size, and token totals. Every eval run is persisted to SQLite.

Gate example:

```bash
uv run rag eval --retrieval-only --gate eval/gates.yaml
```

Retrieval-only mode checks retrieval gates and skips answer gates. Full mode also checks fragment rate, refusal accuracy, verifier score, and citation validity.

---

### `rag compare` — A/B configs side by side

```bash
uv run rag compare [NAMES...] [--variants eval/variants.yaml] [--retrieval-only]
```

Runs the eval once per variant defined in `eval/variants.yaml` (fields: `name`, `mode`, `top_k`, `rerank_top`, `use_reranker`, `small_to_big`, `parent_top_k`, `max_context_chars`, `min_score`, `max_iters`, `collection`) and prints a metric-by-metric table with the best value per row highlighted. The shipped variants include the flat retrieval ablation plus the V3 default:

```bash
uv run rag compare --retrieval-only            # flat modes plus v3-hybrid-rerank
uv run rag compare hybrid-rerank v3-hybrid-rerank
```

A variant's `collection` lets it run against a separately-ingested index — e.g. the same corpus chunked with a different strategy (`rag --collection fixed_idx ingest book.pdf --strategy fixed`, then add a variant with `collection: fixed_idx`).

---

### `rag runs` — inspect logged retrieval runs

```bash
uv run rag runs list [--limit 20]
uv run rag runs show <ID>
```

Every `rag query` and web query is logged to `<db-path>/runs.sqlite3`: question, config, answer, verifier verdict, iteration trace, retrieved chunks with distances, latency, and token usage. `runs show` replays a run in full.

---

### `rag serve` and dashboard — launch the web UI

```bash
uv run rag serve [--host 127.0.0.1] [--port 8000]
```

`rag serve` starts the FastAPI backend on `http://127.0.0.1:8000`. It still serves a small built-in test console at `/`, but the main V3.1 UI is the React dashboard:

```bash
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

Dashboard views:

- **Ask** — query mode, reranker toggle, small-to-big toggle, answer, citations, context, trace
- **Corpus** — upload, document list, reingest, delete
- **Runs** — run history and full JSON details
- **Eval** — retrieval-only or full API eval from the browser
- **Settings** — active models, collections, DB path, API-key presence

API endpoints (errors are returned as proper HTTP status codes with a JSON `detail`):
- `POST /api/ingest` — multipart file upload; optional form fields `strategy`, `chunk_size`, `overlap`
- `POST /api/query` — JSON `{question, mode?, top_k?, rerank_top?, use_reranker?, small_to_big?, min_score?}`
- `GET /api/stats` and `GET /api/config`
- `GET /api/collections`, `DELETE /api/collections/{name}`
- `GET /api/docs`, `GET /api/docs/{id}`, `POST /api/docs/{id}/reingest`, `DELETE /api/docs/{id}`
- `GET /api/runs`, `GET /api/runs/{id}`
- `GET /api/evals`, `GET /api/evals/{id}`, `POST /api/eval`

`POST /api/eval` defaults to retrieval-only. Full evals call the configured LLM and should be run only when `OPENROUTER_API_KEY` is present.

---

### `rag mcp serve` — expose rag-lab to MCP clients

```bash
uv run rag mcp serve
```

The MCP server runs on stdio and exposes corpus/retrieval tools:

- `rag_search` — retrieve candidates and selected context without LLM calls
- `rag_answer` — run the full answer loop
- `rag_ingest` — ingest a file path
- `rag_delete` — delete by exact `doc_id`, file SHA, source path, or filename; requires `confirm=true`
- `rag_docs_list` and `rag_reingest`
- `rag_collections_list`
- `rag_runs_show`
- `rag_eval_run`

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

`EMBEDDING_MODEL` defaults to `intfloat/multilingual-e5-small` from sentence-transformers. Query text is automatically prefixed with `query:` and document text with `passage:` for E5-family models, and embeddings are normalized for cosine search. Runs on CPU only (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`). Model is cached locally after first download.

---

## Vector Store

ChromaDB in persistent mode (`PersistentClient`). Cosine distance metric. Data lives in `~/.local/share/rag-lab/chroma_db` by default, so every working directory sees the same corpus. Override with `RAG_DB_PATH` or per-invocation with `--db-path`. If a legacy `./chroma_db` exists in the current directory, the CLI prints a migration hint.

---

## Runtime Configuration

Runtime settings are read from environment variables. The app does not automatically load a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter API key (preferred) |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (fallback) |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen/qwen3.7-plus` | Model for answer generation |
| `LLM_VERIFIER_MODEL` | (same as `LLM_MODEL`) | Model for verification (can differ) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Local sentence-transformer embedder |
| `EMBEDDING_BATCH_SIZE` | `32` | Embedding batch size |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual cross-encoder for reranking (local, no API) |
| `RERANKER_BATCH_SIZE` | `16` | Reranker batch size |
| `RAG_COLLECTION` | versioned V3 name | Default Chroma collection name |
| `RAG_DB_PATH` | `~/.local/share/rag-lab/chroma_db` | ChromaDB persist directory (also holds `runs.sqlite3`); `--db-path` overrides per invocation |

**Tested OpenRouter model choices:**

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

## Citation Validation

The generator is instructed to cite retrieved context labels such as `[ba_fintech.pdf p.42]`. After every answer, `rag_lab/citations.py` extracts bracketed citations and compares them against the context labels actually sent to the model.

Validation is logged with each run and surfaced in CLI, web, dashboard, eval, and MCP responses:

- `citation_valid`: true only when every citation is present in context
- `citation_count`: number of citations extracted from the answer
- `citation_errors`: missing citations, fake citations, or uncited non-refusal answers
- `citation_validity_rate`: full-eval aggregate used by answer gates

Correct refusals are not forced to cite context, but grounded answers are expected to cite at least one retrieved chunk.

---

## Hybrid Retrieval & Reranking

**BM25 lexical index** (`rag_lab/lexical.py`): built lazily from all chunks in the collection, cached in memory, invalidated on ingest/delete. The tokenizer keeps letters and digits together (module codes like `DLBFMWFT1` stay one token) and handles German umlauts.

**Reciprocal rank fusion**: hybrid mode fetches `top_k` candidates from both the vector index and BM25, then fuses them: `score(chunk) = Σ 1/(60 + rank)` across both lists. A chunk found by both sources outranks chunks found by only one.

**Cross-encoder reranker** (`rag_lab/reranker.py`): the fused candidate pool is rescored by a multilingual cross-encoder that reads the query and chunk *together* — far more accurate than embedding cosine distance, at the cost of one forward pass per candidate. Model set via `RERANKER_MODEL` (default `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`). Disable per query with `--no-rerank`.

**Small-to-big context**: chunks are embedded and ranked as small child chunks, then the best children expand to parent contexts using `parent_id`. The trace records both `candidate_ids` and final `context_ids`, while the prompt uses human-readable citation labels such as `ba_fintech.pdf p.42`.

Measure the effect on your own corpus with `uv run rag compare --retrieval-only` — the shipped variants ablate these stages. The `chunk_hit_rate`/`chunk_mrr` metrics (questions with `expected_chunk_fragments`) are the ones to watch: they require the *right chunk*, not just the right file.

**V3 reference results** (23-question golden set, sample corpus, 4,072 chunks, retrieval-only, measured 2026-07-06):

| Metric | vector | lexical | hybrid | hybrid-rerank | v3-hybrid-rerank |
|---|---|---|---|---|---|
| Hit rate @5 | 95% | 90% | 100% | 100% | **100%** |
| MRR | 0.88 | 0.85 | 0.93 | **0.96** | **0.96** |
| Chunk hit rate @5 | 82% | 64% | 82% | **91%** | **91%** |
| Chunk MRR | 0.75 | 0.59 | 0.76 | **0.77** | **0.77** |
| Context chars avg | 3180 | 2597 | 2977 | 3301 | 8027 |

Takeaways: the multilingual E5 embedder fixes the old vector ceiling, hybrid retrieval reaches 100% source hit even without rerank, and the multilingual reranker lifts chunk hit rate to 91%. Small-to-big does not change child-ranking metrics, but it gives the LLM larger cited context windows.

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

## Evaluation

The eval harness is the lab bench: every pipeline change should be justified by a before/after eval, not a vibe.

**Golden questions** (`eval/questions.yaml`):

```yaml
questions:
  - id: var-definition
    question: What does a one-day 95% VaR of $1 million mean?
    expected_sources: [risk_management.md]   # basenames; ANY match = retrieval hit
    expected_chunk_fragments: ["single day"]
    expected_fragments: ["5%", "single day"] # any-match substrings for the answer
    tags: [factoid, en]
  - id: refusal-tire
    question: How do I change a flat tire?
    expect_refusal: true                     # correct behavior is "I don't know..."
```

**Workflow:**

```bash
uv run rag eval --retrieval-only     # free: hit@k + MRR only, no API calls
uv run rag eval --retrieval-only --gate eval/gates.yaml
uv run rag eval                      # full loop: + fragments, verifier, refusals, cost
uv run rag compare --retrieval-only  # ablation across eval/variants.yaml
uv run rag runs list                 # inspect individual logged runs
```

Eval results and per-query runs are stored in `<db-path>/runs.sqlite3`, so results are comparable across time and config changes.

`expected_sources` measures file-level recall. `expected_chunk_fragments` measures whether the exact supporting chunk made it into the top-k child candidates. After small-to-big expansion, context metrics check whether the final LLM window still contains the expected support.

**Retrieval ablation results** (23 questions incl. hard exact-code, paraphrase, and cross-language cases; sample corpus, 4,072 chunks; `top_k=50`, `k=5`; measured 2026-07-06):

| Metric | vector | lexical | hybrid | hybrid-rerank | v3-hybrid-rerank |
|---|---|---|---|---|---|
| Hit rate @1 | 86% | 81% | 90% | **95%** | **95%** |
| Hit rate @k | 95% | 90% | **100%** | **100%** | **100%** |
| MRR | 0.88 | 0.85 | 0.93 | **0.96** | **0.96** |
| Chunk hit rate @k | 82% | 64% | 82% | **91%** | **91%** |
| Chunk MRR | 0.75 | 0.59 | 0.76 | **0.77** | **0.77** |
| Context chunk hit | 82% | 64% | 82% | **91%** | **91%** |

Reading: multilingual vector retrieval is now competitive with BM25, hybrid is best at source recall, and reranking is still the chunk-precision lever. `v3-hybrid-rerank` is the default retrieval shape because it keeps the child-ranking gains while handing the LLM expanded parent context.

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
├── dashboard/              # React/Vite dashboard
│   ├── src/App.tsx
│   └── src/styles.css
├── data/                   # Sample test documents
│   ├── pdfs/
│   ├── epubs/
│   └── markdowns/
├── eval/
│   ├── questions.yaml      # Golden question set for `rag eval`
│   ├── variants.yaml       # Named configs for `rag compare`
│   └── gates.yaml          # Retrieval/answer metric gates
├── rag_lab/
│   ├── __init__.py
│   ├── config.py           # LLM configuration (env vars)
│   ├── cli.py              # Typer CLI
│   ├── citations.py        # Citation extraction and validation
│   ├── chunker.py          # Fixed and sentence-aware chunking
│   ├── embedder.py         # SentenceTransformer wrapper (CPU)
│   ├── gates.py            # Eval gate loading and evaluation
│   ├── ingestion.py        # Shared ingest metadata/upsert flow
│   ├── lexical.py          # BM25 index and RRF fusion
│   ├── manifest.py         # SQLite document manifest
│   ├── mcp_server.py       # MCP tools for search/answer/ingest/delete/eval
│   ├── reranker.py         # Cross-encoder reranker
│   ├── vector_store.py     # ChromaDB client (collections, upsert, query, keyword_search)
│   ├── retriever.py        # Retrieval loop, generator, refinement, run logging
│   ├── verifier.py         # Grounding auditor (separate LLM call)
│   ├── evaluation.py       # Eval harness: golden questions, metrics, variants
│   ├── runs.py             # SQLite persistence for runs and eval results
│   ├── web.py              # FastAPI web test console
│   └── parsers/
│       ├── __init__.py     # Parser registry, pick_parser()
│       ├── pdf.py          # pypdf text extraction
│       ├── epub.py         # ebooklib + BeautifulSoup
│       └── markdown.py     # markdown-it-py + frontmatter stripping
└── tests/
    ├── test_chunker.py     # Chunker unit tests (offsets, guards, edge cases)
    ├── test_citations.py   # Citation validator
    ├── test_evaluation.py  # Metric math, question/variant loader validation
    ├── test_gates.py       # Metric gates
    ├── test_mcp_server.py  # MCP helper behavior
    ├── test_runs.py        # Run log round-trips
    └── test_web_api.py     # Dashboard API endpoints
```

---

## Known Limitations

- **Full eval requires an API key**: Retrieval-only evals are local. Full answer evals need `OPENROUTER_API_KEY` or another configured provider key.
- **Local model downloads**: The E5 embedder and cross-encoder reranker are local models. First use may download weights, and reranked evals are slower on CPU.
- **Only selected context reaches the LLM**: Exhaustive enumeration still requires keyword mode or a larger `--rerank`/`--parent-top-k`.
- **No OCR**: Scanned/image PDFs produce empty text. Only text-layer PDFs work.
- **Collection safety is manual**: Collections are versioned and deletable, but the tool is still a single-user personal RAG lab, not a multi-tenant service.
- **Re-ingest replaces by content hash**: Re-ingesting a file deletes previous chunks with the same SHA before upserting the new chunks.
- **No streaming**: LLM responses are fully buffered. No token-by-token output.
- **No authentication on web UI**: `rag serve` binds to 127.0.0.1 by default. Changing `--host` exposes it without auth.
