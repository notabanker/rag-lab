# rag-lab

Personal RAG lab for local corpora: PDF + EPUB + Markdown + DOCX -> versioned Chroma collections -> multilingual hybrid retrieval (E5 vectors + BM25 + cross-encoder rerank + small-to-big context) -> cited answers with verifier and eval gates.

## Setup

```bash
cd ~/rag-lab
uv sync
```

LLM calls use an OpenAI-compatible endpoint. Defaults are OpenRouter + Qwen:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_MODEL=qwen/qwen3.7-plus
export LLM_VERIFIER_MODEL=qwen/qwen3.7-plus
```

Local retrieval models are downloaded by sentence-transformers on first use:

```bash
export EMBEDDING_MODEL=intfloat/multilingual-e5-small
export RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

The corpus lives in `~/.local/share/rag-lab/chroma_db` by default (override with
`RAG_DB_PATH` or `--db-path`), so the CLI works from any directory.

The web API refuses to bind a non-loopback host without a token (fail-closed),
and uploads are capped by default:

```bash
export RAG_API_TOKEN=change-me     # required for non-loopback hosts
export RAG_MAX_UPLOAD_MB=200       # per-ingest upload cap
export RAG_EVAL_DIR=./eval         # allowlist root for eval question files
```

Check the active runtime config:

```bash
uv run rag config show
```

Before committing, run the green check (tests + retrieval gate):

```bash
scripts/check.sh
```

## CLI

```bash
# Ingest one file or rebuild the bundled corpus
uv run rag ingest ~/some/book.pdf
uv run rag ingest ~/notes/file.md
uv run rag rebuild data/markdowns/*.md data/pdfs/*.pdf data/epubs/*.epub

# Ask with the V3.1 default: hybrid + rerank + small-to-big
uv run rag query "What is DLBBWME about?" --trace

# Inspect and manage corpus state
uv run rag stats
uv run rag collections list
uv run rag collections delete old_collection
uv run rag docs list
uv run rag docs show <doc_id-or-source>
uv run rag docs reingest <doc_id-or-source>
uv run rag docs delete <doc_id-or-source>

# Retrieval-only evals are free; full evals call the configured LLM
uv run rag eval --retrieval-only --gate eval/gates.yaml
uv run rag compare --retrieval-only
uv run rag eval

# Inspect persisted runs and evals
uv run rag runs list
uv run rag runs show <run_id>
```

## Dashboard

Start the FastAPI backend:

```bash
uv run rag serve --port 8000
```

Start the React dashboard in another shell:

```bash
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The dashboard has Ask, Corpus, Runs, Eval, and Settings views, including upload, delete, reingest, retrieval/full eval, trace, citation validation, and config inspection.

## MCP

Run the MCP server on stdio:

```bash
uv run rag mcp serve
```

Tools include search, answer, ingest, delete with confirmation, document list, reingest, collection list, run show, and eval run.

## Current Baseline

Measured on July 6, 2026 against the bundled 4,072-chunk corpus and 23-question eval set:

| Variant | Hit@5 | MRR | Chunk Hit@5 | Chunk MRR |
|---|---:|---:|---:|---:|
| V3.1 default (`hybrid` + rerank + small-to-big) | 100% | 0.96 | 91% | 0.77 |

The retrieval gate passes:

```bash
uv run rag eval --retrieval-only --gate eval/gates.yaml
```

Full answer gates additionally check fragments, refusal behavior, verifier score, and citation validity. They require `OPENROUTER_API_KEY`.

## More

See [documentation.md](documentation.md) for architecture, CLI/API details, evaluation schema, gates, and the test plan.
