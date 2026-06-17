# rag-lab
Standalone RAG learning project. PDF + EPUB + Markdown → Chroma → /goal retrieval loop with verifier.

## Quick Start

```bash
git clone https://github.com/notabanker/rag-lab.git
cd rag-lab

# On CPU-only machines (skip CUDA torch download):
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv sync

# On machines with GPU/CUDA:
uv sync
```

## LLM Configuration

Set your API key and model. Using OpenRouter with qwen3.7-plus (recommended):

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_MODEL=qwen/qwen3.7-plus
```

Or add them to a `.env` file and source it:

```bash
export $(grep -v '^#' .env | xargs)
```

Any OpenAI-compatible provider works — just change `LLM_BASE_URL` and `LLM_MODEL`.

## Usage

```bash
# Ingest documents
uv run rag ingest ~/Documents/handbook.pdf
uv run rag ingest notes.md --strategy fixed --size 300

# Ask questions (vector search — best for factoid questions)
uv run rag query "How many credit points is the program?" --trace

# Enumerate everything (keyword search — best for listing all items)
uv run rag query "List all modules" --keyword "DLB[A-Z0-9]+" --rerank 100 --max-tokens 3000

# Show stats
uv run rag stats

# Launch web UI
uv run rag serve
# → http://127.0.0.1:8000
```

## Full Documentation

See [documentation.md](documentation.md) for architecture, all CLI flags, keyword retrieval, verifier details, model comparison, and test plan.
