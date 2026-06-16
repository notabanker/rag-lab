# rag-lab
Standalone RAG learning project. PDF + EPUB + Markdown → Chroma → /goal retrieval loop with verifier.

## Setup
```bash
cd ~/code/rag-lab
uv sync
```
In another terminal:
```bash
ollama serve   # if not already running
ollama pull qwen2.5:1.5b
```

## Usage
```bash
# Ingest a file
uv run rag ingest ~/some/book.pdf
uv run rag ingest ~/some/book.epub
uv run rag ingest ~/notes/file.md

# Query
uv run rag query "What is the main risk model in chapter 3?" --trace

# Stats
uv run rag stats
```

## What to test
- T1: Single-PDF roundtrip → 3 questions, expect cited answers
- T2: Multi-format mix → retriever pulls from all 3
- T3: Chunking comparison → `uv run rag ingest book.pdf --strategy fixed` then `--strategy sentence`, compare
- T5: Halluzination → ask about topic NOT in any ingested doc, expect "I don't know"
- T6: Verifier catch → if a query triggers bad answer, check `verifier.score < 8` in output
