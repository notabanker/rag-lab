#!/usr/bin/env bash
# Green check: full test suite + retrieval gate. Run before every commit/push.
# Baseline (v3.1.0, 32-question set): hit@5 97%, MRR 0.94, chunk hit@5 87%, chunk MRR 0.78, context hit 97%.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== pytest =="
uv run pytest -q

echo
echo "== ruff (E9/F: real bugs only) =="
uv run ruff check --select E9,F rag_lab tests

echo
echo "== retrieval gate (eval/gates.yaml) =="
if uv run python -c "
from rag_lab import vector_store
import sys
sys.exit(0 if vector_store.count() > 0 else 1)
" 2>/dev/null; then
    uv run rag eval --retrieval-only --gate eval/gates.yaml
else
    echo "SKIP retrieval gate: no chunks indexed (ingest documents first)"
fi

echo
echo "ALL GREEN"
