#!/usr/bin/env bash
# Green check: full test suite + retrieval gate. Run before every commit/push.
# Baseline (v3.1.0): hit@5 100%, MRR 0.96, chunk hit@5 91%, chunk MRR 0.77.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== pytest =="
uv run pytest -q

echo
echo "== retrieval gate (eval/gates.yaml) =="
uv run rag eval --retrieval-only --gate eval/gates.yaml

echo
echo "ALL GREEN"
