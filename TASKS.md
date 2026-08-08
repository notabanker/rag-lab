# rag-lab — V4 Roadmap Task List

Goal: make rag-lab reliable for daily personal + university use.
Rule: every task is a separate branch/commit, and is only "done" when:

1. `uv run pytest` is green
2. `uv run rag eval --retrieval-only --gate eval/gates.yaml` passes
3. `documentation.md` reflects the change

Effort: **S** = under an hour, **M** = an evening, **L** = a weekend.

---

## Phase 0 — Stabilize the foundation

*Nothing else starts until this phase is done. The whole V3 release is currently uncommitted.*

### 0.1 Commit the V3 work — **S**
- [x] Review `git diff` and `git status`; group changes into logical commits
      (e.g. core retrieval, eval harness, dashboard, MCP server, docs)
- [x] Verify `.gitignore` covers `chroma_db/`, `.run/`, `dashboard/node_modules/`,
      `dashboard/dist/`, `__pycache__/`, `.pytest_cache/`
- [x] Bump `pyproject.toml` version `3.1.0.dev0` → `3.1.0`
- [x] Commit, tag `v3.1.0`

### 0.2 Make "green" checkable in one command — **S**
- [x] Add `scripts/check.sh`: runs `pytest` + retrieval-only eval with gates, exits non-zero on failure
- [x] Record the current baseline metrics in the script output or a `BASELINE.md` note
- [ ] (Optional) git pre-push hook that runs `scripts/check.sh`

### 0.3 Kill the CWD-relative database foot-gun — **M**
Current: `--db-path` defaults to `./chroma_db` ([cli.py:32](rag_lab/cli.py:32)); running from another directory silently creates an empty corpus.
- [x] Add `RAG_DB_PATH` env var to [config.py](rag_lab/config.py)
- [x] Default resolution order: `--db-path` flag → `RAG_DB_PATH` → `~/.local/share/rag-lab/chroma_db` (expanded, created on first use)
- [x] On startup, if legacy `./chroma_db` exists in CWD and differs from the resolved path: print a one-line warning with the migration command (`mv ./chroma_db ~/.local/share/rag-lab/chroma_db`)
- [x] Apply the same resolution in `rag serve` ([web.py](rag_lab/web.py)) and the MCP server ([mcp_server.py](rag_lab/mcp_server.py))
- [x] Update tests that assume `./chroma_db`; add a test for the resolution order
- [x] Update README + documentation.md ("CWD-dependent DB" leaves Known Limitations)

---

## Phase A — Corpus reality

*University materials are scanned PDFs, DOCX, and PPTX. Today, scanned PDFs silently produce zero chunks and DOCX/PPTX can't be ingested at all. This phase is the highest-leverage work in the roadmap.*

### A.1 Parse-quality report on ingest — **M**
*Do this first — it's the instrument that detects the scanned-PDF problem.*
- [x] [parsers/pdf.py](rag_lab/parsers/pdf.py): return per-page character counts alongside the text
- [x] Define a low-yield heuristic (e.g. avg < 50 chars/page, or >50% empty pages) → "probably scanned"
- [x] [ingestion.py](rag_lab/ingestion.py): produce an ingest report — pages/sections, chars extracted, chunks created, warnings
- [x] Persist the report in the SQLite manifest ([manifest.py](rag_lab/manifest.py)) per `file_sha`
- [x] `rag ingest` prints the report; **refuse by default** when yield ≈ 0 ("no text extracted — scanned PDF? see OCR") with `--allow-empty` override
- [x] Surface warnings in `rag docs show`, `POST /api/ingest` response, and the dashboard Corpus view
- [x] Tests: text PDF (clean report), empty-text fixture (warning + refusal)

### A.2 OCR fallback for scanned PDFs — **L**
- [ ] Decide the engine — recommendation: `ocrmypdf` as an optional dependency group
      (`uv sync --group ocr`), since it handles rasterization, deskew, and language packs;
      document `brew install tesseract tesseract-lang` for deu+eng
- [ ] Wire into the PDF parser: only pages flagged low-yield by A.1 get OCR'd (never re-OCR good text layers)
- [ ] Cache OCR output keyed by `file_sha` (OCR is expensive; re-ingest must be cheap) — store beside the manifest
- [ ] `--ocr/--no-ocr` flag on `ingest`/`rebuild`; auto-on when engine is installed, clear error pointing to install docs when not
- [ ] Languages: default `deu+eng`, configurable via `RAG_OCR_LANGS`
- [ ] Tests: tiny scanned-PDF fixture → OCR path produces chunks; cache-hit test
- [ ] documentation.md: "No OCR" leaves Known Limitations

### A.3 DOCX parser — **M**
- [x] Add `python-docx` dependency
- [x] New [parsers/docx.py](rag_lab/parsers): paragraphs + headings (keep heading text inline for chunk context), tables flattened row-per-line
- [x] Register `.docx` in `PARSERS` ([parsers/__init__.py](rag_lab/parsers/__init__.py))
- [x] Citation labels: no page numbers in DOCX — use heading-based section labels (`report.docx §2.3 Methods`) or paragraph ranges; keep [citations.py](rag_lab/citations.py) validation working
- [x] Tests + small `.docx` fixture (headings, table, plain paragraphs)
- [x] Update supported-formats list in docs (dashboard/MCP derive their format list from `PARSERS`, so no separate accept-list edits needed)

### A.4 PPTX parser — **M**
- [ ] Add `python-pptx` dependency
- [ ] New [parsers/pptx.py](rag_lab/parsers): all text frames per slide + speaker notes; one logical section per slide
- [ ] Citation labels: `lecture03.pptx slide 12`
- [ ] Consider slide = natural parent chunk for small-to-big (slides are short; verify chunker behavior on tiny sections)
- [ ] Register `.pptx`; tests + fixture (title, bullets, notes)
- [ ] Update docs, dashboard accept-list, MCP description

### A.5 Eval coverage for new formats — **S**
- [ ] Add a DOCX and a PPTX sample to `data/`
- [ ] Add golden questions targeting them in [eval/questions.yaml](eval/questions.yaml) (incl. one slide-notes question)
- [ ] Re-baseline: run `rag eval --retrieval-only`, update the README metrics table

---

## Phase B — Daily-use friction

*The corpus that stays current is the corpus that stays useful.*

### B.1 Workspaces: `uni` vs `personal` — **L**
Design: a workspace = named bundle of {collection, question set, optional model overrides}, stored in `~/.config/rag-lab/workspaces.toml`. One shared Chroma DB; separation by collection.
- [ ] Workspace registry module: load/save TOML, resolve active workspace
- [ ] `rag workspace list / create <name> / use <name> / show` commands
- [ ] Resolution order: `--workspace` flag → `RAG_WORKSPACE` env → active from config → `default`
- [ ] Workspace maps to a collection name prefix (`uni_` + versioned fingerprint from [config.py](rag_lab/config.py)) so embedding/chunking versioning still applies
- [ ] All CLI commands, [web.py](rag_lab/web.py) endpoints, and MCP tools respect the active workspace; API/MCP accept an optional `workspace` param
- [ ] Per-workspace eval sets: `eval/uni.yaml`, `eval/personal.yaml`; `rag eval --workspace uni` picks the right one automatically
- [ ] Dashboard: workspace switcher in the header; Settings view shows active workspace
- [ ] Tests: resolution order, collection mapping, cross-workspace isolation (ingest into `uni`, assert invisible in `personal`)

### B.2 `rag sync` — incremental folder ingestion — **M**
- [ ] `rag sync <dir>... [--prune] [--dry-run]`: recursive scan for supported extensions
- [ ] Skip unchanged files: compare content SHA against the manifest ([manifest.py](rag_lab/manifest.py) already stores `file_sha`)
- [ ] Re-ingest changed files (existing SHA-replace flow in [ingestion.py](rag_lab/ingestion.py) already handles dedupe)
- [ ] `--prune`: delete indexed docs whose source path no longer exists (confirm unless `--yes`)
- [ ] Summary table: added / updated / unchanged / pruned / failed (with parse-quality warnings from A.1)
- [ ] Respect workspaces (`rag sync ~/uni/semester4 --workspace uni`)
- [ ] Tests: unchanged-skip, changed-reingest, prune, dry-run
- [ ] (Parking lot, not now: `--watch` mode with filesystem events)

### B.3 MCP as the primary daily interface — **M**
- [ ] Add ready-to-paste registration snippets to README (Claude Code `claude mcp add`, Claude Desktop JSON config) using absolute `uv --directory ~/rag-lab run rag mcp serve`
- [ ] Ensure the MCP server works when launched from any CWD (depends on 0.3)
- [ ] Add `workspace` parameter to `rag_search` / `rag_answer` / `rag_ingest` (depends on B.1)
- [ ] Review tool descriptions so an LLM picks the right tool (search vs answer vs keyword enumeration); document the `--keyword` equivalent in `rag_search`
- [ ] Error ergonomics: missing API key or empty collection should return actionable messages, not stack traces
- [ ] End-to-end smoke test: register in Claude Code, run a real uni question through `rag_answer`, verify citations

---

## Phase C — Retrieval quality (strictly eval-driven)

*Nothing in this phase starts without a failing eval question that demands it.*

### C.1 Grow the golden set from real usage — **M**
- [ ] `rag runs promote <run_id>`: convert a logged run ([runs.py](rag_lab/runs.py)) into a question-stub appended to the workspace's eval YAML (question + retrieved sources prefilled; human fills `expected_fragments`)
- [ ] Tag every question `uni` / `personal` + language (`en`/`de`)
- [ ] [evaluation.py](rag_lab/evaluation.py): per-tag metric breakdown in the report (spot "German questions underperform" class regressions)
- [ ] Habit hook: when a real query fails you, promote it before fixing anything

### C.2 Metadata filtering — **L**
- [ ] Ingest: `--meta key=value` (repeatable), e.g. `course=DLBFMWFT1 semester=4 type=slides`; store in chunk metadata + manifest
- [ ] Query: `--filter key=value` → Chroma `where` clause for vector search; post-filter for the BM25 path in [lexical.py](rag_lab/lexical.py)
- [ ] Expose filters in `POST /api/query`, dashboard Ask view, and MCP `rag_search`/`rag_answer`
- [ ] Eval: allow `filters:` on golden questions; add at least 2 filtered questions
- [ ] Tests: filter narrows both retrieval paths; empty-result behavior is a clean "no matches", not a hallucination

### C.3 Parking lot (needs a failing eval to justify)
- Query routing (auto-pick vector/lexical/keyword per question)
- Multi-query expansion / HyDE
- GraphRAG, ColBERT, agentic retrieval
- Streaming responses
- Local LLM fallback (Ollama endpoint is already compatible via `LLM_BASE_URL`)
- `rag sync --watch` filesystem watcher

---

## Suggested execution order

```
0.1 → 0.2 → 0.3          (one sitting: repo is safe, checkable, path-safe)
A.1 → A.3 → A.4 → A.2    (quality report first; DOCX/PPTX before OCR — cheaper, unblock more files)
A.5                       (re-baseline evals)
B.2 → B.1 → B.3          (sync gives immediate value solo; workspaces before MCP polish)
C.1                       (start immediately alongside daily use — it's a habit, not a feature)
C.2                       (when the eval set proves the need)
```
