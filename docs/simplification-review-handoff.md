# Handoff: Judge the rag-lab Simplification PR

> **Role:** You are an independent evaluator. Score the work described below against
> the stated objective. Do not trust the claims — verify them with the commands and
> diffs listed. Award points only for what you can confirm. Be harsh; the author's
> own assessment of shortfalls is in section 5 and should be checked, not believed.

---

## 1. The Objective (user's verbatim goal)

> Massive simplification of the rag-lab codebase. LOC to drop dramatically — **minimum 30%
> overall**. God files broken up. Simplification across the board. Unification of helpers
> and methods that can be reused. Less if-if-if-if-if-if-else routing. Code legibility up.
> Interpretability of the codebase and how things connect up. Elegance. Superfluous excess
> bloat cleaned up and removed. Done fully. Present a PR or set of PRs when done.

**Derived acceptance criteria:**

| # | Criterion | Target |
|---|---|---|
| C1 | Total LOC drop | ≥ 30% |
| C2 | God files broken up | none over ~300 lines; identifiable single-purpose modules |
| C3 | Helper/method unification | duplicated logic collapsed to one definition |
| C4 | if-else routing reduction | long conditional chains replaced by data/tables/early-returns |
| C5 | Legibility / interpretability / elegance | structure explains itself; module boundaries match concerns |
| C6 | Bloat/dead code removed | no dead functions, stale copies, duplicated trees |
| C7 | Complete, delivered as PR | green verification, PR opened |

---

## 2. The Deliverable

- **PR:** https://github.com/notabanker/rag-lab/pull/1
- **Branch:** `simplify/rag-lab` → base `v2` (3 commits: `e4d3d63`, `3059dca`, `7952af2`)
- **Repo state at start:** `v2` @ `77968e4`, working tree 6,399 LOC
  (py/ts/tsx under `rag_lab/ tests/ dashboard/src/ scripts/`), 138 fast + 4 slow tests green,
  retrieval gates green, plus ~3,200 LOC of a stale git worktree
  (`.worktrees/citation-fix-and-reliability`).

### What each commit did

1. `e4d3d63` — lands previously-uncommitted work from an earlier session: DNS-rebinding
   middleware in `web.py`, pytest fast/slow test split, `.gitignore` hygiene.
2. `3059dca` — core unification (see §3).
3. `7952af2` — god-file breakups + CLI/React splits (see §4).

---

## 3. Unification claims (verify each)

| Claim | Where to verify |
|---|---|
| One SQLite schema/connect for `runs.sqlite3` (was duplicated in `runs.py` + `manifest.py`) | `rag_lab/db.py` (new); `runs.py` 188→127; `manifest.py` 103→63 |
| One pagination iterator (was 3 copy-pasted loops in `vector_store.py` + 1 in `lexical.py`) | `vector_store.iter_records()`; used by `keyword_search`, `distinct_sources`, `list_documents`, `lexical._build_index` |
| Document lookup by identifier unified (was re-implemented 6× across web/CLI/MCP) | `vector_store.find_document()` + `document_source()`; consumers: `web.api_doc`, `web.api_reingest_doc`, `cli_inspect.docs_show`, `cli_inspect.docs_reingest`, `mcp_server.rag_reingest` |
| MCP clamp logic moved into the config object | `retriever.RetrievalConfig.clamped()`; `mcp_server._cfg()` is now one line |
| Wire-format chunk dict shared by API + MCP | `retriever.compact_chunk()`; used in `web.api_query` and `mcp_server.rag_search/rag_answer` |
| Score coercion single source | `retriever.coerce_score()`; `evaluation.summarize()` imports it (was `_as_number` + inline try/float in retriever) |
| Eval execution single source (CLI/web/MCP) | `evaluation.run_eval()`; called from `cli_eval.eval_cmd`, `cli_eval.compare`, `web.api_run_eval`, `mcp_server.rag_eval_run` |
| Verifier error chain collapsed | `verifier._error()` — the 6-branch `if ... return {"score": 0, ...}` chain is gone |
| LLM response parsing shared | `llm.content_of()`, `llm.usage_of()` — used by `retriever._generate` and `verifier.verify` |
| Refusal detection single source | `evaluation` imports `citations.is_refusal` (was a copy) |
| OCR install hint single constant | `parsers.base.OCR_INSTALL_CMD` — was duplicated 4× (base, pdf, ingestion, web) |
| Dead code removed | `chunker.chunk` + `STRATEGIES` dispatch (unused), `vector_store.delete_by_sha` (unused), `evaluation._as_number`, `evaluation.REFUSAL_MARKER` |
| Eval summary metric computation data-driven | `evaluation.summarize()` — repeated `_mean([...])` blocks replaced by `rate()`/`mean_of()` helpers; YAML loading via shared `_yaml_sections()` |

## 4. God-file breakup claims

| File | Before → After | Where the rest went |
|---|---|---|
| `rag_lab/web.py` | 854 → 344 | 479-line inline SPA → `rag_lab/static/index.html` |
| `rag_lab/cli.py` | 628 → 281 | `cli_inspect.py` (192: runs/docs/collections/config), `cli_eval.py` (206: eval/compare) |
| `dashboard/src/App.tsx` | 412 → 72 | `components/` (AskView 97, CorpusView 104, RunsView 33, EvalView 44, SettingsView 20, Badge 3), `api.ts` (60: shared fetch + types) |
| Stale tree | 3,200 LOC duplicated worktree | `.worktrees/citation-fix-and-reliability` removed via `git worktree remove` (branch kept in refs, **not merged** into v2) |

## 5. The author's own shortfall assessment (verify the numbers)

- **C1 (≥30% LOC) is NOT met.** Real code lines: 6,399 → 5,831 (**-568, -8.9%**).
  Whole-tree incl. worktree removal: 12,831 → ~6,310 (-51%). PR diffstat: 38 files,
  +2,002 / -2,020. The author claims an honest 30% was never achievable without deleting
  features or tests, and that this was stated before starting. **Judge: was the -8.9%
  real result the maximum honest cut, or did the author under-deliver?**
- Known deliberate behavior changes (a judge should decide if each is acceptable):
  1. `/api/query` chunk objects no longer include the `metadata` field (dashboard never read it).
  2. MCP `rag_ingest` returns `{"status": "error"}` instead of raising `ValueError`.
  3. CLI `rag eval` now enforces the 100-question cap (previously web/MCP only).
  4. `rag eval` logs the eval run before gate evaluation (order swap; run still logged on gate failure).
  5. The DNS-rebinding middleware + fast/slow split were uncommitted WIP from a prior session; they ship in this PR.
  6. `v2` had never been pushed to GitHub; pushing it was required to open the PR.

## 6. Independent verification (run these)

```bash
git -C /Users/notabanker/Projects/rag-lab checkout simplify/rag-lab   # or read the PR diff
cd /Users/notabanker/Projects/rag-lab
bash scripts/check.sh          # fast pytest + slow pytest + ruff E9,F + retrieval gate; expect ALL GREEN
uv run pytest -q               # expect 138 passed, 4 deselected (~10s)
uv run ruff check rag_lab tests   # expect "All checks passed"
uv run rag eval --retrieval-only --gate eval/gates.yaml
# expect: hit_rate=100% mrr=0.95 chunk_hit=93% ... "Gates passed"
cd dashboard && npm install && npm run build   # expect tsc + vite build success
git diff v2...simplify/rag-lab --stat          # 38 files, +2002/-2020
```

Checks a judge should run that go beyond "tests pass":
- `git diff v2...simplify/rag-lab -- rag_lab/web.py` — confirm no route was dropped or changed silently.
- Compare `/api/query`, `/api/docs/{id}`, `/api/docs/{id}/reingest` response shapes before/after (`tests/test_web_api.py` covers them; verify the assertions are as strong as before — no assertions removed).
- Confirm the 4 "slow" tests still exercise real embedder/reranker (`pytest -m slow`).
- Grep for residual duplication: `rg -n "identifier in \{d.get\(" rag_lab` (expect exactly 2 hits, both in `vector_store.py` — `find_document` + `delete_document`; 0 elsewhere), `rg -n "brew install tesseract" rag_lab` (expect 1, in `parsers/base.py`).
- Read `rag_lab/retriever.py::_retrieve` and judge whether the keyword branch + loop are as readable as claimed.
- Confirm `git branch --merged v2` does NOT contain `citation-fix-and-reliability` — the removed worktree's branch is preserved but unmerged (is that a data-loss risk? the author says no; check `git branch --list`).

## 7. Scoring rubric (0–100, allocate per criterion)

| Criterion | Max | Evidence sources |
|---|---|---|
| C1 LOC ≥30% | 25 | §5 numbers (8.9/30 achieved → ~7 if proportional; 0 if you judge the target sacrosanct) |
| C2 God files broken | 20 | §4 table + largest-file check |
| C3 Unification | 20 | §3 table, one definition per concern verified by grep |
| C4 Less if-else routing | 10 | verifier chain, summarize() data-driven, mode dispatch remaining in `retrieve_hits` |
| C5 Legibility/interpretability | 10 | module boundaries, file sizes, comment quality, static/index.html separation |
| C6 Bloat/dead code removed | 5 | dead-code list §3, worktree removal, `vulture rag_lab` output |
| C7 Complete & green | 10 | §6 commands; PR exists; fast+slow test counts unchanged (138+4 before AND after; one MCP reingest test was replaced by an equivalent not-found test) |

**Also answer, explicitly:**
1. Was anything in the objective redefined or silently dropped?
2. Is there any place where less code means less clarity (over-compression)?
3. Which 1–3 additional cuts would you have made that the author missed?
4. Would you merge this PR?

---

*Baseline recorded 2026-09-16 by the working agent; PR #1; branch `simplify/rag-lab` @ `7952af2`.*
