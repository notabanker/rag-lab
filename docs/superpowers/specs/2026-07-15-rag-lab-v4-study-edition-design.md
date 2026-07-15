# rag-lab V4 "Study Edition" — Design

**Date:** 2026-07-15
**Status:** Approved by user (brainstorming session)
**Supersedes:** extends `TASKS.md` (V4 roadmap); does not replace it

## Goal

Make rag-lab the daily tool for studying and personal knowledge work. Four
outcomes, all in scope for V4:

1. **Ingest real university materials** — scanned PDFs (OCR), DOCX, PPTX
2. **Active study tools** — flashcards, quiz/exam-prep sessions, summaries,
   built-in spaced repetition
3. **Zero-friction daily use** — `rag sync`, workspaces, MCP polish
4. **Better answers** — metadata filtering, golden set grown from real usage

The user calls this "v2 of the lab"; in repo versioning it is the **V4
release** (repo is at v3.1.0).

## Decisions made

| Decision | Choice | Rejected alternatives |
|---|---|---|
| Relationship to TASKS.md | Evolve the existing V4 roadmap; study layer added as Phase S | Fresh rethink; separate workstream |
| Study-layer architecture | Inside `rag_lab` as `rag_lab/study/` package | Separate `rag-study` service (two services for one user); stateless + Anki-only (drops SRS/quiz tracking) |
| Study interface | MCP tools first, dashboard Study view last | CLI/TUI-only; dashboard-first |
| Spaced repetition | Built-in SM-2, plus Anki export | Anki-only; FSRS (parking lot) |
| Study state storage | Separate `study.sqlite` beside the manifest | Extending the manifest DB (re-ingest must never touch review history) |
| Metadata filtering (TASKS.md C.2) | Promoted to a prerequisite of Phase S | Leaving it parked "until an eval demands it" — course-scoped decks/quizzes require it |

## Phasing

```
Phase A   (TASKS.md, unchanged)  A.3 DOCX → A.4 PPTX → A.2 OCR → A.5 re-baseline
Phase B   (TASKS.md, unchanged)  B.2 rag sync → B.1 workspaces → B.3 MCP polish
Phase C2' (promoted from C.2)    Metadata filtering: ingest --meta course=… semester=… type=…;
                                 --filter on query; exposed in API/MCP/dashboard/eval
Phase S   (new)                  S.1 card engine → S.2 Anki export → S.3 SM-2 scheduler + MCP review
                                 → S.4 quiz sessions → S.5 summaries → S.6 dashboard Study view
Phase C1  (TASKS.md C.1, habit)  rag runs promote — continuous, not a phase gate
```

Done-criteria per task (unchanged from TASKS.md): `uv run pytest` green,
`uv run rag eval --retrieval-only --gate eval/gates.yaml` passes,
`documentation.md` updated, one branch/commit per task.

## Phase S — Study layer design

### Storage: `study.sqlite`

Lives in the data dir beside the manifest (same resolution rules as
`RAG_DB_PATH`). Separate file because learning state has a different
lifecycle than corpus state: re-ingesting a changed document must never
wipe review history. Cards reference the corpus by `doc_id` + `chunk_ids`;
after re-ingest, stale references are flagged lazily, never deleted.

Tables:

- **cards** — id, workspace, course, doc_id, chunk_ids, front, back,
  citation, content_hash (dedupe), status (`active`/`suspended`/`stale`),
  created_at, model
- **reviews** — card_id, timestamp, rating (`again`/`hard`/`good`/`easy`),
  interval, ease factor, due_at. Append-only log; scheduler state derives
  from the most recent row per card.
- **quiz_sessions** / **quiz_items** — session id, scope (course/doc
  filter); per item: generated question, source chunk_ids, user answer,
  grade (0–10), feedback, missed_points
- **summaries** — file_sha, scope, summary_md, created_at. A cache;
  regeneration is explicit.

### Modules (small, single-purpose files)

- `study/store.py` — SQLite access; the only module touching study.sqlite
- `study/cards.py` — generation pipeline: retrieve parent chunks for a
  doc/course scope → LLM emits candidate cards as JSON (front, back, source
  chunk ids) → schema-validate → citation-validate against chunk text
  (reuse the `citations.py` approach) → content-hash dedupe → store.
  Cards failing citation validation are **dropped, never stored**.
- `study/scheduler.py` — plain SM-2 as pure functions:
  `(last_state, rating) -> (new_interval, ease, due_at)`. FSRS parked.
- `study/quiz.py` — session flow: pick chunks weighted toward weak review
  history → generate question → grade the user's answer against source
  text using the existing verifier pattern (score + missed points,
  grounded only in chunk text)
- `study/summarize.py` — map-reduce over parent chunks per doc/section,
  inline citations, cached by file_sha

### Surfaces

- **MCP tools (daily driver):** `study_generate_cards`, `study_due`,
  `study_review` (card id + rating → next due), `study_quiz_start`,
  `study_quiz_answer`, `study_summarize`, `study_stats` (due counts, weak
  topics per course)
- **CLI:** `rag cards generate/list/export`, `rag study due`, `rag quiz`,
  `rag summarize` — thin wrappers over the same functions
- **Anki export:** `rag cards export --anki deck.apkg` via `genanki` as an
  optional dependency group (like OCR); TSV fallback with zero extra deps
- **Dashboard Study view (S.6, last):** due-card review with keyboard
  shortcuts, quiz mode, weak-spot chart. REST endpoints mirror the MCP
  tools so S.6 is UI-only work.

### LLM cost control

Card generation and summaries are explicit batch operations per doc/course,
never implicit. Grading is one small call per quiz answer. Everything is
stored or cached; nothing regenerates silently.

## Error handling

- **Card generation:** LLM output schema-validated; one retry on malformed
  JSON, then fail with an actionable message; never partial-store. Dropped
  cards reported in the generation summary
  (`12 generated, 2 dropped: citation mismatch`).
- **Quiz grading:** grader failure stores the item ungraded and
  re-gradable; the user's typed answer is never lost.
- **Missing API key / empty corpus / empty scope:** actionable one-liners,
  same convention Phase B.3 establishes for MCP.
- **Stale card references after re-ingest:** flagged `stale` on next touch,
  surfaced in `study_stats`; user decides to regenerate or keep.

## Testing

Existing patterns: pytest, fake LLM via monkeypatch (as in
`tests/test_retrieval.py`), tmp-path SQLite.

- `scheduler.py`: pure-function unit tests incl. first review, lapse,
  ease clamps
- `store.py`: dedupe by content_hash, append-only reviews, workspace
  isolation
- `cards.py` / `quiz.py`: fake LLM with canned JSON — happy path,
  malformed-JSON retry, citation-failure drop
- One MCP smoke test per new tool

## Evaluation gates

- **Hard gate** (joins `scripts/check.sh`): 100% of stored cards pass
  citation validation; scheduler round-trip property test
- **Soft gate** (LLM-judged, on demand): sample of generated cards judged
  "answerable from cited chunk alone"; small golden set for quiz grading
  (known-good and known-wrong answers must be separated)
- Pedagogical card quality stays subjective; suspend-rate per doc in
  `study_stats` is the practical signal

## Parking lot

- FSRS scheduler
- Own review UI beyond the dashboard view (mobile, TUI)
- `rag sync --watch`, query routing, HyDE, GraphRAG (unchanged from
  TASKS.md C.3)
