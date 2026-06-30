# Importance Filter — Implementation Plan

## Problem

Old Hermes froze because 1672 files got indexed indiscriminately — hourly scribe logs, casual chat, repeated state files. Search returned noise instead of signal.

## Solution

Three changes. No new files. ~50 lines total.

---

## T1 — Session Writer: Skip Unimportant Sessions

**File:** `vault_bridge/vault_bridge/session_writer.py`

**Change:** Add importance check before writing L2.

```python
def _is_important(task: str, decisions: list[str] | None, lessons: list[str] | None,
                  messages: list[dict]) -> bool:
    """Returns False for purely conversational sessions with no substance."""
    if task:
        return True
    if decisions:
        return True
    if lessons:
        return True
    if len(messages) < 2:
        return False
    content = " ".join(m.get("content", "") for m in messages).lower()
    keywords = ["project", "config", "agent", "persona", "vault", "system",
                "memory", "plutos", "stoa", "hermes", "francesca", "rag",
                "deploy", "build", "fix", "error", "decision", "strategy",
                "bitcoin", "portfolio", "risk", "trade", "market",
                "architecture", "pipeline", "deploy", "mcp", "tool"]
    return any(kw in content for kw in keywords)
```

**If important:** write to `L2-episodic/agent-logs/` as today (gets indexed by watcher).

**If not important:** write to `L2-episodic/drafts/` (excluded from indexing — already in `.rag-ignore`). Still write L1 summary.

**Why:** Zero API cost. Rule-based check on words that signal project/agent/persona relevance. Casual chats like "how are you" or "what's the weather" have none of these keywords → go to drafts.

---

## T2 — Compressor: LLM Importance Scoring

**File:** `vault_bridge/vault_bridge/compressor.py`

**Change:** Add `importance` field to the LLM JSON output prompt.

```json
"importance": <integer 1-10>
```

**Scoring guide added to system prompt:**
```
- 1-3: transient chatter, status polls, repeated info — skip indexing
- 4-6: useful context but not critical — index with low priority
- 7-8: decisions, lessons, new insights — index fully
- 9-10: mission-critical (system changes, persona updates, major decisions) — index + flag as high priority
```

**If score ≥ 7:** write L2 daily summary to `daily/` (indexed) + L3 delta to `knowledge/` (JSON, not indexed anyway).

**If score < 7:** write L2 daily summary to `drafts/daily-{date}.md` (not indexed). Still write L3 delta (knowledge is knowledge, even if low-importance day should be recorded).

**Why:** One extra field in existing LLM call. Zero additional API cost. LLM already reads the entire day — scoring it is trivial.

---

## T3 — Skip L1 Indexing

**File:** `rag_lab/rag_lab/vault_config.py`

**Change:** Remove L1 from `DEFAULT_LAYERS`.

```python
DEFAULT_LAYERS = [
    LayerConfig("l2", "l2_episodic", "L2-episodic", "Episodic memory"),
    LayerConfig("l3", "l3_semantic", "L3-semantic", "Semantic memory"),
]
```

**Also update `.rag-ignore`:** add `L2-episodic/drafts/` to exclusion patterns.

**Why:** Hermes already reads `today.md` directly for current context. L1 is transient — indexing it returns yesterday's scratchpad mixed with today's, creating confusion.

---

## T4 — Watcher: Importance Frontmatter Check

**File:** `rag_lab/rag_lab/watcher.py`

**Change:** In `_index_single_file`, check for `importance: low` in frontmatter.

```python
fm, body = extract_frontmatter(raw)
if fm.get("importance") == "low":
    return  # skip indexing, file stays on filesystem
```

**Why:** Belt and suspenders. If something gets written to an indexed directory with `importance: low` in frontmatter, the watcher skips it. The compressor can set this for daily summaries that scored < 7.

---

## T5 — Compressor: Write `importance` to Daily Summary Frontmatter

**File:** `vault_bridge/vault_bridge/compressor.py`

**Change:** When writing L2 daily summary, include YAML frontmatter with importance score.

```python
daily_md = f"""---
importance: {importance_score}
date: {date}
---

# Daily Summary — {date}

{summary_body}"""
```

**Why:** The watcher (T4) reads this and decides whether to index. The Obsidian user can see at a glance which days mattered.

---

## Impact Summary

| | Before | After |
|---|---|---|
| Indexed files | ~1672 | ~800-1000 (estimated) |
| Casual chat in search | Yes | No (goes to drafts/) |
| L1 noise in search | Yes | No (L1 excluded) |
| API cost increase | — | $0 (importance scored in existing LLM call) |
| Hermes search latency | Bloated | Focused |

## Task List

| # | Task | File | Lines |
|---|---|---|---|
| T1 | Session importance gate | `session_writer.py` | +20 |
| T2 | LLM importance scoring in compressor | `compressor.py` | +10 |
| T3 | Remove L1 from DEFAULT_LAYERS | `vault_config.py` | -2 |
| T4 | Watcher frontmatter check | `watcher.py` + `indexer.py` | +10 |
| T5 | Compressor writes importance frontmatter | `compressor.py` | +10 |

## Tests

| Test | What it checks |
|---|---|
| `test_is_important_with_task` | Session with task → True |
| `test_is_important_with_decision` | Session with decision → True |
| `test_is_important_casual_chat` | "how are you" → False |
| `test_is_important_project_keyword` | "deploy the pipeline" → True |
| `test_is_important_empty` | No messages → False |
| `test_session_drafts_for_unimportant` | Casual session → file in drafts/, not agent-logs/ |
| `test_compressor_importance_field` | LLM response has importance key |
| `test_watcher_skips_low_importance` | File with `importance: low` → not indexed |
| `test_l1_not_in_default_layers` | DEFAULT_LAYERS has l2, l3 only |
