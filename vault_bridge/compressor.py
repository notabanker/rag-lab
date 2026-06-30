import json
import os
from datetime import datetime

import httpx

from .config import get_vault_root
from .reader import read_l3_deltas, read_recent_sessions, read_today
from .writer import _atomic_write

COMPRESSOR_SYSTEM = """You are a memory compression agent for the Stoa AI system.

You receive today's working memory and session logs.
Produce two outputs:

1. A daily summary (markdown) — condensed overview of the day's events, decisions, and context.
2. Knowledge deltas (JSON) — what new entities, relationships, decisions, and lessons were created today.

Score the day's overall importance from 1-10:
- 1-3: transient chatter, status polls, repeated info — skip indexing
- 4-6: useful context but not critical — index with low priority
- 7-8: decisions, lessons, new insights — index fully
- 9-10: mission-critical (system changes, persona updates, major decisions)

Be concise. Focus on what matters for long-term memory. Drop transient chatter."""


def _llm_synthesize(prompt: str) -> dict:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": COMPRESSOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.1,
        "stream": False,
    }

    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()

    return json.loads(raw)


def run(date: str = None) -> dict:
    """
    Nightly memory compression.

    1. Read L1 today.md
    2. Read today's L2 agent-logs
    3. Read recent L3 deltas (for context)
    4. LLM synthesis → L2 daily summary + L3 deltas
    5. Atomic writes for both outputs

    Returns {l2_path, l3_path, summary_chars, entities_extracted}
    """
    date = date or datetime.now().strftime("%Y-%m-%d")

    today = read_today()
    if not today.strip():
        return {"error": "No L1 content to compress", "date": date}

    sessions = read_recent_sessions(days=1)
    recent_deltas = read_l3_deltas(days=3)

    sessions_text = ""
    for s in sessions:
        content = s["content"]
        if len(content) > 2000:
            content = content[:2000] + "\n...\n"
        sessions_text += f"\n### {s['date']}\n{content}\n---\n"

    prompt = f"""DATE: {date}

TODAY'S WORKING MEMORY:
{today[:3000]}

TODAY'S SESSIONS:
{sessions_text[:4000]}

RECENT KNOWLEDGE DELTAS (for context):
{json.dumps(recent_deltas, indent=2)[:2000]}

Produce a JSON object with exactly these keys:
- "importance": integer 1-10 (overall importance of this day's content for long-term memory)
- "daily_summary": string (markdown, the day's condensed summary with ## sections for Overview, Key Events, Decisions Made, Agent Activity)
- "new_entities": list of {{"name": "...", "type": "...", "description": "..."}}
- "updated_entities": list of {{"name": "...", "field": "...", "new_value": "..."}}
- "decisions_captured": list of {{"decision": "...", "rationale": "...", "context": "..."}}
- "lessons_learned": list of {{"lesson": "...", "source": "..."}}

Return ONLY valid JSON, no surrounding text."""

    try:
        result = _llm_synthesize(prompt)
    except Exception as e:
        return {"error": str(e), "date": date}

    vault = get_vault_root()
    importance = result.get("importance", 5)

    daily_body = result.get("daily_summary", f"Compression failed.")
    daily_md = f"---\nimportance: {importance}\ndate: {date}\n---\n\n# Daily Summary — {date}\n\n{daily_body}"

    if importance >= 7:
        l2_dir = vault / "L2-episodic" / "daily"
    else:
        l2_dir = vault / "L2-episodic" / "drafts"
    l2_dir.mkdir(parents=True, exist_ok=True)
    l2_path = l2_dir / f"{date}.md"
    _atomic_write(l2_path, daily_md)

    l3_dir = vault / "L3-semantic" / "knowledge"
    l3_path = l3_dir / f"l3_delta_{date}.json"
    delta = {
        "date": date,
        "importance": importance,
        "new_entities": result.get("new_entities", []),
        "updated_entities": result.get("updated_entities", []),
        "decisions_captured": result.get("decisions_captured", []),
        "lessons_learned": result.get("lessons_learned", []),
    }
    _atomic_write(l3_path, json.dumps(delta, indent=2, default=str))

    return {
        "l2_path": str(l2_path),
        "l3_path": str(l3_path),
        "summary_chars": len(daily_body),
        "importance": importance,
        "indexed": importance >= 7,
        "entities_extracted": len(delta.get("new_entities", [])) + len(delta.get("updated_entities", [])),
    }
