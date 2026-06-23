import httpx
import json
import re
from .config import get_provider

VERIFIER_SYSTEM = """You are a strict grounding auditor.

You will be given a QUESTION, CONTEXT chunks that an LLM used, and an ANSWER.
Judge the answer along three dimensions:

(a) GROUNDED: Is every claim in ANSWER directly supported by a CONTEXT chunk?
(b) COMPLETE: Does ANSWER address the QUESTION fully?
(c) HONEST: Does ANSWER admit gaps when CONTEXT is insufficient?

Reply ONLY in this JSON format (no prose outside the JSON):
{
  "score": <integer 1-10>,
  "grounded": <true|false>,
  "issues": [<list of specific issues, or empty>],
  "verdict": "<one of: GROUNDED | PARTIAL | UNGROUNDED>"
}"""

_JSON_RE = re.compile(r'\{[\s\S]*\}', re.DOTALL)


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```(?:json)?\s*\n?', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\n?```\s*$', '', raw, flags=re.MULTILINE)
    match = _JSON_RE.search(raw)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def verify(question: str, answer: str, chunks: list[dict], provider: str = None, model: str = None) -> dict:
    prov = get_provider(provider)
    model = model or prov.model
    chat_url = f"{prov.base_url}/chat/completions"

    context = "\n\n---\n\n".join(
        f"[chunk {c.get('id', '?')} | {(c.get('metadata') or {}).get('source', '?')}]\n{c.get('text', '')[:800]}"
        for c in chunks
    )
    user_prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}"

    headers = {
        "Content-Type": "application/json",
    }
    if prov.api_key:
        headers["Authorization"] = f"Bearer {prov.api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 400,
        "temperature": 0.01,
        "stream": False,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(chat_url, json=payload, headers=headers)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
        return _extract_json(raw)
    except Exception as e:
        return {"score": 0, "grounded": False, "issues": [f"verifier error: {e}"], "verdict": "ERROR"}
