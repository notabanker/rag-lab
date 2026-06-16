import httpx
import json
from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

CHAT_URL = f"{DEEPSEEK_BASE_URL}/chat/completions"

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

def verify(question: str, answer: str, chunks: list[dict], model: str = None) -> dict:
    model = model or DEEPSEEK_MODEL
    context = "\n\n---\n\n".join(
        f"[chunk {c['id']} | {c['metadata'].get('source', '?')}]\n{c['text'][:800]}"
        for c in chunks
    )
    user_prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 400,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(CHAT_URL, json=payload, headers=headers)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        return {"score": 0, "grounded": False, "issues": [f"verifier error: {e}"], "verdict": "ERROR"}
