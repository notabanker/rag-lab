"""Shared LLM chat wrapper: retries, timeouts, headers."""
import logging
import time

import httpx

from .config import LLM_BASE_URL, LLM_MAX_RETRIES, LLM_TIMEOUT, require_api_key

log = logging.getLogger("rag_lab.llm")

CHAT_URL = f"{LLM_BASE_URL}/chat/completions"


def content_of(body: dict) -> str:
    return (body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


def usage_of(body: dict) -> dict:
    usage = body.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def chat(
    payload: dict,
    timeout: float | None = None,
    retries: int | None = None,
    purpose: str = "llm",
) -> dict:
    """POST /chat/completions with retry/backoff on 429/5xx/network errors.

    Returns the parsed JSON body; raises RuntimeError once retries are spent
    (callers must degrade gracefully, never crash the request).
    """
    timeout = timeout if timeout is not None else LLM_TIMEOUT
    retries = retries if retries is not None else LLM_MAX_RETRIES
    headers = {
        "Authorization": f"Bearer {require_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/notabanker/rag-lab",
        "X-Title": "rag-lab",
    }
    last_error: RuntimeError | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(CHAT_URL, json=payload, headers=headers)
            if r.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(f"{purpose}: HTTP {r.status_code}")
            elif r.status_code >= 400:
                raise RuntimeError(f"{purpose}: HTTP {r.status_code}: {r.text[:200]}")
            else:
                return r.json()
        except (httpx.HTTPError, ValueError) as e:
            last_error = RuntimeError(f"{purpose}: {e}")
        if attempt < retries:
            delay = 0.5 * (2 ** attempt)
            log.warning("%s attempt %d/%d failed (%s), retrying in %.1fs", purpose, attempt + 1, retries, last_error, delay)
            time.sleep(delay)
    assert last_error is not None
    raise last_error
