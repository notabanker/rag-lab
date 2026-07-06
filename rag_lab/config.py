import os
import re

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.7-plus").strip()
LLM_VERIFIER_MODEL = os.environ.get("LLM_VERIFIER_MODEL", "").strip() or LLM_MODEL
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "").strip() or "intfloat/multilingual-e5-small"
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "").strip() or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_BATCH_SIZE = int(os.environ.get("RERANKER_BATCH_SIZE", "16"))
INDEX_VERSION = "v3"
CHUNKING_VERSION = "sentence-v2-parent"

def _slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "default"

def collection_fingerprint(
    embedding_model: str = None,
    chunking_version: str = None,
) -> str:
    model = embedding_model or EMBEDDING_MODEL
    chunking = chunking_version or CHUNKING_VERSION
    return f"{INDEX_VERSION}-{_slug(model)}-{_slug(chunking)}"

DEFAULT_COLLECTION = os.environ.get("RAG_COLLECTION", "").strip() or f"rag_lab_{collection_fingerprint()}"

def get_api_key() -> str:
    return OPENROUTER_API_KEY or DEEPSEEK_API_KEY

def require_api_key() -> str:
    key = get_api_key()
    if not key:
        raise RuntimeError("No API key set — export OPENROUTER_API_KEY or DEEPSEEK_API_KEY")
    return key
