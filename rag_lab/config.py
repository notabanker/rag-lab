import os
import re

def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None

# Read at import: these pin the identity of the index (collection fingerprint,
# embedding model). They must be stable for the lifetime of the process.
LLM_BASE_URL = _env_str("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = _env_str("LLM_MODEL", "qwen/qwen3.7-plus")
LLM_VERIFIER_MODEL = _env_str("LLM_VERIFIER_MODEL", "") or LLM_MODEL
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "") or "intfloat/multilingual-e5-small"
RERANKER_MODEL = _env_str("RERANKER_MODEL", "") or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)
RERANKER_BATCH_SIZE = _env_int("RERANKER_BATCH_SIZE", 16)

# LLM call behavior (mutable per process; defaults sane for OpenRouter).
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 180)            # seconds, generator
LLM_VERIFIER_TIMEOUT = _env_int("LLM_VERIFIER_TIMEOUT", 120)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)      # on 429/5xx/network errors

# OCR (scanned PDFs): languages for tesseract, "deu+eng" typical.
RAG_OCR_LANGS = _env_str("RAG_OCR_LANGS", "deu+eng")

# Upload guard for the web API.
MAX_UPLOAD_BYTES = _env_int("RAG_MAX_UPLOAD_MB", 200) * 1024 * 1024
MAX_INGEST_CHUNKS = 100_000

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

DEFAULT_COLLECTION = _env_str("RAG_COLLECTION", "") or f"rag_lab_{collection_fingerprint()}"

def default_db_path() -> str:
    """Resolution order: --db-path flag (handled by callers) > RAG_DB_PATH > user data dir."""
    env = _env_str("RAG_DB_PATH")
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser("~"), ".local", "share", "rag-lab", "chroma_db")

def eval_dir() -> str:
    """Root directory for golden-question files accepted by API/MCP eval runs."""
    env = _env_str("RAG_EVAL_DIR")
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.getcwd(), "eval")

def get_api_key() -> str:
    # Read live so later env changes and tests take effect.
    return _env_str("OPENROUTER_API_KEY") or _env_str("DEEPSEEK_API_KEY")

def require_api_key() -> str:
    key = get_api_key()
    if not key:
        raise RuntimeError("No API key set — export OPENROUTER_API_KEY or DEEPSEEK_API_KEY")
    return key

def get_api_token() -> str:
    """Bearer token for the web API when bound beyond loopback. Empty = localhost mode."""
    return _env_str("RAG_API_TOKEN")

def is_loopback_host(host: str) -> bool:
    # 0.0.0.0 binds ALL interfaces — serving on it without a token is not
    # loopback-only, so the serve guard treats it as remote (token required).
    return host in {"127.0.0.1", "localhost", "::1"} or host.startswith("127.")
