import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat").strip()
LLM_VERIFIER_MODEL = os.environ.get("LLM_VERIFIER_MODEL", "").strip() or LLM_MODEL

def get_api_key() -> str:
    return OPENROUTER_API_KEY or DEEPSEEK_API_KEY
