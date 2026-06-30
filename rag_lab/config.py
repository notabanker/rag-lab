import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    embed_model: Optional[str] = None


_PROVIDER_META = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "key_env": "OPENROUTER_API_KEY",
    },
    "tokenrouter": {
        "base_url": "https://api.tokenrouter.ai/v1",
        "model": "deepseek/deepseek-chat",
        "key_env": "TOKENROUTER_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:1.5b",
        "key_env": None,
    },
}

PROVIDERS = list(_PROVIDER_META.keys())
DEFAULT_PROVIDER = "deepseek"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def _resolve_provider_name() -> str:
    return os.environ.get("RAG_PROVIDER", DEFAULT_PROVIDER)


def _resolve_embed_model() -> str:
    return os.environ.get("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def get_provider(name: Optional[str] = None) -> ProviderConfig:
    name = name or _resolve_provider_name()
    if name not in _PROVIDER_META:
        raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDER_META)}")
    meta = _PROVIDER_META[name]
    key_env = meta.get("key_env")
    api_key = os.environ.get(key_env, "").strip() if key_env else ""
    if name == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", meta["base_url"])
    else:
        base_url = meta["base_url"]
    return ProviderConfig(
        name=name,
        api_key=api_key,
        base_url=base_url,
        model=meta["model"],
        embed_model=meta.get("embed_model"),
    )


def get_embed_model(provider: Optional[str] = None) -> str:
    prov = get_provider(provider)
    return prov.embed_model or _resolve_embed_model()
