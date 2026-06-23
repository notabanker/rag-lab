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


PROVIDERS = {
    "deepseek": ProviderConfig(
        name="deepseek",
        api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat",
    ),
    "tokenrouter": ProviderConfig(
        name="tokenrouter",
        api_key=os.environ.get("TOKENROUTER_API_KEY", "").strip(),
        base_url="https://api.tokenrouter.ai/v1",
        model="deepseek/deepseek-chat",
    ),
    "ollama": ProviderConfig(
        name="ollama",
        api_key="",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        model="qwen2.5:1.5b",
    ),
}

DEFAULT_PROVIDER = os.environ.get("RAG_PROVIDER", "deepseek")
DEFAULT_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")


def get_provider(name: Optional[str] = None) -> ProviderConfig:
    name = name or DEFAULT_PROVIDER
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS)}")
    return PROVIDERS[name]


def get_embed_model(provider: Optional[str] = None) -> str:
    prov = get_provider(provider)
    return prov.embed_model or DEFAULT_EMBED_MODEL
