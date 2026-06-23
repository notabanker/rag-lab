import pytest

from rag_lab.config import DEFAULT_PROVIDER, PROVIDERS, get_provider


class TestGetProvider:
    def test_deepseek(self):
        prov = get_provider("deepseek")
        assert prov.name == "deepseek"
        assert prov.base_url == "https://api.deepseek.com/v1"

    def test_openrouter(self):
        prov = get_provider("openrouter")
        assert prov.name == "openrouter"
        assert "openrouter" in prov.base_url

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_default(self):
        prov = get_provider()
        assert prov.name == DEFAULT_PROVIDER

    def test_all_registered(self):
        assert "deepseek" in PROVIDERS
        assert "openrouter" in PROVIDERS
        assert "tokenrouter" in PROVIDERS
        assert "ollama" in PROVIDERS
