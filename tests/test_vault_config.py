import pytest

from rag_lab.vault_config import (
    DEFAULT_EXCLUDE_PATTERNS,
    load_exclude_patterns,
    resolve_layer,
    should_index,
)


class TestResolveLayer:
    def test_l1_excluded(self):
        assert resolve_layer("vault/L1-working/today.md", "vault") is None

    def test_l2(self):
        assert resolve_layer("vault/L2-episodic/agent-logs/session.md", "vault") == "l2_episodic"

    def test_l3(self):
        assert resolve_layer("vault/L3-semantic/entities/test.md", "vault") == "l3_semantic"

    def test_l3_knowledge(self):
        assert resolve_layer("vault/L3-semantic/knowledge/arch.md", "vault") == "l3_semantic"

    def test_none_for_unknown(self):
        assert resolve_layer("vault/other/file.md", "vault") is None

    def test_none_outside_vault(self):
        assert resolve_layer("/tmp/outside.md", "/Users/vault") is None

    def test_empty_relative(self):
        assert resolve_layer("vault", "vault") is None


class TestLoadExcludePatterns:
    def test_defaults_when_no_file(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        patterns = load_exclude_patterns(str(vault))
        for p in DEFAULT_EXCLUDE_PATTERNS:
            assert p in patterns

    def test_custom_patterns(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / ".rag-ignore").write_text("custom_dir/\n")
        patterns = load_exclude_patterns(str(vault))
        assert "custom_dir/" in patterns
        assert ".git/" in patterns

    def test_comments_skipped(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / ".rag-ignore").write_text("# this is a comment\nreal_dir/\n")
        patterns = load_exclude_patterns(str(vault))
        assert "real_dir/" in patterns
        assert "# this is a comment" not in patterns

    def test_blank_lines_skipped(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / ".rag-ignore").write_text("\n\nreal_dir/\n\n")
        patterns = load_exclude_patterns(str(vault))
        assert "real_dir/" in patterns
        assert "" not in patterns


class TestShouldIndex:
    @pytest.fixture
    def vault(self, tmp_path):
        v = tmp_path / "vault"
        v.mkdir()
        return str(v)

    def test_markdown_in_l3(self, vault):
        assert should_index(f"{vault}/L3-semantic/entities/test.md", vault, DEFAULT_EXCLUDE_PATTERNS)

    def test_excluded_dir(self, vault):
        assert not should_index(f"{vault}/personal/secret.md", vault, DEFAULT_EXCLUDE_PATTERNS)

    def test_excluded_nested(self, vault):
        assert not should_index(f"{vault}/personal/deep/nested.md", vault, DEFAULT_EXCLUDE_PATTERNS)

    def test_git_excluded(self, vault):
        assert not should_index(f"{vault}/.git/config", vault, DEFAULT_EXCLUDE_PATTERNS)

    def test_drafts_excluded(self, vault):
        assert not should_index(f"{vault}/L2-episodic/drafts/session.md", vault, DEFAULT_EXCLUDE_PATTERNS)

    def test_ds_store_excluded(self, vault):
        assert not should_index(f"{vault}/.DS_Store", vault, DEFAULT_EXCLUDE_PATTERNS)

    def test_outside_vault(self, vault):
        assert not should_index("/tmp/outside.md", vault, DEFAULT_EXCLUDE_PATTERNS)

    def test_custom_glob(self, vault):
        patterns = DEFAULT_EXCLUDE_PATTERNS + ["chiron/"]
        assert not should_index(f"{vault}/L3-semantic/chiron/megabible/test.md", vault, patterns)
