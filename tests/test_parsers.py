import tempfile
from pathlib import Path

from rag_lab.parsers import pick_parser
from rag_lab.parsers.markdown import parse_markdown
from rag_lab.parsers.pdf import parse_pdf


class TestMarkdown:
    def test_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Hello\n\nThis is **bold** text.\n")
            f.flush()
            path = f.name
        try:
            result = parse_markdown(path)
            assert "Hello" in result
            assert "bold" in result
        finally:
            Path(path).unlink()

    def test_frontmatter_stripped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("---\ntitle: test\n---\n\n# Content here\n")
            f.flush()
            path = f.name
        try:
            result = parse_markdown(path)
            assert "title" not in result
            assert "Content here" in result
        finally:
            Path(path).unlink()

    def test_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name
        try:
            result = parse_markdown(path)
            assert result == ""
        finally:
            Path(path).unlink()


class TestPickParser:
    def test_pdf(self):
        fn = pick_parser("test.pdf")
        assert fn is parse_pdf

    def test_epub(self):
        fn = pick_parser("test.epub")
        from rag_lab.parsers.epub import parse_epub
        assert fn is parse_epub

    def test_md(self):
        fn = pick_parser("test.md")
        assert fn is parse_markdown

    def test_unknown(self):
        import pytest
        with pytest.raises(ValueError, match="Unsupported"):
            pick_parser("test.xyz")
