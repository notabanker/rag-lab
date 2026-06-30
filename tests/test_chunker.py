import pytest

from rag_lab.chunker import chunk, chunk_document, chunk_fixed, chunk_paragraph, chunk_sentence


class TestChunkFixed:
    def test_basic(self):
        chunks = chunk_fixed("hello world", size=5, overlap=2)
        assert len(chunks) == 3
        assert chunks[0].text == "hello"
        assert chunks[1].text == "lo wo"
        assert chunks[2].text == "world"

    def test_single_chunk(self):
        chunks = chunk_fixed("ab", size=5, overlap=2)
        assert len(chunks) == 1
        assert chunks[0].text == "ab"

    def test_exact_size(self):
        chunks = chunk_fixed("12345", size=5, overlap=2)
        assert len(chunks) == 1

    def test_overlap_validation(self):
        with pytest.raises(ValueError, match="must be greater than overlap"):
            chunk_fixed("test", size=5, overlap=10)

    def test_offsets(self):
        chunks = chunk_fixed("abcdefgh", size=3, overlap=1)
        assert chunks[0].start == 0
        assert chunks[0].end == 3
        assert chunks[1].start == 2
        assert chunks[1].end == 5

    def test_empty(self):
        chunks = chunk_fixed("", size=5, overlap=2)
        assert len(chunks) == 0


class TestChunkSentence:
    def test_basic(self):
        chunks = chunk_sentence("Hello world. Foo bar. Baz qux.", target_size=20, overlap=1)
        assert len(chunks) >= 1
        assert any("Hello" in c.text for c in chunks)

    def test_single_sentence(self):
        chunks = chunk_sentence("Hello world.", target_size=512, overlap=1)
        assert len(chunks) == 1

    def test_no_overlap(self):
        chunks = chunk_sentence("A. B. C. D.", target_size=8, overlap=0)
        assert all(c.start <= c.end for c in chunks)

    def test_offsets_monotonic(self):
        text = "First sentence. Second one. Third here. Fourth now."
        chunks = chunk_sentence(text, target_size=30, overlap=1)
        for i in range(len(chunks) - 1):
            assert chunks[i].start <= chunks[i + 1].start, f"non-monotonic at chunk {i}"
            assert chunks[i].end <= chunks[i + 1].end, f"end went backward at chunk {i}"


class TestChunkDocument:
    def test_basic(self):
        chunks = chunk_document("Hello world\n\nFoo bar")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world\n\nFoo bar"

    def test_empty(self):
        chunks = chunk_document("")
        assert len(chunks) == 0

    def test_whitespace_only(self):
        chunks = chunk_document("   \n  ")
        assert len(chunks) == 0

    def test_strips_surrounding_whitespace(self):
        chunks = chunk_document("  Hello  ")
        assert chunks[0].text == "Hello"

    def test_offsets(self):
        chunks = chunk_document("  Hello world  ")
        assert chunks[0].start == 2
        assert chunks[0].end == 13


class TestChunkParagraph:
    def test_basic(self):
        chunks = chunk_paragraph("Alpha.\n\nBeta.\n\nGamma.", max_chars=10, overlap_paragraphs=0)
        assert len(chunks) >= 2

    def test_single_paragraph(self):
        chunks = chunk_paragraph("Single paragraph.", max_chars=1000)
        assert len(chunks) == 1
        assert "Single paragraph" in chunks[0].text

    def test_empty(self):
        chunks = chunk_paragraph("")
        assert len(chunks) == 0

    def test_whitespace_only(self):
        chunks = chunk_paragraph("  \n\n  \n  ")
        assert len(chunks) == 0

    def test_no_split_when_under_max(self):
        chunks = chunk_paragraph("P1.\n\nP2.", max_chars=2000)
        assert len(chunks) == 1
        assert "P1." in chunks[0].text
        assert "P2." in chunks[0].text

    def test_offsets_monotonic(self):
        text = "AAAA.\n\nBBBB.\n\nCCCC.\n\nDDDD.\n\nEEEE."
        chunks = chunk_paragraph(text, max_chars=10, overlap_paragraphs=1)
        for i in range(len(chunks) - 1):
            assert chunks[i].start <= chunks[i + 1].start, f"non-monotonic at chunk {i}"
            assert chunks[i].end <= chunks[i + 1].end, f"end went backward at chunk {i}"

    def test_overlap_preserves_context(self):
        text = "First paragraph here.\n\nSecond paragraph text.\n\nThird chunk content."
        chunks = chunk_paragraph(text, max_chars=20, overlap_paragraphs=1)
        assert len(chunks) >= 2
        assert "Second paragraph" in chunks[1].text

    def test_large_max_single_chunk(self):
        chunks = chunk_paragraph("P1.\n\nP2.\n\nP3.", max_chars=10000)
        assert len(chunks) == 1

    def test_separator_variants(self):
        chunks = chunk_paragraph("A.\n\n\n\nB.\n\nC.", max_chars=1000)
        assert len(chunks) == 1
        assert "A." in chunks[0].text
        assert "B." in chunks[0].text
        assert "C." in chunks[0].text


class TestChunkDispatch:
    def test_dispatch_fixed(self):
        chunks = chunk("hello world", strategy="fixed", size=5, overlap=2)
        assert len(chunks) > 0

    def test_dispatch_sentence(self):
        chunks = chunk("Hello world. Foo bar.", strategy="sentence", target_size=20, overlap=1)
        assert len(chunks) > 0

    def test_dispatch_document(self):
        chunks = chunk("hello world", strategy="document")
        assert len(chunks) == 1

    def test_dispatch_paragraph(self):
        chunks = chunk("A.\n\nB.", strategy="paragraph", max_chars=2, overlap_paragraphs=0)
        assert len(chunks) >= 2

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk("test", strategy="invalid")
