import pytest

from rag_lab.chunker import chunk, chunk_fixed, chunk_sentence


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


class TestChunkDispatch:
    def test_dispatch_fixed(self):
        chunks = chunk("hello world", strategy="fixed", size=5, overlap=2)
        assert len(chunks) > 0

    def test_dispatch_sentence(self):
        chunks = chunk("Hello world. Foo bar.", strategy="sentence", target_size=20, overlap=1)
        assert len(chunks) > 0

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk("test", strategy="invalid")
