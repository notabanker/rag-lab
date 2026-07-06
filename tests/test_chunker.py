import pytest

from rag_lab.chunker import chunk_fixed, chunk_sentence


def test_fixed_rejects_overlap_ge_size():
    with pytest.raises(ValueError):
        chunk_fixed("x" * 1000, size=64, overlap=64)
    with pytest.raises(ValueError):
        chunk_fixed("x" * 1000, size=64, overlap=100)


def test_fixed_rejects_bad_size():
    with pytest.raises(ValueError):
        chunk_fixed("abc", size=0)
    with pytest.raises(ValueError):
        chunk_fixed("abc", size=10, overlap=-1)


def test_fixed_covers_full_text():
    text = "abcdefghij" * 50
    chunks = chunk_fixed(text, size=64, overlap=8)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for c in chunks:
        assert text[c.start:c.end] == c.text


def test_fixed_empty_text():
    assert chunk_fixed("") == []


def test_sentence_offsets_match_source():
    text = "Alpha beta gamma. " * 10 + "Delta epsilon zeta! " * 10
    chunks = chunk_sentence(text, target_size=100)
    assert chunks
    for c in chunks:
        assert text[c.start:c.end] == c.text


def test_sentence_empty_text():
    assert chunk_sentence("") == []


def test_sentence_single_long_sentence():
    text = "word " * 200  # no sentence terminators
    chunks = chunk_sentence(text.strip(), target_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == text.strip()


def test_sentence_rejects_bad_target_size():
    with pytest.raises(ValueError):
        chunk_sentence("Some text.", target_size=0)
