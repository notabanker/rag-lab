import pytest

from rag_lab.chunker import Chunk
from rag_lab.vector_store import count, get_collection, init_store, query, query_multi, upsert


@pytest.fixture(autouse=True)
def _clean_store(tmp_path):
    db = tmp_path / "chroma_db"
    db.mkdir()
    init_store(str(db))
    yield
    from rag_lab.vector_store import shutdown
    shutdown()


class TestMultiCollection:
    def test_get_collection_creates(self):
        coll = get_collection("test_coll")
        assert coll is not None
        assert count("test_coll") == 0

    def test_get_collection_reuses(self):
        a = get_collection("test_reuse")
        b = get_collection("test_reuse")
        assert a is b

    def test_upsert_to_named_collection(self):
        chunks = [Chunk(text="hello", start=0, end=5)]
        upsert(chunks, [[0.1, 0.2, 0.3, 0.4]], [{"source": "t"}], ["id1"], collection="test_named")
        assert count("test_named") == 1
        assert count() == 0

    def test_query_named_collection(self):
        chunks = [Chunk(text="hello world", start=0, end=11)]
        upsert(chunks, [[0.1, 0.2, 0.3, 0.4]], [{"source": "t"}], ["id2"], collection="test_query")
        hits = query([0.1, 0.2, 0.3, 0.4], top_k=5, collection="test_query")
        assert len(hits) >= 1
        assert hits[0]["text"] == "hello world"

    def test_query_multi_collections(self):
        upsert(
            [Chunk(text="alpha", start=0, end=5)],
            [[0.5, 0.1, 0.3, 0.7]],
            [{"source": "a"}],
            ["id_a"],
            collection="coll_a",
        )
        upsert(
            [Chunk(text="beta", start=0, end=4)],
            [[0.1, 0.2, 0.3, 0.4]],
            [{"source": "b"}],
            ["id_b"],
            collection="coll_b",
        )
        hits = query_multi([0.1, 0.15, 0.3, 0.4], ["coll_a", "coll_b"], top_k=5)
        assert len(hits) >= 2
        collections_found = {h.get("collection") for h in hits}
        assert "coll_a" in collections_found
        assert "coll_b" in collections_found

    def test_query_multi_top_k_limit(self):
        upsert(
            [Chunk(text="x", start=0, end=1)],
            [[0.1, 0.2, 0.3, 0.4]],
            [{"source": "x"}],
            ["id_x"],
            collection="col_one",
        )
        upsert(
            [Chunk(text="y", start=0, end=1)],
            [[0.2, 0.1, 0.3, 0.4]],
            [{"source": "y"}],
            ["id_y"],
            collection="col_two",
        )
        hits = query_multi([0.1, 0.15, 0.3, 0.4], ["col_one", "col_two"], top_k=1)
        assert len(hits) == 1

    def test_query_empty_collection(self):
        get_collection("empty_coll")
        hits = query([0.1, 0.2], top_k=5, collection="empty_coll")
        assert hits == []

    def test_count_named(self):
        chunks = [Chunk(text="a", start=0, end=1)]
        upsert(chunks, [[0.1, 0.2, 0.3, 0.4]], [{"s": "t"}], ["id3"], collection="count_coll")
        assert count("count_coll") == 1
