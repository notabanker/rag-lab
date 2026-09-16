from rag_lab import lexical, vector_store
from rag_lab.chunker import Chunk


def test_tokenize_keeps_codes_and_umlauts():
    assert lexical.tokenize("Modul DLBFMWFT1: FinTech") == ["modul", "dlbfmwft1", "fintech"]
    assert lexical.tokenize("Prüfung: Klausur, 90 Min.") == ["prüfung", "klausur", "90", "min"]
    assert lexical.tokenize("...") == []


def _hit(id_, source="a.md"):
    return {"id": id_, "text": f"text {id_}", "metadata": {"source": source}, "distance": 0.1}


def test_rrf_overlap_beats_single_source():
    vector = [_hit("shared"), _hit("v-only")]
    lex = [_hit("l-only"), _hit("shared")]
    fused = lexical.rrf_fuse({"vector": vector, "lexical": lex})
    assert fused[0]["id"] == "shared"  # 1/(60+1) + 1/(60+2) beats any single source
    assert fused[0]["source_ranks"] == {"vector": 1, "lexical": 2}
    ids = [h["id"] for h in fused]
    assert set(ids) == {"shared", "v-only", "l-only"}


def test_rrf_deterministic_tiebreak_and_limit():
    a = [_hit("x"), _hit("y")]
    b = [_hit("y"), _hit("x")]  # symmetric ranks -> equal scores
    fused = lexical.rrf_fuse({"a": a, "b": b})
    assert [h["id"] for h in fused] == ["x", "y"]  # tie broken by id
    assert len(lexical.rrf_fuse({"a": a, "b": b}, limit=1)) == 1


def test_rrf_empty_inputs():
    assert lexical.rrf_fuse({"vector": [], "lexical": []}) == []


def _setup_store(store, docs: dict[str, str]):
    lexical.invalidate()
    chunks = [Chunk(text=t, start=0, end=len(t)) for t in docs.values()]
    ids = list(docs.keys())
    # fake 2-d embeddings; BM25 does not use them
    vecs = [[float(i), 1.0] for i in range(len(ids))]
    metas = [{"source": f"{i}.md", "file_sha": "t", "chunk_idx": n} for n, i in enumerate(ids)]
    vector_store.upsert(chunks, vecs, metas, ids)


def test_bm25_exact_code_ranks_first(store):
    _setup_store(store, {
        "c1": "Das Modul DLBXYZ99 behandelt Zahlungsverkehr und Banken.",
        "c2": "Monte Carlo simulation uses random sampling for risk analysis.",
        "c3": "Banken und Versicherungen sind Finanzintermediäre.",
    })
    hits = lexical.bm25_search("DLBXYZ99")
    assert hits and hits[0]["id"] == "c1"
    assert hits[0]["bm25_score"] > 0


def test_bm25_cache_invalidation_on_upsert(store):
    _setup_store(store, {"c1": "alpha beta gamma", "c3": "delta epsilon"})
    assert lexical.bm25_search("zeta") == []
    # add a new doc; the index must pick it up (invalidate hook + count change).
    # Three docs keep BM25 idf positive (a term in 1 of 2 docs scores exactly 0).
    _setup_store(store, {"c1": "alpha beta gamma", "c2": "zeta eta theta", "c3": "delta epsilon"})
    hits = lexical.bm25_search("zeta")
    assert hits and hits[0]["id"] == "c2"


def test_bm25_empty_collection(store):
    lexical.invalidate()
    assert lexical.bm25_search("anything") == []


def test_document_listing_and_delete(store):
    lexical.invalidate()
    chunks = [
        Chunk(text="alpha beta gamma", start=0, end=16),
        Chunk(text="delta epsilon", start=0, end=13),
        Chunk(text="zeta eta theta", start=0, end=14),
    ]
    metas = [
        {"source": "c1.md", "doc_id": "c1", "file_sha": "c1", "chunk_idx": 0},
        {"source": "c2.md", "doc_id": "c2", "file_sha": "c2", "chunk_idx": 0},
        {"source": "c3.md", "doc_id": "c3", "file_sha": "c3", "chunk_idx": 0},
    ]
    vector_store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], metas, ["c1", "c2", "c3"])
    docs = vector_store.list_documents()
    assert len(docs) == 3
    removed = vector_store.delete_document("c2.md")
    assert removed == 1
    assert all(d["basename"] != "c2.md" for d in vector_store.list_documents())
