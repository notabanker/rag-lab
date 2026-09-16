"""End-to-end retrieval with the REAL embedder and cross-encoder.

Every other test monkeypatches the models away, so nothing verifies that
`retrieve_hits` actually retrieves: the e5 query/passage prefixes could be
dropped, fusion could lose a source list, and rerank could stop running —
all silently, with the rest of the suite still green.

Each assertion below was mutation-checked: breaking the layer it names
turns it red. One case could not be made mutation-sensitive and was left out rather
than written as decoration: no query over a corpus this small makes BM25
necessary for top-1, because the vector path resolves rare codes and exact
identifiers on its own once the surrounding text is similar. Removing the
lexical source from fusion breaks nothing visible here. That coverage
belongs to eval/questions.yaml at real corpus scale.

A 'reranking rescues a wrong fusion ranking' case was tried and dropped:
with a decoy strong enough to beat RRF the cross-encoder prefers it too,
and with a weaker decoy fusion is already correct. The balance between
the two is narrower than BM25's sensitivity to corpus composition, so
any such test would be flaky. Reranking is covered below instead by the
fact that skipping it leaves hits without a rerank_score.

Slow (loads two models), so it is excluded from the default `pytest` run
and executed explicitly by scripts/check.sh.
"""
import pytest

from rag_lab import embedder, ingestion, lexical, vector_store
from rag_lab.retriever import RetrievalConfig, retrieve_hits

pytestmark = pytest.mark.slow

CORPUS = {
    "raft.md": (
        "# Raft\n\n"
        "Raft is a consensus algorithm. A single leader accepts client "
        "commands and replicates the log to followers. If the leader fails, "
        "an election chooses a new one after a randomized timeout."
    ),
    # Distractor sharing everyday vocabulary ("agree", "ordering",
    # "several") with the retrieval queries, so top-1 is not free.
    "party.md": (
        "# Office party\n\n"
        "We ordered several large pizzas and arranged the seating so the "
        "whole team could agree on a date for the celebration."
    ),
    "lcr.md": (
        "# Basel III\n\n"
        "The Liquidity Coverage Ratio requires banks to hold high-quality "
        "liquid assets sufficient to survive a thirty-day stress scenario."
    ),
    "ernaehrung.md": (
        "# Ernährung\n\n"
        "Vitamin D wird bei Sonnenlicht in der Haut gebildet. Eine "
        "ausgewogene Ernährung liefert ausreichend Mineralstoffe."
    ),
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    vector_store.init_store(str(tmp / "db"))
    lexical.invalidate()
    for name, text in CORPUS.items():
        path = tmp / name
        path.write_text(text)
        ingestion.ingest_file(str(path))
    assert vector_store.count() == len(CORPUS), "expected one chunk per short doc"
    return tmp


def _top(question: str, **cfg) -> str:
    hits = retrieve_hits(question, RetrievalConfig(top_k=50, **cfg))
    return hits[0]["metadata"]["source"].rsplit("/", 1)[-1] if hits else "NONE"


def test_german_question_finds_german_document(corpus):
    """Multilingual path: a German query reaches the German passage although
    every other document is English. Red if the model or prefixes break."""
    assert _top("Wie entsteht Vitamin D im Körper?") == "ernaehrung.md"


def test_vector_half_survives_fusion(corpus):
    """`distributed agreement protocol` shares no token with raft.md, so
    BM25 returns nothing at all. Hybrid must still answer from the vector
    list — red if fusion ever drops the vector source."""
    assert _top("distributed agreement protocol", mode="lexical") == "NONE"
    assert _top("distributed agreement protocol", mode="hybrid") == "raft.md"


def test_rerank_scores_annotate_the_whole_pool(corpus):
    hits = retrieve_hits("liquidity requirements for banks", RetrievalConfig())
    assert hits[0]["metadata"]["source"].endswith("lcr.md")
    scores = [h["rerank_score"] for h in hits]
    assert scores == sorted(scores, reverse=True), "rerank did not order by score"


def test_query_and_passage_prefixes_differ(corpus):
    """e5 needs 'query: ' / 'passage: '. Dropping a prefix costs accuracy
    silently; identical vectors here would prove it was lost."""
    text = "Wie entsteht Vitamin D im Körper?"
    assert embedder.embed_query(text) != embedder.embed([text], input_type="document")[0]
