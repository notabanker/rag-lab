from rag_lab import runs, vector_store


def _init(tmp_path):
    vector_store.init_store(str(tmp_path / "db"))


def _result():
    return {
        "answer": "The answer [c1].",
        "verifier": {"score": 9, "verdict": "GROUNDED", "grounded": True, "issues": []},
        "iterations": 1,
        "trace": [{"iter": 1, "verifier_score": 9}],
        "citation_validation": {"citation_valid": True, "citation_count": 1, "citation_errors": []},
        "chunks": [
            {"id": "c1", "distance": 0.12, "metadata": {"source": "a.pdf"}},
            {"id": "c2", "distance": 0.30, "metadata": {"source": "b.md"}},
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40},
    }


def test_log_and_get_run(tmp_path):
    _init(tmp_path)
    rid = runs.log_run("What is X?", {"top_k": 20}, _result(), latency_ms=1234)
    r = runs.get_run(rid)
    assert r["question"] == "What is X?"
    assert r["config"] == {"top_k": 20}
    assert r["verifier"]["score"] == 9
    assert r["latency_ms"] == 1234
    assert r["prompt_tokens"] == 100
    assert r["completion_tokens"] == 40
    assert r["citation_validation"]["citation_valid"] is True
    assert [c["chunk_id"] for c in r["chunks"]] == ["c1", "c2"]
    assert r["chunks"][0]["rank"] == 1
    assert r["chunks"][0]["source"] == "a.pdf"


def test_list_runs_order_and_limit(tmp_path):
    _init(tmp_path)
    for i in range(3):
        runs.log_run(f"q{i}", {}, _result(), latency_ms=i)
    rows = runs.list_runs(limit=2)
    assert len(rows) == 2
    assert rows[0]["question"] == "q2"  # newest first


def test_get_run_missing(tmp_path):
    _init(tmp_path)
    assert runs.get_run(999) is None


def test_log_eval_roundtrip(tmp_path):
    _init(tmp_path)
    eid = runs.log_eval("baseline", {"top_k": 20}, {"hit_rate": 0.9}, [{"id": "q1"}])
    evals = runs.list_evals()
    assert evals[0]["id"] == eid
    assert evals[0]["variant"] == "baseline"
    assert evals[0]["summary"]["hit_rate"] == 0.9


def test_persistence_across_connections(tmp_path):
    _init(tmp_path)
    rid = runs.log_run("persist?", {}, _result(), latency_ms=1)
    # fresh connection path: re-init with the same dir
    vector_store.init_store(str(tmp_path / "db"))
    assert runs.get_run(rid)["question"] == "persist?"
