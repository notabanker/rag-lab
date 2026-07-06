"""Eval harness: golden questions, retrieval/answer metrics, variant comparison."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time

import yaml

REFUSAL_MARKER = "don't know from the provided documents"

# ---------- Golden question set ----------

@dataclass
class EvalQuestion:
    id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_fragments: list[str] = field(default_factory=list)
    expected_chunk_fragments: list[str] = field(default_factory=list)  # must appear in retrieved chunk TEXT
    expect_refusal: bool = False
    tags: list[str] = field(default_factory=list)

def load_questions(path: str) -> list[EvalQuestion]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Questions file not found: {path}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise ValueError(f"{path}: expected a top-level 'questions:' list")
    questions = []
    seen_ids = set()
    for i, raw in enumerate(data["questions"], 1):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: question #{i} is not a mapping")
        qid = raw.get("id")
        text = raw.get("question")
        if not qid or not text:
            raise ValueError(f"{path}: question #{i} needs both 'id' and 'question'")
        if qid in seen_ids:
            raise ValueError(f"{path}: duplicate question id '{qid}'")
        seen_ids.add(qid)
        q = EvalQuestion(
            id=str(qid),
            question=str(text),
            expected_sources=list(raw.get("expected_sources") or []),
            expected_fragments=list(raw.get("expected_fragments") or []),
            expected_chunk_fragments=list(raw.get("expected_chunk_fragments") or []),
            expect_refusal=bool(raw.get("expect_refusal", False)),
            tags=list(raw.get("tags") or []),
        )
        if not q.expect_refusal and not q.expected_sources:
            raise ValueError(f"{path}: question '{qid}' needs 'expected_sources' (or 'expect_refusal: true')")
        questions.append(q)
    if not questions:
        raise ValueError(f"{path}: no questions defined")
    return questions

# ---------- Variants ----------

@dataclass
class Variant:
    name: str
    mode: str = "hybrid"
    top_k: int = 50
    rerank_top: int = 5
    use_reranker: bool = True
    small_to_big: bool = True
    parent_top_k: int = 5
    max_context_chars: int = 12000
    min_score: int = 8
    max_iters: int = 3
    collection: str | None = None

    def config(self) -> dict:
        return asdict(self)

    def to_retrieval_config(self):
        from .retriever import RetrievalConfig
        return RetrievalConfig(
            mode=self.mode, top_k=self.top_k, rerank_top=self.rerank_top,
            use_reranker=self.use_reranker, small_to_big=self.small_to_big,
            parent_top_k=self.parent_top_k, max_context_chars=self.max_context_chars,
            min_score=self.min_score, max_iters=self.max_iters,
        )

def load_variants(path: str) -> list[Variant]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Variants file not found: {path}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("variants"), list):
        raise ValueError(f"{path}: expected a top-level 'variants:' list")
    variants = []
    seen = set()
    known = {
        "name", "mode", "top_k", "rerank_top", "use_reranker", "small_to_big",
        "parent_top_k", "max_context_chars", "min_score", "max_iters", "collection",
    }
    for i, raw in enumerate(data["variants"], 1):
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ValueError(f"{path}: variant #{i} needs a 'name'")
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{path}: variant '{raw['name']}' has unknown keys: {sorted(unknown)}")
        if raw["name"] in seen:
            raise ValueError(f"{path}: duplicate variant name '{raw['name']}'")
        seen.add(raw["name"])
        variants.append(Variant(**raw))
    if not variants:
        raise ValueError(f"{path}: no variants defined")
    return variants

# ---------- Metrics ----------

def _hit_matches(hit: dict, expected_basenames: set[str]) -> bool:
    source = (hit.get("metadata") or {}).get("source", "")
    return Path(source).name in expected_basenames

def retrieval_metrics(hits: list[dict], expected_sources: list[str], k: int,
                      chunk_fragments: list[str] = None) -> dict:
    """Source-level hit@k over the first k hits; MRR over the full hit list.
    If chunk_fragments given, also chunk-level metrics: the retrieved chunk TEXT
    itself must contain one of the fragments (much stricter than source-level)."""
    expected = {Path(s).name for s in expected_sources}
    first_rank = next((i + 1 for i, h in enumerate(hits) if _hit_matches(h, expected)), None)
    out = {
        "hit_at_1": any(_hit_matches(h, expected) for h in hits[:1]),
        "hit_at_3": any(_hit_matches(h, expected) for h in hits[:3]),
        "hit_at_5": any(_hit_matches(h, expected) for h in hits[:5]),
        "hit_at_10": any(_hit_matches(h, expected) for h in hits[:10]),
        "hit_at_k": any(_hit_matches(h, expected) for h in hits[:k]),
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "first_rank": first_rank,
    }
    if chunk_fragments:
        frags = [f.lower() for f in chunk_fragments]
        def chunk_match(h):
            text = (h.get("text") or "").lower()
            return any(f in text for f in frags)
        c_first = next((i + 1 for i, h in enumerate(hits) if chunk_match(h)), None)
        out["chunk_hit_at_k"] = any(chunk_match(h) for h in hits[:k])
        out["chunk_mrr"] = (1.0 / c_first) if c_first else 0.0
        out["chunk_first_rank"] = c_first
    return out

def is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in (answer or "").lower()

def fragment_match(answer: str, fragments: list[str]) -> bool | None:
    """Any-match, case-insensitive. None when the question defines no fragments."""
    if not fragments:
        return None
    low = (answer or "").lower()
    return any(f.lower() in low for f in fragments)

def _prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}

# ---------- Eval loop ----------

def evaluate(
    questions: list[EvalQuestion],
    cfg=None,
    retrieval_only: bool = False,
) -> dict:
    from .retriever import RetrievalConfig, retrieve, retrieve_hits, select_context_chunks
    cfg = cfg or RetrievalConfig()

    per_question = []
    for q in questions:
        entry = {"id": q.id, "question": q.question, "tags": q.tags, "expect_refusal": q.expect_refusal}
        if not q.expect_refusal:
            start = time.perf_counter()
            hits = retrieve_hits(q.question, cfg)
            context = select_context_chunks(hits, cfg)
            entry["retrieval_latency_ms"] = int((time.perf_counter() - start) * 1000)
            entry["context_chars"] = sum(len(c.get("text") or "") for c in context)
            entry["context_count"] = len(context)
            entry.update(retrieval_metrics(
                hits, q.expected_sources, k=cfg.rerank_top,
                chunk_fragments=q.expected_chunk_fragments,
            ))
            entry.update(_prefix_metrics(retrieval_metrics(
                context, q.expected_sources, k=len(context) or 1,
                chunk_fragments=q.expected_chunk_fragments,
            ), "context"))
        if not retrieval_only:
            result = retrieve(q.question, cfg, log=False)
            answer = result.get("answer", "")
            citation_validation = result.get("citation_validation") or {}
            entry["answer"] = answer
            entry["verifier_score"] = (result.get("verifier") or {}).get("score")
            entry["iterations"] = result.get("iterations")
            entry["latency_ms"] = result.get("latency_ms")
            entry["usage"] = result.get("usage", {})
            entry["citation_validation"] = citation_validation
            entry["citation_valid"] = citation_validation.get("citation_valid")
            entry["citation_count"] = citation_validation.get("citation_count")
            entry["citation_errors"] = citation_validation.get("citation_errors", [])
            refused = is_refusal(answer)
            if q.expect_refusal:
                entry["refusal_correct"] = refused
            else:
                entry["refused"] = refused
                fm = fragment_match(answer, q.expected_fragments)
                if fm is not None:
                    entry["fragment_matched"] = fm
        per_question.append(entry)

    return {
        "per_question": per_question,
        "summary": summarize(per_question),
        "config": {**asdict(cfg), "retrieval_only": retrieval_only},
    }

def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None

def _as_number(v) -> float | None:
    """Coerce verifier scores that arrive as strings; None if not numeric."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def summarize(per_question: list[dict], include_tags: bool = True) -> dict:
    answered = [e for e in per_question if not e["expect_refusal"]]
    refusals = [e for e in per_question if e["expect_refusal"]]
    hits = [e["hit_at_k"] for e in answered if "hit_at_k" in e]
    summary = {
        "questions": len(per_question),
        "hit_rate": _mean([1.0 if h else 0.0 for h in hits]),
        "hit_at_1": _mean([1.0 if e["hit_at_1"] else 0.0 for e in answered if "hit_at_1" in e]),
        "hit_at_3": _mean([1.0 if e["hit_at_3"] else 0.0 for e in answered if "hit_at_3" in e]),
        "hit_at_5": _mean([1.0 if e["hit_at_5"] else 0.0 for e in answered if "hit_at_5" in e]),
        "hit_at_10": _mean([1.0 if e["hit_at_10"] else 0.0 for e in answered if "hit_at_10" in e]),
        "mrr": _mean([e.get("mrr") for e in answered if "mrr" in e]),
        "chunk_hit_rate": _mean([
            1.0 if e["chunk_hit_at_k"] else 0.0 for e in answered if "chunk_hit_at_k" in e
        ]),
        "chunk_mrr": _mean([e.get("chunk_mrr") for e in answered if "chunk_mrr" in e]),
        "context_hit_rate": _mean([
            1.0 if e["context_hit_at_k"] else 0.0 for e in answered if "context_hit_at_k" in e
        ]),
        "context_mrr": _mean([e.get("context_mrr") for e in answered if "context_mrr" in e]),
        "context_chunk_hit_rate": _mean([
            1.0 if e["context_chunk_hit_at_k"] else 0.0 for e in answered if "context_chunk_hit_at_k" in e
        ]),
        "context_chunk_mrr": _mean([
            e.get("context_chunk_mrr") for e in answered if "context_chunk_mrr" in e
        ]),
        "fragment_rate": _mean([
            1.0 if e["fragment_matched"] else 0.0
            for e in answered if e.get("fragment_matched") is not None
        ]),
        "verifier_mean": _mean([_as_number(e.get("verifier_score")) for e in answered]),
        "refusal_accuracy": _mean([
            1.0 if e.get("refusal_correct") else 0.0 for e in refusals if "refusal_correct" in e
        ]),
        "citation_validity_rate": _mean([
            1.0 if e.get("citation_valid") else 0.0
            for e in per_question if e.get("citation_valid") is not None
        ]),
        "citation_count_mean": _mean([
            e.get("citation_count") for e in per_question if e.get("citation_count") is not None
        ]),
        "retrieval_latency_ms_mean": _mean([
            e.get("retrieval_latency_ms") for e in per_question if e.get("retrieval_latency_ms") is not None
        ]),
        "latency_ms_mean": _mean([e.get("latency_ms") for e in per_question if e.get("latency_ms") is not None]),
        "context_chars_mean": _mean([e.get("context_chars") for e in answered if e.get("context_chars") is not None]),
        "prompt_tokens": sum((e.get("usage") or {}).get("prompt_tokens", 0) for e in per_question),
        "completion_tokens": sum((e.get("usage") or {}).get("completion_tokens", 0) for e in per_question),
    }
    if include_tags:
        tag_summary = {}
        for e in answered:
            for tag in e.get("tags", []):
                tag_summary.setdefault(tag, []).append(e)
        summary["by_tag"] = {tag: summarize(rows, include_tags=False) for tag, rows in tag_summary.items()}
    return summary
