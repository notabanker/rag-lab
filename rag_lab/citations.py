"""Citation parsing and validation against retrieved context labels."""
from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_SPLIT_RE = re.compile(r"\s*(?:;|,)\s*")


def normalize(label: str) -> str:
    return " ".join((label or "").strip().split())


def context_labels(chunks: list[dict]) -> list[str]:
    labels = []
    for chunk in chunks:
        label = chunk.get("citation") or (chunk.get("metadata") or {}).get("citation") or chunk.get("id")
        if label:
            labels.append(normalize(str(label)))
    return labels


def extract(answer: str, allowed_labels: list[str] = None) -> list[str]:
    """Extract bracketed citations from answer text.

    If a bracket contains comma/semicolon-separated labels and the whole bracket is
    not an allowed label, split it into separate citations.
    """
    allowed = {normalize(x) for x in (allowed_labels or [])}
    out = []
    for raw in _CITATION_RE.findall(answer or ""):
        label = normalize(raw)
        if label in allowed:
            out.append(label)
            continue
        parts = [normalize(p) for p in _SPLIT_RE.split(label) if normalize(p)]
        out.extend(parts or [label])
    return out


def is_refusal(answer: str) -> bool:
    return "don't know from the provided documents" in (answer or "").lower()


def validate(answer: str, chunks: list[dict], require_citation: bool = True) -> dict:
    allowed = context_labels(chunks)
    allowed_set = set(allowed)
    cited = extract(answer, allowed)
    errors = []

    if require_citation and chunks and not cited and not is_refusal(answer):
        errors.append("answer has no citations")

    invalid = [c for c in cited if c not in allowed_set]
    for c in invalid:
        errors.append(f"citation not in context: {c}")

    valid_count = len(cited) - len(invalid)
    citation_count = len(cited)
    return {
        "citation_count": citation_count,
        "citation_valid": not errors,
        "citation_validity_rate": (valid_count / citation_count) if citation_count else (1.0 if not errors else 0.0),
        "citations": cited,
        "context_citations": allowed,
        "citation_errors": errors,
    }
