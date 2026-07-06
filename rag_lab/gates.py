"""Metric gate loading and evaluation for eval reports."""
from pathlib import Path

import yaml

SECTIONS = ("retrieval", "answer")


def load_gates(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Gates file not found: {path}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    unknown = set(data) - set(SECTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown gate section(s): {sorted(unknown)}")
    for section, metrics in data.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"{path}: section '{section}' must be a mapping")
        for metric, threshold in metrics.items():
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"{path}: threshold for '{section}.{metric}' must be numeric")
    return data


def evaluate_gates(summary: dict, gates: dict, include_answer: bool = False) -> dict:
    sections = ["retrieval"] + (["answer"] if include_answer else [])
    checked = []
    failures = []
    skipped = []
    for section in sections:
        for metric, threshold in (gates.get(section) or {}).items():
            value = summary.get(metric)
            item = {"section": section, "metric": metric, "threshold": threshold, "value": value}
            if value is None:
                failures.append({**item, "reason": "metric missing"})
            elif value < threshold:
                failures.append({**item, "reason": "below threshold"})
            else:
                checked.append(item)
    if not include_answer:
        skipped.extend({"section": "answer", "metric": metric} for metric in (gates.get("answer") or {}))
    return {
        "passed": not failures,
        "checked": checked,
        "failures": failures,
        "skipped": skipped,
    }
