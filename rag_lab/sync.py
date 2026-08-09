"""Incremental folder sync: classify files against the manifest, then apply.

build_plan is pure (read-only: scans dirs, hashes files, reads the manifest
list passed in). apply_plan performs the ingest/delete work and returns
counts. The CLI owns confirmation prompts and rendering.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import ingestion, manifest, vector_store
from .parsers import PARSERS


@dataclass
class SyncPlan:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _supported(path: Path) -> bool:
    return path.suffix.lower() in PARSERS


def _scan_files(dirs: list[str]) -> list[str]:
    files = []
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            raise ValueError(f"Not a directory: {d}")
        for dirpath, _, filenames in os.walk(root):
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if _supported(p):
                    files.append(str(p))
    return sorted(set(files))


def _under_dirs(path: str, dirs: list[str]) -> bool:
    abs_path = Path(path).resolve()
    return any(abs_path.is_relative_to(Path(d).resolve()) for d in dirs)


def build_plan(dirs: list[str], docs: list[dict]) -> SyncPlan:
    """Classify every supported file under `dirs` against the manifest docs.

    unchanged: same source path AND same sha. updated: source present but
    content changed. added: not in the manifest. pruned: manifest source
    under `dirs` whose file no longer exists on disk (only sources inside
    the synced dirs — never anything ingested elsewhere).
    """
    plan = SyncPlan()
    by_source = {d["source"]: d for d in docs}
    for src in _scan_files(dirs):
        try:
            sha = ingestion.content_sha(Path(src).read_bytes())
        except OSError as e:
            plan.failed.append(f"{src}: {e}")
            continue
        doc = by_source.get(src)
        if doc is None:
            plan.added.append(src)
        elif doc["file_sha"] != sha:
            plan.updated.append(src)
        else:
            plan.unchanged.append(src)
    for src in sorted(by_source):
        if _under_dirs(src, dirs) and not Path(src).exists():
            plan.pruned.append(src)
    return plan


def apply_plan(
    plan: SyncPlan,
    *,
    dry_run: bool = False,
    ocr: str = "auto",
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
    parent_size: int = 4,
) -> dict:
    """Execute the plan. dry_run only returns what would happen."""
    if dry_run:
        return {
            "added": len(plan.added),
            "updated": len(plan.updated),
            "unchanged": len(plan.unchanged),
            "pruned": len(plan.pruned),
            "failed": 0,
        }
    counts = {"added": 0, "updated": 0, "unchanged": len(plan.unchanged),
              "pruned": 0, "failed": 0}
    for src in plan.added + plan.updated:
        old = manifest.get_document(src)
        try:
            ingestion.ingest_file(
                src, ocr=ocr, strategy=strategy, chunk_size=chunk_size,
                overlap=overlap, parent_size=parent_size,
            )
        except Exception as e:
            plan.failed.append(f"{src}: {e}")
            counts["failed"] += 1
            continue
        if old is not None:
            # Replace, don't accumulate: drop the previous sha's chunks AFTER
            # the new version is safely in — a failed ingest (e.g. a rewritten
            # file that now extracts no text) must leave the old version
            # intact. Same atomic-swap doctrine as ingestion.ingest_file.
            vector_store.delete_document(old["file_sha"])
        counts["updated" if old is not None else "added"] += 1
    for src in plan.pruned:
        vector_store.delete_document(src)
        counts["pruned"] += 1
    return counts
