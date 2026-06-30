import hashlib
import logging
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from . import chunker, embedder, vector_store
from .metadata import extract_frontmatter, frontmatter_to_metadata
from .vault_config import load_exclude_patterns, resolve_layer, should_index

logger = logging.getLogger("rag_lab.watcher")


def _index_single_file(file_path: str, vault_root: str, embed_model: str = None):
    fp = Path(file_path)
    try:
        rel = str(fp.relative_to(vault_root))
    except ValueError:
        logger.warning(f"File outside vault root, skipping: {file_path}")
        return

    collection = resolve_layer(file_path, vault_root)
    if collection is None:
        logger.debug(f"Cannot resolve layer for {file_path}, skipping")
        return

    try:
        raw = fp.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return

    fm, body = extract_frontmatter(raw)

    if len(body) <= 1000:
        chunks = chunker.chunk(body, strategy="document")
    else:
        chunks = chunker.chunk(body, strategy="paragraph", max_chars=1000, overlap_paragraphs=1)

    if not chunks:
        return

    file_sha = hashlib.sha256(rel.encode()).hexdigest()[:12]
    meta_base = frontmatter_to_metadata(fm, rel, collection)

    metadatas = []
    ids = []
    for i, ch in enumerate(chunks):
        m = dict(meta_base)
        m["chunk_idx"] = i
        m["chunk_count"] = len(chunks)
        metadatas.append(m)
        ids.append(f"{file_sha}-{i}")

    vector_store.delete_by_source(file_sha, collection=collection)
    vecs = embedder.embed([c.text for c in chunks], model_name=embed_model)
    vector_store.upsert(chunks, vecs, metadatas, ids, collection=collection)

    logger.info(f"Indexed {rel} → {collection} ({len(chunks)} chunks)")


def _remove_single_file(file_path: str, vault_root: str):
    fp = Path(file_path)
    try:
        rel = str(fp.relative_to(vault_root))
    except ValueError:
        return

    collection = resolve_layer(file_path, vault_root)
    if collection is None:
        return

    file_sha = hashlib.sha256(rel.encode()).hexdigest()[:12]
    n = vector_store.delete_by_source(file_sha, collection=collection)
    if n:
        logger.info(f"Removed {n} chunks for deleted file {rel}")


class VaultIndexHandler(FileSystemEventHandler):
    def __init__(self, vault_root: str, embed_model: str = None, debounce_sec: float = 2.0):
        super().__init__()
        self.vault_root = str(Path(vault_root).resolve())
        self.embed_model = embed_model
        self.debounce_sec = debounce_sec
        self._pending: dict[str, float] = {}
        self._exclude = load_exclude_patterns(self.vault_root)

    def on_created(self, event):
        self._schedule(event.src_path)

    def on_modified(self, event):
        self._schedule(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith('.md'):
            return
        _remove_single_file(event.src_path, self.vault_root)

    def _schedule(self, path: str):
        if not path.endswith('.md'):
            return
        if not should_index(path, self.vault_root, self._exclude):
            return
        if path not in self._pending:
            self._pending[path] = time.time()

    def process_pending(self):
        now = time.time()
        ready = []
        for path, first_seen in list(self._pending.items()):
            if now - first_seen >= self.debounce_sec:
                ready.append(path)
                del self._pending[path]
        for path in ready:
            try:
                _index_single_file(path, self.vault_root, self.embed_model)
            except Exception as e:
                logger.error(f"Failed to index {path}: {e}")


def watch(vault_path: str, embed_model: str = None, debounce: float = 2.0):
    handler = VaultIndexHandler(vault_path, embed_model=embed_model, debounce_sec=debounce)
    observer = Observer()
    observer.schedule(handler, vault_path, recursive=True)
    observer.start()
    logger.info(f"Watching {vault_path} for changes...")
    try:
        while True:
            time.sleep(1)
            handler.process_pending()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
