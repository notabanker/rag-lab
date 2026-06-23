from .chunker import Chunk, chunk
from .embedder import embed
from .embedder import shutdown as embedder_shutdown
from .retriever import retrieve
from .retriever import shutdown as retriever_shutdown
from .vector_store import init_store
from .vector_store import shutdown as vector_store_shutdown

VERSION = "0.1.0"


def shutdown():
    retriever_shutdown()
    embedder_shutdown()
    vector_store_shutdown()


__all__ = ["VERSION", "chunk", "Chunk", "embed", "retrieve", "init_store", "shutdown"]
