import logging
import sys


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root = logging.getLogger("rag_lab")
    root.setLevel(level)
    root.addHandler(handler)
    if not verbose:
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
