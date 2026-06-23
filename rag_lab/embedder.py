import os
import threading

from sentence_transformers import SentenceTransformer

from .config import get_embed_model

_MODELS: dict[str, SentenceTransformer] = {}
_lock = threading.Lock()


def _configure_env():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


_configure_env()


def get_model(model_name: str = None) -> SentenceTransformer:
    model_name = model_name or get_embed_model()
    if model_name not in _MODELS:
        with _lock:
            if model_name not in _MODELS:
                _MODELS[model_name] = SentenceTransformer(model_name, device="cpu")
    return _MODELS[model_name]


def embed(texts: list[str], model_name: str = None, batch_size: int = 32) -> list[list[float]]:
    model = get_model(model_name)
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def shutdown():
    global _MODELS
    with _lock:
        _MODELS.clear()
