import os
# Force CPU (no GPU assumed)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

_MODELS = {}

def model_name() -> str:
    return EMBEDDING_MODEL

def _prefixed(texts: list[str], input_type: str) -> list[str]:
    """Apply model-specific query/document prefixes where the model expects them."""
    model = model_name().lower()
    if "e5" not in model:
        return texts
    prefix = "query: " if input_type == "query" else "passage: "
    return [t if t.lower().startswith(("query: ", "passage: ")) else prefix + t for t in texts]

def get_model():
    name = model_name()
    if name not in _MODELS:
        _MODELS[name] = SentenceTransformer(name, device="cpu")
    return _MODELS[name]

def embed(texts: list[str], batch_size: int = None, input_type: str = "document") -> list[list[float]]:
    if input_type not in {"document", "query"}:
        raise ValueError("input_type must be 'document' or 'query'")
    model = get_model()
    vectors = model.encode(
        _prefixed(texts, input_type),
        batch_size=batch_size or EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()

def embed_query(text: str) -> list[float]:
    return embed([text], input_type="query")[0]
