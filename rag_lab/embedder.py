import os
# Force CPU (no GPU assumed)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _MODEL

def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()
