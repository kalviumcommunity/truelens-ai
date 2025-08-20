from sentence_transformers import SentenceTransformer
import numpy as np
from utils.config import EMBEDDING_MODEL

# load model once
_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL

def get_embeddings(texts, convert_to_numpy=True):
    """
    texts: list[str] or str
    returns: np.ndarray (n, dim)
    """
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    model = _get_model()
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    if single:
        return emb[0]
    return emb