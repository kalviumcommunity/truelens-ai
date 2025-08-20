# core/vectorstore.py
import os
import faiss
import numpy as np
import pickle

INDEX_DIR = "data/index"
os.makedirs(INDEX_DIR, exist_ok=True)

_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
_META_PATH = os.path.join(INDEX_DIR, "metadatas.pkl")

def build_faiss_index(embeddings: np.ndarray, metadatas: list):
    """
    embeddings: np.ndarray (n, dim)
    metadatas: list of dicts (same len as embeddings)
    Saves index + metadatas to disk.
    """
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, _INDEX_PATH)
    with open(_META_PATH, "wb") as f:
        pickle.dump(metadatas, f)
    return index

def load_faiss_index():
    if not os.path.exists(_INDEX_PATH) or not os.path.exists(_META_PATH):
        return None, None
    index = faiss.read_index(_INDEX_PATH)
    with open(_META_PATH, "rb") as f:
        metadatas = pickle.load(f)
    return index, metadatas

def search(query_emb: np.ndarray, top_k: int = 4):
    index, metadatas = load_faiss_index()
    if index is None:
        raise RuntimeError("FAISS index not found. Build it first.")
    query_emb = query_emb.astype('float32').reshape(1, -1)
    D, I = index.search(query_emb, top_k)  # D = similarity scores, I = indices
    results = []
    for score, idx in zip(D[0], I[0]):
        md = metadatas[idx]
        results.append({"score": float(score), "metadata": md})
    return results