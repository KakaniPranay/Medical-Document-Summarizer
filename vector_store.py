# vector_store.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Tuple, Dict

class FaissStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # build empty index after seeing a sample embedding
        sample = self.model.encode("hello", convert_to_numpy=True)
        self.dim = sample.shape[0]
        self.index = faiss.IndexFlatIP(self.dim)
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []

    def reset(self):
        # re-create index and clear texts
        self.index = faiss.IndexFlatIP(self.dim)
        self.texts = []
        self.metadatas = []

    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        if not texts:
            return
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        self.index.add(embs)
        self.texts.extend(texts)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in texts])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append((self.texts[idx], self.metadatas[idx]))
        return results