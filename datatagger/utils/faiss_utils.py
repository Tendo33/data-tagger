import os
import pickle
from typing import List, Optional, Tuple

import faiss
import numpy as np


class FaissClient:
    def __init__(
        self,
        index_file: str = "faiss.index",
        meta_file: str = "faiss_meta.pkl",
        dim: int = 1024,
    ):
        self.index_file = index_file
        self.meta_file = meta_file
        self.dim = dim
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "rb") as f:
                self.metas = pickle.load(f)
        else:
            self.metas = []

    def insert_embeddings(
        self, embeddings: List[List[float]], metas: Optional[List[str]] = None
    ):
        n = len(embeddings)
        if metas is None:
            metas = ["" for _ in range(n)]
        self.index.add(np.array(embeddings, dtype="float32"))
        self.metas.extend(metas)
        self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metas, f)

    def search(self, embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        distances, indices = self.index.search(
            np.array([embedding], dtype="float32"), top_k
        )
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metas):
                results.append((self.metas[idx], float(dist)))
        return results

    def get_embedding_by_meta(self, meta: str) -> Optional[List[float]]:
        """
        根据 meta 内容返回对应的 embedding（向量），如有多个匹配返回第一个，找不到返回 None。
        """
        if meta in self.metas:
            idx = self.metas.index(meta)
            return self.index.reconstruct(idx).tolist()
        return None
