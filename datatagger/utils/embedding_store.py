from typing import List, Optional
from datatagger.utils.faiss_utils import FaissClient
from datatagger.utils.milvus_utils import MilvusClient


class EmbeddingStore:
    def __init__(
        self,
        faiss_client: Optional[FaissClient] = None,
        milvus_client: Optional[MilvusClient] = None,
    ):
        self.faiss_client = faiss_client
        self.milvus_client = milvus_client

    def insert(
        self,
        embeddings: List[List[float]],
        metas: Optional[List[str]] = None,
        use_faiss: bool = False,
        use_milvus: bool = False,
    ):
        if use_faiss and self.faiss_client:
            self.faiss_client.insert_embeddings(embeddings, metas)
        if use_milvus and self.milvus_client:
            self.milvus_client.insert_embeddings(embeddings, metas)
