from typing import List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections


class MilvusClient:
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "embeddings",
        dim: int = 768,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self._connect()
        self._create_collection_if_not_exists()

    def _connect(self):
        connections.connect(alias="default", host=self.host, port=self.port)

    def _create_collection_if_not_exists(self):
        if self.collection_name in [col for col in Collection.list()]:
            self.collection = Collection(self.collection_name)
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(
                name="meta", dtype=DataType.VARCHAR, max_length=512, is_primary=False
            ),
        ]
        schema = CollectionSchema(fields, description="Embedding collection")
        self.collection = Collection(self.collection_name, schema)
        self.collection.load()

    def insert_embeddings(
        self, embeddings: List[List[float]], metas: Optional[List[str]] = None
    ):
        n = len(embeddings)
        if metas is None:
            metas = [""] * n
        # None 对应 id
        self.collection.insert([None, embeddings, metas])
        self.collection.flush()

    def search(self, embedding: List[float], top_k: int = 5):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["meta"],
        )
        return results
