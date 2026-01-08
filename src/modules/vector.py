from qdrant_client import QdrantClient, models
import uuid

from src.config import settings

class VectorDB:
    def __init__(self, url):
        self.client = QdrantClient(url=url)
        self.col = "federated_docs"
        if not self.client.collection_exists(self.col):
            self.client.create_collection(
                self.col, 
                vectors_config=models.VectorParams(size=settings.EMBEDDING_DIMENSION, distance=models.Distance.COSINE)
            )

    def upsert(self, text, vector, meta, id=None):
        """Create / Update vector."""
        if id is None:
            id = str(uuid.uuid4())
            
        self.client.upsert(
            collection_name=self.col,
            points=[models.PointStruct(
                id=id, 
                vector=vector, 
                payload={"text": text, **meta}
            )]
        )

    def search(self, vector, limit=5, file_filter=None):
        """Read: Semantic search with optional file filtering."""
        query_filter = None
        if file_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=file_filter)
                    )
                ]
            )

        return self.client.query_points(
            collection_name=self.col, 
            query=vector, 
            limit=limit,
            query_filter=query_filter
        ).points

    def get_by_ids(self, ids):
        """Read: Retrieve specific points by ID."""
        if not ids:
            return []
        
        # Qdrant retrieve API
        points = self.client.retrieve(
            collection_name=self.col,
            ids=ids,
            with_payload=True,
            with_vectors=False
        )
        return points

    def clear(self):
        """Delete all vectors in the collection."""
        # Easiest way is to recreate collection or delete all points
        self.client.delete(
            collection_name=self.col,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[])
            )
        )