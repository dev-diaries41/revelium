from smartscan.embeds import EmbeddingStore
from smartscan import GetResult, QueryResult
from typing import Any
import chromadb

# Consider changing EmbeddingStore to return dict[str, List]
class ChromaDBEmbeddingStore(EmbeddingStore[Any, chromadb.CollectionMetadata]):
    def __init__(self, chroma_colletion: chromadb.Collection):
        self.chroma_colletion = chroma_colletion

    def add(self, items):
        ids, embeddings, metadatas = [], [], []
        for item in items:
            ids.append(item.item_id)
            embeddings.append(item.embedding)
            metadatas.append(item.metadata)
        self.chroma_colletion.add(ids, embeddings, metadatas)
    
    def get(self, ids = None, filter = None, limit = None, offset = None, include = ["metadatas"]) ->  GetResult:
        result = self.chroma_colletion.get(ids, where=filter, limit=limit, offset=offset, include = include or [])
        return GetResult(ids=result["ids"], embeddings=result['embeddings'], metadatas=result['metadatas'])
    
    def update(self, items):
        ids, embeddings, metadatas = [], [], []
        for item in items:
            ids.append(item.item_id)
            embeddings.append(item.embedding)
            metadatas.append(item.metadata)
        self.chroma_colletion.update(ids, embeddings, metadatas)
    
    def upsert(self, items):
        ids, embeddings, metadatas = [], [], []
        for item in items:
            ids.append(item.item_id)
            embeddings.append(item.embedding)
            metadatas.append(item.metadata)
        return self.chroma_colletion.upsert(ids, embeddings, metadatas)
    
    def delete(self, ids = None, filter = None):
        return self.chroma_colletion.delete(ids,filter)
    
    def query(self, query_embeds, filter = None, limit = 10, include = ["metadatas"]) -> GetResult:
        result =  self.chroma_colletion.query(query_embeddings=query_embeds, where=filter, include=include, n_results=limit)
        return QueryResult(ids=result["ids"], embeddings=result['embeddings'], metadatas=result['metadatas'], sims=result['distances'])
    
    def count(self, filter = None):
        return self.chroma_colletion.count()      