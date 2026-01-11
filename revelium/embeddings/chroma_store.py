from smartscan.embeds import EmbeddingStore
from smartscan import GetResult, QueryResult
import chromadb

# Consider changing EmbeddingStore to return dict[str, List]
class ChromaDBEmbeddingStore(EmbeddingStore[str, chromadb.CollectionMetadata]):
    def __init__(self, chroma_colletion: chromadb.Collection):
        self.chroma_colletion = chroma_colletion

    def add(self, items):
        ids, embeddings, metadatas, documents = [], [], [], []
        for item in items:
            ids.append(item.item_id)
            embeddings.append(item.embedding)
            metadatas.append(item.metadata)
            documents.append(item.data)
        self.chroma_colletion.add(ids, embeddings, metadatas, documents=documents)
    
    def get(self, ids = None, filter = None, limit = None, offset = None, include = ["metadatas", "embeddings"]) ->  GetResult[str, chromadb.CollectionMetadata]:
        result = self.chroma_colletion.get(ids, where=filter, limit=limit, offset=offset, include = include)
        return GetResult(ids=result["ids"], embeddings=result['embeddings'], metadatas=result['metadatas'], datas=result['documents'])
    
    # Updates prevent embeddings
    def update(self, items):
        ids, embeddings, metadatas = [], [], []
        has_any_embedding = any(item.embedding is not None for item in items)

        for item in items:
            ids.append(item.item_id)
            metadatas.append(item.metadata)
            if has_any_embedding:
                embeddings.append(item.embedding)
        
        if has_any_embedding:
            self.chroma_colletion.update(ids, embeddings, metadatas)
        else:
            self.chroma_colletion.update(ids, metadatas=metadatas)

    
    def delete(self, ids = None, filter = None):
        return self.chroma_colletion.delete(ids,filter)
    
    def query(self, query_embeds, filter = None, limit = 10, include = ["metadatas", "embeddings"]) -> QueryResult[str, chromadb.CollectionMetadata]:
        result = self.chroma_colletion.query(query_embeddings=query_embeds, where=filter, include=include, n_results=limit)
        return QueryResult(ids=result["ids"], embeddings=result['embeddings'], metadatas=result['metadatas'], sims=result['distances'], datas=result['documents'])
    
    def count(self, filter = None):
        return self.chroma_colletion.count()      