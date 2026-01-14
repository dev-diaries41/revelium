
from revelium.providers.types import TextEmbeddingModel
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
import chromadb

# helps ensure each collection get embeddings of the right size
def get_embedding_collection_name( type: str, model: TextEmbeddingModel, embed_dim: int) -> str:
    return f"{type}_{model}_{embed_dim}_collection"
    
def get_embedding_store(chroma_path, prefix:str, model: TextEmbeddingModel, embedding_dim: int ):
    client = chromadb.PersistentClient(path=chroma_path, settings=chromadb.Settings(anonymized_telemetry=False))
    return ChromaDBEmbeddingStore(client.get_or_create_collection(
                get_embedding_collection_name(prefix, model, embedding_dim))
            ) 