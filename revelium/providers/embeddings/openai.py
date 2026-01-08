
import numpy as np

from typing import Literal
from smartscan import TextEmbeddingProvider    
from openai import OpenAI


class OpenAITextEmbedder(TextEmbeddingProvider):
    def __init__(self, api_key: str, model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-small", max_tokenizer_length: int = 8191):
        self.openai = OpenAI(api_key=api_key)
        self._embedding_dim = 3072 if model == "text-embedding-3-large" else 1536
        self._max_len = max_tokenizer_length
        self.model = model
        
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, data: str) -> np.ndarray:
        response = self.openai.embeddings.create(
            input=data,
            model=self.model
        )
        return response.data[0].embedding
    

    def embed_batch(self, data: list[str])-> np.ndarray:
        return np.stack([self.embed(text) for text in data], axis=0)
    
    def close_session(self):
        pass

    def init(self):
        pass
    
    def is_initialized(self):
        return True
    