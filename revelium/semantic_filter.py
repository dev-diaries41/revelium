
import numpy as np
from smartscan.providers import  TextEmbeddingProvider
from smartscan.processor import BatchProcessor, ProcessorListener

class SemanticContentFilter(BatchProcessor[list[tuple[int, str]], list[int]]):
    def __init__(self, text_embedder: TextEmbeddingProvider, threshold: float, criteria_embedding: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.text_embedder = text_embedder
        self.threshold = threshold
        self.criteria_embedding = criteria_embedding
        self.match_ids = []
        
    def on_batch_complete(self, batch):
        for ids in batch:
            self.match_ids.extend(ids)
        
    def on_process(self, item):
        ids, comments = zip(*((comment[0], comment[1]) for comment in item))
        batch_embeds = self.text_embedder.embed_batch(comments)
        sims = np.dot(batch_embeds, self.criteria_embedding)
        indices = [int(np.where(sims == sim)[0][0]) for sim in sims if sim > self.threshold]  
        return [ids[idx] for idx in indices]
    

class SemanticContentFilterListener(ProcessorListener[list[tuple[int, str]], list[int]]):
    def on_error(self, e, item):
        print(e)
    def on_batch_complete(self, batch):
        print(batch)