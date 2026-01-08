from datetime import datetime
from typing import List, Dict, Any
from numpy import ndarray

from revelium.utils import with_time
from revelium.prompts.prompt_indexer import PromptIndexer
from revelium.prompts.store import PromptStore
from revelium.prompts.types import Prompt
from revelium.tokens import embedding_token_cost

from smartscan.classify import  IncrementalClusterer, calculate_cluster_accuracy
from smartscan.providers import  TextEmbeddingProvider
from smartscan import ItemEmbedding, BaseCluster, ClusterMetadata, Assignments, ModelName
from smartscan.embeds import EmbeddingStore

## DEV ONLY

class ReveliumLocalClient():
    def __init__(
            self,
            text_embedder: TextEmbeddingProvider,
            prompt_store: PromptStore,
            embedding_store: EmbeddingStore,
            indexer: PromptIndexer,
            clusterer: IncrementalClusterer,
                 ):
        self.text_embedder = text_embedder
        self.prompt_store = prompt_store
        self.clusterer = clusterer
        self.indexer = indexer
        self.embedding_store = embedding_store
    
    async def index(self, prompts: List[Prompt]):
        return await self.indexer.run(prompts)
    
    def cluster(self, ids: List[str], embeddings: List[ndarray]):
        return self.clusterer.cluster(ids, embeddings)
               
    def update_prompts(self, assignments: Assignments):
        prompt_ids = assignments.keys()
        prompts = self.prompt_store.get_by_ids(prompt_ids)
        updated_promtps = [Prompt(prompt_id=p.prmopt_id, content=p.content, created_at=p.created_at, updated_at= datetime.now(), cluster_id=assignments[p.prompt_id])  for p in prompts]
        self.prompt_store.update(updated_promtps)

    def update_clusters(self, clusters: Dict[str, BaseCluster]):
        cluster_embeddings = [ItemEmbedding[Any, ClusterMetadata](c.prototype_id, c.embedding, metadata={"prototype_size":c.prototype_size, "cohesion_score": c.cohesion_score}) for c in clusters.values()]
        self.embedding_store.upsert(cluster_embeddings)

    def calculate_cluster_accuracy(self, labelled_cluster_counts: Dict[str, int]):
        cluster_ids = labelled_cluster_counts.keys()
        predict_count = {}
        for cluster_id in cluster_ids:
            predict_count[cluster_id] = self.prompt_store.count(cluster_id=cluster_id)
        return calculate_cluster_accuracy(labelled_cluster_counts, predict_count)
    
    def calculate_prompt_cost(self, prompt: Prompt, price_per_1m_tokens: float, model: str | ModelName):
        return embedding_token_cost(prompt.content, price_per_1m_tokens, model)


