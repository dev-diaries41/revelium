from datetime import datetime
from typing import List, Dict, Any
from dataclasses import asdict
from numpy import ndarray

from revelium.utils import with_time
from revelium.prompts.prompt_indexer import PromptIndexer
from revelium.prompts.store import PromptStore
from revelium.prompts.types import Prompt
from revelium.tokens import embedding_token_cost

from smartscan.classify import  IncrementalClusterer, calculate_cluster_accuracy
from smartscan.providers import  TextEmbeddingProvider
from smartscan import ItemEmbedding, BaseCluster, ClusterMetadata, Assignments, ClusterMerges, ModelName
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
               
    async def update_prompts(self, assignments: Assignments, merges: ClusterMerges):
        prompt_ids = [str(k) for k in assignments.keys()]
        prompts = await self.prompt_store.get_by_ids(prompt_ids)
        updated_at = datetime.now()
        updated_prompts: list[Prompt] = []

        for p in prompts:
            original_cluster = assignments[p.prompt_id]

            if not merges:
                new_cluster = original_cluster
            else:
                new_cluster = next(
                    (mid for mid, clusters in merges.items()
                    if original_cluster in clusters),
                    original_cluster,
                )

            updated_prompts.append(
                Prompt(
                    p.prompt_id,
                    p.content,
                    created_at=p.created_at,
                    updated_at=updated_at,
                    cluster_id=new_cluster,
                )
            )

        await self.prompt_store.update(updated_prompts)


    def update_clusters(self, clusters: Dict[str, BaseCluster], merges: ClusterMerges | None = None):
        """
        Update the embedding store with clusters, applying merges if provided.
        Old clusters that have been merged are removed from the store.
        """
        effective_clusters: Dict[str, BaseCluster] = clusters.copy()

        if merges:
            merged_ids = {cid for targets in merges.values() for cid in targets}
            for mid in merged_ids:
                effective_clusters.pop(mid, None)

        cluster_embeddings = [
            ItemEmbedding[Any, ClusterMetadata](
                c.prototype_id,
                c.embedding,
                metadata={**asdict(c.metadata), "label": c.label}  # include label in stored metadata
            )
            for c in effective_clusters.values()
        ]

        if merges:
            self.embedding_store.delete(list(merged_ids))

        self.embedding_store.upsert(cluster_embeddings)


    def calculate_cluster_accuracy(self, labelled_cluster_counts: Dict[str, int]):
        cluster_ids = labelled_cluster_counts.keys()
        predict_count = {}
        for cluster_id in cluster_ids:
            predict_count[cluster_id] = self.prompt_store.count(cluster_id=cluster_id)
        return calculate_cluster_accuracy(labelled_cluster_counts, predict_count)
    
    def calculate_prompt_cost(self, prompt: Prompt, price_per_1m_tokens: float, model: str | ModelName):
        return embedding_token_cost(prompt.content, price_per_1m_tokens, model)


