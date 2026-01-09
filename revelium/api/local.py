from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from numpy import ndarray

from revelium.utils.decorators import with_time
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.store import PromptStore
from revelium.prompts.types import Prompt
from revelium.tokens import embedding_token_cost
from revelium.providers.llm.llm_client import LLMClient
from revelium.schemas.label import LLMClassificationResult
from smartscan.classify import  IncrementalClusterer, calculate_cluster_accuracy
from smartscan.providers import  TextEmbeddingProvider
from smartscan import ItemEmbedding, BaseCluster, ClusterMetadata, Assignments, ClusterMerges, ItemId,ClassificationResult,  ModelName
from smartscan.embeds import EmbeddingStore

## DEV ONLY

class Revelium():
    def __init__(
            self,
            text_embedder: TextEmbeddingProvider,
            prompt_store: PromptStore,
            embedding_store: EmbeddingStore,
            indexer: PromptIndexer,
            clusterer: IncrementalClusterer,
            llm: LLMClient,
            api_key:Optional[str] = None,
                 ):
        self.text_embedder = text_embedder
        self.prompt_store = prompt_store
        self.clusterer = clusterer
        self.indexer = indexer
        self.embedding_store = embedding_store
        self.llm = llm

        self.api_key = api_key
        if self.api_key:
            #TODO integrate paid api
            pass

    
    async def index(self, prompts: List[Prompt]):
        return await self.indexer.run(prompts)
    
    async def label_prompts(self, cluster_id: str, sample_size: int) -> LLMClassificationResult:
        prompts = await self.prompt_store.get(cluster_id=cluster_id, limit=sample_size)
        sample_prompts = [p.content for p in prompts]
        input_prompt = f"""## ClusterId: {cluster_id}\nCluster sample_prompts \n\n {sample_prompts}"""
        return self.llm.generate_json(input_prompt, LLMClassificationResult)
        
    def cluster(self, ids: List[str], embeddings: List[ndarray]):
        return self.clusterer.cluster(ids, embeddings)
               
    async def update_prompts(self, assignments: Assignments, merges: Optional[ClusterMerges] = None):
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
        print(f"length of updated: {len(updated_prompts)}")

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


    def calculate_cluster_accuracy(self, true_labels: Dict[ItemId, str],predicted_clusters: Assignments):
        return calculate_cluster_accuracy(true_labels, predicted_clusters)
    
    def calculate_prompt_cost(self, prompt: Prompt, price_per_1m_tokens: float, model: str | ModelName):
        return embedding_token_cost(prompt.content, price_per_1m_tokens, model)


