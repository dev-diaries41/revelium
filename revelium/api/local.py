import os
import chromadb

from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from numpy import ndarray

from smartscan import ItemEmbedding, BaseCluster, ClusterMetadata, Assignments, ClusterMerges, ItemId
from smartscan.classify import  IncrementalClusterer, calculate_cluster_accuracy
from smartscan.providers import  MiniLmTextEmbedder

from revelium.constants import MINILM_MODEL_PATH, DB_DIR, MINILM_MAX_TOKENS
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.types import Prompt
from revelium.tokens import embedding_token_cost
from revelium.schemas.llm import LLMClassificationResult
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.indexer_listener import ProgressBarIndexerListener
from revelium.prompts.store import AsyncSQLitePromptStore
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.providers.llm.openai import OpenAIClient
from revelium.providers.types import TextEmbeddingModel
from revelium.providers.embeddings.openai import OpenAITextEmbedder
from revelium.schemas.llm import LLMClientConfig
from revelium.schemas.revelium_config import ReveliumConfig
from revelium.utils.decorators import with_time


class Revelium():
    UNLABELLED = "unlabelled"  # class-level constant

    def __init__(self, config: ReveliumConfig):
        os.makedirs(DB_DIR, exist_ok=True)
        self.config = config
        self.text_embedder = self._get_text_embedder(config.text_embedder, config.provider_api_key)
        self.llm = OpenAIClient(config.provider_api_key, LLMClientConfig(model_name=config.provider_model, system_prompt=config.system_prompt))
        self.clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.9, benchmarking=config.benchmarking)
        
        chroma_client = chromadb.PersistentClient(path=config.chromadb_path, settings=chromadb.Settings(anonymized_telemetry=False))
        cluster_embed_collection = chroma_client.get_or_create_collection(name=self._get_embedding_collection_name("cluster", config.text_embedder, self.text_embedder.embedding_dim))
        prompt_embed_collection = chroma_client.get_or_create_collection(name=self._get_embedding_collection_name("prompt", config.text_embedder, self.text_embedder.embedding_dim))
        
        self.cluster_embedding_store = ChromaDBEmbeddingStore(cluster_embed_collection)
        self.prompt_embedding_store = ChromaDBEmbeddingStore(prompt_embed_collection)
        
        self.prompt_store = AsyncSQLitePromptStore(config.prompt_store_path)
        
        self.indexer = PromptIndexer(self.text_embedder, listener=ProgressBarIndexerListener(), prompt_store=self.prompt_store, embeddings_store=self.prompt_embedding_store, batch_size=100, max_concurrency=4)
    
    async def index(self, prompts: List[Prompt]):
        return await self.indexer.run(prompts)
    
    async def label_prompts(self, cluster_id: str, sample_size: int) -> LLMClassificationResult:
        existing_labels = self._get_existing_labels()
        prompts = await self.prompt_store.get(cluster_id=cluster_id, limit=sample_size)
        sample_prompts = [p.content for p in prompts]
        input_prompt = self._get_prompt(cluster_id, existing_labels, sample_prompts)
        return self.llm.generate_json(input_prompt, LLMClassificationResult)
        
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
        # print(f"length of updated: {len(updated_prompts)}")

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
                metadata={**asdict(c.metadata), "label": c.label or self.UNLABELLED} 
            )
            for c in effective_clusters.values()
        ]

        if merges:
            self.cluster_embedding_store.delete(list(merged_ids))

        self.cluster_embedding_store.upsert(cluster_embeddings)


    def calculate_cluster_accuracy(self, true_labels: Dict[ItemId, str],predicted_clusters: Assignments):
        return calculate_cluster_accuracy(true_labels, predicted_clusters)
    
    def calculate_prompt_cost(self, prompt: Prompt, price_per_1m_tokens: float, model: str | TextEmbeddingModel):
        return embedding_token_cost(prompt.content, price_per_1m_tokens, model)

    # helps ensure each collection get embeddings of the right size
    def _get_embedding_collection_name(self, type: str, model: TextEmbeddingModel, embed_dim: int):
        return f"{type}_{model}_{embed_dim}_collection"
    
    def _get_text_embedder(self, model: TextEmbeddingModel, provider_api_key: Optional[str] = None):
        if model == ("text-embedding-3-large" or "text-embedding-3-small"):
            if provider_api_key is None:
                raise ValueError("Missing OpenAI API key")
            return OpenAITextEmbedder(provider_api_key, model=model)
        else:
            return MiniLmTextEmbedder(MINILM_MODEL_PATH, MINILM_MAX_TOKENS)

    def _get_existing_labels(self) -> list[str]:
        q = self.cluster_embedding_store.get(include=['metadatas'], filter={"$ne": self.UNLABELLED})
        return [m.get("label") for m in q.metadatas]

    def _get_prompt(self, cluster_id: str, existing_labels: list[str], sample_prompts: list[str]):
        return f"""## ClusterId: {cluster_id}\n\n##Existing labels {existing_labels} Cluster sample_prompts \n\n {sample_prompts}"""
