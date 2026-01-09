import os
import chromadb

from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from smartscan import ItemEmbedding, BaseCluster, ClusterMetadata, Assignments, ClusterMerges, ItemId, TextEmbeddingProvider
from smartscan.classify import  IncrementalClusterer, calculate_cluster_accuracy
from smartscan.providers import  MiniLmTextEmbedder
from smartscan.embeds import EmbeddingStore
from smartscan.processor import ProcessorListener

from revelium.constants import MINILM_MODEL_PATH, DB_DIR, MINILM_MAX_TOKENS
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.types import Prompt
from revelium.tokens import embedding_token_cost
from revelium.schemas.llm import LLMClassificationResult
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.indexer_listener import ProgressBarIndexerListener
from revelium.prompts.store import AsyncSQLitePromptStore, PromptStore
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.providers.llm.openai import OpenAIClient
from revelium.providers.types import TextEmbeddingModel
from revelium.providers.embeddings.openai import OpenAITextEmbedder
from revelium.schemas.llm import LLMClientConfig
from revelium.providers.llm.llm_client import LLMClient
from revelium.schemas.revelium_config import ReveliumConfig
from revelium.utils.decorators import with_time


class Revelium():
    UNLABELLED = "unlabelled"  # class-level constant
    CLUSTER_TYPE = "cluster"
    PROMPT_TYPE = "prompt"

    def __init__(self, 
                
        config: Optional[ReveliumConfig] = None,                 
        text_embedder: Optional[TextEmbeddingProvider] = None,
        llm_client: Optional[LLMClient] = None,
        clusterer: Optional[IncrementalClusterer] = None,
        indexer_listener: Optional[ProcessorListener] = None,
        prompt_store: Optional[PromptStore] = None,
        cluster_embedding_store: Optional[EmbeddingStore] = None,
        prompt_embedding_store: Optional[EmbeddingStore] = None,
                 ):
        os.makedirs(DB_DIR, exist_ok=True)
        self.config = config or ReveliumConfig()
        self.text_embedder = text_embedder or self._get_text_embedder(self.config.text_embedder, self.config.provider_api_key)
        self.llm = llm_client or  OpenAIClient(self.config.provider_api_key, LLMClientConfig(model_name=self.config.provider_model, system_prompt=self.config.system_prompt))
        self.clusterer = clusterer or IncrementalClusterer(default_threshold=0.55, sim_factor=0.9, benchmarking=config.benchmarking)  
        
        if not cluster_embedding_store or prompt_embedding_store:
            client = chromadb.PersistentClient(path=self.config.chromadb_path, settings=chromadb.Settings(anonymized_telemetry=False))

        self.cluster_embedding_store = cluster_embedding_store or  ChromaDBEmbeddingStore(
            client.get_or_create_collection(
                self._get_embedding_collection_name(self.CLUSTER_TYPE, self.config.text_embedder, self.text_embedder.embedding_dim)
                )
            )
        self.prompt_embedding_store = prompt_embedding_store or ChromaDBEmbeddingStore(
                client.get_or_create_collection(
                    self._get_embedding_collection_name(self.PROMPT_TYPE, self.config.text_embedder, self.text_embedder.embedding_dim)
                )
            ) 
        
        self.prompt_store = prompt_store or AsyncSQLitePromptStore(self.config.prompt_store_path)
        self.indexer =  PromptIndexer(self.text_embedder, listener= indexer_listener or ProgressBarIndexerListener(), prompt_store=self.prompt_store, embeddings_store=self.prompt_embedding_store, batch_size=100, max_concurrency=4)
    
    async def index(self, prompts: List[Prompt]):
        return await self.indexer.run(prompts)
    
    async def label_prompts(self, cluster_id: str, sample_size: int) -> LLMClassificationResult:
        existing_labels = self._get_existing_labels()
        prompts = await self.prompt_store.get(cluster_id=cluster_id, limit=sample_size)
        sample_prompts = [p.content for p in prompts]
        input_prompt = self._get_prompt(cluster_id, existing_labels, sample_prompts)
        return self.llm.generate_json(input_prompt, LLMClassificationResult)
        
    def cluster(self):
        ids, embeddings = self._paginate_prompt_embed_store()
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
    
    def _paginate_prompt_embed_store(self):
        ids, embeddings = [], []
        limit = 500
        offset = 0
        count = self.prompt_embedding_store.count()

        while len(ids) < count:
            query_result = self.prompt_embedding_store.get(include=['embeddings'], offset=offset, limit=limit)
            ids.extend(query_result.ids)
            embeddings.extend(query_result.embeddings)
            offset += limit
        return ids, embeddings