import os
import chromadb

from numpy import ndarray
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from smartscan import ItemEmbedding, BaseCluster, ClusterMetadata, Assignments, ClusterMerges, ItemId, TextEmbeddingProvider
from smartscan.classify import  IncrementalClusterer, calculate_cluster_accuracy
from smartscan.providers import  MiniLmTextEmbedder
from smartscan.embeds import EmbeddingStore
from smartscan.processor import ProcessorListener
from smartscan.embeds.types import ItemEmbeddingUpdate
from revelium.constants import MINILM_MODEL_PATH, DB_DIR, MINILM_MAX_TOKENS
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.types import Prompt, PromptMetadata
from revelium.tokens import embedding_token_cost
from revelium.schemas.llm import LLMClassificationResult
from revelium.prompts.indexer import PromptIndexer
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.providers.llm.openai import OpenAIClient
from revelium.providers.types import TextEmbeddingModel
from revelium.providers.embeddings.openai import OpenAITextEmbedder
from revelium.schemas.llm import LLMClientConfig
from revelium.providers.llm.llm_client import LLMClient
from revelium.schemas.revelium_config import ReveliumConfig
from revelium.utils import  paginated_read, paginated_read_until_empty


class Revelium():
    UNLABELLED = "unlabelled"
    CLUSTER_TYPE = "cluster"
    PROMPT_TYPE = "prompt"

    def __init__(self, 
        config: Optional[ReveliumConfig] = None,                 
        text_embedder: Optional[TextEmbeddingProvider] = None,
        llm_client: Optional[LLMClient] = None,
        clusterer: Optional[IncrementalClusterer] = None,
        indexer_listener: Optional[ProcessorListener] = None,
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
        
        self.indexer =  PromptIndexer(self.text_embedder, listener=indexer_listener, embeddings_store=self.prompt_embedding_store, batch_size=100, max_concurrency=4)
    
    async def index(self, prompts: List[Prompt]):
        return await self.indexer.run(prompts)
    
    def update_index_listenr(self, index_listener: ProcessorListener) -> bool:
        self.indexer.listener = index_listener
        return self.indexer.listener == index_listener
    
    def label_prompts(self, cluster_id: str, sample_size: int) -> LLMClassificationResult:
        existing_labels = self._get_all_cluster_labels()
        prompts = self.prompt_embedding_store.get(filter={"cluster_id": cluster_id},  limit=sample_size, include=['documents'])
        sample_prompts = [content for content in prompts.datas]
        input_prompt = self._get_prompt(cluster_id, existing_labels, sample_prompts)
        return self.llm.generate_json(input_prompt, LLMClassificationResult)

        
    def cluster(self):
        ids, embeddings = self._get_all_prompt_embeddings()
        return self.clusterer.cluster(ids, embeddings)
               
    async def update_prompts(self, assignments: Assignments, merges: ClusterMerges):
        prompt_ids = [str(k) for k in assignments.keys()]
        metadatas = self._get_prompts_metadata(prompt_ids)
        updated_at = datetime.now().isoformat()
        updated_prompts: list[ItemEmbedding] = []

        for prompt_id, metadata in zip(prompt_ids, metadatas):
            original_cluster = assignments[prompt_id]

            if not merges:
                new_cluster = original_cluster
            else:
                new_cluster = next(
                    (mid for mid, clusters in merges.items()
                    if original_cluster in clusters),
                    original_cluster,
                )

            updated_prompts.append(
                ItemEmbeddingUpdate(
                    prompt_id,
                    metadata=asdict(PromptMetadata(cluster_id=new_cluster, created_at=metadata.created_at, updated_at=updated_at))
                )
            )
        # print(f"length of updated: {len(updated_prompts)}")

        self.prompt_embedding_store.update(updated_prompts)


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

        self.cluster_embedding_store.update(cluster_embeddings)


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

    def _get_prompt(self, cluster_id: str, existing_labels: list[str], sample_prompts: list[str]):
        return f"""## ClusterId: {cluster_id}\n\n##Existing labels {existing_labels} Cluster sample_prompts \n\n {sample_prompts}"""
    
    def _get_all_prompt_embeddings(self):
        count = self.prompt_embedding_store.count()
        ids: list[str] = []
        embeddings: list[ndarray] = []
        for batch in paginated_read(
            lambda offset, limit: self.prompt_embedding_store.get(
                include=["embeddings"],
                offset=offset,
                limit=limit,
            ),
            total=count,
            limit=500,
            ):
            ids.extend(batch.ids)
            embeddings.extend(batch.embeddings)
        return ids, embeddings
    

    def _get_prompts_metadata(self, ids: list[str]):
        metadatas: list[PromptMetadata] = []
        for batch in paginated_read(
            lambda offset, limit: self.prompt_embedding_store.get(
                ids = ids,
                include=["metadatas"],
                offset=offset,
                limit=limit,
            ),
            total=len(ids),
            limit=500,
            ):
            metadatas.extend([ PromptMetadata(**m) for m in batch.metadatas])
        return metadatas
    
    def _get_all_cluster_labels(self):
        labels: list[str] = []
        for batch in paginated_read_until_empty(
            lambda offset, limit: self.cluster_embedding_store.get(
                include=['metadatas'], filter={"label": {"$ne": self.UNLABELLED}},
                limit=limit,
                offset=offset
                ),
            limit=500,
            ):
            labels.extend([m.get("label") for m in batch.metadatas])
        return labels