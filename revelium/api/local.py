import os
import chromadb

from openai.types import ResponsesModel
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from numpy import ndarray
from pydantic import  BaseModel, Field

from revelium.utils.decorators import with_time
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.types import Prompt
from revelium.tokens import embedding_token_cost
from revelium.schemas.label import LLMClassificationResult
from smartscan.classify import  IncrementalClusterer, calculate_cluster_accuracy
from smartscan import ItemEmbedding, BaseCluster, ClusterMetadata, Assignments, ClusterMerges, ItemId,  ModelName

from smartscan.classify import IncrementalClusterer
from smartscan.providers import  MiniLmTextEmbedder

from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.indexer_listener import DefaultIndexerListener
from revelium.prompts.store import AsyncSQLitePromptStore
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.providers.llm.openai import OpenAIClient
from revelium.providers.types import TextEmbeddingModel
from revelium.providers.embeddings.openai import OpenAITextEmbedder
from revelium.schemas.model import ModelConfig
from server.constants import MINILM_MODEL_PATH, DB_DIR

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_SYSTEM_PROMPT = "Your objective is to label prompt messages from clusters and label them, returning ClassificationResult. Labels should be one word max 3 words."
DEFAULT_CHROMADB_PATH = os.path.join(DB_DIR, "revelium_chromadb")
DEFAULT_PROMPTS_PATH = os.path.join(DB_DIR, "prompts.db")
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
MINILM_MAX_TOKENS = 512

class ReveliumConfig(BaseModel):
    api_key: Optional[str] = Field(default=None)
    text_embedder: TextEmbeddingModel = Field(default="all-minilm-l6-v2")
    provider_model: ResponsesModel = Field(default=DEFAULT_OPENAI_MODEL)
    provider_api_key: Optional[str] = Field(default=None)
    chromadb_path: str = Field(default=DEFAULT_CHROMADB_PATH)
    prompt_store_path: str = Field(default=DEFAULT_PROMPTS_PATH)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    benchmarking: bool = Field(default=False)

class Revelium():
    def __init__(self, config: ReveliumConfig):
        os.makedirs(DB_DIR, exist_ok=True)
        self.config = config
        self.llm = OpenAIClient(OPENAI_API_KEY, ModelConfig(model_name=DEFAULT_OPENAI_MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT))
        self.text_embedder = self._get_text_embedder(config.text_embedder, config.provider_api_key)
        self.chroma_client = chromadb.PersistentClient(path=config.chromadb_path, settings=chromadb.Settings(anonymized_telemetry=False))

        self.clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.9, benchmarking=config.benchmarking)
        
        cluster_embed_collection = self.chroma_client.get_or_create_collection(name=self._get_embedding_collection_name("cluster", config.text_embedder, self.text_embedder.embedding_dim))
        prompt_embed_collection = self.chroma_client.get_or_create_collection(name=self._get_embedding_collection_name("prompt", config.text_embedder, self.text_embedder.embedding_dim))
        self.cluster_embedding_store = ChromaDBEmbeddingStore(cluster_embed_collection)
        self.prompt_embedding_store = ChromaDBEmbeddingStore(prompt_embed_collection)
        self.prompt_store = AsyncSQLitePromptStore(config.prompt_store_path)
        self.indexer = PromptIndexer(self.text_embedder, listener=DefaultIndexerListener(), prompt_store=self.prompt_store, embeddings_store=self.prompt_embedding_store, batch_size=100, max_concurrency=4)

        self.api_key = config.api_key
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
            self.cluster_embedding_store.delete(list(merged_ids))

        self.cluster_embedding_store.upsert(cluster_embeddings)


    def calculate_cluster_accuracy(self, true_labels: Dict[ItemId, str],predicted_clusters: Assignments):
        return calculate_cluster_accuracy(true_labels, predicted_clusters)
    
    def calculate_prompt_cost(self, prompt: Prompt, price_per_1m_tokens: float, model: str | ModelName):
        return embedding_token_cost(prompt.content, price_per_1m_tokens, model)

    # helps ensure each collection get embeddings of the right size
    def _get_embedding_collection_name(self, type: str, model: TextEmbeddingModel, embed_dim: int):
        return f"{type}_{model}_{embed_dim}_collection"
    
    def _get_text_embedder(self, model: TextEmbeddingModel, provider_api_key: Optional[str] = None):
        if model == ("text-embedding-3-large" or "text-embedding-3-small"):
            if provider_api_key is None:
                raise ValueError("Missing OpenAI API key")
            return OpenAITextEmbedder(OPENAI_API_KEY, model=model)
        else:
            return MiniLmTextEmbedder(MINILM_MODEL_PATH, MINILM_MAX_TOKENS)

