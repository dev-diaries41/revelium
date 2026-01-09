import os

from dotenv import load_dotenv

load_dotenv()

from smartscan.classify import IncrementalClusterer
from smartscan.providers import  MiniLmTextEmbedder

from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.indexer_listener import DefaultIndexerListener
from revelium.prompts.store import AsyncSQLitePromptStore
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.api.local import Revelium
from revelium.providers.llm.openai import OpenAIClient
from revelium.schemas.model import ModelConfig

from server.constants import MINILM_MODEL_PATH

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_collection_name(model: str, embed_dim: int):
    return f"{model}_prompts_{embed_dim}"

def get_revelium_client(chromaclient, prompt_store_path, collection_name) -> Revelium:
    llm = OpenAIClient(OPENAI_API_KEY, ModelConfig(model_name="gpt-5-mini", system_prompt="Your objective is to label prompt messages from clusters and label them, returning ClassificationResult. Labels should be one word max 3 words."))
    minilm_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH, 512)
    minilm_embedder.init()
    clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.9, benchmarking=True)
    collection = chromaclient.get_or_create_collection(name=collection_name)
    embedding_store = ChromaDBEmbeddingStore(collection)
    prompt_store = AsyncSQLitePromptStore(prompt_store_path)
    indexer = PromptIndexer(minilm_embedder, listener=DefaultIndexerListener(), prompt_store=prompt_store, embeddings_store=embedding_store, batch_size=100, max_concurrency=4)
    return Revelium(
        text_embedder=minilm_embedder,
        prompt_store=prompt_store,
        embedding_store=embedding_store, 
        indexer=indexer, 
        clusterer=clusterer,
        llm=llm

        )
