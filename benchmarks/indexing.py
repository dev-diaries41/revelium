import json
import chromadb
import asyncio
import os

from dataclasses import asdict
from smartscan.providers import  MiniLmTextEmbedder
from revelium.prompts.prompt_indexer import PromptIndexer, DefaultPromptIndexerListener
from revelium.prompts.store import AsyncSQLitePromptStore
from revelium.prompts.types import Prompt
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.data import get_placeholder_prompts

from server.constants import MINILM_MODEL_PATH, DB_DIR


BENCHMARK_DIR = "output/benchmarks"
BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "indexing_speed.jsonl")
BENCHMARK_PROMPT_STORE_PATH = os.join(BENCHMARK_DIR, "prompts.db")
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking
async def main(labelled_prompts: list[Prompt]):
    text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH, 512)
    text_embedder.init()
    client = chromadb.PersistentClient(path=DB_DIR, settings=chromadb.Settings(anonymized_telemetry=False))
    prompt_store = AsyncSQLitePromptStore("db/prompts.db")
    collection = client.get_or_create_collection(name=f"cluster_collection", metadata={"description": "Cluster Collection"})
    embedding_store = ChromaDBEmbeddingStore(collection)
    indexer = PromptIndexer(text_embedder, 512, listener=DefaultPromptIndexerListener(), prompt_store=prompt_store, embeddings_store=embedding_store)
    
    index_results = await indexer.run(labelled_prompts)
    print(f"time_elpased: {result.time_elapsed} | processed: {result.total_processed}")

    result = {k: v for k, v in asdict(index_results).items() if k != "error"}

    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(result, indent=None) + "\n")


asyncio.run(main(get_placeholder_prompts()))